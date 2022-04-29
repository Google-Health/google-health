"""Tools for performing statistical analysis on ROC curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import scipy.stats
import sklearn.metrics


def _binarize(x, variable_name='values'):
  """Casts a boolean vector to {0, 1} and validates its values."""
  binarized = np.array(x, dtype=np.int32)
  if set(binarized) - set([0, 1]):
    raise ValueError('%s must be in {0, 1}' % variable_name)
  return binarized


def _delong_covariance(y_true, y_scores, sample_weight=None):
  """Estimates the covariance matrix for a set of ROC-AUC scores."""

  y_true = _binarize(y_true, 'true labels')

  if sample_weight is None:
    sample_weight = np.ones_like(y_true, dtype=np.int32)
  else:
    sample_weight = _binarize(sample_weight, 'sample weight')
    if len(sample_weight) != len(y_true):
      raise ValueError()
    elif not sample_weight.sum():
      raise ValueError('No nonzero weights found.')

  y_scores = np.array(y_scores)
  if y_scores.ndim == 1:
    # If there's just one score, add a singleton dimension.
    y_scores = np.expand_dims(y_scores, 1)
  elif y_scores.ndim != 2:
    raise ValueError('Unexpected shape for y_scores: %r' % y_scores.shape)

  num_obs, num_scores = y_scores.shape
  if num_obs != len(y_true):
    raise ValueError('y_true and y_scores must have the same length!')

  y_scores_valid = y_scores[sample_weight == 1, :]
  y_true_valid = y_true[sample_weight == 1]
  num_positives = y_true_valid.sum()
  num_negatives = len(y_true_valid) - num_positives

  point_estimates = []
  d_01s = []
  d_10s = []
  for score_idx in range(num_scores):
    score_pos_neg_matrix = np.array((
        range(len(y_true_valid)),
        y_true_valid,
        1 - y_true_valid,
        y_scores_valid[:, score_idx],
    ),
                                    dtype=np.float64)
    # Positives and negatives are sorted by the score while avoiding bias.
    # Using two ways to break the tie to take the average later.
    # 1. Treat positive larger than negative for tie breaking.
    neg_first_order = score_pos_neg_matrix[:3,
                                           np.lexsort(score_pos_neg_matrix[
                                               1::2, :])]
    # 2. Treat negative larger than positive for tie breaking.
    pos_first_order = score_pos_neg_matrix[:3, np.lexsort(score_pos_neg_matrix)]

    # Up to each point, how many positives and negatives are there.
    cumsum_neg_first_order = neg_first_order[1:].cumsum(axis=1)
    cumsum_pos_first_order = pos_first_order[1:].cumsum(axis=1)

    # For each positive, how many negatives are equal or smaller .
    le_neg_count = cumsum_neg_first_order[1, neg_first_order[1, :] > 0]
    # For each positive, how many negatives are smaller.
    lt_neg_count = cumsum_pos_first_order[1, pos_first_order[1, :] > 0]
    # For each negative, how many positives are equal or greater.
    ge_pos_count = num_positives - cumsum_neg_first_order[
        0, neg_first_order[2, :] > 0]
    # For each negative, how many positives are greater.
    gt_pos_count = num_positives - cumsum_pos_first_order[
        0, pos_first_order[2, :] > 0]

    # Taking the average of the two count methods.
    d01 = (le_neg_count + lt_neg_count) / 2 / num_negatives
    d10 = (ge_pos_count + gt_pos_count) / 2 / num_positives

    # Sorting by index to restore original order.
    d01 = d01[np.argsort(neg_first_order[0, neg_first_order[1, :] > 0])]
    d10 = d10[np.argsort(neg_first_order[0, neg_first_order[2, :] > 0])]

    # Equivalent to sklearn.metrics.roc_auc_score(y_true, y_score).
    point_estimates.append(d01.mean())

    # The notation d_01 and d_10 comes from [2]; see docstring.
    # For each positive score, the fraction of negatives that are smaller.
    d_01s.append(d01)

    # For each negative score, the fraction of positives that are larger.
    d_10s.append(d10)

  s_01 = np.cov(d_01s, ddof=1)
  s_10 = np.cov(d_10s, ddof=1)
  covariance_matrix = s_01 / num_positives + s_10 / num_negatives

  return point_estimates, covariance_matrix


def delong_interval(y_true, y_score, sample_weight=None, coverage=0.95):
  """Computes a confidence interval on the AUC-ROC using DeLong's method.

  See [1] for the original formulation, and [2] for discussion/simulation.

  [1] DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or
  more correlated receiver operating characteristic cuvers: a nonparametric
  approach. Biometrics. 1988;44: 837-845.

  [2] Gensheng Qin, Hotilovac L. Comparison of non-parametric confidence
  intervals for the area under the ROC curve of a continuous-score diagnostic
  test. Stat Methods Med Res. 2008;17: 207-221.

  Args:
    y_true: An array of boolean outcomes.
    y_score: An array of continuous-valued scores. Ordinal values are
      acceptable.
    sample_weight: An optional mask of binary sample weights. Must have nonzero
      entries.
    coverage: The size of the confidence interval. Should be in (0, 1]. The
      default is 95%.

  Returns:
    (lower, upper) the endpoints of an equitailed confidence interval for
    the area under the ROC curve.

  Raises:
    ValueError: if inputs are invalid.
  """
  if coverage > 1.0 or coverage <= 0.0:
    raise ValueError('coverage level must be in (0, 1]')

  point_estimates, covariance_matrix = _delong_covariance(
      y_true, y_score, sample_weight=sample_weight)

  point_estimate = point_estimates[0]
  variance = float(covariance_matrix)

  standard_error = np.sqrt(variance)
  z = scipy.stats.norm.isf((1 - coverage) / 2.0)
  lower = max(point_estimate - z * standard_error, 0.0)
  upper = min(point_estimate + z * standard_error, 1.0)
  return (lower, upper)


class TestResult(
    collections.namedtuple('TestResult',
                           ['effect', 'ci', 'statistic', 'pvalue'])):
  """The result of a hypothesis test."""


def _one_sided_p_value(z):
  """Computes the 1-sided p-value for the standard normal distribution."""
  return scipy.stats.norm.sf(z)


def _two_sided_p_value(z):
  """Computes the 2-sided p-value for the standard normal distribution."""
  return 2 * scipy.stats.norm.cdf(-np.abs(z))


def delong_test(y_true,
                y_score_1,
                y_score_2,
                sample_weight=None,
                coverage=0.95,
                margin=0.0):
  """Compares the area under two correlated ROC curves using DeLong's method.

  The curves are correlated in the sense that they are based on the same set
  of underlying truth values.

  DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or
  more correlated receiver operating characteristic cuvers: a nonparametric
  approach. Biometrics. 1988;44: 837-845.

  Args:
    y_true: An array of boolean outcomes.
    y_score_1: An array of continuous-valued scores. Ordinal values are
      acceptable.
    y_score_2: A second array of continuous-valued scores. Ordinal values are
      acceptable.
    sample_weight: An optional mask of binary sample weights. Must have nonzero
      entries.
    coverage: The size of the confidence interval for the difference between the
      AUC produced by `y_score_2` and `y_score_1`. Should be in (0, 1]. The
      default is 95%.
    margin: A positive noninferiority margin. When supplied and nonzero, the
      p-value refers to the one-sided test of the null hypothesis in which
      score_2 is at least this much worse than score_1.

  Returns:
    A named tuple with fields:
      effect: The estimated difference in the AUC-ROC between the
        two scores. A positive value means that y_score_2 is more discriminating
        than y_score_1.
      ci: A (lower, upper) confidence interval for the true difference in AUCs.
      statistic: The value of the z-statistic.
      pvalue: The p-value associated with the test. Unless a non-inferiority
        is specified, this is two-tailed.

  Raises:
    ValueError: if inputs are invalid.
  """
  if coverage > 1.0 or coverage <= 0.0:
    raise ValueError('coverage level must be in (0, 1]')

  point_estimates, covariance_matrix = _delong_covariance(
      y_true=y_true,
      y_scores=np.column_stack((y_score_1, y_score_2)),
      sample_weight=sample_weight)

  var1, var2 = np.diag(covariance_matrix)
  cov, = np.diag(covariance_matrix, 1)

  observed_effect_size = point_estimates[1] - point_estimates[0]
  variance = var1 + var2 - 2 * cov
  if not variance:
    raise ValueError('Variance estimate is zero! Are the scores equivalent?')
  standard_error = np.sqrt(variance)
  z_statistic = (observed_effect_size + margin) / standard_error

  if margin:
    p_value = _one_sided_p_value(z_statistic)
  else:
    p_value = _two_sided_p_value(z_statistic)

  z = scipy.stats.norm.isf((1 - coverage) / 2.0)
  lower = max(observed_effect_size - z * standard_error, -1.0)
  upper = min(observed_effect_size + z * standard_error, 1.0)
  return TestResult(
      effect=observed_effect_size,
      ci=(lower, upper),
      statistic=z_statistic,
      pvalue=p_value)


def lroc_curve(y_true, y_score, localization_success, sample_weight):
  """Computes points on the localization ROC (LROC) curve.

  For each score, true positives require correct localization.

  See https://pubmed.ncbi.nlm.nih.gov/24174485/

  Args:
    y_true: An array of Boolean outcomes.
    y_score: An array of continuous-valued scores. Ordinal values are
      acceptable.
    localization_success: A Boolean indicator of localization success for each
      example. Note that values for which `y_true` is 0 will be ignored and can
      be null.
    sample_weight: An optional mask of binary sample weights. Must have nonzero
      entries.

  Returns:
    (fpr, tpr, threshes) just like sklearn.metric.roc_curve.
  """

  y_true = _binarize(y_true)
  sample_weight = np.array(sample_weight, dtype=np.float32)
  localization_success = np.array(localization_success)
  _binarize(localization_success[y_true > 0])

  fpr, tpr, threshes = sklearn.metrics.roc_curve(
      y_true, y_score, sample_weight=sample_weight)

  denominator = np.sum(y_true * sample_weight)
  if not denominator:
    raise ValueError('No positives with nonzero weight!')

  tpr_localized = []
  for thresh, raw_tpr in zip(threshes, tpr):
    is_positive = (y_score >= thresh) & (localization_success > 0)
    numerator = np.nansum(
        np.array(is_positive, dtype=np.float32) * sample_weight * y_true)

    hit_rate = numerator / denominator
    assert hit_rate <= raw_tpr
    tpr_localized.append(hit_rate)

  return fpr, np.array(tpr_localized), threshes


def _rotation_matrix(angle):
  """Returns a 2x2 matrix for rotating counter clockwise.

  When using this to rotate the _axes_ by a specified angle, one should rotate
  the cartesian coordinates by the reverse amount, i.e. `-angle`.

  Args:
    angle: the rotation angle in radians.
  """

  return np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])


def average_roc_curves(y_true,
                       y_scores,
                       sample_weight=None,
                       method='sens',
                       num_samples=100):
  """Computes nonparametric ROC curves and averages them into one composite.

  For an explanation of this methodology, see Chen & Samuelson (2014).
  https://doi.org/10.1259/bjr.20140016

  Args:
    y_true: An array of boolean outcomes.
    y_scores: An iterable of arrays of continuous-valued scores. Each item
      should have the same length as `y_true`.
    sample_weight: Optional vector of sample weights.
    method: A string indicating how the averaging should be performed:
      'sens': average sensitivity at fixed specificity.
      'spec': average specificity at fixed sensitivity.
      'diagonal': average along diagonal lines of fixed (sensitivity -
        specificity).
    num_samples: the number of samples over which to perform the averaging.

  Returns:
    fpr: The false positive rates (abscisssa) for the averaged curve.
    tpr: The true positive rates (ordinate) for the averaged curve.
  """

  if method == 'sens':
    angle = 0.0
    max_sample = 1.0
  elif method == 'spec':
    angle = np.pi / 2
    max_sample = 1.0
  elif method == 'diagonal':
    angle = np.pi / 4
    max_sample = np.sqrt(2)
  else:
    raise ValueError(
        "`method` parameter must be one of {'sens', 'spec', 'diagonal'}")

  sample_points = np.linspace(0, max_sample, num=num_samples)

  coordinate_transform = _rotation_matrix(-angle)
  inverse_transform = _rotation_matrix(angle)

  to_average = []
  for y_score in y_scores:
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        y_true, y_score, sample_weight=sample_weight)

    cartesian = np.array([fpr, tpr])
    rotated = np.dot(coordinate_transform, cartesian)

    sampled = np.interp(sample_points, rotated[0], rotated[1])
    to_average.append(sampled)

  averaged = np.mean(np.array(to_average), axis=0)
  cartesian_average = np.dot(inverse_transform,
                             np.array([sample_points, averaged]))
  mean_fpr, mean_tpr = cartesian_average
  valid_range = (mean_fpr < 1.0) & (mean_fpr > 0) & (mean_tpr < 1.0) & (
      mean_tpr > 0)

  mean_fpr = np.append(0, np.append(mean_fpr[valid_range], 1))
  mean_tpr = np.append(0, np.append(mean_tpr[valid_range], 1))

  return mean_fpr, mean_tpr

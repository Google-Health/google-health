"""Tools for analyzing and simulating reader study data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import scipy.stats


def auc_to_mu(auc):
  """Returns a `mu` value corresponding to an area under the ROC curve.

  Under the signal detection theory framework, when the positive and negative
  score distributions are modeled as Gaussians with unit variance, this function
  finds a separation in means that yields a given AUC.
  mu is equivalent to the sensitivity index, d'
  (https://en.wikipedia.org/wiki/Sensitivity_index).

  Args:
    auc: (float) Area under an ROC curve.

  Returns:
    Float value representing the separation of score distributions
    between negative and positive scores for a labeler (an algorithm or
    group of readers who assign continuous suspicion scores to a set
    of cases). The AUC is given by PHI(mu / sqrt(2)), where PHI is the
    cumulative distribution function of the normal distribution.
  """
  return np.sqrt(2) * scipy.stats.norm.ppf(auc)


def simulate_single_modality(disease,
                             mu,
                             delta_mu,
                             sigma_r,
                             sigma_c,
                             num_readers,
                             rng=np.random):
  """Simulates a single-modality reader data according to the Roe-Metz model.

  (See simulate_model_vs_readers for an API with a friendlier parameterization
  of the score distribution.)

  This is the sort of data you might acquire when seeking to compare the
  performance of multiple readers with a standalone algorithm.

  Continuous "suspicion scores" are produced by both model and readers,
  but they can be thresholded to simulate binary predictions.

  Model scores are simulated by sampling two unit-variance normal distributions
  separated by mu. Model scores are assumed to be deterministic for a given
  case.

  Reader scores are simulated by sampling from two unit variance normal
  distributions separated by mu + delta_mu. For any specific radiologist, the
  two score distributions also have unit variance.
  The reader scores for a given case are correlated with those of the algorithm.

  See chapter 23 and the online supplement in:
  Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
  Foundations, Modeling, and Applications with R-Based Examples.
  CRC Press; 2017.

  Args:
    disease: An array of indicator variables (whose elements are in {0, 1})
      giving the disease state for each case. The number of cases will be
      inferred from the length of this variable.
    mu: The separation of the model score distributions for diseased and
      non-diseased patients. Larger values correspond to better models. The
      model AUC is given by PHI(mu / sqrt(2)).
    delta_mu: The change in separation of the score distributions of the average
      reader with that of the model. When this value is positive, readers are
      more discriminating than the algorithm. The AUC of the average reader is
      given by PHI((mu + delta_mu) / sqrt(2)).
    sigma_r: The standard deviation of the random effect for reader id. This
      should be small relative to mu.
    sigma_c: A value in [0, 1] that determines the correlation between model and
      reader scores. If sigma_c = 1, the average reader score is the same as the
      model score.
    num_readers: The number of readers to simulate.
    rng: An optional instance of numpy.random.RandomState.

  Returns:
    A pair of arrays:
      model_score: A vector of model scores for each case.
      reader_scores: A [num_cases x num_readers] array of reader scores.

  """
  if sigma_c < 0 or sigma_c > 1:
    raise ValueError('sigma_c should be in [0, 1]')

  disease = np.array(disease, dtype=np.int32)
  if set(disease) - set([0, 1]):
    raise ValueError('disease indicators must be in {0, 1}')

  num_cases = len(disease)
  case_random_effect = rng.randn(num_cases)
  model_score = disease * mu + case_random_effect

  def reader_score():
    """Generates the scores for a single reader."""
    reader_ranef_negative, reader_ranef_positive = sigma_r * rng.randn(2)
    error_term = np.sqrt(1 - sigma_c**2) * rng.randn(num_cases)
    reader_score = (mu + delta_mu) * disease
    reader_score += reader_ranef_negative * (1 - disease)
    reader_score += reader_ranef_positive * disease
    reader_score += sigma_c * case_random_effect
    reader_score += error_term
    return reader_score

  reader_scores = np.column_stack([reader_score() for _ in range(num_readers)])

  return model_score, reader_scores


def simulate_model_vs_readers(disease,
                              model_auc,
                              reader_auc,
                              sigma_r,
                              sigma_c,
                              num_readers,
                              rng=np.random):
  """Simulates a single-modality reader data according to the Roe-Metz model.

  This is the sort of data you might acquire when seeking to compare the
  performance of multiple readers with a standalone algorithm.

  Continuous "suspicion scores" are produced by both a model and the readers,
  but they can be thresholded to simulate binary predictions.

  Model scores are simulated by sampling two unit-variance normal distributions
  separated by mu. Model scores are assumed to be deterministic for a given
  case.

  Reader scores are simulated by sampling from two unit variance normal
  distributions separated by mu + delta_mu. For any specific radiologist, the
  two score distributions also have unit variance.
  The reader scores for a given case are correlated with those of the algorithm.

  See chapter 23 and the online supplement in:
  Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
  Foundations, Modeling, and Applications with R-Based Examples.
  CRC Press; 2017.

  Args:
    disease: An array of indicator variables (whose elements are in {0, 1})
      giving the disease state for each case. The number of cases will be
      inferred from the length of this variable.
    model_auc: The expected discrimination performance of the model, in terms of
      area under the ROC curve.
    reader_auc: The expected discrimination performance of the average reader,
      in terms of area under the ROC curve.
    sigma_r: The standard deviation of the random effect for reader id. This
      should be small relative to mu.
    sigma_c: A value in [0, 1] that determines the correlation between model and
      reader scores. If sigma_c = 1, the average reader score is the same as the
      model score.
    num_readers: The number of readers to simulate.
    rng: An optional instance of numpy.random.RandomState.

  Returns:
    A pair of arrays:
      model_score: A vector of model scores for each case.
      reader_scores: A [num_cases x num_readers] array of reader scores.

  """
  mu = auc_to_mu(model_auc)
  delta_mu = auc_to_mu(reader_auc) - mu
  return simulate_single_modality(
      disease,
      mu,
      delta_mu,
      sigma_r=sigma_r,
      sigma_c=sigma_c,
      num_readers=num_readers,
      rng=rng)


def _get_roe_metz_variances(structure):
  """Gets variances for the random effects in the Roe-Metz model.

  These settings correspond to rows from table 2 in:
  Hillis SL. Relationship between Roe and Metz simulation model for
  multireader diagnostic data and Obuchowski-Rockette model parameters.
  Stat Med. 2018;37: 2067-2093.

  Args:
    structure: A string in {'HL', 'LL', 'LH', 'HH', 'deterministic'}. The last
      choice produces no per-reader effects and should be used for debugging
      only.

  Returns:
    A dictionary of variance values. Within reader and modality, the total
    variance equals 1.0.
  """
  if structure == 'HL':
    return dict(
        var_case=0.3,
        var_modality_case=0.3,
        var_reader_case=0.2,
        var_pure_error=0.2,
        var_reader=0.0082,
        var_modality_reader=0.0082,
    )
  elif structure == 'LL':
    return dict(
        var_case=0.1,
        var_modality_case=0.1,
        var_reader_case=0.2,
        var_pure_error=0.6,
        var_reader=0.0082,
        var_modality_reader=0.0082,
    )
  elif structure == 'HH':
    return dict(
        var_case=0.3,
        var_modality_case=0.3,
        var_reader_case=0.2,
        var_pure_error=0.2,
        var_reader=0.0447,
        var_modality_reader=0.0447,
    )
  elif structure == 'LH':
    return dict(
        var_case=0.1,
        var_modality_case=0.1,
        var_reader_case=0.2,
        var_pure_error=0.6,
        var_reader=0.0447,
        var_modality_reader=0.0447,
    )
  elif structure == 'deterministic':
    return dict(
        var_case=1.0,
        var_modality_case=0.0,
        var_reader_case=0.0,
        var_pure_error=0.0,
        var_reader=0.0,
        var_modality_reader=0.0,
    )
  else:
    raise ValueError(
        'Unrecognized structure %s. '
        'Allowable values are "LL", "HL", "LH", and "HH"' % structure)


def simulate_dual_modality_from_mu(disease,
                                   mu,
                                   delta_mu,
                                   structure,
                                   num_readers,
                                   b=1,
                                   rng=np.random):
  """Simulates reader study data on two modalities.

  Simulates reader study decision variables using the constrained
  unequal-variance Roe-Metz model.
  This was introduced in [1], but formulae were excerpted from Hillis [2].
  It's essentially the canonical Roe-Metz model [3], but the variance of the
  scores for diseased cases is inflated by a constant factor.

  [1] Hillis SL. Simulation of unequal-variance binormal multireader ROC
  decision data: an extension of the Roe and Metz simulation model.
  Acad Radiol. 2012;19: 1518-1528.

  [2] Hillis SL. Relationship between Roe and Metz simulation model for
  multireader diagnostic data and Obuchowski-Rockette model parameters.
  Stat Med. 2018;37: 2067-2093.

  [3] Roe CA, Metz CE. Dorfman-Berbaum-Metz method for statistical analysis of
  multireader, multimodality receiver operating characteristic data: validation
  and computer simulation. Acad Radiol. 1997;4:298-303.

  For his simulations, Hillis [2] used parameters
    (mu, b) in {0.821, 1.831, 3.661} x {0.856, 0.711, 0.551},

  while Roe & Metz [3] used parameters
    (mu, b) in {0.75,  1.50,  2.50 } x {1}

  Args:
    disease: An array of indicator variables (whose elements are in {0, 1})
      giving the disease state for each case. The number of cases will be
      inferred from the length of this variable.
    mu: The typical separation of the score distributions for diseased and
      non-diseased cases.
    delta_mu: The median difference in decision variables for diseased and
      non-diseased cases for modality 1 versus modality 0. Modality 0 has
      separation mu, while modality 1 has separation mu + delta_mu.
    structure: A setting for the variance parameters; one of {'HL', 'LL', 'HH',
      'LH'}. These letters correspond to high/low data correlation and high/low
      reader variance. See Table 2 of [2].
    num_readers: The number of readers to simulate.
    b: The variance inflation factor for the diseased cases. Variances for
      diseased cases will be multiplied by 1/b^2. This value is typically at
      most 1 (the default).
    rng: An optional instance of numpy.random.RandomState.

  Returns:
    separations: An array giving the difference in the mean decision variable
      for diseased and non-diseased cases for each reader x modality pair.
      It has shape [num_readers, 2].
    scores: An array of continuous-valued scores of shape
      [num_cases, num_readers, 2].
  """
  disease = np.array(disease, dtype=np.int32)
  if set(disease) - set([0, 1]):
    raise ValueError('disease indicators must be in {0, 1}')

  num_cases = len(disease)

  variances = _get_roe_metz_variances(structure)
  sd_reader = np.sqrt(variances.pop('var_reader'))
  sd_case = np.sqrt(variances.pop('var_case'))
  sd_modality_reader = np.sqrt(variances.pop('var_modality_reader'))
  sd_reader_case = np.sqrt(variances.pop('var_reader_case'))
  sd_modality_case = np.sqrt(variances.pop('var_modality_case'))
  sd_pure_error = np.sqrt(variances.pop('var_pure_error'))
  assert not variances, variances

  # Accounts for different variance in disease cases, based on the b-factor.
  sd_multiplier = (1 + (1 / b - 1) * disease)

  # C in equation 9 from [2].
  case_ranef = sd_case * sd_multiplier * rng.randn(num_cases)

  # tau-C in equation 9 from [2].
  modality_by_case_ranefs = [
      sd_modality_case * sd_multiplier * rng.randn(num_cases) for _ in range(2)
  ]

  reader_scores = np.zeros((num_cases, num_readers, 2), dtype=np.float32)
  separations = np.zeros((num_readers, 2), dtype=np.float32)
  for reader_idx in range(num_readers):
    # R in equation 9 from [2].
    reader_ranef_positive, reader_ranef_negative = sd_reader * rng.randn(2)

    # R-C in equation 9 from [2].
    reader_by_case_ranef = sd_reader_case * sd_multiplier * rng.randn(num_cases)

    for modality_idx in range(2):
      # tau-R in equation 9 from [2].
      modality_by_reader_ranef_positive, modality_by_reader_ranef_negative = (
          sd_modality_reader * rng.randn(2))

      # Median separation between positive and negative cases for this
      # reader x modality. Determines the expected value of the AUC.
      separation = mu + delta_mu * modality_idx
      separation += (reader_ranef_positive - reader_ranef_negative)
      separation += (
          modality_by_reader_ranef_positive - modality_by_reader_ranef_negative)
      separations[reader_idx, modality_idx] = separation

      # The last two terms (epsilon) in equation 9 from [2].
      pure_error = sd_pure_error * sd_multiplier * rng.randn(num_cases)

      reader_score = (mu + delta_mu * modality_idx) * disease
      reader_score += reader_ranef_positive * disease
      reader_score += reader_ranef_negative * (1.0 - disease)
      reader_score += modality_by_reader_ranef_positive * disease
      reader_score += modality_by_reader_ranef_negative * (1.0 - disease)
      reader_score += case_ranef
      reader_score += reader_by_case_ranef
      reader_score += modality_by_case_ranefs[modality_idx]
      reader_score += pure_error
      reader_scores[:, reader_idx, modality_idx] = reader_score

  return separations, reader_scores


def simulate_dual_modality(disease,
                           modality_0_auc,
                           modality_1_auc,
                           structure,
                           num_readers,
                           b=1,
                           rng=np.random):
  """Simulates reader study data on two modalities.

  Simulates reader study decision variables using the constrained
  unequal-variance Roe-Metz model.
  This was introduced in [1], but formulae were excerpted from Hillis [2].
  It's essentially the canonical Roe-Metz model [3], but the variance of the
  scores for diseased cases is inflated by a constant factor.

  [1] Hillis SL. Simulation of unequal-variance binormal multireader ROC
  decision data: an extension of the Roe and Metz simulation model.
  Acad Radiol. 2012;19: 1518-1528.

  [2] Hillis SL. Relationship between Roe and Metz simulation model for
  multireader diagnostic data and Obuchowski-Rockette model parameters.
  Stat Med. 2018;37: 2067-2093.

  [3] Roe CA, Metz CE. Dorfman-Berbaum-Metz method for statistical analysis of
  multireader, multimodality receiver operating characteristic data: validation
  and computer simulation. Acad Radiol. 1997;4:298-303.

  For his simulations, Hillis [2] used parameters
    (mu, b) in {0.821, 1.831, 3.661} x {0.856, 0.711, 0.551},

  while Roe & Metz [3] used parameters
    (mu, b) in {0.75,  1.50,  2.50 } x {1}

  Args:
    disease: An array of indicator variables (whose elements are in {0, 1})
      giving the disease state for each case. The number of cases will be
      inferred from the length of this variable.
    modality_0_auc: The discrimination performance of the average reader in
      modality 0, given in terms of AUC-ROC.
    modality_1_auc: The discrimination performance of the average reader in
      modality 1, given in terms of AUC-ROC.
    structure: A setting for the variance parameters; one of {'HL', 'LL', 'HH',
      'LH'}. These letters correspond to high/low data correlation and high/low
      reader variance. See Table 2 of [2].
    num_readers: The number of readers to simulate.
    b: The variance inflation factor for the diseased cases. Variances for
      diseased cases will be multiplied by 1/b^2. This value is typically at
      most 1 (the default).
    rng: An optional instance of numpy.random.RandomState.

  Returns:
    separations: An array giving the difference in the mean decision variable
      for diseased and non-diseased cases for each reader x modality pair.
      It has shape [num_readers, 2].
    scores: An array of continuous-valued scores of shape
      [num_cases, num_readers, 2].
  """
  mu = auc_to_mu(modality_0_auc)
  delta_mu = auc_to_mu(modality_1_auc) - mu
  return simulate_dual_modality_from_mu(
      disease,
      mu,
      delta_mu,
      structure=structure,
      num_readers=num_readers,
      b=b,
      rng=rng)


class TestResult(
    collections.namedtuple('TestResult',
                           ['effect', 'ci', 'statistic', 'dof', 'pvalue'])):
  """The results of the ORH procedure hypothesis test."""


def _two_sided_p_value(t, df):
  """Computes the 2-sided p-value for a t-statisic with the specified d.o.f."""
  return 2 * scipy.stats.t.cdf(-np.abs(t), df=df)


def _one_sided_p_value(t, df):
  """Computes the 1-sided p-value for a t-statisic with the specified d.o.f."""
  return scipy.stats.t.sf(t, df=df)


def _test_result(effect, margin, se, dof, coverage):
  """Computes the test results based on the t-distribution."""
  t_stat = (effect + margin) / se
  if margin:
    p_value = _one_sided_p_value(t_stat, dof)
  else:
    p_value = _two_sided_p_value(t_stat, dof)
  t_alpha = scipy.stats.t.isf((1 - coverage) / 2.0, dof)
  lower = effect - t_alpha * se
  upper = effect + t_alpha * se
  return TestResult(
      effect=effect,
      ci=(lower, upper),
      statistic=t_stat,
      dof=dof,
      pvalue=p_value)


def _jackknife_covariance_model_vs_readers(disease, model_score, reader_scores,
                                           fom_fn):
  """Estimates the reader covariance matrix of the difference figure-of-merit.

  See equation 22.8 in
  Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
  Foundations, Modeling, and Applications with R-Based Examples.
  CRC Press; 2017.

  Args:
    disease: An array of ground-truth labels for each case, with shape
      [num_cases,].
    model_score: An array of model predictions for each case, with shape
      [num_cases,].
    reader_scores: A matrix of reader scores for each case, with shape
      [num_cases, num_readers].
    fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
      yielding a scalar summary value. Examples are
      sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.

  Returns:
    A [num_readers x num_readers] covariance matrix.
  """
  num_cases = len(disease)
  model_fom_jk = []
  reader_fom_jk = []

  for jk_idx in range(num_cases):
    disease_jk = np.delete(disease, jk_idx)
    model_score_jk = np.delete(model_score, jk_idx)
    reader_scores_jk = np.delete(reader_scores, jk_idx, axis=0)
    model_fom_jk.append(fom_fn(disease_jk, model_score_jk))
    reader_fom_jk.append([
        fom_fn(disease_jk, reader_score) for reader_score in reader_scores_jk.T
    ])
  difference_foms = np.expand_dims(model_fom_jk, 1) - np.array(reader_fom_jk)
  covariances = np.cov(difference_foms, rowvar=False, ddof=1)
  return covariances * (num_cases - 1)**2 / num_cases


def model_vs_readers_orh_index(example_indices,
                               reader_indices,
                               index_fom_fn,
                               coverage=0.95,
                               margin=0):
  """Wraps model_vs_readers_orh in a more general API.

  Args:
    example_indices: The set of example indices on which the metric should be
      computed.
    reader_indices: The reader ids that should be included in the analysis. This
      should not include -1.
    index_fom_fn:  A callable that accepts indices: (indices, reader_id), where
      indices is a vector of example indices on which to compute the metric. (In
      general this may be a subset of `example_indices`.) reader_id indicates
      which reader the metric is computed for. -1 is a special case, and should
      be reserved for the model.
    coverage: The size of the confidence interval. Should be in (0, 1]. The
      default is 0.95.
    margin: A positive noninferiority margin. When supplied and nonzero, the
      p-value refers to the one-sided test of the null hypothesis in which the
      model is at least this much worse than the average human reader. The units
      depend on the figure-of-merit function.

  Returns:
    A TestResult (see model_vs_readers_orh).
  """

  def fom_fn(indices, reader_idx_vector):
    """Wraps index_fom_fn to acccept (disease, score) vectors."""
    idx_values = set(reader_idx_vector)
    assert len(idx_values) == 1
    idx = idx_values.pop()
    assert idx in ([-1] + list(reader_indices)), idx
    unrecognized = set(indices) - set(example_indices)
    assert unrecognized == set(), unrecognized

    return index_fom_fn(indices, idx)

  assert -1 not in reader_indices
  disease = np.array(example_indices)
  model_score = -1 * np.ones_like(disease)
  reader_scores = np.column_stack(
      [idx * np.ones_like(disease) for idx in reader_indices])

  return model_vs_readers_orh(
      disease,
      model_score,
      reader_scores,
      fom_fn,
      coverage=coverage,
      margin=margin)


def model_vs_readers_orh(disease,
                         model_score,
                         reader_scores,
                         fom_fn,
                         coverage=0.95,
                         margin=0):
  """Performs the ORH procedure to compare a standalone model against readers.

  This function uses the Obuchowski-Rockette-Hillis analysis to compare the
  quality of a model's predictions with that of a panel of readers that all
  interpreted the same cases. I.e., the reader data occurs in a dense matrix of
  shape [num_cases, num_readers], and the model has been applied to these same
  cases.

  This tool can be used with an arbitrary 'figure of merit' (FOM) defined on the
  labels and the scores; scores can be binary, ordinal or continuous.
  It tests the null hypothesis that the average difference in the FOM between
  the readers and the model is 0.


  See chapter 22 of:
  Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
  Foundations, Modeling, and Applications with R-Based Examples.
  CRC Press; 2017.

  This implementation has been benchmarked against RJafroc:
  https://cran.r-project.org/web/packages/RJafroc/index.html.

  Args:
    disease: An array of ground-truth labels for each case, with shape
      [num_cases,].
    model_score: An array of model predictions for each case, with shape
      [num_cases,].
    reader_scores: A matrix of reader scores for each case, with shape
      [num_cases, num_readers].
    fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
      yielding a scalar summary value. Examples are
      sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.
    coverage: The size of the confidence interval. Should be in (0, 1]. The
      default is 0.95.
    margin: A positive noninferiority margin. When supplied and nonzero, the
      p-value refers to the one-sided test of the null hypothesis in which the
      model is at least this much worse than the average human reader. The units
      depend on the figure-of-merit function.

  Returns:
    A named tuple with fields:
      effect: The estimated difference in the FOM between the model and the
        readers. A positive effect means the model has a higher value
        than the average reader.
      ci: A (lower, upper) confidence interval for the true difference
        in the FOM.
      statistic: The value of the t-statistic.
      dof: The degrees of freedom for the t-statistic.
      pvalue: The p-value associated with the test.
  """
  if margin < 0:
    raise ValueError('margin parameter should be nonnegative.')

  num_cases, num_readers = reader_scores.shape
  if len(disease) != num_cases or len(model_score) != num_cases:
    raise ValueError(
        'disease, model_score and reader_scores must have the same size '
        'in the first dimension.')

  model_fom = fom_fn(disease, model_score)
  radiologist_foms = [
      fom_fn(disease, rad_scores) for rad_scores in reader_scores.T
  ]
  observed_effect_size = model_fom - np.mean(radiologist_foms)

  covariances = _jackknife_covariance_model_vs_readers(disease, model_score,
                                                       reader_scores, fom_fn)
  off_diagonals = []
  for offset in range(1, num_readers):
    off_diagonals.extend(np.diag(covariances, k=offset))
  cov2 = np.mean(off_diagonals)

  # msr = mean squared reader difference
  msr = np.var(radiologist_foms - model_fom, ddof=1)
  se = np.sqrt((msr + max(num_readers * cov2, 0)) / num_readers)
  dof = (num_readers - 1) * ((msr + max(num_readers * cov2, 0)) / msr)**2

  return _test_result(observed_effect_size, margin, se, dof, coverage)


def _jackknife_covariance_dual_modality(disease, reader_scores, fom_fn):
  """Estimates the reader-modality covariance matrix.

  See section 10.3 of Chakraborty DP. Observer Performance Methods for
  Diagnostic Imaging: Foundations, Modeling, and Applications with R-Based
  Examples. CRC Press; 2017.

  Args:
    disease: An array of ground-truth labels for each case, with shape
      [num_cases,].
    reader_scores: A matrix of reader scores, with shape [num_cases,
      num_readers, num_modalities.
    fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
      yielding a scalar summary value. Examples are
      sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.

  Returns:
    covariance: A square covariance matrix of size
      (num_readers * num_modalities) x (num_readers * num_modalities).
      Treatments and readers are interleaved, as described by `indices`, below.
    indices: A list of pairs (i, j) giving the modality and reader indices
      for the rows and columns of the covariance matrix.
  """
  num_cases, num_readers, num_modalities = reader_scores.shape
  # Here, jk denotes jackknife. The jackknife samples are the figures-of-merit
  # computed with one case omitted.
  jk_samples = []
  indices = []
  for modality_idx in range(num_modalities):
    for reader_idx in range(num_readers):
      score = reader_scores[:, reader_idx, modality_idx]
      indices.append((modality_idx, reader_idx))
      fom_jk = []
      for case_idx in range(num_cases):
        disease_jk = np.delete(disease, case_idx)
        score_jk = np.delete(score, case_idx)
        fom_jk.append(fom_fn(disease_jk, score_jk))
      jk_samples.append(fom_jk)
  covariances = np.cov(jk_samples, rowvar=True, ddof=1)
  covariances *= (num_cases - 1)**2 / float(num_cases)
  return covariances, indices


def dual_modality_orh(disease,
                      reader_scores,
                      fom_fn,
                      coverage=0.95,
                      margin=0,
                      verbose=True):
  """Performs the ORH procedure to compare readers in two conditions.

  This function uses Obuchowski-Rockette-Hillis (ORH) analysis to compare reader
  performance under two conditions: modality 0 and modality 1.
  A common application might be to compare reader performance with and without
  algorithmic assistance.

  This tool analyzes data from a fully-crossed mutlireader, multicase (MRMC)
  study in which a panel of readers interprets the same set of cases under
  two modalities. The reader scores can thus be arranged into a dense array
  of shape [num_cases, num_readers, 2].

  This tool can be used with an arbitrary 'figure of merit' (FOM) defined on the
  labels and the reader scores; scores can be binary, ordinal or continuous.
  It tests the null hypothesis that the modality effect is zero.
  I.e., reader performance does not differ between the two conditions.

  See chapter 10 of:
  Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
  Foundations, Modeling, and Applications with R-Based Examples.
  CRC Press; 2017.

  This implementation has been benchmarked against RJafroc:
  https://cran.r-project.org/web/packages/RJafroc/index.html.

  For the use of the margin parameter, see:
  Chen et al. (2012) https://doi.org/10.1016/j.acra.2012.04.011

  Args:
    disease: An array of ground-truth labels for each case, with shape
      [num_cases,].
    reader_scores: A matrix of reader scores for each case, under both
      conditions. It has shape [num_cases, num_readers, 2].
    fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
      yielding a scalar summary value. Examples are
      sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.
    coverage: The size of the confidence interval. Should be in (0, 1]. The
      default is 0.95.
    margin: A positive noninferiority margin. When supplied and nonzero, the
      p-value refers to the one-sided test of the null hypothesis in which the
      model is at least this much worse than the average human reader. The units
      depend on the figure-of-merit function.
    verbose: A boolean indicating whether intermediate quantities should be
      printed. Defaults to True.

  Returns:
    A named tuple with fields:
      effect: The estimated difference in the FOM between modality 1 and
        modality 0. A positive value indicates that performance in modality 1
        is greater than that in modality 0.
      ci: A (lower, upper) confidence interval for the modality effect.
      statistic: The value of the t-statistic.
      dof: The degrees of freedom for the t-statistic.
      pvalue: The p-value associated with the test.
  """
  if margin < 0:
    raise ValueError('margin parameter should be nonnegative.')

  num_cases, num_readers, num_modalities = reader_scores.shape
  if num_modalities != 2:
    raise ValueError('Only two modalities are supported.')

  if len(disease) != num_cases:
    raise ValueError(
        'disease, model_score and reader_scores must have the same size '
        'in the first dimension.')

  reader_modality_foms = np.zeros((num_readers, 2), dtype=np.float32)
  for reader_idx in range(num_readers):
    for modality_idx in range(2):
      fom = fom_fn(disease, reader_scores[:, reader_idx, modality_idx])
      reader_modality_foms[reader_idx, modality_idx] = fom

  reader_foms = np.mean(reader_modality_foms, axis=1)
  modality_foms = np.mean(reader_modality_foms, axis=0)
  average_fom = np.mean(modality_foms)

  assert len(reader_foms) == num_readers
  assert len(modality_foms) == 2

  # mstr = mean squared reader/modality difference; equation 10.43
  mstr = 0.0
  for reader_idx in range(num_readers):
    for modality_idx in range(2):
      summand = reader_modality_foms[reader_idx, modality_idx]
      summand -= modality_foms[modality_idx]
      summand -= reader_foms[reader_idx]
      summand += average_fom
      mstr += summand**2
  mstr /= num_readers - 1

  # Estimate covariance terms according to Equation 10.31
  covmat, indices = _jackknife_covariance_dual_modality(disease, reader_scores,
                                                        fom_fn)
  cov2_samples = []
  cov3_samples = []
  for row_idx in range(2 * num_readers):
    modality, reader = indices[row_idx]
    for col_idx in range(row_idx + 1):
      modality_prime, reader_prime = indices[col_idx]
      if reader != reader_prime:
        if modality == modality_prime:
          cov2_samples.append(covmat[row_idx, col_idx])
        else:
          cov3_samples.append(covmat[row_idx, col_idx])
  cov2 = np.mean(cov2_samples)
  cov3 = np.mean(cov3_samples)

  if verbose:
    print('mstr', mstr)
    print('cov2 * 10^5', cov2 * 1e5)
    print('cov3 * 10^5', cov3 * 1e5)

  observed_effect_size = modality_foms[1] - modality_foms[0]

  # Equation 10.45
  dof = (mstr + max(num_readers * (cov2 - cov3), 0))**2
  dof /= (mstr**2) / (num_readers - 1)

  # Equation 10.48
  se = np.sqrt(2 * (mstr + num_readers * max(cov2 - cov3, 0)) / num_readers)

  return _test_result(observed_effect_size, margin, se, dof, coverage)


def two_treatment_orh(*args, **kwargs):
  """An alias for `dual_modality_orh`, kept for backward compatibility."""
  return dual_modality_orh(*args, **kwargs)

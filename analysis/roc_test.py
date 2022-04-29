"""Tests for roc.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import sklearn.metrics
import roc


def generate_test_data(rng, num_scores=1, scale_factor=1):
  num_positives = int(100 * scale_factor)
  num_negatives = int(200 * scale_factor)
  y_true = [0] * num_negatives + [1] * num_positives

  y_scores = []
  for _ in range(num_scores):
    y_score = np.concatenate(
        (rng.randn(num_negatives), 1.0 + rng.randn(num_positives)))
    y_scores.append(y_score)
  if num_scores == 1:
    y_scores = y_scores[0]

  y_ordinal = rng.randint(20, size=num_positives + num_negatives)
  sample_weight = rng.rand(num_positives + num_negatives) > 0.1
  return y_true, y_scores, y_ordinal, sample_weight


class DeLongIntervalTest(parameterized.TestCase):

  def setUp(self):
    super(DeLongIntervalTest, self).setUp()
    rng = np.random.RandomState(1987)
    self.y_true, self.y_score, self.y_ordinal, self.weights = (
        generate_test_data(rng))
    self.rng = rng

  def testFullCoverage(self):
    """When the coverage is 100% the CI should be (0.0, 1.0)."""
    ci = roc.delong_interval(self.y_true, self.y_score, coverage=1.0)
    np.testing.assert_allclose((0.0, 1.0), ci)

  def testPerfectDiscriminator(self):
    """When the classifier is perfect, the CI should be (1.0, 1.0)."""
    ci = roc.delong_interval(self.y_true, self.y_true)
    np.testing.assert_allclose((1.0, 1.0), ci)

  @parameterized.parameters(
      itertools.product((0.5, 0.75, 0.95, 0.99), (True, False)))
  def testMidpointMatchesSklearn(self, coverage, ordinals):
    """The midpoint of the CI should be equal to sklearn's AUC.

    Args:
      coverage: The coverage of the confidence interval. Note that if coverage
        is 1.0, the resulting confidence interval will be (0, 1), so the
        midpoint will be the meaningless value of 0.5.
      ordinals: whether to use an ordinal score or not. These types of scores
        tend to have "ties"; multiple observations share the same score.
    """
    if ordinals:
      y_score = self.y_ordinal
    else:
      y_score = self.y_score

    lower, upper = roc.delong_interval(
        self.y_true, y_score, sample_weight=self.weights, coverage=coverage)
    midpoint = 0.5 * (lower + upper)
    sklearn_auc = sklearn.metrics.roc_auc_score(
        self.y_true, y_score, sample_weight=self.weights)
    np.testing.assert_allclose(sklearn_auc, midpoint)

  def testMatchesRPackage(self):
    """Benchmark against a reference implementation in the R language.

    Xavier Robin, Natacha Turck, Alexandre Hainard, et al. (2011)
    pROC: an open-source package for R and S+ to analyze and compare ROC curves
    BMC Bioinformatics, 7, 77. DOI: 10.1186/1471-2105-12-77.

    R code:
    ```
    > library(pROC)
    > x <- c(seq(0, 98, 2), seq(1, 99, 2))
    > y <- c(rep(0, 50), rep(1, 50))
    > ci(roc(response=y, predictor=x))
    95% CI: 0.3957-0.6243 (DeLong)
    ```
    """
    y_true = 50 * [0] + 50 * [1]
    y_score = list(range(0, 100, 2)) + list(range(1, 100, 2))
    ci = roc.delong_interval(y_true, y_score, coverage=0.95)
    np.testing.assert_allclose((0.3957154, 0.6242846), ci, atol=1e-6)

  def testNoWeights(self):
    """No nonzero weights."""
    with self.assertRaises(ValueError):
      roc.delong_interval(
          self.y_true, self.y_score, sample_weight=np.zeros_like(self.y_true))

  def testBadLabels(self):
    """Outcome labels are not binary."""
    y_true = self.rng.randint(3, size=len(self.y_true))
    with self.assertRaises(ValueError):
      roc.delong_interval(y_true, self.y_score, sample_weight=self.weights)


class DeLongTestTest(absltest.TestCase):

  def setUp(self):
    super(DeLongTestTest, self).setUp()
    rng = np.random.RandomState(1987)
    num_positives = 50
    num_negatives = 100
    self.y_true = [0] * num_negatives + [1] * num_positives
    self.y_score_1 = np.concatenate((rng.randn(num_negatives),
                                     1.0 + rng.randn(num_positives)))
    self.y_score_2 = np.concatenate((rng.randn(num_negatives),
                                     2.0 + rng.randn(num_positives)))

    self.weights = rng.rand(num_positives + num_negatives) > 0.1
    self.y_ordinal = rng.randint(20, size=num_positives + num_negatives)
    self.rng = rng

  def testOneSidedPValue(self):
    # Compare to a pre-computed table from
    # https://www.medcalc.org/manual/values_of_the_normal_distribution.php
    np.testing.assert_allclose(
        0.05, roc._one_sided_p_value(1.644854), atol=1e-6)

  def testTwoSidedPValue(self):
    # Compare to a pre-computed table from
    # https://www.medcalc.org/manual/values_of_the_normal_distribution.php
    np.testing.assert_allclose(
        0.05, roc._two_sided_p_value(1.959964), atol=1e-6)

  def testMarginTrend(self):
    """Tests that a larger margin leads to a smaller p-value."""
    result1 = roc.delong_test(
        self.y_true, self.y_score_1, self.y_score_2, margin=0.01)
    result2 = roc.delong_test(
        self.y_true, self.y_score_1, self.y_score_2, margin=0.05)
    self.assertLess(result2.pvalue, result1.pvalue)

  def testCoverageTrend(self):
    """Tests that higher coverage leads to a wider interval."""
    result98 = roc.delong_test(
        self.y_true, self.y_score_1, self.y_score_2, coverage=0.98)
    result99 = roc.delong_test(
        self.y_true, self.y_score_1, self.y_score_2, coverage=0.99)
    lower99, upper99 = result99.ci
    lower98, upper98 = result98.ci
    self.assertLess(lower99, lower98)
    self.assertGreater(upper99, upper98)

  def testMatchesRPackage(self):
    """Benchmark against a reference implementation in the R language.

    Xavier Robin, Natacha Turck, Alexandre Hainard, et al. (2011)
    pROC: an open-source package for R and S+ to analyze and compare ROC curves
    BMC Bioinformatics, 7, 77. DOI: 10.1186/1471-2105-12-77.

    R code:
    ```
    > library(pROC)
    > roc.test(response = c(rep(0, 20), rep(1, 30)),
    +          predictor1 = (1:50 %% 4),
    +          predictor2 = (1:50 %% 8),
    +          method = "delong")
    Z = -0.27058646, p-value = 0.7867091
    sample estimates:
    AUC of roc1  AUC of roc2
    0.5000000000 0.5233333333
    """
    result = roc.delong_test(20 * [0] + 30 * [1],
                             np.arange(1, 51) % 4,
                             np.arange(1, 51) % 8)
    np.testing.assert_allclose(result.effect, 0.02333333333)
    np.testing.assert_allclose(result.pvalue, 0.7867091)
    np.testing.assert_allclose(result.statistic, 0.27058646)
    lower, upper = result.ci
    self.assertGreaterEqual(lower, -1.0)
    self.assertLessEqual(upper, 1.0)


class LrocTest(absltest.TestCase):

  def testMatchesSklearn(self):
    """Tests that values match the ROC curve when localization is perfect."""
    rng = np.random.RandomState(1999)
    y_true, y_score, _, sample_weight = generate_test_data(rng)

    fpr, tpr, threshes = sklearn.metrics.roc_curve(
        y_true, y_score, sample_weight=sample_weight)

    localization_success = np.ones_like(y_true)
    fpr_loc, tpr_loc, threshes_loc = roc.lroc_curve(
        y_true, y_score, localization_success, sample_weight=sample_weight)

    np.testing.assert_allclose(fpr, fpr_loc, atol=1e-4)
    np.testing.assert_allclose(tpr, tpr_loc, atol=1e-4)
    np.testing.assert_allclose(threshes, threshes_loc, atol=1e-4)

  def testNoWeights(self):
    """Tests that an error is raised when no positives have nonzero weight."""
    rng = np.random.RandomState(2000)
    y_true, y_score, _, sample_weight = generate_test_data(rng)

    localization_success = np.ones_like(y_true)
    zero_weight = np.zeros_like(sample_weight)
    with self.assertRaises(ValueError):
      roc.lroc_curve(
          y_true, y_score, localization_success, sample_weight=zero_weight)

  def testToyExample(self):
    """Tests that FPR saturates due to localization error."""
    y_true = [1, 1, 1, 1]
    y_score = [1, 2, 3, 4]
    localization_success = [1, 1, 0, 1]
    sample_weight = [1, 1, 1, 1]
    _, tpr, _ = roc.lroc_curve(y_true, y_score, localization_success,
                               sample_weight)
    np.testing.assert_allclose(tpr.max(), 0.75, atol=1e-6)

  def testNullValues(self):
    """Confirms that null localization results are disallowed on positives."""
    y_true = [1, 1, 1, 1]
    y_score = [1, 2, 3, 4]
    localization_success = [None, 1, 0, 1]
    sample_weight = [1, 1, 1, 1]

    with self.assertRaises(TypeError):
      roc.lroc_curve(y_true, y_score, localization_success, sample_weight)


class BinarizationTest(parameterized.TestCase):

  def testBinarizationFailure(self):
    with self.assertRaisesRegex(ValueError, 'integers must be in {0, 1}'):
      roc._binarize([1, 2, 3], 'integers')

    with self.assertRaises(Exception):
      roc._binarize([1, False, np.nan])

  @parameterized.named_parameters(
      ('Integers', [1, 0, 0, 1]), ('Floats', [1.0, 0.0, 0.0, 1.0]),
      ('Bools', [True, False, False, True]), ('MixedType', [True, 0, 0.0, 1]))
  def testBinarization(self, values):
    np.testing.assert_equal(roc._binarize(values), [1, 0, 0, 1])


class AverageROCTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Identity', 0.0, [[1, 0], [0, 1]]),
      ('NinetyDegrees', np.pi / 2, [[0, -1], [1, 0]]))
  def testRotationMatrix(self, angle, expected):
    matrix = roc._rotation_matrix(angle)
    np.testing.assert_allclose(matrix, expected, atol=1e-5)

  @parameterized.parameters('sense', 'speck')
  def testBadMethodRaises(self, method):
    y_true, y_score, _, sample_weight = generate_test_data(
        np.random.RandomState(200))
    with self.assertRaisesRegex(ValueError, 'method'):
      roc.average_roc_curves(y_true, [y_score], sample_weight, method=method)

  def assertUniformSpacing(self, array):
    # Skip endpoints because they may be redundant due to padding with {0, 1}.
    self.assertLen(np.unique(np.round(np.diff(array[1:-1]), decimals=4)), 1)

  def testFprSpacing(self):
    """Averaging along vertical lines results in uniform spacing on y-axis."""
    y_true, y_scores, _, sample_weight = generate_test_data(
        np.random.RandomState(201), num_scores=5)
    fpr, unused_tpr = roc.average_roc_curves(
        y_true, y_scores, sample_weight, method='sens')
    self.assertUniformSpacing(fpr)

  def testTprSpacing(self):
    """Averaging along horizontal lines results in uniform spacing on y-axis."""
    y_true, y_scores, _, sample_weight = generate_test_data(
        np.random.RandomState(202), num_scores=5)
    unused_fpr, tpr = roc.average_roc_curves(
        y_true, y_scores, sample_weight, method='spec')
    self.assertUniformSpacing(tpr)

  def testEndpoints(self):
    y_true, y_scores, _, sample_weight = generate_test_data(
        np.random.RandomState(203), num_scores=5)
    fpr, tpr = roc.average_roc_curves(
        y_true, y_scores, sample_weight, method='diagonal')
    self.assertEqual(fpr[0], 0.0)
    self.assertEqual(fpr[-1], 1.0)
    self.assertEqual(tpr[0], 0.0)
    self.assertEqual(tpr[-1], 1.0)

  @parameterized.parameters('sens', 'spec', 'diagonal')
  def testAUCs(self, method):
    """Tests that average of AUCs is roughly the AUC of the average curve."""
    y_true, y_scores, _, sample_weight = generate_test_data(
        np.random.RandomState(203), num_scores=12, scale_factor=10)
    fpr, tpr = roc.average_roc_curves(
        y_true,
        y_scores,
        sample_weight=sample_weight,
        method=method,
        num_samples=500)
    average_of_aucs = np.mean([
        sklearn.metrics.roc_auc_score(
            y_true, y_score, sample_weight=sample_weight)
        for y_score in y_scores
    ])
    auc_of_average = sklearn.metrics.auc(fpr, tpr)
    np.testing.assert_allclose(average_of_aucs, auc_of_average, atol=1e-4)


if __name__ == '__main__':
  absltest.main()

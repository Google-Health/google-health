"""Tests for reader_study.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import numpy as np
import scipy.stats
import sklearn.metrics
import reader_study


class ReaderStudyTest(unittest.TestCase):

  def simulate_reader_study(self, **update_params):

    num_readers = 10
    num_cases = 10000
    num_positives = 5000
    disease = (
        np.array(([1] * num_positives) + ([0] * (num_cases - num_positives))))

    params = dict(
        # cad's separation of the negative & positive distributions
        mu=1.0,

        # increased separation possessed by the radiologists
        delta_mu=0.1,

        # reader standard deviation
        sigma_r=0.075,

        # expected correlation coefficient between CAD and readers
        sigma_c=0.6,
    )
    params.update(update_params)

    rng = np.random.RandomState(1771)
    model_score, reader_scores = reader_study.simulate_single_modality(
        disease, num_readers=num_readers, rng=rng, **params)

    self.assertEqual((num_cases,), model_score.shape)
    self.assertEqual((num_cases, num_readers), reader_scores.shape)

    auc_fn = sklearn.metrics.roc_auc_score

    theoretical_model_auc = scipy.stats.norm.cdf(params['mu'] / np.sqrt(2))
    empirical_model_auc = auc_fn(disease, model_score)
    np.testing.assert_allclose(
        theoretical_model_auc, empirical_model_auc, atol=0.02)

    theoretical_reader_auc = scipy.stats.norm.cdf(
        (params['mu'] + params['delta_mu']) / np.sqrt(2))
    average_reader_auc = np.mean(
        np.apply_along_axis(lambda x: auc_fn(disease, x), 0, reader_scores))
    np.testing.assert_allclose(
        theoretical_reader_auc, average_reader_auc, atol=0.02)

    # For each disease state, model scores have unit variance.
    np.testing.assert_allclose(
        np.var(model_score[:num_positives], ddof=1), 1.0, atol=0.05)
    np.testing.assert_allclose(
        np.var(model_score[num_positives:], ddof=1), 1.0, atol=0.05)

    # For each radiologist, the scores for each disease state
    # have unit variance.
    for reader_idx in range(num_readers):
      positive_scores = reader_scores[:num_positives, reader_idx]
      negative_scores = reader_scores[num_positives:, reader_idx]
      np.testing.assert_allclose(1.0, np.var(positive_scores, ddof=1), atol=0.1)
      np.testing.assert_allclose(1.0, np.var(negative_scores, ddof=1), atol=0.1)
      np.testing.assert_allclose(
          params['sigma_c'],
          np.corrcoef(model_score[:num_positives], positive_scores)[0, 1],
          atol=0.1)
      np.testing.assert_allclose(
          params['sigma_c'],
          np.corrcoef(model_score[num_positives:], negative_scores)[0, 1],
          atol=0.1)

  def testSimulateReaderStudy(self):
    for sigma_c in (0.4, 0.6, 0.8):
      for mu in (0.75, 1.0, 1.25):
        for delta_mu in (-0.15, 0.0, 0.15):
          self.simulate_reader_study(sigma_c=sigma_c, mu=mu, delta_mu=delta_mu)

  def _testRoeMetzVariance(self, structure):
    variances = reader_study._get_roe_metz_variances(structure)
    total_variance = 0.0
    for key in ('var_case', 'var_modality_case', 'var_reader_case',
                'var_pure_error'):
      total_variance += variances[key]
    self.assertEqual(total_variance, 1.0)
    self.assertEqual(variances['var_reader'], variances['var_modality_reader'])

  def testRoeMetzVarianceHL(self):
    self._testRoeMetzVariance('HL')

  def testRoeMetzVarianceLH(self):
    self._testRoeMetzVariance('LH')

  def testRoeMetzVarianceHH(self):
    self._testRoeMetzVariance('HH')

  def testRoeMetzVarianceLL(self):
    self._testRoeMetzVariance('LL')

  def testRoeMetzVarianceDeterministic(self):
    self._testRoeMetzVariance('deterministic')

  def testSimulateTwoModalities(self):
    rng = np.random.RandomState(1941)
    num_cases = 50000
    num_positives = 10000
    num_readers = 12
    disease = [1] * num_positives + (num_cases - num_positives) * [0]

    separations, scores = reader_study.simulate_dual_modality(
        disease=disease,
        mu=1.0,
        delta_mu=0.5,
        structure='HL',
        num_readers=num_readers,
        b=0.5,
        rng=rng)

    for reader_idx in range(num_readers):
      for modality_idx in range(2):
        var_positves = np.var(
            scores[:num_positives, reader_idx, modality_idx], ddof=1)
        # Since b=0.5, the variance for the positive cases should be
        # 1 / b ** 2 = 4
        np.testing.assert_allclose(4.0, var_positves, atol=0.2)

        var_negatives = np.var(
            scores[num_positives:, reader_idx, modality_idx], ddof=1)
        np.testing.assert_allclose(1.0, var_negatives, atol=0.1)

        # The variance of the positives and the negatives sum to 5.
        expected_auc = scipy.stats.norm.cdf(
            separations[reader_idx, modality_idx] / np.sqrt(5))
        actual_auc = sklearn.metrics.roc_auc_score(
            disease, scores[:, reader_idx, modality_idx])
        np.testing.assert_allclose(expected_auc, actual_auc, atol=0.025)

  def testOneSidedPValue(self):
    self.assertEqual(0.5, reader_study._one_sided_p_value(0, df=1))
    self.assertEqual(0.5, reader_study._one_sided_p_value(0, df=2))
    self.assertEqual(0.5, reader_study._one_sided_p_value(0, df=3))

    # Compare to a pre-computed table from
    # https://math.tutorvista.com/statistics/t-distribution-table.html
    np.testing.assert_allclose(
        0.05, reader_study._one_sided_p_value(2.3534, df=3), atol=0.0001)

  def testTwoSidedPValue(self):
    self.assertEqual(1.0, reader_study._two_sided_p_value(0, df=1))
    self.assertEqual(1.0, reader_study._two_sided_p_value(0, df=2))
    self.assertEqual(1.0, reader_study._two_sided_p_value(0, df=3))

    # Compare to a pre-computed value from
    # https://www.tutorialspoint.com/statistics/t_distribution_table.htm#:~:text=2.3534-,3.1824,-4.5407
    np.testing.assert_allclose(
        0.05, reader_study._two_sided_p_value(3.1824, df=3), atol=0.0001)

  def testModelVsReadersORH(self):
    disease = np.array(([1] * 200) + ([0] * 800))
    rng = np.random.RandomState(1987)
    model_score, reader_scores = reader_study.simulate_single_modality(
        disease,
        num_readers=10,
        mu=1.0,
        delta_mu=0,
        sigma_r=0.075,
        sigma_c=0.6,
        rng=rng)
    result = reader_study.model_vs_readers_orh(
        disease,
        model_score,
        reader_scores,
        fom_fn=sklearn.metrics.roc_auc_score)

    # The following values were produced by the RJafroc implementation.
    # Note that RJafroc subtracts model from readers, while we do the reverse.
    np.testing.assert_allclose(result.effect, -0.00584125)
    np.testing.assert_allclose(result.pvalue, 0.6560453)
    np.testing.assert_allclose(result.ci, (-0.03233184, 0.02064934), atol=1e-9)
    np.testing.assert_allclose(result.dof, 31.00905)

  def testModelVsReadersOrhIndexFn(self):
    disease = np.array(([1] * 200) + ([0] * 800))
    rng = np.random.RandomState(1987)
    model_score, reader_scores = reader_study.simulate_single_modality(
        disease,
        num_readers=10,
        mu=1.0,
        delta_mu=0,
        sigma_r=0.075,
        sigma_c=0.6,
        rng=rng)

    fom_fn = sklearn.metrics.roc_auc_score

    result = reader_study.model_vs_readers_orh(
        disease, model_score, reader_scores, fom_fn=fom_fn)

    def index_fom_fn(indices, reader_idx):
      if reader_idx == -1:
        y_score = model_score
      else:
        y_score = reader_scores[:, reader_idx]

      return fom_fn(disease[indices], y_score[indices])

    np.testing.assert_allclose(
        index_fom_fn(list(range(len(disease))), -1),
        fom_fn(disease, model_score))

    for i in range(10):
      np.testing.assert_allclose(
          index_fom_fn(list(range(len(disease))), i),
          fom_fn(disease, reader_scores[:, i]))

    # The following values were produced by the RJafroc implementation.
    # Note that RJafroc subtracts model from readers, while we do the reverse.
    np.testing.assert_allclose(result.effect, -0.00584125)
    np.testing.assert_allclose(result.pvalue, 0.6560453)
    np.testing.assert_allclose(result.ci, (-0.03233184, 0.02064934), atol=1e-9)
    np.testing.assert_allclose(result.dof, 31.00905)

  def testTwoTreatmentORH(self):
    rng = np.random.RandomState(1987)
    disease = 200 * [0] + 200 * [1]
    num_readers = 10
    _, scores = reader_study.simulate_dual_modality(
        disease=disease,
        mu=0.821,
        delta_mu=0,
        structure='HL',
        num_readers=num_readers,
        b=1,
        rng=rng)
    result = reader_study.two_treatment_orh(
        disease, scores, fom_fn=sklearn.metrics.roc_auc_score, verbose=False)
    # The following values were produced by the RJafroc implementation.
    # Note that RJafroc subtracts the second treatment from the first, while we
    # do the reverse.
    np.testing.assert_allclose(result.effect, 0.00451, atol=1e-6)
    np.testing.assert_allclose(result.pvalue, 0.8502715, atol=1e-6)
    np.testing.assert_allclose(
        result.ci, (-0.04284442691, 0.05186442691), atol=1e-6)
    np.testing.assert_allclose(result.dof, 85.5241922, atol=1e-4)

  def testORHEquivalence(self):
    """Tests equivalency of two approaches to model vs. readers comparison."""
    disease = np.array(([1] * 200) + ([0] * 800))
    rng = np.random.RandomState(1987)
    model_score, reader_scores = reader_study.simulate_single_modality(
        disease,
        num_readers=10,
        mu=1.0,
        delta_mu=0,
        sigma_r=0.075,
        sigma_c=0.6,
        rng=rng)

    # In model_vs_readers_orh, the effect size is computed as
    # `model - readers`
    result1 = reader_study.model_vs_readers_orh(
        disease,
        model_score,
        reader_scores,
        fom_fn=sklearn.metrics.roc_auc_score)

    # In two_treatment_orh, the effect size is computed as
    # `modality_at_index_1 - modality_at_index_0`.
    tiled_model_scores = np.tile(np.expand_dims(model_score, -1), (1, 10))
    stacked_scores = np.stack((reader_scores, tiled_model_scores), -1)
    # input shape should be (num_cases, num_readers, num_modalities)
    assert stacked_scores.shape == (1000, 10, 2)

    result2 = reader_study.two_treatment_orh(
        disease, stacked_scores, fom_fn=sklearn.metrics.roc_auc_score,
        verbose=False)

    np.testing.assert_allclose(result1.effect, result2.effect, atol=1e-4)
    np.testing.assert_allclose(result1.ci, result2.ci, atol=1e-4)
    np.testing.assert_allclose(result1.statistic, result2.statistic, atol=1e-4)
    np.testing.assert_allclose(result1.dof, result2.dof, atol=1e-4)
    np.testing.assert_allclose(result1.pvalue, result2.pvalue, atol=1e-4)


if __name__ == '__main__':
  unittest.main()

import numpy as np
from sklearn import linear_model
from sklearn import metrics

import unittest
import eval_utils


  class TestEvalUtils(unittest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)
    self.x = np.array([[1, 2], [3, 4]])
    self.y_reg = np.array([2, 4])
    self.y_cls = np.array([0, 1])

  def test_create_linear_probe(self):
    probe = eval_utils.create_linear_probe(0.1, is_regression=True)
    with self.subTest(name='check_model_type'):
      self.assertIsInstance(probe, linear_model.Ridge)
    with self.subTest(name='check_regularization_parameter'):
      self.assertEqual(probe.alpha, 0.1)

    probe = eval_utils.create_linear_probe(
        0.1, is_regression=False, use_sgd_classifier=True
    )
    with self.subTest(name='check_model_type'):
      self.assertIsInstance(probe, linear_model.SGDClassifier)
    with self.subTest(name='check_regularization_parameter'):
      self.assertEqual(probe.alpha, 0.1)
    with self.subTest(name='check_loss_type'):
      self.assertEqual(probe.loss, 'log')
    with self.subTest(name='check_class_weight'):
      self.assertEqual(probe.class_weight, 'balanced')

    probe = eval_utils.create_linear_probe(
        0.1, is_regression=False, use_sgd_classifier=False
    )
    with self.subTest(name='check_model_type'):
      self.assertIsInstance(probe, linear_model.LogisticRegression)
    with self.subTest(name='check_regularization_parameter'):
      self.assertEqual(probe.C, 0.1)
    with self.subTest(name='check_class_weight'):
      self.assertEqual(probe.class_weight, 'balanced')

  def test_predict_with_probe(self):
    probe = linear_model.Ridge().fit(self.x, self.y_reg)
    predictions = eval_utils.predict_with_probe(
        probe, self.x, is_regression=True
    )
    self.assertEqual(predictions.shape, (2,))

    probe = linear_model.LogisticRegression().fit(self.x, self.y_cls)
    predictions = eval_utils.predict_with_probe(
        probe, self.x, is_regression=False
    )
    with self.subTest(name='check_predictions_shape'):
      self.assertEqual(predictions.shape, (2,))
    with self.subTest(name='check_value_in_0_1_range'):
      self.assertTrue(all(0 <= p <= 1 for p in predictions))

  def test_find_reg_coef_with_best_metric(self):
    cv_scores = {0.1: 0.8, 0.01: 0.9, 1.0: 0.7}

    best_alpha = eval_utils.find_reg_coef_with_best_metric(
        cv_scores, lower_is_better=True
    )
    with self.subTest(name='check_alpha_is_highest'):
      self.assertEqual(best_alpha, 1.0)

    best_alpha = eval_utils.find_reg_coef_with_best_metric(
        cv_scores, lower_is_better=False
    )
    with self.subTest(name='check_alpha_is_lowest'):
      self.assertEqual(best_alpha, 0.01)

  def test_compute_metrics_for_probe(self):
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    mae = eval_utils.compute_metrics_for_probe(
        y_true, y_pred, is_regression=True
    )
    self.assertIsInstance(mae, float)
    self.assertAlmostEqual(mae, 0.1)

    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.8, 0.7, 0.2])
    auc = eval_utils.compute_metrics_for_probe(
        y_true, y_score, is_regression=False
    )
    with self.subTest(name='check_auc_type'):
      self.assertIsInstance(auc, float)
    with self.subTest(name='check_auc_value'):
      self.assertAlmostEqual(auc, metrics.roc_auc_score(y_true, y_score))

  def test_train_linear_probe_with_participant_level_crossval(self):
    n_samples = 100
    n_features = 10
    n_participants = 20

    features = np.random.randn(n_samples, n_features)
    participant_ids = np.repeat(
        range(n_participants), n_samples // n_participants
    )

    labels_reg = np.random.randn(n_samples)
    probe_reg = eval_utils.train_linear_probe_with_participant_level_crossval(
        features, labels_reg, participant_ids, is_regression=True
    )
    with self.subTest(name='check_model_type_ridge'):
      self.assertIsInstance(probe_reg, linear_model.Ridge)

    labels_cls = np.random.randint(0, 2, n_samples)
    probe_cls = eval_utils.train_linear_probe_with_participant_level_crossval(
        features, labels_cls, participant_ids, is_regression=False
    )
    with self.subTest(name='check_model_type_logistic_regression'):
      self.assertIsInstance(
          probe_cls,
          (linear_model.SGDClassifier, linear_model.LogisticRegression),
      )

  def test_input_validation(self):
    with self.assertRaises(AssertionError):
      features = np.random.randn(10, 5)
      labels = np.random.randn(9)
      participant_ids = np.arange(10)
      eval_utils.train_linear_probe_with_participant_level_crossval(
          features, labels, participant_ids, is_regression=True
      )


if __name__ == '__main__':
  unittest.main()

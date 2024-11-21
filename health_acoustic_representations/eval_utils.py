"""Eval utils.

Train linear models on top of frozen embeddings.
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import utils as sk_utils


LinearProbe = (
    linear_model.LogisticRegressionCV
    | linear_model.RidgeCV
    | linear_model.Ridge
    | model_selection.GridSearchCV
    | linear_model.SGDClassifier
    | linear_model.LogisticRegression
    | linear_model.ElasticNetCV
    | linear_model.ElasticNet
)


def create_linear_probe(
    regularization_coef: float,
    is_regression: bool,
    use_sgd_classifier: bool = True,
) -> LinearProbe:
  """Creates linear probe."""
  if is_regression:
    return linear_model.Ridge(alpha=regularization_coef)
  else:
    if use_sgd_classifier:
      return linear_model.SGDClassifier(
          loss='log',
          penalty='l2',
          alpha=regularization_coef,
          class_weight='balanced',
          max_iter=1_000_000,
          tol=1e-3,
          random_state=42,
      )
    else:
      return linear_model.LogisticRegression(
          C=regularization_coef,
          class_weight='balanced',
          penalty='l2',
          max_iter=10_000_000,
      )


def predict_with_probe(
    probe: LinearProbe,
    features: np.ndarray,
    is_regression: bool,
) -> np.ndarray:
  """Computes trained linear probe's predictions."""
  # pytype: disable=attribute-error
  if is_regression:
    return probe.predict(features)
  else:
    return probe.predict_proba(features)[:, 1]
  # pytype: enable=attribute-error


def find_reg_coef_with_best_metric(
    cv_scores: dict[float, float], lower_is_better: bool
) -> float:
  """Finds regularization coef with best metric.

  Args:
    cv_scores: A map between regularization coefficients and the cross-validated
      performance (ROCAUC for classification or MAE for regression) on the
      held-out folds.
    lower_is_better: A boolean indicating if the metric is best when lowest
      (True) or highest (False).

  Returns:
    The key of `cv_scores` corresponding to the best metric.
  """
  best_alpha = -1.0
  if lower_is_better:
    best_metric = 1e50
  else:
    best_metric = 0
  for alpha, metric in cv_scores.items():
    if lower_is_better:
      if metric < best_metric:
        best_alpha = alpha
        best_metric = metric
    else:
      if metric > best_metric:
        best_alpha = alpha
        best_metric = metric
  return best_alpha


def compute_metrics_for_probe(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_regression: bool,
) -> float:
  if is_regression:
    return metrics.mean_absolute_error(y_true=y_true, y_pred=y_score)
  else:
    return metrics.roc_auc_score(y_true=y_true, y_score=y_score)


def train_linear_probe_with_participant_level_crossval(
    features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    is_regression: bool,
    n_folds: int = 5,
    use_sgd_classifier: bool = True,
    stratify_per_label: bool = True,
) -> LinearProbe:
  """Trains a linear probe using cross-validated l2 penalization parameter."""
  assert features.shape[0] == labels.shape[0] == participant_ids.shape[0]

  if is_regression:
    label_by_participant_ids = (
        pd.DataFrame({'participant_id': participant_ids, 'label': labels})
        .groupby('participant_id')
        .mean()
    )
    label_by_participant_ids = label_by_participant_ids.label.to_dict()
  else:
    label_by_participant_ids = dict(zip(participant_ids, labels))
  unique_participant_ids = np.array(list(set(participant_ids)))
  unique_labels = np.array(
      [label_by_participant_ids[k] for k in unique_participant_ids]
  )
  if stratify_per_label and not is_regression:
    folds = list(
        model_selection.StratifiedKFold(
            n_folds, shuffle=True, random_state=43
        ).split(unique_participant_ids, unique_labels)
    )
  else:
    folds = list(
        model_selection.KFold(n_folds, shuffle=True, random_state=43).split(
            unique_participant_ids
        )
    )

  cv_scores = {}
  for alpha in np.logspace(-5, 5, num=50):
    cross_validated_metrics = 0

    for random_seed, (train_idx, test_idx) in enumerate(folds):

      # `train_idx` and `test_idx` are arrays of integers corresponding to
      # indices within `unique_participant_ids`. They take values in
      # `range(len(unique_participant_ids))`.
      train_unique_participant_idx = set(unique_participant_ids[train_idx])
      test_unique_participant_idx = set(unique_participant_ids[test_idx])
      assert not (test_unique_participant_idx & train_unique_participant_idx)

      keep_train = [
          pid in train_unique_participant_idx for pid in participant_ids
      ]
      train_fold_features = features[keep_train] * 1
      train_fold_labels = labels[keep_train] * 1
      train_fold_features, train_fold_labels = sk_utils.shuffle(
          train_fold_features, train_fold_labels, random_state=random_seed
      )

      keep_test = [
          pid in test_unique_participant_idx for pid in participant_ids
      ]
      test_fold_features = features[keep_test]
      test_fold_labels = labels[keep_test]

      lr = create_linear_probe(
          regularization_coef=alpha,
          is_regression=is_regression,
          use_sgd_classifier=use_sgd_classifier,
      )
      lr.fit(train_fold_features, train_fold_labels)

      predictions = predict_with_probe(
          probe=lr,
          features=test_fold_features,
          is_regression=is_regression,
      )

      cross_validated_metrics += compute_metrics_for_probe(
          y_true=test_fold_labels,
          y_score=predictions,
          is_regression=is_regression,
      )

    cv_scores[alpha] = cross_validated_metrics / n_folds

  best_alpha = find_reg_coef_with_best_metric(
      cv_scores=cv_scores, lower_is_better=is_regression
  )
  lr = create_linear_probe(
      regularization_coef=best_alpha,
      is_regression=is_regression,
      use_sgd_classifier=use_sgd_classifier,
  )

  lr.fit(features, labels)
  return lr

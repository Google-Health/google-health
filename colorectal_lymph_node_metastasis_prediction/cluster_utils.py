"""Utils need to generate machine-learned features via clustering.

For example use, see demo.ipynb.
"""

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
import statsmodels.api as sm


_CASE_ID = 'case_id'


def train_k_means_model(embedding_dict, k, batch_size=10000):
  """Generate KMeans models with k clusters.

  Args:
    embedding_dict: dict mapping case id: embeddings. Embeddings have shape
      [num_patches, emb_dimensions].
    k: number of clusters.
    batch_size: size of batch to use for MiniBatchKMeans training.

  Returns:
    Trained kmeans model.
  """
  x = []
  for case_id in embedding_dict:
    x.append(embedding_dict[case_id])
  x = np.concatenate(x)
  print(f'Embeddings shape: {x.shape}')

  return MiniBatchKMeans(
      n_clusters=k, random_state=0, batch_size=batch_size).fit(x)


def get_cluster_quantitation_df(embedding_dict, model):
  """Get case-level cluster quantitation vectors.

  Computes the fraction of patches for each case that belong to each cluster.

  Args:
    embedding_dict: dict mapping case id: embeddings. Embeddings have shape
      [num_patches, emb_dimensions].
    model: trained kmeans model.

  Returns:
    pd.DataFrame of cluster quantitation vectors.
  """
  cq = {}
  for case_id, embeddings in embedding_dict.items():
    cluster_distances = model.transform(embeddings)
    cq[case_id] = _distances_to_cluster_quantitation(cluster_distances)

  df = pd.DataFrame.from_dict(cq, orient='index')
  cols = list(df.columns)
  df[_CASE_ID] = df.index
  df = df[[_CASE_ID] + cols]
  return df


def select_top_clusters(
    df_train,
    df_valid,
    label_col,
    baseline_cols,
    cluster_cols,
    n):
  """Select top clusters and return these clusters with respective AUCs.

  The set of n `cluster_cols` that lead to the greatest gain in AUC over
  `baseline_cols` on `df_valid` are chosen via forward stepwise selection.

  Args:
    df_train: pd.Dataframe with training data.
    df_valid: pd.Dataframe with validation data.
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    cluster_cols: a list of column names in `df` corresponding to cluster
      quantitation features.
    n: number of clusters to select.

  Returns:
    pd.DataFrame of cluster ids and AUCs.
  """
  cluster_cols = cluster_cols.copy()
  selected_cluster_cols = []
  results = []

  for i in range(n):
    cluster_id, auc = _select_next_cluster(
        df_train=df_train,
        df_valid=df_valid,
        label_col=label_col,
        baseline_cols=baseline_cols,
        selected_cluster_cols=selected_cluster_cols,
        candidate_cluster_cols=cluster_cols)
    selected_cluster_cols.append(cluster_id)
    cluster_cols.remove(cluster_id)
    results.append({'order': i, 'cluster_id': cluster_id, 'auc': auc})
  return pd.DataFrame(results)


def likelihood_ratio_test(
    df,
    label_col,
    baseline_cols,
    cluster_cols):
  """Likelihood ratio test for significance of `cluster_cols`.

  Likelihood ratio test comparing the full model fit on the combination of
  `baseline_cols` and `cluster_cols` and the null model fit only on
  `baseline_cols`.

  Args:
    df: pd.Dataframe with data on which to fit logistic regression models.
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    cluster_cols: a list of column names in `df` corresponding to cluster
      quantitation features.

  Returns:
    p-value of likelihood ratio test between alternative hypothesis (with
    clusters) model and null hypothesis (without clusters) model.
  """
  lr_alt, _, _ = train_eval_lr(
      df_train=df,
      df_valid=df,
      label_col=label_col,
      baseline_cols=baseline_cols,
      cluster_cols=cluster_cols)
  lr_null, _, _ = train_eval_lr(
      df_train=df,
      df_valid=df,
      label_col=label_col,
      baseline_cols=baseline_cols,
      cluster_cols=[])

  def lrt(ll_alt, ll_null, k):
    test_stat = 2 * (ll_alt - ll_null)
    return scipy.stats.chi2.sf(test_stat, k)

  return lrt(lr_alt.llf, lr_null.llf, len(cluster_cols))


def get_odds_ratios_p_values(df, label_col, baseline_cols, cluster_cols):
  """Train LR model and return odds ratios and p-values of model parameters.

  Args:
    df: pd.Dataframe with data on which to fit logistic regression models.
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    cluster_cols: a list of column names in `df` corresponding to cluster
      quantitation features.

  Returns:
    pd.Dataframe containing model parameters with their odds ratios and
    p-values.
  """
  lr, _, _ = train_eval_lr(
      df_train=df,
      df_valid=df,
      label_col=label_col,
      baseline_cols=baseline_cols,
      cluster_cols=cluster_cols)

  point = np.exp(lr.params).apply(lambda x: f'{x:.2f}')
  lower = np.exp(lr.conf_int()[0]).apply(lambda x: f'{x:.2f}')
  upper = np.exp(lr.conf_int()[1]).apply(lambda x: f'{x:.2f}')
  or_ci = point + ' ' + '[' + lower + ',  ' + upper + ']'

  def p_value_to_str(p, num_digits=3):
    min_value = 10**(-num_digits)
    if p < min_value:
      return f'<{min_value}'
    pattern = f'%.{str(num_digits)}f'
    return pattern % p

  p = lr.pvalues.apply(p_value_to_str)
  return pd.DataFrame({'OR': or_ci, 'p': p})


def get_eval_aucs(df_train, df_valid, label_col, baseline_cols, cluster_cols):
  """Evaluate models' predictive performance.

  Trains logistic regression models with baseline_cols, cluster_cols, and both
  sets of features and computes AUC on separate validation dataset.

  Args:
    df_train: pd.Dataframe with training data.
    df_valid: pd.Dataframe with validation data (e.g., `validation` or `test`).
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df_train` and `df_test`
      corresponding to baseline features.

    cluster_cols: a list of column names in `df_train` and `df_test`
      corresponding to cluster quantitation features.


  Returns:
    pd.Dataframe containing AUCs.
  """
  _, _, auc_baseline = train_eval_lr(
      df_train=df_train,
      df_valid=df_valid,
      label_col=label_col,
      baseline_cols=baseline_cols,
      cluster_cols=[])
  _, _, auc_cluster = train_eval_lr(
      df_train=df_train,
      df_valid=df_valid,
      label_col=label_col,
      baseline_cols=[],
      cluster_cols=cluster_cols)
  _, _, auc_all = train_eval_lr(
      df_train=df_train,
      df_valid=df_valid,
      label_col=label_col,
      baseline_cols=baseline_cols,
      cluster_cols=cluster_cols)
  return pd.DataFrame({
      'Baseline features only': [auc_baseline],
      'Cluster features only': [auc_cluster],
      'Baseline + cluster features': [auc_all]}, index=['AUC'])


def train_eval_lr(df_train, df_valid, label_col,
                  baseline_cols, cluster_cols):
  """Train and evaluate LR.

  Train logistic regression model on `df_train` using specified `baseline_cols`
  and `cluster_cols`, then evaluate performance on `df_valid`.

  Args:
    df_train: pd.Dataframe with training data.
    df_valid: pd.Dataframe with validation data.
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    cluster_cols: a list of column names in `df` corresponding to cluster
      quantitation features.

  Returns:
    tuple: (LR model, validation set predictions, AUC evaluated on `df_valid`).
  """
  x_train, y_train = _get_lr_data(
      df_train,
      label_col,
      baseline_cols=baseline_cols,
      cluster_cols=cluster_cols)
  x_valid, y_valid = _get_lr_data(
      df_valid,
      label_col,
      baseline_cols=baseline_cols,
      cluster_cols=cluster_cols)
  lr = train_lr(x_train, y_train)
  y_hat = lr.predict(x_valid)
  auc = sklearn.metrics.roc_auc_score(y_valid, y_hat)
  return lr, y_hat, auc


def train_lr(x, y):
  """Returns trained logistic regression model."""
  return sm.Logit(y, x).fit(disp=0)


def _distances_to_cluster_quantitation(cluster_distances):
  """Converts distances to cluster centroids to cluster quantitation vector.

  Args:
    cluster_distances: ndarray with shape [num_patches, k].

  Returns:
    ndarray with shape [k] reflecting percent of patches assigned to each
    cluster in the case.
  """
  if len(cluster_distances.shape) != 2:
    raise ValueError('Expect cluster distances to be of rank 2')
  k = cluster_distances.shape[1]
  min_distances = cluster_distances.min(axis=1, keepdims=True)
  min_distances = np.tile(min_distances, [1, k])
  cluster_quants = np.mean((cluster_distances == min_distances), axis=0)
  assert cluster_quants.shape == (k,)
  return cluster_quants


def _get_lr_data(df, label_col, baseline_cols=None, cluster_cols=None):
  """Get X and y np.arrays for fitting logistic regression.

  Args:
    df: pd.Dataframe with labels, baseline features, and cluster features.
    label_col: name of column in `df` to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    cluster_cols: a list of column names in `df` corresponding to cluster
      quantitation features.

  Returns:
    Tuple of (features, labels).
  """
  x = df[baseline_cols + cluster_cols]
  y = df[label_col]
  return x, y


def _select_next_cluster(
    df_train,
    df_valid,
    label_col,
    baseline_cols,
    selected_cluster_cols,
    candidate_cluster_cols):
  """Select next best candidate cluster.

  Selects the next cluster from `candidate_cluster_cols` that gives the best AUC
  on `df_valid` when added to a logistic regression model trained on `df_train`
  using `baseline_cols` and `selected_cluster_cols` as features.

  Args:
    df_train: pd.Dataframe with training data.
    df_valid: pd.Dataframe with validation data.
    label_col: column to use for labels.
    baseline_cols: a list of column names in `df` corresponding to baseline
      features.
    selected_cluster_cols: a list of column names in `df` corresponding to
      candidate cluster quantitation features that have already been added to
      the model.
    candidate_cluster_cols: a list of column names in `df` corresponding to
      cluster quantitation features that are candidates for being added to the
      model.

  Returns:
    Tuple: (next best cluster, AUC obtained when using this cluster).
  """
  aucs = {}
  AUC_INDEX = 2  # pylint: disable=invalid-name
  for cluster_col in candidate_cluster_cols:
    assert cluster_col not in selected_cluster_cols
    cluster_cols = selected_cluster_cols + [cluster_col]
    aucs[cluster_col] = train_eval_lr(
        df_train=df_train,
        df_valid=df_valid,
        label_col=label_col,
        baseline_cols=baseline_cols,
        cluster_cols=cluster_cols)[AUC_INDEX]
  top_cluster_id = max(aucs, key=aucs.get)
  return top_cluster_id, aucs[top_cluster_id]

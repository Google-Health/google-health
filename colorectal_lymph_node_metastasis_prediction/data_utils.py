"""Data-processing utils need to prep data for feature generation and selection.

For example use, see demo.ipynb.
"""

import pandas as pd


def bin_age(age, start_cutoff=60, stop_cutoff=80, increment=10):
  """Categorize age by bins.

  Args:
    age: age to bin.
    start_cutoff: first cutoff to use forming bins.
    stop_cutoff: last cutoff to use forming bins.
    increment: difference in age between bins.

  Returns:
    string representing binned age.
  """
  if pd.isnull([age]):
    return age
  age = float(age)

  last_cutoff = 0
  for age_cutoff in range(start_cutoff, stop_cutoff + increment, increment):
    if age < age_cutoff:
      return f'{last_cutoff}-{age_cutoff-1}'
    last_cutoff = age_cutoff
  return f'>={stop_cutoff}'


def prep_features(df, feature_cols, label_cols):
  """Returns a pd.DataFrame suitable for modeling.

  1) Select only desired `feature_cols`.
  2) Remove rows with nan values.
  3) Convert categorical features to dummy variables.
  4) Remove constant `feature_cols`.

  Args:
    df: pd.DataFrame containing `feature_cols`, `label_cols`.
    feature_cols: a list of columns in `df` containing regression features.
    label_cols: a list of column names in `df` containing labels. These columns
      are not coded as dummies if they are categorical.

  Returns:
    tuple of (dataframe, expanded feature cols).
  """
  # Select subset of required columns.
  df = df.copy()[label_cols + feature_cols]
  # Remove rows with missing values.
  n_rows = df.shape[0]
  df = df.dropna()
  delta = n_rows - df.shape[0]
  if delta > 0:
    print('Dropped %d rows due to missing values.' % delta)

  # Convert categorical cols to dummy vars.
  df_labels = df[label_cols]
  df = pd.get_dummies(df[feature_cols], drop_first=True)
  expanded_feature_cols = list(df.columns)

  # Remove constant feature columns
  df = df.loc[:, (df != df.iloc[0]).any()]
  delta = set(expanded_feature_cols) - set(df.columns)
  if delta:
    print('Dropped %s constant feature columns.' % list(delta))

  # Add labels to regression
  df = pd.concat([df_labels, df], axis=1)
  return df, expanded_feature_cols

"""Tests for stage2_features."""

import unittest

import numpy as np

import stage2_features

# Test tolerance to 2 decimal digits.
_absolute_tolerance = 0.01


class Stage2FeaturesTest(unittest.TestCase):

  def test_np_tf_featurizer(self):
    # This heatmap corresponds to:
    # [[np.nan, G1, G1, G2],
    #  [np.nan, G1, G2, G3]],
    # where G1..3 = Grade 1..3, and
    # [[np.nan, B,   IC, IC],
    #  [np.nan, CIS, IC, IC]],
    # where B = benign, IC = invasive carcinoma, and CIS = carcinoma in situ.
    # pyformat: disable
    heatmap = np.array([
        [[np.nan, 1.0, 1.0, 0.0],
         [np.nan, 1.0, 0.0, 0.0]],  # Grade 1
        [[np.nan, 0.0, 0.0, 1.0],
         [np.nan, 0.0, 1.0, 0.0]],  # Grade 2
        [[np.nan, 0.0, 0.0, 0.0],
         [np.nan, 0.0, 0.0, 1.0]],  # Grade 3
    ])
    ic_heatmap = np.array([
        [[np.nan, 1.0, 0.1, 0.2],
         [np.nan, 0.1, 0.0, 0.0]],  # Benign
        [[np.nan, 0.0, 0.8, 0.7],
         [np.nan, 0.2, 0.8, 0.9]],  # Invasive Carcinoma
        [[np.nan, 0.0, 0.1, 0.1],
         [np.nan, 0.7, 0.2, 0.1]],  # Carcinoma In Situ
    ])
    # pyformat: enable
    tmap = {
        'heatmap': np.moveaxis(heatmap, 0, -1),
        'ic_heatmap': np.moveaxis(ic_heatmap, 0, -1)
    }
    # First three are area within IC, latter three are outside of IC.
    expected_feature = np.array([0.25, 0.5, 0.25, 1.0, 0.0, 0.0])

    actual_feature = stage2_features.np_tf_featurizer(tmap)

    np.testing.assert_allclose(
        actual_feature, expected_feature, atol=_absolute_tolerance)

  def test_np_tf_7d_heatmap(self):
    # This heatmap corresponds to:
    # [[np.nan, G0, G1, G2],
    #  [np.nan, G1, G2, G3]],
    # where G1..3 = NP/TF Grade 1..3, and
    # [[np.nan, B,   IC, IC],
    #  [np.nan, CIS, IC, IC]],
    # where B = benign, IC = invasive carcinoma, and CIS = carcinoma in situ.
    # pyformat: disable
    heatmap = np.array([
        [[np.nan, 1.0, 0.0, 0.0],
         [np.nan, 0.0, 0.0, 0.0]],  # Grade 0
        [[np.nan, 0.0, 1.0, 0.0],
         [np.nan, 1.0, 0.0, 0.0]],  # Grade 1
        [[np.nan, 0.0, 0.0, 1.0],
         [np.nan, 0.0, 1.0, 0.0]],  # Grade 2
        [[np.nan, 0.0, 0.0, 0.0],
         [np.nan, 0.0, 0.0, 1.0]],  # Grade 3
    ])
    ic_heatmap = np.array([
        [[np.nan, 1.0, 0.0, 0.0],
         [np.nan, 0.1, 0.0, 0.0]],  # Benign
        [[np.nan, 0.0, 0.8, 0.7],
         [np.nan, 0.2, 0.8, 0.9]],  # Invasive Carcinoma
        [[np.nan, 0.0, 0.1, 0.1],
         [np.nan, 0.7, 0.2, 0.1]],  # Carcinoma In Situ
    ])
    # pyformat: enable
    tmap = {
        'heatmap': np.moveaxis(heatmap, 0, -1),
        'ic_heatmap': np.moveaxis(ic_heatmap, 0, -1)
    }

    # First three are area within IC, latter three are outside of IC.
    expected_feature = np.array([0.25, 0.5, 0.25, 1.0, 0.0, 0.0])

    actual_feature = stage2_features.np_tf_featurizer(tmap)

    np.testing.assert_allclose(actual_feature, expected_feature)

  def test_mc_featurizer(self):
    # 101 elements so that n-th percentile is exactly n.
    n_density_elem = 101
    # Two rows: one for actual density, and one for non-tissue area.
    density_map = np.zeros((2, n_density_elem))
    density_map[0, :] = np.arange(n_density_elem)
    # Add Nan for non-tissue area.
    density_map[1, :] = np.nan

    tmap = {'density': density_map}
    # Current feature set is 5th, 25th, 50th, 75th, and 95th percentile.
    expected_feature = np.array([5, 25, 50, 75, 95])

    actual_feature = stage2_features.mc_featurizer(tmap)

    np.testing.assert_allclose(
        actual_feature, expected_feature, atol=_absolute_tolerance)


if __name__ == '__main__':
  unittest.main()

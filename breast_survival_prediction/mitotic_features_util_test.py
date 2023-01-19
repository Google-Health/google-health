"""Tests for mitotic_features_util."""

import unittest

import numpy as np

import mitotic_features_util

# Expect mitosis to be full 3x3 detection, so this heatmap has two mitoses
# at top, left and bottom, right corners.
# pylint: disable=bad-whitespace
_HEATMAP = [[1, 1, 1,   0, 0, 1, 0],
            [1, 1, 1,   1, 0, 0, 0],
            [1, 1, 1,   0, 1, 1, 1],
            [0, 0, 0,   0, 1, 1, 1],
            [1, 0, 0, 0.5, 1, 1, 1]]  # pyformat: disable
_MASK = [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0]]  # pyformat: disable
# pylint: enable=bad-whitespace
_EXPECTED_DETECTION = [(1, 1), (3, 5)]
_EXPECTED_DENSITY_MAP = [[1, 0, 0],
                         [0, 0, 1]]  # pyformat: disable
_EXPECTED_MASKED_DETECTION = [(1, 1)]
_EXPECTED_MASKED_DENSITY_MAP = [[1, np.nan, np.nan],
                                [0, np.nan, np.nan]]  # pyformat: disable
_WINDOW_SIZE = 2
_STRIDE = 2
_DETECTION_TH = 0.7
_MORPH_ERODE_SIZE = 3


class MitoticFeaturesUtilTest(unittest.TestCase):

  def test_mitosis_detection(self):
    heatmap = np.array(_HEATMAP)
    expected_detection = np.array(_EXPECTED_DETECTION)
    actual_detection = mitotic_features_util.heatmap_to_list(
        heatmap,
        detection_th=_DETECTION_TH,
        mask=None,
        morph_erode_size=_MORPH_ERODE_SIZE)
    # Sort by row index.
    actual_detection = sorted(actual_detection, key=lambda x: x[0])
    # Tolerate up to half pixel error.
    np.testing.assert_allclose(expected_detection, actual_detection, atol=0.55)

  def test_density_calculation(self):
    # Applying 2x2 windows to the detection of heatmap above.
    expected_density_map = np.array(_EXPECTED_DENSITY_MAP) / (_WINDOW_SIZE**2)
    actual_density_map = mitotic_features_util.calculate_density(
        _EXPECTED_DETECTION, (5, 7), _WINDOW_SIZE, _STRIDE, mask=None)
    np.testing.assert_allclose(
        expected_density_map, actual_density_map, atol=0.01)


if __name__ == '__main__':
  unittest.main()

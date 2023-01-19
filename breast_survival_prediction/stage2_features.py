"""Features for Nottingham Stage 2 models."""

from typing import Mapping, Sequence, Union
import numpy as np

# Number of possible NF/TF grades
_NUM_NPTF_GRADES = 3


def _calc_normalized_area(heatmap: np.ndarray) -> np.ndarray:
  """Calculates normalized area for each class in the heatmap.

  Args:
    heatmap: 3D array of heatmap (channel order: HWC).

  Returns:
    1D array of normalized area for each channel of the heatmap.
  """
  area = np.nansum(heatmap, axis=(0, 1))
  normalized_area = area / (np.nansum(area) + np.finfo(float).eps)
  return normalized_area


def np_tf_featurizer(
    tmap: Mapping[str, Union[Sequence[Sequence[float]],
                             np.ndarray]]) -> np.ndarray:
  """Featurization for Nuclear Pleomorphism (NP) and Tubule Formation (TF).

  Args:
    tmap: Dictionary of tensors. Expected to contain two 3D heatmaps of
      probabilities (channel order: HWC) representing NP/TF model output (keyed
      'heatmap' and the invasive carcinoma segmentation model output (keyed
      'ic_heatmap'). Channels of first heatmap represent NP/TF1, NP/TF2, NP/TF3.
      While the ic_heatmap represents Benign, Invasive Carcinoma, Carcinoma in
      situ. The first heatmap can optionally contain NP/TF0 which indicate
      redundant non-invasive carcinoma segmentation by the NP/TF model. Both
      heatmaps are expected to be of the same size.

  Returns:
    Normalized area of NP/TF grade 1, 2, 3 within tumor area and outside tumor
      area resulting in 6 numbers in total.
  """

  ic_heatmap = tmap['ic_heatmap']
  ic_positive_mask = np.argmax(ic_heatmap, axis=-1) == 1
  ic_positive_mask = np.expand_dims(ic_positive_mask, -1)

  nptf_heatmap = tmap['heatmap']
  # Some heatmap may have extra dim in front, so selecting only the last
  # _NUM_NPTF_GRADES channels.
  nptf_heatmap = nptf_heatmap[..., -_NUM_NPTF_GRADES:]
  # Heatmap within invasive tumor.
  heatmap_ic = nptf_heatmap * ic_positive_mask
  # heatmap outside invasive tumor.
  heatmap_non_ic = nptf_heatmap * (1 - ic_positive_mask)
  area_ic = _calc_normalized_area(heatmap_ic)
  area_non_ic = _calc_normalized_area(heatmap_non_ic)
  return np.concatenate((area_ic, area_non_ic))


def mc_featurizer(
    tmap: Mapping[str, Union[Sequence[Sequence[float]],
                             np.ndarray]]) -> np.ndarray:
  """Featurization for Mitotic Count (MC).

  Args:
    tmap: Mapping of precomputed mitotic features. Expected to contain density
      map (a 2D array) under the key 'density'.

  Returns:
    The 5th, 25th, 50th, 75th, and 95th percentiles of the density map
  """
  density_map = tmap['density']
  # drop NaN
  density_map = density_map[~np.isnan(density_map)]
  if not density_map.size:
    density_map = np.array([0])
  # An alternative is to include histogram and total count.
  return np.percentile(density_map, [5, 25, 50, 75, 95])

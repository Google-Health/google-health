"""Utilities function for calculating second-stage features for mitotic model."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
import PIL.Image


def heatmap_to_list(
    hm: np.ndarray,
    detection_th: float,
    mask: Optional[np.ndarray] = None,
    morph_erode_size: Optional[int] = None) -> List[Tuple[float]]:
  """Detect mitosis on the heatmap.

  Args:
    hm: 2D heatmap output from the mitotic model.
    detection_th: Probability threshold for detection (float between 0-1).
    mask: Area not to consider such as out of tissue or out of tumor.
    morph_erode_size: Size of morphological eroding structuring element. This is
      used as the clean up step.

  Returns:
    List of (row, column) heatmap-coordinate of the detected centroid.
  """
  binarized_hm = hm > detection_th
  if mask is not None:
    binarized_hm = binarized_hm * mask
  if morph_erode_size and morph_erode_size > 0:
    binarized_hm = cv2.morphologyEx(
        binarized_hm.astype('uint8'), cv2.MORPH_ERODE,
        np.ones(morph_erode_size))
  # into one connected component, see
  # https://docs.opencv.org/3.4/dd/d46/imgproc_8hpp.html
  # Index 3 is the centroid among other info.
  detected_centroids = cv2.connectedComponentsWithStats(
      binarized_hm.astype('uint8'), 8, cv2.CV_32S
  )[3]
  # The centroids include the background label, so we slice from the second
  # element onward.
  detected_centroids = detected_centroids[1:]
  # convert to (row, column) format so that this is consistent with indexing.
  return [(pt[1], pt[0]) for pt in detected_centroids]


def calculate_density(detected_centroids: List[Tuple[float]],
                      heatmap_shape: List[int],
                      window_size: int,
                      stride: int,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
  """Calculate density map.

  Args:
    detected_centroids: list of detected centroids in original heatmap's
      coordinate. This can be the output of the heatmap_to_list function.
    heatmap_shape: original heatmap shape (row, column).
    window_size: size of windows to calculate density.
    stride: overlap between each density windows. This is what control the size
      of density map, similar to prediction_size in the inference pipeline.
    mask: binary mask that specify area to compute density. If set, things
      outside of mask will get NaN.

  Returns:
    numpy array representing the density map.
  """
  density_shape = (int(heatmap_shape[0] // stride),
                   int(heatmap_shape[1] // stride))
  window_area = window_size**2
  density_map = np.zeros(density_shape)
  density_map[:] = np.nan

  def _is_inside(pt, tl):
    y, x = pt[0], pt[1]
    yb, xb = tl[0], tl[1]
    return (x >= xb) and (x <= xb + window_size) and (y >= yb) and (
        y <= yb + window_size)

  if mask is not None:
    # PIL expects (width, height), but the shape is
    # (row, column) = (height, width)
    m_pil = PIL.Image.fromarray(mask).resize(
        (density_shape[1], density_shape[0]), PIL.Image.Resampling.NEAREST
    )
    mask = np.array(m_pil)
  for i in range(density_shape[0]):
    for j in range(density_shape[1]):
      topleft = [i * stride, j * stride]
      if (mask is not None) and (not mask[i][j]):
        continue
      inbox = [_is_inside(pt, topleft) for pt in detected_centroids]
      density_map[i][j] = float(np.sum(inbox))
  return density_map / window_area

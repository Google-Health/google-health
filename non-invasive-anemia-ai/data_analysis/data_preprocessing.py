# Copyright (c) 2026, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Data Preprocessing utilities for Non Invasive Anemia AI."""

import io
import itertools
from typing import Any, Callable, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from data_analysis import config


def read_image_bytes(
    image_path_or_bytes: Union[str, bytes, io.BytesIO],
) -> bytes:
  """Fetches raw image bytes from a file path or in-memory byte stream.

  Args:
    image_path_or_bytes: File path, bytes, or BytesIO object containing the
      image.

  Returns:
    Raw image bytes.

  Raises:
    TypeError: If the input type is not supported.
  """
  if isinstance(image_path_or_bytes, bytes):
    return image_path_or_bytes
  if isinstance(image_path_or_bytes, io.BytesIO):
    return image_path_or_bytes.getvalue()
  if isinstance(image_path_or_bytes, str):
    with open(image_path_or_bytes, "rb") as f:
      return f.read()
  raise TypeError(f"Unsupported image input type: {type(image_path_or_bytes)}")


def load_image(
    image_path: Union[str, bytes, io.BytesIO],
    convert_to_bgr: bool = True,
    convert_to_grayscale: bool = False,
) -> np.ndarray:
  """Loads an OpenCV image from a file path or bytes buffer.

  Args:
    image_path: File path or bytes buffer containing the image.
    convert_to_bgr: Whether to convert decoded image from RGB to BGR.
    convert_to_grayscale: Whether to convert image to grayscale.

  Returns:
    Loaded numpy array representation of the image.
  """
  image_bytes = read_image_bytes(image_path)
  nparr = np.frombuffer(image_bytes, np.uint8)

  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if convert_to_bgr and img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if convert_to_grayscale and img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  return img


def detect_reference_line_in_conjunctiva_image(
    image: np.ndarray,
    line_config=config.LINE_DETECTION_CONFIG,
) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
  """Detects dominant straight reference lines in the conjunctiva image.

  First, uses Canny edge detection over the image. Next, uses Hough line
  transform over the detected edges to detect straight lines in the image
  (in Polar coordinates).

  Args:
    image: Grayscale image numpy array.
    line_config: Line detection configuration.

  Returns:
    A tuple ((x1, y1), (x2, y2), theta) representing the primary detected line.
  """
  edges = cv2.Canny(
      image,
      line_config.canny_lower_threshold,
      line_config.canny_upper_threshold,
  )
  detected_lines = cv2.HoughLines(
      edges,
      line_config.hough_rho_resolution,
      line_config.hough_theta_resolution,
      line_config.hough_threshold,
  )

  if detected_lines is None:
    _, image_width = image.shape[:2]
    return (
        (0, 0),
        (image_width, 0),
        0.0,
    )

  rho, theta = np.asarray(
      detected_lines[0][0],
      dtype=np.float64,
  )

  cosine_theta = np.cos(theta)
  sine_theta = np.sin(theta)

  x_origin = cosine_theta * rho
  y_origin = sine_theta * rho

  line_length = line_config.line_extension_length

  first_point = (
      int(x_origin - line_length * sine_theta),
      int(y_origin + line_length * cosine_theta),
  )

  second_point = (
      int(x_origin + line_length * sine_theta),
      int(y_origin - line_length * cosine_theta),
  )

  return first_point, second_point, theta


def zero_fill_above_line(
    image_rgb: np.ndarray,
    x2: int,
    y1: int,
    y2: int,
    denom: float,
    extra_rows: int = config.EYE_CROP_CONFIG.zero_fill_extra_rows,
) -> None:
  """Fills pixels above the detected reference line with zeros (black) in-place."""
  height, width, _ = image_rgb.shape
  for x, y in itertools.product(range(width), range(height)):
    if (x - x2) * (y2 - y1) / denom + y2 - y >= 0:
      image_rgb[y : y + extra_rows + 1, x, :] = 0


def extract_eye_region(
    image_path: Union[str, bytes, io.BytesIO, np.ndarray],
    bbox_limit_fn: Callable[[Any], Any] = np.min,
    align_eye: bool = True,
    zero_fill: bool = False,
    crop_config=config.EYE_CROP_CONFIG,
    line_config=config.LINE_DETECTION_CONFIG,
) -> np.ndarray:
  """Heuristic to closely crop the eye/conjunctiva bounding box.

  A sheet of paper covers the forehead. The lower edge of the paper is used as a
  reference line for cropping the eye region.

  Args:
    image_path: File path, bytes, or numpy array of the input image.
    bbox_limit_fn: Function to aggregate coordinate limits (default: np.min).
    align_eye: Whether to rotate the cropped region along the detected line.
    zero_fill: Whether to fill pixels above detected line with black.
    crop_config: Configuration for cropping.
    line_config: Configuration for line detection.

  Returns:
    Cropped (and optionally aligned) eye bounding box image array.
  """
  if isinstance(image_path, np.ndarray):
    image_rgb = image_path.copy()
  else:
    image_rgb = load_image(image_path, convert_to_bgr=False)

  gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

  height, width, _ = image_rgb.shape

  # If the height is less than the width then it needs to be rotated because
  # expected image is captured in portrait mode
  if height < width:
    image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

    # Update height & width
    height, width, _ = image_rgb.shape

  (x1, y1), (x2, y2), theta = detect_reference_line_in_conjunctiva_image(
      gray_image, line_config=line_config
  )

  # If `zero_fill` is `True`, we will fill all the pixels above the detected
  # paper edge with zero (black). This may not be visible especially if the
  # bounding box is aligned to the paper edge but in the case that the paper has
  # not been held straight, some pixel values will be replaced with black.
  denom = x2 - x1 if x2 != x1 else 1e-5
  if zero_fill:
    zero_fill_above_line(
        image_rgb,
        x2=x2,
        y1=y1,
        y2=y2,
        denom=denom,
        extra_rows=crop_config.zero_fill_extra_rows,
    )

  # Find top left & right corners of the cropped eye bounding box
  # Equation of a line between two points is:
  # y = (y2 - y1) * (x - x1) / (x2 - x1) + y1
  crop_start = min(crop_config.approximate_crop_start_x, width - 1)
  crop_end = min(crop_config.approximate_crop_end_x, width)
  if crop_start >= crop_end:
    crop_start = 0
    crop_end = width

  y1_new = (y2 - y1) * (crop_start - x1) / denom + y1
  y2_new = (y2 - y1) * (crop_end - x1) / denom + y1

  # Find the absolute difference - this value is used to crop out the paper if
  # the bounding box is rotated
  y_diff = np.round(np.abs(y1_new - y2_new)).astype(int)

  # Find the y-coordinates of the crop
  y = int(np.clip(bbox_limit_fn([y1_new, y2_new]), 0, height - 1))
  image_bbox = image_rgb[y:, crop_start:crop_end, :]

  # Rotate the bounding box of the eye to line up with the edge of the paper
  if align_eye and image_bbox.size > 0:
    bbox_h, bbox_w = image_bbox.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D(
        center=(bbox_w / 2, bbox_h / 2), angle=theta, scale=1
    )
    rotated_bbox = cv2.warpAffine(
        image_bbox,
        rotation_mat,
        image_bbox.shape[1::-1],
        flags=cv2.INTER_LINEAR,
    )
    y_diff = min(y_diff, max(0, rotated_bbox.shape[0] - 1))
    return rotated_bbox[y_diff:, :]
  else:
    return image_bbox


def resize_image(
    image: Union[Image.Image, np.ndarray],
    crop_h: int = config.EYE_CROP_CONFIG.target_height,
    crop_w: int = config.EYE_CROP_CONFIG.target_width,
) -> Image.Image:
  """Resizes image to target dimensions.

  Args:
    image: PIL Image or numpy array.
    crop_h: Target height (default: 256).
    crop_w: Target width (default: 256).

  Returns:
    Resized PIL Image.
  """
  if isinstance(image, np.ndarray):
    image = Image.fromarray(image)
  resample_mode = Image.Resampling.NEAREST
  return image.resize((crop_w, crop_h), resample=resample_mode)


def convert_rgb_to_lab(
    image: Union[Image.Image, np.ndarray],
) -> np.ndarray:
  """Converts an RGB PIL image or numpy array into CIE-LAB color space.

  Args:
    image: PIL Image or numpy array.

  Returns:
    CIE-LAB color space numpy array.
  """
  if isinstance(image, Image.Image):
    image_np = np.array(image)
  else:
    image_np = image
  return cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

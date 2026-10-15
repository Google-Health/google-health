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
"""Unit tests for data_preprocessing module."""

from unittest import mock

from absl.testing import absltest
import cv2
import numpy as np
from PIL import Image

from data_analysis import config
from data_analysis import data_preprocessing


class DataPreprocessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Create a synthetic test RGB image
    self.test_img_np = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a diagonal line to trigger Hough line detection
    cv2.line(self.test_img_np, (50, 100), (550, 400), (255, 255, 255), 5)
    success, buffer = cv2.imencode('.png', self.test_img_np)
    self.assertTrue(success)
    self.test_image_bytes = buffer.tobytes()

  def test_load_image_from_bytes(self):
    """Tests loading an image from bytes."""
    img = data_preprocessing.read_image_bytes(self.test_image_bytes)
    self.assertIsNotNone(img)

  def test_load_image_grayscale(self):
    """Tests loading an image in grayscale."""
    img = data_preprocessing.load_image(
        self.test_image_bytes, convert_to_bgr=False, convert_to_grayscale=True
    )
    self.assertIsNotNone(img)
    self.assertLen(img.shape, 2)
    self.assertEqual(img.shape, (480, 640))

  def test_detect_lines_in_conjunctiva_image(self):
    """Tests detecting lines in a conjunctiva image."""
    gray = cv2.cvtColor(self.test_img_np, cv2.COLOR_RGB2GRAY)

    line_config = config.LineDetectionConfig(
        canny_lower_threshold=50,
        canny_upper_threshold=150,
        hough_rho_resolution=1,
        hough_theta_resolution=np.pi / 180,
        hough_threshold=20,
        line_extension_length=10,
    )

    # Mock the OpenCV functions so the test is deterministic.
    rho = 100.0
    theta = np.pi / 4

    expected_edges = np.zeros_like(gray)
    expected_hough_lines = np.array(
        [[[rho, theta]]],
        dtype=np.float32,
    )

    with mock.patch.object(
        cv2,
        'Canny',
        return_value=expected_edges,
    ) as mock_canny, mock.patch.object(
        cv2,
        'HoughLines',
        return_value=expected_hough_lines,
    ) as mock_hough_lines:

      (x1, y1), (x2, y2), actual_theta = (
          data_preprocessing.detect_reference_line_in_conjunctiva_image(
              gray,
              line_config=line_config,
          )
      )

    self.assertEqual(x1, 63)
    self.assertEqual(y1, 77)
    self.assertEqual(x2, 77)
    self.assertEqual(y2, 63)
    self.assertAlmostEqual(actual_theta, np.pi / 4, places=5)

    mock_canny.assert_called_once_with(
        gray,
        50,
        150,
    )
    mock_hough_lines.assert_called_once_with(
        expected_edges,
        1,
        np.pi / 180,
        20,
    )

  def test_extract_eye_region(self):
    """Tests extracting the eye region from a conjunctiva image."""
    bbox = data_preprocessing.extract_eye_region(
        self.test_img_np, align_eye=True, zero_fill=False
    )
    self.assertIsNotNone(bbox)
    self.assertGreater(bbox.size, 0)
    self.assertLen(bbox.shape, 3)

  def test_extract_eye_region_zero_fill(self):
    """Tests extracting the eye region with zero filling."""
    bbox = data_preprocessing.extract_eye_region(
        self.test_img_np, align_eye=False, zero_fill=True
    )
    self.assertIsNotNone(bbox)
    self.assertGreater(bbox.size, 0)

  def test_resize_image(self):
    """Tests resizing an image."""
    pil_img = Image.fromarray(self.test_img_np)
    resized = data_preprocessing.resize_image(pil_img, crop_h=256, crop_w=256)
    self.assertEqual(resized.size, (256, 256))

  def test_convert_rgb_to_lab(self):
    """Tests converting an RGB image to LAB color space."""
    lab = data_preprocessing.convert_rgb_to_lab(self.test_img_np)
    self.assertEqual(lab.shape, self.test_img_np.shape)


if __name__ == '__main__':
  absltest.main()

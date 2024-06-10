import enum
import io
import os
from typing import Mapping

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image

import image_utils


class BitWidth(enum.Enum):
  B16 = '16-bits'
  B8 = '8-bits'


_TEST_DATA_DIR = 'google_health/ct_dicom/testdata'

_NP_IMAGE_FILENAME_BY_BIT_WIDTH = {
    BitWidth.B8: os.path.join(_TEST_DATA_DIR, 'img_8c1.npy'),
    BitWidth.B16: os.path.join(_TEST_DATA_DIR, 'img_16c1.npy'),
}
_PNG_IMAGE_FILENAME_BY_BIT_WIDTH = {
    BitWidth.B8: os.path.join(_TEST_DATA_DIR, 'img_8c1.png'),
    BitWidth.B16: os.path.join(_TEST_DATA_DIR, 'img_16c1.png'),
}
_PIXEL_ERROR_MARGIN = 0.00001
# Some encoder tests are parameterized by bit width.
_DTYPE_BY_BIT_WIDTH = {
    BitWidth.B8: np.uint8,
    BitWidth.B16: np.uint16,
}
_TEST_PARAMS = tuple((bit_width,) for bit_width in BitWidth)


def _GetPixelStats(encoded_string: str) -> Mapping[str, np.array]:
  """Returns, min, max and average pixel value in the encoded_string."""
  decoded_array = Image.open(io.BytesIO(encoded_string))
  npdecoded_array = np.array(decoded_array)
  return {
      'min': np.min(npdecoded_array),
      'max': np.max(npdecoded_array),
      'ave': np.average(npdecoded_array),
  }


class TestEncodePng(parameterized.TestCase):
  """Unit tests for `encode_png()`."""

  @classmethod
  def setUpClass(cls):
    """Preloads test resources."""
    super().setUpClass()
    cls._NP_IMAGE_BY_BIT_WIDTH = {}
    cls._PNG_IMAGE_BY_BIT_WIDTH = {}

    for bit_width in BitWidth:
      logging.info('Loading: %s', _NP_IMAGE_FILENAME_BY_BIT_WIDTH[bit_width])
      with gfile.Open(_NP_IMAGE_FILENAME_BY_BIT_WIDTH[bit_width], 'rb') as f:
        cls._NP_IMAGE_BY_BIT_WIDTH[bit_width] = np.load(f)

      logging.info('Loading: %s', _PNG_IMAGE_FILENAME_BY_BIT_WIDTH[bit_width])
      with gfile.Open(_PNG_IMAGE_FILENAME_BY_BIT_WIDTH[bit_width], 'rb') as f:
        cls._PNG_IMAGE_BY_BIT_WIDTH[bit_width] = f.read()

  @parameterized.parameters(*_TEST_PARAMS)
  def testSuccess_Range(self, bit_width):
    """Tests image (w, h) = (4, 2) with maximum range of values for uint*."""
    self.assertIn(bit_width, _DTYPE_BY_BIT_WIDTH)
    dtype = _DTYPE_BY_BIT_WIDTH[bit_width]

    test_array = np.array(
        [[0, 1, 2, 3], [np.iinfo(dtype).max, 12000, 100, 150]]
    ).astype(dtype)
    png_text = image_utils.encode_png(test_array)
    result_array = np.array(Image.open(io.BytesIO(png_text)))
    np.testing.assert_array_equal(test_array, result_array)

  @parameterized.parameters(*_TEST_PARAMS)
  def testSuccess_Idempotence(self, bit_width):
    """Tests that `decode(encode(*))` is an identity op."""
    self.assertIn(bit_width, self._PNG_IMAGE_BY_BIT_WIDTH)
    loaded_png_bytes = self._PNG_IMAGE_BY_BIT_WIDTH[bit_width]

    canonical_pixels = self.DecodePng(loaded_png_bytes)
    encoded_png_bytes = image_utils.encode_png(canonical_pixels)
    actual_pixels = self.DecodePng(encoded_png_bytes)
    self.assertEqual(canonical_pixels.dtype, actual_pixels.dtype)
    self.assertEqual(canonical_pixels.shape, actual_pixels.shape)
    self.assertTrue(np.array_equal(canonical_pixels, actual_pixels))

  @parameterized.parameters(*_TEST_PARAMS)
  def testSuccess_Regression(self, bit_width):
    """Captures difference in outputs from OpenCV-based encoder."""
    self.assertIn(bit_width, self._PNG_IMAGE_BY_BIT_WIDTH)
    canonical_png_bytes = self._PNG_IMAGE_BY_BIT_WIDTH[bit_width]
    self.assertIn(bit_width, self._NP_IMAGE_BY_BIT_WIDTH)
    test_png_bytes = image_utils.encode_png(
        self._NP_IMAGE_BY_BIT_WIDTH[bit_width]
    )

    canonical_pixel_stats = _GetPixelStats(canonical_png_bytes)
    test_pixel_stats = _GetPixelStats(test_png_bytes)

    logging.info('Canonical pixels: %s', str(canonical_pixel_stats))
    logging.info('Instance pixels: %s', str(test_pixel_stats))
    logging.info(
        'Diff in average pixel values: %f',
        test_pixel_stats['ave'] - canonical_pixel_stats['ave'],
    )

    self.assertEqual(test_pixel_stats['min'], canonical_pixel_stats['min'])
    self.assertEqual(test_pixel_stats['max'], canonical_pixel_stats['max'])
    self.assertAlmostEqual(
        test_pixel_stats['ave'],
        canonical_pixel_stats['ave'],
        delta=_PIXEL_ERROR_MARGIN,
    )

  @parameterized.parameters(np.int32, np.uint32, np.int16, np.int8)
  def testFailure_Dtype(self, dtype):
    """Tests failure to convert to PNG for invalid image dimensions."""
    array = np.array([[0, 1], [2, 4]], dtype=dtype)
    with self.assertRaisesRegex(
        ValueError, 'Pixels must be either `uint8` or `uint16`.'
    ):
      image_utils.encode_png(array)

  def testFailure_Dimensions(self):
    """Tests failure to convert with wrong input dimensions."""
    test_array_3d = np.ones([2, 2, 2]).astype(np.uint16)
    with self.assertRaisesRegex(ValueError, 'Array must be 2-D.'):
      image_utils.encode_png(test_array_3d)

  def testEncodeRaisesErrorWithBadInput(self):
    with self.assertRaisesRegex(ValueError, 'empty image'):
      image_utils.encode_png(np.zeros((50, 0), dtype=np.uint16))

  def testConversionToPNGImage16bitExtremes(self):
    # Test image wXh = 4X2 with maximum range of values for unsigned int 16.
    test_array = np.array(
        [[0, 1, 2, 3], [np.iinfo(np.uint16).max, 12000, 100, 150]]
    ).astype(np.uint16)

    png_bytes = image_utils.encode_png(test_array)
    result_array = np.array(Image.open(io.BytesIO(png_bytes)))
    np.testing.assert_array_equal(test_array, result_array)

  def testConversionToPNGImage8bitExtremes(self):
    # Test image wXh = 4X2 with maximum range of values for unsigned int 8.
    test_array = np.array(
        [[0, 1, 2, 3], [np.iinfo(np.uint8).max, 128, 64, 32]]
    ).astype(np.uint8)

    png_bytes = image_utils.encode_png(test_array)
    result_array = np.array(Image.open(io.BytesIO(png_bytes)))
    np.testing.assert_array_equal(test_array, result_array)

  def DecodePng(self, png_bytes: bytes) -> np.ndarray:
    """Converts an encoded 16-bit grayscale PNG to a 2D uint16 array."""
    # The use of np.uint8 here is for the png bytes, not the pixel values.
    byte_array = np.frombuffer(png_bytes, np.uint8)
    pixel_array = np.array(Image.open(io.BytesIO(byte_array))).astype(np.uint16)
    self.assertEqual(pixel_array.dtype, np.uint16)
    self.assertEqual(pixel_array.ndim, 2)
    return pixel_array


if __name__ == '__main__':
  absltest.main()

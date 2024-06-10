"""Utilities for image encoding."""

import io

import numpy as np
import png

_NUM_BITS_PER_BYTE = 8


def encode_png(array: np.ndarray) -> bytes:
  """Converts an unsigned integer 2-D NumPy array to a PNG-encoded string.

  Unsigned 8-bit and 16-bit images are supported.

  Args:
    array: Array to be encoded.

  Returns:
    PNG-encoded string.

  Raises:
    ValueError: If any of the following occurs:
      - `array` is empty.
      - `array` is not 2-D.
      - `array` data type is unsupported.
  """
  supported_types = frozenset([np.uint8, np.uint16])
  # Sanity checks.
  if not array.size:
    raise ValueError(f'Received an empty image with shape {array.shape}.')
  if array.ndim != 2:
    raise ValueError(f'Array must be 2-D. Actual dimensions: {array.ndim}')
  if array.dtype.type not in supported_types:
    raise ValueError(
        'Pixels must be either `uint8` or `uint16`. '
        f'Actual type: {array.dtype.name!r}'
    )

  # Actual conversion.
  writer = png.Writer(
      width=array.shape[1],
      height=array.shape[0],
      greyscale=True,
      bitdepth=_NUM_BITS_PER_BYTE * array.dtype.itemsize,
  )
  output_data = io.BytesIO()
  writer.write(output_data, array.tolist())
  return output_data.getvalue()

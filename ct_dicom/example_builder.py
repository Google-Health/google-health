"""Example Preparation Routines based on Pydicom for running CT models."""

import datetime
from typing import Sequence

import numpy as np
import pydicom
import tensorflow as tf

import dicom_utils
import image_utils

# The minimum encoded pixel data in Houndsfield units. Anything lower is clipped
# to this value.
MIN_HU = -1024


# TODO(b/339471206): Add regression test for `create_ct_tfexample()`.
def create_ct_tfexample(
    dicom_series: Sequence[pydicom.Dataset], dataset_name: str = 'adhoc',
    strict_check: bool = False
) -> tf.train.Example:
  """Create a CT tf.example for inference based on a single series as input.

  Creates the core precursor tf.example produced upon DICOM export for
  volumetric images in the CT pipeline. This allows for loaded pydicom images
  to be used to make inference / deployment example creation easier.

  Args:
    dicom_series: A list of pydicom series as input to create the example.
    dataset_name: The dataset-level name given to the key created for the
      example. Stored under 'volume/id'.
    strict_check: If True, raise ValueError if the DICOM series as error.
      Otherwise, return a tf.train.Example with possibly invalid DICOM series.

  Returns:
    example: A tf.example in CT format for inference.
  """

  # Dedupe and sort incoming slices.
  dicom_series, _ = dicom_utils.dedupe_series(dicom_series, strict_check)

  dicom_images_dict = dicom_utils.map_by_series_instance_uid(
      dicom_series, sort_values=True
  )
  dicom_images_series_uid = list(dicom_images_dict.keys())[0]
  sorted_dicom_images = dicom_images_dict[dicom_images_series_uid]

  if not sorted_dicom_images:
    raise ValueError('No DICOM images found.')

  # Filter out derived images.
  filtered_dicom_images = [
      image
      for image in sorted_dicom_images
      if 'DERIVED' not in image.get('ImageType', [])
  ]

  if not filtered_dicom_images:
    raise ValueError('Series contains only derived images.')

  # Verify slice locations are consecutive (i.e. DICOM list is complete).
  spacing = dicom_utils.try_get_average_slice_spacing(filtered_dicom_images)
  depth = len(filtered_dicom_images)
  patient_id = 'UNKNOWN'
  if 'PatientID' in filtered_dicom_images[0]:
    patient_id = filtered_dicom_images[0].PatientID
  study_uid = filtered_dicom_images[0].StudyInstanceUID
  # Extract Age from the DICOM.
  bucketized_age_value = None
  try:
    if 'PatientAge' in filtered_dicom_images[0]:
      patient_age_as = filtered_dicom_images[0].PatientAge
      age_value = ''.join(x for x in patient_age_as if x.isdigit())
      age_value = float(age_value)
      bucketized_age_value = int(np.floor(age_value / 5.0))
  except ValueError:
    bucketized_age_value = None

  # Extract image PNG values for example.
  instances_png = []
  widths = set()
  heights = set()
  pixel_widths = set()
  pixel_heights = set()
  for a_dicom in filtered_dicom_images:
    heights.add(int(a_dicom.Rows))
    widths.add(int(a_dicom.Columns))
    pixel_heights.add(float(a_dicom.PixelSpacing[0]))  # Row / Column
    pixel_widths.add(float(a_dicom.PixelSpacing[1]))
    intercept = float(a_dicom.RescaleIntercept)
    slope = float(a_dicom.RescaleSlope)
    pixel_data = a_dicom.pixel_array

    hu = pixel_data * slope + intercept  # Cast to float.
    np.clip(hu, MIN_HU, None, hu)
    hu += 1024
    hu = hu.astype('uint16')
    instances_png.append(image_utils.encode_png(hu))

  if (
      len(widths) != 1
      or len(heights) != 1
      or len(pixel_widths) != 1
      or len(pixel_heights) != 1
  ):
    raise ValueError('Images of individual slices are of different dimensions.')
  (width,) = widths
  (height,) = heights
  (pixel_width,) = pixel_widths
  (pixel_height,) = pixel_heights

  # Create the tf.example
  example = tf.train.Example()
  f_dict = example.features.feature
  f_dict['volume/encoded'].bytes_list.value[:] = instances_png
  f_dict['volume/voxelsize'].float_list.value[:] = [
      pixel_width,
      pixel_height,
      spacing,
  ]
  f_dict['volume/width'].int64_list.value.append(width)
  f_dict['volume/height'].int64_list.value.append(height)
  f_dict['volume/depth'].int64_list.value.append(depth)
  if bucketized_age_value is not None:
    f_dict['AgeIn5YBuckets'].int64_list.value[:] = [bucketized_age_value]

  key = '%s/%s/%s/%s' % (
      dataset_name,
      patient_id,
      study_uid,
      dicom_images_series_uid,
  )
  f_dict['volume/id'].bytes_list.value.append(str.encode(key))

  # Used for prior classification in older pipeline
  # Get study date. Assume it's the same for all slices.
  study_date_value = 0
  if 'StudyDate' in filtered_dicom_images[0]:
    study_date = str(filtered_dicom_images[0].StudyDate)
    if len(study_date) == 8:
      dt = datetime.datetime(
          int(study_date[:4]),
          int(study_date[4:6]),
          int(study_date[6:]),
          0,
          0,
          0,
      )
      study_date_value = (
          dt - datetime.datetime(1970, 1, 1)
      ).total_seconds() * 1000000
  f_dict['volume/stack/STUDY_DATE/value'].int64_list.value.append(
      int(study_date_value)
  )

  return example

"""DICOM utilities based on pydicom for sorting and examining DICOM data."""

import collections
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pydicom

# Expect that the axial spacing between slices is consistent to a factor of 40%.
_SLICE_SPACING_TOLERANCE_RATIO = 0.4

# Index of the axial (Z) dimension for slice spacing computations. Used for
# indexing into the Image Position (Patient) (0020,0032) Attribute value.
_IMG_POS_PAT_ZCOORD = 2


def validate_slice_spacing(dicoms: Sequence[pydicom.Dataset]) -> None:
  """Verifies slice spacing based on sanity checks on the average spacing.

  The following requirements are validated:
  - At least 2 DICOMs in `dicoms` to infer slice spacing.
  - Slices are sorted in increasing order of axial dimension of Image Position.
  - No duplicate slices.
  - The max slice spacing is no more than 50% of the min slice spacing.

  Args:
    dicoms: Sequence of DICOM images in increasing order of the axial dimension
      values for the Image Position (Patient) (0020,0032) Attribute. Must have
      at least 2 DICOMs.

  Raises:
    ValueError: If any one of the requirements (in the description) fail.
  """
  slice_positions = tuple(
      dicom.ImagePositionPatient[_IMG_POS_PAT_ZCOORD] for dicom in dicoms
  )
  slice_spacings = np.array(
      [cur - prev for prev, cur in zip(slice_positions, slice_positions[1:])]
  )
  if not slice_spacings.size:
    raise ValueError(f'Too few DICOMs ({len(dicoms)}) to infer slice spacing.')
  if np.any(np.isclose(slice_spacings, 0.0)):
    raise ValueError(
        'DICOM slices are not ordered in increasing value of axial dimension of'
        ' the Image Position (Patient) (0020,0032) Attribute.'
    )

  min_slice_spacing = np.min(slice_spacings)
  max_slice_spacing = np.max(slice_spacings)
  try:
    spacing_factor = (max_slice_spacing - min_slice_spacing) / min_slice_spacing
  except ZeroDivisionError as e:
    raise ValueError(
        'Found a pair of duplicate or non-axially aligned slices.'
    ) from e

  if spacing_factor > _SLICE_SPACING_TOLERANCE_RATIO:
    raise ValueError(
        f'CT Instance spacing ratio {spacing_factor:.2f} exceeds the allowed'
        f' {_SLICE_SPACING_TOLERANCE_RATIO:.2f} tolerance. Max spacing:'
        f' {max_slice_spacing:.2f}mm, Min spacing: {min_slice_spacing:.2f}mm'
    )


def try_get_average_slice_spacing(
    dicoms: Sequence[pydicom.Dataset],
) -> float:
  """Returns an average of the slice spacing.

  Exceptions from `validate_slice_spacing()` are passed through to the caller.

  Args:
    dicoms: Sequence of DICOM images in increasing order of the axial dimension
      values for the Image Position (Patient) (0020,0032) Attribute. Must have
      at least 2 DICOMs.

  Returns:
    The average slice spacing.
  """
  validate_slice_spacing(dicoms)
  assert len(dicoms) > 1
  return (
      dicoms[-1].ImagePositionPatient[_IMG_POS_PAT_ZCOORD]
      - dicoms[0].ImagePositionPatient[_IMG_POS_PAT_ZCOORD]
  ) / (len(dicoms) - 1)


def dedupe_series(
    dicom_datasets: Sequence[pydicom.Dataset],
    strict_check: bool = False
) -> Tuple[Sequence[pydicom.Dataset], bool]:
  """Deduplicates slices of a single series by acquisition and instance number.

   In some cases unrelated slices are grouped into the same series UID.
   This attempts to remove duplicates by selecting the acquisition with the
   greatest number of slices followed by having unique instance numbers.

  Args:
    dicom_datasets: List of pydicom datasets to be de-duped. Note, this assumes
      that all belong to the same series instance UID.
    strict_check: If True, raise ValueError if the DICOM series as error.
      Otherwise, return DICOMs with possibly invalid DICOM series (e.g. missing
      AcquisitionNumber).

  Returns:
    final_dicoms: List of deduped cases.
    needed_correction: Set to True iff dicoms were eliminated.
  """
  needed_correction = False

  # Get the acquisitions with the most slices and ensure a single series.
  series_uid = set()
  acquisitions = {}
  for a_dicom in dicom_datasets:
    series_uid.add(a_dicom.SeriesInstanceUID)
    a_acquisition_number = -1
    if 'AcquisitionNumber' not in a_dicom and strict_check:
      raise ValueError('DICOM does not have AcquisitionNumber metadata.')
    elif 'AcquisitionNumber' in a_dicom:
      a_acquisition_number = a_dicom.AcquisitionNumber
    if a_acquisition_number not in acquisitions:
      acquisitions[a_acquisition_number] = []
    acquisitions[a_acquisition_number].append(a_dicom)
  most_slices = max(acquisitions, key=lambda k: len(acquisitions[k]))

  if len(series_uid) != 1:
    raise ValueError(
        'Got {len(series_uid)} unique series. Function only operates on a'
        ' single series.'
    )

  # Dedupe by instance number.
  final_dicoms = []
  instance_numbers = set()
  for instance in acquisitions[most_slices]:
    if instance.InstanceNumber not in instance_numbers:
      instance_numbers.add(instance.InstanceNumber)
      final_dicoms.append(instance)

  if len(dicom_datasets) != len(final_dicoms):
    needed_correction = True
  return final_dicoms, needed_correction


def _sort_series_by_image_position_patient(
    dicoms: Sequence[pydicom.Dataset],
) -> Tuple[pydicom.Dataset, ...]:
  """Sorts a series of pydicom data by Z of ImagePositionPatient (Axial CT)."""
  return tuple(
      sorted(dicoms, key=lambda d: d.ImagePositionPatient[_IMG_POS_PAT_ZCOORD])
  )


def _map_by_dicom_attribute(
    dicoms: Iterable[pydicom.Dataset],
    attr: str,
    attr_transformer: Callable[[Any], Any] = str,
    sort_values=True,
) -> Mapping[bytes, Sequence[pydicom.Dataset]]:
  """Maps DICOMs by Attribute Value (optionally sorted by Series Number."""
  dicoms_by_attribute = collections.defaultdict(list)
  for d in dicoms:
    dicoms_by_attribute[attr_transformer(getattr(d, attr))].append(d)

  if sort_values:
    for attribute_value, unsorted_dicoms in dicoms_by_attribute.items():
      dicoms_by_attribute[attribute_value] = (
          _sort_series_by_image_position_patient(unsorted_dicoms)
      )
  return dicoms_by_attribute


def map_by_series_instance_uid(
    dicoms: Iterable[pydicom.Dataset], sort_values=True
) -> Mapping[bytes, Sequence[pydicom.Dataset]]:
  """Get DICOMs mapped by Series Instance UID (0020, 000E).

  Each UID maps to a list of DICOMs, which may optionally be sorted in
  increasing order of the value of the third (Z) dimension of the Image Position
  (Patient) Attribute (0020, 0032).

  Args:
    dicoms: Input DICOM datasets to map.
    sort_values: Sort the mapped DICOM by the Z coordinate of the Image Position
      (Patient) Attribute. See method docstring for details.

  Returns:
    Mapping of Series Number Attribute values to DICOM Dataset sequences.
  """
  return _map_by_dicom_attribute(
      dicoms, 'SeriesInstanceUID', sort_values=sort_values
  )


def map_by_series_number(
    dicoms: Iterable[pydicom.Dataset], sort_values: bool = True
) -> Mapping[bytes, Sequence[pydicom.Dataset]]:
  """Get DICOMs mapped by Series Number (0020, 0011).

  Each UID maps to a list of DICOMs, which may optionally be sorted in
  increasing order of the value of the third (Z) dimension of the Image Position
  (Patient) Attribute (0020, 0032).

  Args:
    dicoms: Input DICOM datasets to map.
    sort_values: Sort the mapped DICOM by the Z coordinate of the Image Position
      (Patient) Attribute. See method docstring for details.

  Returns:
    Mapping of Series Number Attribute values to DICOM Dataset sequences.
  """
  return _map_by_dicom_attribute(
      dicoms, 'SeriesNumber', attr_transformer=int, sort_values=sort_values
  )

"""DICOM utilities based on pydicom for sorting and examining DICOM data."""

import collections
from typing import Any, Callable, Dict, Iterable, List, Tuple
import pydicom


def dedupe_series(
    dicom_data_list: List[pydicom.Dataset]
) -> Tuple[List[pydicom.Dataset], bool]:
  """Deduplicates slices of a single series by acquisition and instance number.

   In some cases unrelated slices are grouped into the same series UID.
   This attempts to remove duplicates by selecting the acquisition with the
   greatest number of slices followed by having unique instance numbers.

  Args:
    dicom_data_list: List of pydicom datasets to be de-duped. Note, this
      assumes that all belong to the same series instance UID.

  Returns:
    final_dicoms: List of deduped cases.
    needed_correction: Set to True if dicoms were eliminated.
  """
  needed_correction = False

  # Get the acquisitions with the most slices and ensure a single series.
  series_uid = set()
  acquisitions = {}
  for a_dicom in dicom_data_list:
    series_uid.add(a_dicom.SeriesInstanceUID)
    if 'AcquisitionNumber' not in a_dicom:
      raise ValueError('DICOM does not have AcquisitionNumber metadata.')
    if a_dicom.AcquisitionNumber not in acquisitions:
      acquisitions[a_dicom.AcquisitionNumber] = []
    acquisitions[a_dicom.AcquisitionNumber].append(a_dicom)
  most_slices = max(acquisitions, key=lambda k: len(acquisitions[k]))

  if len(series_uid) != 1:
    raise ValueError(
        'Got %d unique series. Function only operates on a single series.' %
        len(series_uid))

  # Dedupe by instance number.
  final_dicoms = []
  instance_numbers = set()
  for instance in acquisitions[most_slices]:
    if instance.InstanceNumber not in instance_numbers:
      instance_numbers.add(instance.InstanceNumber)
      final_dicoms.append(instance)

  if len(dicom_data_list) != len(final_dicoms):
    needed_correction = True
  return final_dicoms, needed_correction


def sort_series_by_image_position_patient(dicoms: List[pydicom.Dataset]):
  """Sorts a series of pydicom data by Z of ImagePositionPatient (Axial CT)."""
  dicoms.sort(key=lambda d: d.ImagePositionPatient[2])


def verify_series_by_slice_location(dicoms: List[pydicom.Dataset]) -> float:
  """Verify a complete sorted and dedupped stack by checking slice location."""
  if len(dicoms) <= 1:
    raise ValueError('Got %d dicoms. Number of dicoms must be > 1.' %
                     len(dicoms))

  slice_spacing = []
  for i in range(1, len(dicoms)):
    spacing = round(
        dicoms[i].ImagePositionPatient[2] -
        dicoms[i - 1].ImagePositionPatient[2], 3)
    slice_spacing.append(spacing)

  if len(set(slice_spacing)) != 1:
    raise ValueError(
        'Got %d unique slice spacing values. Missing or duplicate slices in '
        'selected stack.' % len(set(slice_spacing)))
  return slice_spacing[0]


def _get_dicom_attr_dict(
    dicoms: Iterable[pydicom.Dataset],
    attr: str,
    attr_transformer: Callable[[Any], Any] = str,
    sort_values=True) -> Dict[bytes, List[pydicom.Dataset]]:
  """Returns subset of given pydicoms."""
  dicom_attr_dict = collections.defaultdict(list)
  for d in dicoms:
    dicom_attr_dict[attr_transformer(getattr(d, attr))].append(d)

  if sort_values:
    for series in dicom_attr_dict.values():
      sort_series_by_image_position_patient(series)
  return dicom_attr_dict


def get_series_uid_dict(dicoms: Iterable[pydicom.Dataset],
                        sort_values=True) -> Dict[bytes, List[pydicom.Dataset]]:
  """Get list of DICOMs organized by a dictionary of series UID."""
  return _get_dicom_attr_dict(
      dicoms, 'SeriesInstanceUID', sort_values=sort_values)


def get_series_num_dict(dicoms: Iterable[pydicom.Dataset],
                        sort_values=True) -> Dict[bytes, List[pydicom.Dataset]]:
  return _get_dicom_attr_dict(
      dicoms, 'SeriesNumber', attr_transformer=int, sort_values=sort_values)


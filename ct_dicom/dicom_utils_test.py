import copy

from absl.testing import absltest
import pydicom

import dicom_utils


def _make_axial_spaced_dicoms(z_pos: float) -> pydicom.Dataset:
  dataset = pydicom.Dataset()
  dataset.ImagePositionPatient = [0, 0, z_pos]
  return dataset


class DicomUtilsTest(absltest.TestCase):

  def testDedupe(self):
    pydicom_image = pydicom.Dataset()
    pydicom_image.SeriesInstanceUID = '1.22.333.4444.55555'

    dicoms_to_dedupe = []
    pydicom_image.InstanceNumber = 1
    pydicom_image.AcquisitionNumber = 1
    dicoms_to_dedupe.append(copy.deepcopy(pydicom_image))
    pydicom_image.InstanceNumber = 2
    pydicom_image.AcquisitionNumber = 2
    dicoms_to_dedupe.append(copy.deepcopy(pydicom_image))
    pydicom_image.InstanceNumber = 1
    pydicom_image.AcquisitionNumber = 1
    dicoms_to_dedupe.append(copy.deepcopy(pydicom_image))
    filtered_dicoms, changed = dicom_utils.dedupe_series(dicoms_to_dedupe)
    self.assertTrue(changed)
    self.assertLen(filtered_dicoms, 1)
    self.assertEqual(filtered_dicoms[0].InstanceNumber, 1)

  def testGetAverageSliceSpacingReturnsAverage(self):
    dicoms = [
        _make_axial_spaced_dicoms(0),
        _make_axial_spaced_dicoms(1),
        _make_axial_spaced_dicoms(2.099),
        _make_axial_spaced_dicoms(3.1),
    ]
    self.assertAlmostEqual(
        dicom_utils.try_get_average_slice_spacing(dicoms),
        1.0333,
        delta=1e-3,
    )

  def testGetAverageSliceSpacingRaisesOnDuplicateSlicing(self):
    dicoms = [
        _make_axial_spaced_dicoms(0),
        _make_axial_spaced_dicoms(1),
        _make_axial_spaced_dicoms(1.01),
        _make_axial_spaced_dicoms(3.1),
        _make_axial_spaced_dicoms(5.1),
    ]
    with self.assertRaisesRegex(ValueError, 'spacing ratio (.*)208.00(.*)'):
      dicom_utils.validate_slice_spacing(dicoms)


if __name__ == '__main__':
  absltest.main()

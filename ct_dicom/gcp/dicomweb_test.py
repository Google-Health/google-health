r"""Unit and integration tests for `dicomweb.py`.

The integration tests call the CHC DICOMweb API and require generation of GCP
Credentials. These credentials may be supplied on command-line when invoking the
test via `--test-args`. In the absence of these arguments, the tests try to
obtain Application Default Credentials. If this fails too, the integration tests
are skipped.

See `auth.py` for supported Credentials and the applicable command-line
arguments.

The integration tests require the principal for the Credentials to have read
access to the NIH Chest X-ray DICOM Dataset:
https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest

An example invocation using Bazel with Access Token credentials would look like:
  bazel test :dicomweb_test \
      --test_arg="--access_token=`gcloud auth print-access-token`"
"""

import io
from typing import Optional

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from google import auth as gauth
from google.auth import credentials as gcredentials
from google.auth.transport import requests as grequests
import pydicom

from gcp import auth
from gcp import dicomweb


# NIH Chest X-ray dataset:
# https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
_PROJECT_ID = 'chc-nih-chest-xray'
_LOCATION = 'us-central1'
_DATASET_ID = 'nih-chest-xray'
_DICOM_STORE_ID = 'nih-chest-xray'

# Expected results for `test_search_all_instance_uids`.
_EXPECTED_STUDY_INSTANCE_UID_COUNT = 112120
_EXPECTED_STUDY_INSTANCE_UID_SUBSET = (
    '1.3.6.1.4.1.11129.5.5.152282914531363032949713603336202516605481',
    '1.3.6.1.4.1.11129.5.5.184301693334578016850836775758484230512396',
    '1.3.6.1.4.1.11129.5.5.164250117850161943194161905688563403884135',
    '1.3.6.1.4.1.11129.5.5.140385430840977525703440794320485211181362',
    '1.3.6.1.4.1.11129.5.5.167706292467338973012608016343615400174590',
    '1.3.6.1.4.1.11129.5.5.163417649223033709931105279768318099333246',
    '1.3.6.1.4.1.11129.5.5.198762697972258983626924507783975291419048',
    '1.3.6.1.4.1.11129.5.5.179387610138005701821131995952817087438990',
    '1.3.6.1.4.1.11129.5.5.171239031143542988682131441419389147590039',
    '1.3.6.1.4.1.11129.5.5.147597744515408142786448222471485994263174',
)
_EXPECTED_SERIES_INSTANCE_UID_COUNT = 112120
_EXPECTED_SERIES_INSTANCE_UID_SUBSET = (
    '1.3.6.1.4.1.11129.5.5.189954393471739102829266165312094856233636',
    '1.3.6.1.4.1.11129.5.5.143558942555202490607264725181630351940542',
    '1.3.6.1.4.1.11129.5.5.176710413805170297742754213182615158224592',
    '1.3.6.1.4.1.11129.5.5.173636723534566295092170234024171745083941',
    '1.3.6.1.4.1.11129.5.5.117069920031875109817919872105931584298369',
    '1.3.6.1.4.1.11129.5.5.197548155589040844108767552896301599265032',
    '1.3.6.1.4.1.11129.5.5.129746959800038691773791850432598606633014',
    '1.3.6.1.4.1.11129.5.5.112816532724284921242285811820367699999976',
    '1.3.6.1.4.1.11129.5.5.124760182693964961804727823562642962497744',
    '1.3.6.1.4.1.11129.5.5.164199707373855088457902663345054608406163',
)
_EXPECTED_SOP_INSTANCE_UID_COUNT = 112120
_EXPECTED_SOP_INSTANCE_UID_SUBSET = (
    '1.3.6.1.4.1.11129.5.5.166102574194965874627091290058724004650583',
    '1.3.6.1.4.1.11129.5.5.174595247921557872296549148592856328357393',
    '1.3.6.1.4.1.11129.5.5.161604609646646508448240207505375450718327',
    '1.3.6.1.4.1.11129.5.5.110781759534414118628869516274778470110585',
    '1.3.6.1.4.1.11129.5.5.131674849861941389292183794407648121636881',
    '1.3.6.1.4.1.11129.5.5.166584992838825321228887195374285748055797',
    '1.3.6.1.4.1.11129.5.5.149201334290302566289032704513923100758696',
    '1.3.6.1.4.1.11129.5.5.127088735465040101685300160676229389796677',
    '1.3.6.1.4.1.11129.5.5.188630119197320110714891789392510733491306',
    '1.3.6.1.4.1.11129.5.5.111308039523360824950858441125001747832163',
)


# Expected results for `test_search_instance_uids_within`.
_QUERY_STUDY_INSTANCE_UID = (
    '1.3.6.1.4.1.11129.5.5.152282914531363032949713603336202516605481'
)
_QUERY_SERIES_INSTANCE_UID = (
    '1.3.6.1.4.1.11129.5.5.177548625476944771058786684820803450871597'
)
_EXPECTED_SERIES_INSTANCE_UIDS = (
    '1.3.6.1.4.1.11129.5.5.177548625476944771058786684820803450871597',
)
_EXPECTED_SOP_INSTANCE_UIDS = (
    '1.3.6.1.4.1.11129.5.5.179440910029755481671257528401401954169381',
)


def skip_if_session_is_none(f):

  def wrapper(*args, **kwargs):
    if args[0].session is None:
      args[0].skipTest(
          'Failed to create Authorized Session. Currently, this requires'
          ' passing an Access Token via the command line flag `--access_token`.'
      )
    return f(*args, **kwargs)

  return wrapper


@absltest.skipThisClass('Base class')
class CHCSessionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._session = None
    self._create_authorized_session()

  def _create_credentials(self) -> Optional[gcredentials.Credentials]:
    try:
      return auth.create_gcp_credentials()
    except gauth.exceptions.DefaultCredentialsError:
      logging.error(
          'Failed to initialize Application Default Credentials. Some tests may'
          ' be skipped.'
      )

  def _create_authorized_session(self) -> None:
    credentials = self._create_credentials()
    if credentials is not None and credentials.valid:
      self._session = dicomweb.create_authorized_session(credentials)

  @property
  def session(self) -> Optional[grequests.AuthorizedSession]:
    return self._session


class SearchTest(CHCSessionTest, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='study',
          search_fn=dicomweb.search_study_instance_uids,
          limit=5000,
          expected_count=_EXPECTED_STUDY_INSTANCE_UID_COUNT,
          expected_subset=_EXPECTED_STUDY_INSTANCE_UID_SUBSET,
      ),
      dict(
          testcase_name='series',
          search_fn=dicomweb.search_series_instance_uids,
          limit=5000,
          expected_count=_EXPECTED_SERIES_INSTANCE_UID_COUNT,
          expected_subset=_EXPECTED_SERIES_INSTANCE_UID_SUBSET,
      ),
      dict(
          testcase_name='sop',
          search_fn=dicomweb.search_sop_instance_uids,
          limit=50000,
          expected_count=_EXPECTED_SOP_INSTANCE_UID_COUNT,
          expected_subset=_EXPECTED_SOP_INSTANCE_UID_SUBSET,
      ),
  )
  @skip_if_session_is_none
  def test_search_all_instance_uids(
      self, search_fn, limit, expected_count, expected_subset
  ):
    actual_instance_uids = set(
        search_fn(
            _PROJECT_ID,
            _LOCATION,
            _DATASET_ID,
            _DICOM_STORE_ID,
            self.session,
            limit=limit,
        )
    )
    self.assertLen(actual_instance_uids, expected_count)
    self.assertContainsSubset(expected_subset, actual_instance_uids)

  @parameterized.named_parameters(
      dict(
          testcase_name='study:series',
          search_fn=dicomweb.search_series_instance_uids,
          query_instance_uid=_QUERY_STUDY_INSTANCE_UID,
          expected_instance_uids=_EXPECTED_SERIES_INSTANCE_UIDS,
      ),
      dict(
          testcase_name='study:sop',
          search_fn=dicomweb.search_sop_instance_uids,
          query_instance_uid=_QUERY_STUDY_INSTANCE_UID,
          expected_instance_uids=_EXPECTED_SOP_INSTANCE_UIDS,
      ),
  )
  @skip_if_session_is_none
  def test_search_instance_uids_within(
      self, search_fn, query_instance_uid, expected_instance_uids
  ):
    actual_instance_uids = set(
        search_fn(
            _PROJECT_ID,
            _LOCATION,
            _DATASET_ID,
            _DICOM_STORE_ID,
            self.session,
            query_instance_uid,
            limit=100,
        )
    )
    self.assertEqual(set(expected_instance_uids), actual_instance_uids)

  @skip_if_session_is_none
  def test_search_instance_uids_within_study_and_series(self):
    actual_sop_instance_uids = set(
        dicomweb.search_sop_instance_uids(
            _PROJECT_ID,
            _LOCATION,
            _DATASET_ID,
            _DICOM_STORE_ID,
            self.session,
            study_instance_uid=_QUERY_STUDY_INSTANCE_UID,
            series_instance_uid=_QUERY_SERIES_INSTANCE_UID,
            limit=100,
        )
    )
    self.assertEqual(set(_EXPECTED_SOP_INSTANCE_UIDS), actual_sop_instance_uids)

  def test_search_sop_instance_uids_under_series_without_study(self):
    with self.assertRaisesRegex(ValueError, 'specified'):
      tuple(
          dicomweb.search_sop_instance_uids(
              _PROJECT_ID,
              _LOCATION,
              _DATASET_ID,
              _DICOM_STORE_ID,
              self.session,
              series_instance_uid=_QUERY_SERIES_INSTANCE_UID,
              limit=100,
          )
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='study',
          search_fn=dicomweb.search_study_instance_uids,
          limit=5001,
          regex='Study Instances',
      ),
      dict(
          testcase_name='series',
          search_fn=dicomweb.search_series_instance_uids,
          limit=5001,
          regex='Series Instances',
      ),
      dict(
          testcase_name='sop',
          search_fn=dicomweb.search_sop_instance_uids,
          limit=50001,
          regex='SOP Instances',
      ),
  )
  def test_search_instance_uids_limit_too_high(self, search_fn, limit, regex):
    with self.assertRaisesRegex(ValueError, regex):
      tuple(
          search_fn(
              _PROJECT_ID,
              _LOCATION,
              _DATASET_ID,
              _DICOM_STORE_ID,
              self.session,
              limit=limit,
          )
      )


class DownloadTest(CHCSessionTest):

  @skip_if_session_is_none
  def test_download_multipart_dicom_series(self):
    multipart_bytes = tuple(
        dicomweb.download_multipart_dicom_series(
            _PROJECT_ID,
            _LOCATION,
            _DATASET_ID,
            _DICOM_STORE_ID,
            self.session,
            _QUERY_STUDY_INSTANCE_UID,
            _QUERY_SERIES_INSTANCE_UID,
        )
    )

    self.assertLen(multipart_bytes, 1)
    dicom_dataset = pydicom.filereader.dcmread(io.BytesIO(multipart_bytes[0]))
    self.assertEqual(dicom_dataset.StudyInstanceUID, _QUERY_STUDY_INSTANCE_UID)
    self.assertEqual(
        dicom_dataset.SeriesInstanceUID, _QUERY_SERIES_INSTANCE_UID
    )
    self.assertEqual(
        dicom_dataset.SOPInstanceUID, _EXPECTED_SOP_INSTANCE_UIDS[0]
    )


if __name__ == '__main__':
  auth.define_flags()
  absltest.main()

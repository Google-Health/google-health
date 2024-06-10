"""Google Cloud Healthcare (CHC) DICOMweb utilities."""

import http
from typing import Iterable, Optional

import dicomweb_client.ext.gcp.uri as gcp_uri
import dicomweb_client.uri as dicomweb_uri
from google.auth import credentials as gcredentials
from google.auth.transport import requests
from requests_toolbelt.multipart import decoder


# Well-known constants from https://www.dicomstandard.org/.
_STUDY_INSTANCE_UID_TAG = '0020000D'
_SERIES_INSTANCE_UID_TAG = '0020000E'
_SOP_INSTANCE_UID_TAG = '00080018'

_SERIES_INSTANCE_UID_SEARCH_SUFFIX = 'series'
_STUDY_INSTANCE_UID_SEARCH_SUFFIX = 'studies'
_SOP_INSTANCE_UID_SEARCH_SUFFIX = 'instances'

_VALUE_KEY = 'Value'

# Scope requirements from:
# https://cloud.google.com/healthcare-api/docs/reference/rest/v1/projects.locations.datasets.dicomStores/searchForInstances#authorization-scopes
_AUTHORIZATION_SCOPES = ['https://www.googleapis.com/auth/cloud-healthcare']

# Search result limits for the CHC DICOMweb API:
# https://cloud.google.com/healthcare-api/docs/dicom#search_parameters
_MAX_LIMIT_STUDY = 5000
_MAX_LIMIT_SERIES = 5000
_MAX_LIMIT_SOP = 50000
_MAX_OFFSET = 1000000

_MAX_REFRESH_ATTEMPTS = 10
_REQUEST_TIMEOUT_SECONDS = 600


def create_authorized_session(
    credentials: gcredentials.Credentials,
) -> requests.AuthorizedSession:
  """Creates a Session authorized for Cloud Healthcare API interactions.

  Args:
    credentials: Google Auth credentials. For further details, see
      https://googleapis.dev/python/google-auth/latest/index.html.

  Returns:
    Credentials object with the requisite API scope.
  """
  authorization_scopes = _AUTHORIZATION_SCOPES
  scoped_credentials = gcredentials.with_scopes_if_required(
      credentials, authorization_scopes
  )
  return requests.AuthorizedSession(
      scoped_credentials, max_refresh_attempts=_MAX_REFRESH_ATTEMPTS
  )


def download_multipart_dicom_series(
    project_id: str,
    location: str,
    dataset_id: str,
    dicom_store_id: str,
    session: requests.AuthorizedSession,
    study_instance_uid: str,
    series_instance_uid: str,
) -> Iterable[bytes]:
  """Downloads all SOP Instances (DICOMs) within a Series Instance UID.

  The request accepts a multipart MIME response from the CHC DICOMweb API to
  reduce the:
  - Latency associated with making one API call per Instance.
  - API quota usage while downloading all Instances within a Series.

  Args:
    project_id: The GCP Project containing the DICOM Store to query.
    location: The regional location associated with the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/regions).
    dataset_id: The Dataset containing the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
    dicom_store_id: The DICOM Store to query.
    session: An Google Auth session authorized to use the CHC DICOMweb API.
    study_instance_uid: The Study Instance UID containing the Series Instance
      UID to download.
    series_instance_uid: The Series Instance UID containing the SOP Instances
      (DICOMs) to download.

  Yields:
    DICOM bytes associated with each Instance contained within the input Series
    Instance UID.
  """
  dicomweb_path = str(
      dicomweb_uri.URI(
          str(
              gcp_uri.GoogleCloudHealthcareURL(
                  project_id, location, dataset_id, dicom_store_id
              )
          ),
          study_instance_uid,
          series_instance_uid,
      )
  )
  # The "type" specification ensures that only DICOM bytes are returned by the
  # server. This way, any variations due to implementation of the download
  # requests would be flagged by this snippet raising an HTTP 406.
  headers = {
      'Accept': (
          'multipart/related; transfer-syntax=1.2.840.10008.1.2.1;'
          ' type="application/dicom"'
      )
  }
  response = session.get(
      dicomweb_path, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
  )
  response.raise_for_status()

  for part in decoder.MultipartDecoder.from_response(response).parts:
    yield part.content


def search_study_instance_uids(
    project_id: str,
    location: str,
    dataset_id: str,
    dicom_store_id: str,
    session: requests.AuthorizedSession,
    limit: int = 100,
) -> Iterable[str]:
  """Recovers all Study Instance UIDs from a CHC DICOM Store.

  Args:
    project_id: The GCP Project containing the DICOM Store to query.
    location: The regional location associated with the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/regions).
    dataset_id: The Dataset containing the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
    dicom_store_id: The DICOM Store to query.
    session: An Google Auth session authorized to use the CHC DICOMweb API.
    limit: The number of Study Instance UIDs in the DICOM Store could be large.
      The UIDs are recovered in a paginated fashion, where each page of results
      (one page per query) includes at most `limit` values. The higher this
      value, the fewer the total number of requests, but each response would be
      larger. Depending on your network connection, set this value in the range
      1 through 5000 (both inclusive). This parameter impacts the speed and
      network bandwidth utilization, but not the values returned by the method.

  Yields:
    Study Instance UIDs from the DICOM Store.

  Raises:
    ValueError: If `limit` exceeds the max value of 5000 allowed by the CHC
      DICOMweb API (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
  """
  if limit > _MAX_LIMIT_STUDY:
    raise ValueError(
        f'Request limit {limit} exceeds the CHC Search query request limit of'
        f' {_MAX_LIMIT_STUDY} for Study Instances.'
    )
  yield from _search_dicom_data(
      project_id,
      location,
      dataset_id,
      dicom_store_id,
      _STUDY_INSTANCE_UID_SEARCH_SUFFIX,
      _STUDY_INSTANCE_UID_TAG,
      session,
      limit,
  )


def search_series_instance_uids(
    project_id: str,
    location: str,
    dataset_id: str,
    dicom_store_id: str,
    session: requests.AuthorizedSession,
    study_instance_uid: Optional[str] = None,
    limit: int = 100,
) -> Iterable[str]:
  """Recovers all Series Instance UIDs from a CHC DICOM Store.

  The scope may be restricted to all Series within a fixed Study Instance
  UIDs (see `study_instance_uid` below).

  Args:
    project_id: The GCP Project containing the DICOM Store to query.
    location: The regional location associated with the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/regions).
    dataset_id: The Dataset containing the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
    dicom_store_id: The DICOM Store to query.
    session: An Google Auth session authorized to use the CHC DICOMweb API.
    study_instance_uid: If provided, restricts the returned Series Instance UIDs
      to within this Study Instance UID.
    limit: The number of Study Instance UIDs in the DICOM Store could be large.
      The UIDs are recovered in a paginated fashion, where each page (query)
      includes at most `limit` values. The higher this value, the fewer the
      total number of requests, but each response would be larger. Depending on
      your network connection, set this value in the range 1 through 5000 (both
      inclusive).

  Yields:
    Series Instance UIDs from the DICOM Store (optionally within the scope of
    the input `study_instance_uid`, if provided).

  Raises:
    ValueError: If `limit` exceeds the max value of 5000 allowed by the CHC
      DICOMweb API (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
  """
  if limit > _MAX_LIMIT_SERIES:
    raise ValueError(
        f'Request limit {limit} exceeds the CHC Search query request limit of'
        f' {_MAX_LIMIT_SERIES} for Series Instances.'
    )
  search_suffix = (
      _SERIES_INSTANCE_UID_SEARCH_SUFFIX
      if study_instance_uid is None
      else f'studies/{study_instance_uid}/series'
  )
  yield from _search_dicom_data(
      project_id,
      location,
      dataset_id,
      dicom_store_id,
      search_suffix,
      _SERIES_INSTANCE_UID_TAG,
      session,
      limit,
  )


def search_sop_instance_uids(
    project_id: str,
    location: str,
    dataset_id: str,
    dicom_store_id: str,
    session: requests.AuthorizedSession,
    study_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    limit: int = 1000,
) -> Iterable[str]:
  """Recovers all SOP Instance UIDs from a CHC DICOM Store.

  The scope may be restricted to all Series within a fixed:
  - Study Instance UID.
  - Study and Series Instance UID pair.

  (see `study_instance_uid` and `series_instance_uid` below).

  Args:
    project_id: The GCP Project containing the DICOM Store to query.
    location: The regional location associated with the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/regions).
    dataset_id: The Dataset containing the DICOM Store (c.f.
      https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores)
    dicom_store_id: The DICOM Store to query.
    session: An Google Auth session authorized to use the CHC DICOMweb API.
    study_instance_uid: If provided, restricts the returned SOP Instance UIDs to
      within this Study Instance UID.
    series_instance_uid: If provided, restricts the returned Series Instance
      UIDs to within this Series Instance UID. The associated Study Instance UID
      must also be specified.
    limit: The number of SOP Instance UIDs in the DICOM Store could be large.
      The UIDs are recovered in a paginated fashion, where each page (query)
      includes at most `limit` values. The higher this value, the fewer the
      total number of requests, but each response would be larger. Depending on
      your network connection, set this value in the range 1 through 5000 (both
      inclusive).

  Yields:
    SOP Instance UIDs from the DICOM Store (optionally within the scope of
    the input `study_instance_uid` and/or `series_instance_uid`, if provided).

  Raises:
    ValueError: If
      - `limit` exceeds the max value of 50000 allowed by the CHC DICOMweb API
        (c.f.
        https://cloud.google.com/healthcare-api/docs/projects-datasets-data-stores).
      - `series_instance_uid` is specified by `study_instance_uid` is not.
  """
  if limit > _MAX_LIMIT_SOP:
    raise ValueError(
        f'Request limit {limit} exceeds the CHC Search query request limit of'
        f' {_MAX_LIMIT_SOP} for SOP Instances.'
    )

  if study_instance_uid is None and series_instance_uid is None:
    search_suffix = _SOP_INSTANCE_UID_SEARCH_SUFFIX
  elif series_instance_uid is None:
    search_suffix = f'studies/{study_instance_uid}/instances'
  elif study_instance_uid is None:
    raise ValueError(
        'Study Instance UID must be provided if Series Instance UID is'
        ' specified.'
    )
  else:
    search_suffix = (
        f'studies/{study_instance_uid}/series/{series_instance_uid}/instances'
    )
  yield from _search_dicom_data(
      project_id,
      location,
      dataset_id,
      dicom_store_id,
      search_suffix,
      _SOP_INSTANCE_UID_TAG,
      session,
      limit,
  )


def _search_dicom_data(
    project_id: str,
    location: str,
    dataset_id: str,
    dicom_store_id: str,
    query_suffix: str,
    dicom_tag: str,
    session: requests.AuthorizedSession,
    limit: int,
) -> Iterable[str]:
  """Generates DICOM UIDs from a CHC DICOM Store."""
  assert limit > 0

  uri = gcp_uri.GoogleCloudHealthcareURL(
      project_id, location, dataset_id, dicom_store_id
  )
  base_dicomweb_query_path = f'{uri}/{query_suffix}?includefield={dicom_tag}'
  headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

  # The CHC offset limit puts an upper bound on the Instance count, which is
  # also used to limit the number of iterations.
  for offset in range(0, _MAX_OFFSET, limit):
    dicomweb_query_path = (
        f'{base_dicomweb_query_path}&offset={offset}&limit={limit}'
    )

    response = session.get(
        dicomweb_query_path, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    # CHC DICOMweb API does not set a Warning response header on the last
    # available page:
    # https://cloud.google.com/healthcare-api/docs/dicom#search_parameters
    if response.status_code == http.HTTPStatus.NO_CONTENT:
      return

    for instance in response.json():
      assert dicom_tag in instance
      assert _VALUE_KEY in instance[dicom_tag]

      for value in instance[dicom_tag][_VALUE_KEY]:
        yield value

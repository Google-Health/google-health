"""GCP authentication utilities for binary runfiles."""

from absl import flags
from absl import logging
from google import auth
from google.auth import credentials
from google.auth.compute_engine import credentials as gce_credentials
from typing_extensions import override

_ACCESS_TOKEN_FLAG_NAME = 'access_token'
_GCE_FLAG_NAME = 'use_gce_credentials'


_WARNING_MSG_TEMPLATE = (
    '--%s flag not set. define_flags() has not been called before calling'
    ' create_gcp_credentials().'
)


class _AccessTokenCredentials(credentials.Credentials):

  def __init__(self, bearer_token: str) -> None:
    super().__init__()
    self.token = bearer_token

  @override
  def refresh(self, _) -> None:
    pass


def define_flags() -> None:
  """Defines command line flags for users to specify GCP Credentials.

  The method may be safely called multiple times, since a flag is defined only
  if it does not already exist.

  The following flags are defined:
  - "access_token"
  - "use_gce_credentials"
  """
  if _ACCESS_TOKEN_FLAG_NAME not in flags.FLAGS:
    flags.DEFINE_string(
        _ACCESS_TOKEN_FLAG_NAME,
        None,
        'The OAuth2 Access Token to access the DICOM Store. Primarily meant for'
        ' toy/test applications. Cannot be used in conjunction with other '
        'credentials.',
    )
  if _GCE_FLAG_NAME not in flags.FLAGS:
    flags.DEFINE_boolean(
        _GCE_FLAG_NAME,
        False,
        'If true, use GCE Credentials. Cannot be used in conjunction with other'
        ' credentials.',
    )


def create_gcp_credentials() -> credentials.Credentials:
  """Creates GCP credentials, depending on which command line flag is set.

  To define command line flags to specify all supported credentials, the
  `define_flags()` method must be called first.

  If `define_flags()` is not called or no flag is set, it defaults to using
  Application Default Credentials.

  The supported credential types are:
  - Access Token (flag: "access_token")
  - GCE (flag: "use_gce_credentials")
  - Application Default (no other flags set)

  Returns:
    A GCP Credentials instance.

  Raises:
    ValueError: If more than one command line Credentials flags is set.
  """
  num_credentials_flags_set = 0

  try:
    access_token = flags.FLAGS[_ACCESS_TOKEN_FLAG_NAME].value
  except KeyError:
    access_token = None
    logging.warning(_WARNING_MSG_TEMPLATE, _ACCESS_TOKEN_FLAG_NAME)
  num_credentials_flags_set += access_token is not None

  try:
    use_gce_credentials = flags.FLAGS[_GCE_FLAG_NAME].value
  except KeyError:
    use_gce_credentials = False
    logging.warning(_WARNING_MSG_TEMPLATE, _GCE_FLAG_NAME)
  num_credentials_flags_set += use_gce_credentials

  if num_credentials_flags_set > 1:
    raise ValueError('At most one credential type can be set.')

  if access_token is not None:
    logging.info('Using Access Token credentials.')
    return _AccessTokenCredentials(access_token)

  if use_gce_credentials:
    logging.info('Using GCE Credentials.')
    return gce_credentials.Credentials()

  logging.info('Using Application Default Credentials.')
  return auth.default()[0]

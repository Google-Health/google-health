from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from google import auth as gauth
from google.auth.compute_engine import credentials as gce_credentials

from gcp import auth

_TOO_MANY_FLAGS_PARAMS = (
    dict(
        testcase_name='gce_access',
        use_gce_credentials=True,
        access_token='dummy_access_token',
    ),
)


class DefineFlagsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    auth.define_flags()

  def test_access_token(self):
    self.assertIn('access_token', flags.FLAGS)

  def test_use_gce_credentials(self):
    self.assertIn('use_gce_credentials', flags.FLAGS)


class GenerateGCPCredentials(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    auth.define_flags()

  def test_application_default_credentials(self):
    """ADC is used when no other credentials flags are set."""
    with mock.patch.object(gauth, 'default', autospec=True) as mock_default_fn:
      auth.create_gcp_credentials()
      mock_default_fn.assert_called_once()

  @flagsaver.flagsaver(use_gce_credentials=True)
  def test_use_gce_credentials(self):
    """GCE credentials used when `use_gce_credentials` flag is set."""
    with mock.patch.object(
        gce_credentials, 'Credentials', autospec=True
    ) as mock_credentials_fn:
      auth.create_gcp_credentials()
      mock_credentials_fn.assert_called_once_with()

  def test_use_gce_credentials_flag_not_defined(self):
    del flags.FLAGS.use_gce_credentials
    with mock.patch.object(gauth, 'default', autospec=True):
      auth.create_gcp_credentials()

  @flagsaver.flagsaver(access_token='dummy_access_token')
  def test_access_token(self):
    """Access Token is used when `access_token` flag is set."""
    with mock.patch.object(
        auth, '_AccessTokenCredentials', autospec=True
    ) as mock_access_token_cls:
      auth.create_gcp_credentials()
      mock_access_token_cls.assert_called_once_with('dummy_access_token')

  def test_access_token_flag_not_defined(self):
    del flags.FLAGS.access_token
    with mock.patch.object(gauth, 'default', autospec=True):
      auth.create_gcp_credentials()

  @parameterized.named_parameters(*_TOO_MANY_FLAGS_PARAMS)
  def test_more_than_one_credential_requested(self, **kwargs):
    with flagsaver.flagsaver(**kwargs):
      with self.assertRaisesRegex(ValueError, 'one credential'):
        auth.create_gcp_credentials()


if __name__ == '__main__':
  absltest.main()

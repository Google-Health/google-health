import json
import os
from unittest import mock

import google.auth.credentials
from google.cloud.aiplatform.aiplatform import gapic
import numpy as np

import api_utils


class TestMakePrediction(unittest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/fake_credentials.json"
    with open("/tmp/fake_credentials.json", "w") as f:
      # Fake file
      d = {
          "account": "",
          "client_id": "fergfggthyt-grht4thhrtyhy.apps.googleusercontent.com",
          "client_secret": "d-grteghrthy",
          "refresh_token": "1//freghthhyy-getrhythrythyr-egthhrtyhtrth",
          "type": "authorized_user",
          "universe_domain": "googleapis.com",
      }
      f.write(json.dumps(d))

  @mock.patch.object(gapic.PredictionServiceClient, "predict")
  def test_raw_audio_endpoint_success(self, mock_predict):
    mock_predict.return_value = mock.MagicMock(
        predictions=[[0.1] * 512, [0.9] * 512]
    )
    instances = np.random.rand(2, 32000)
    result = api_utils.make_prediction(
        api_utils.RAW_AUDIO_ENDPOINT_PATH, instances,
    )
    self.assertEqual(result.shape, (2, 512))
    mock_predict.assert_called_once_with(
        endpoint=api_utils.RAW_AUDIO_ENDPOINT_PATH, instances=instances.tolist()
    )

  @mock.patch.object(gapic.PredictionServiceClient, "predict")
  def test_gcs_uri_endpoint_success(self, mock_predict):
    mock_predict.return_value = mock.MagicMock(
        predictions=[[0.1] * 512, [0.9] * 512]
    )
    instances = ["gs://bucket/file1.wav", "gs://bucket/file2.wav"]
    gcs_bucket_name = "bucket"
    gcs_creds = mock.MagicMock(spec=google.auth.credentials.Credentials)
    gcs_creds.token = "mocked_token"

    result = api_utils.make_prediction(
        api_utils.GCS_URI_ENDPOINT_PATH,
        instances,
        gcs_bucket_name,
        gcs_creds,
    )
    self.assertEqual(result.shape, (2, 512))
    expected_instances = api_utils._get_prediction_instances(
        image_uris=instances,
        gcs_bucket_name=gcs_bucket_name,
        gcs_creds=gcs_creds,
    )
    mock_predict.assert_called_once_with(
        endpoint=api_utils.GCS_URI_ENDPOINT_PATH, instances=expected_instances
    )

  def test_raw_audio_endpoint_invalid_instances_type(self):
    instances = ["invalid", "instances"]
    with self.assertRaisesRegex(ValueError, "must be a numpy array"):
      api_utils.make_prediction(
          api_utils.RAW_AUDIO_ENDPOINT_PATH,
          instances,
      )

  def test_raw_audio_endpoint_invalid_instances_shape(self):

    instances = np.random.rand(2, 1000)
    with self.assertRaisesRegex(ValueError, "must be a numpy array of shape"):
      api_utils.make_prediction(
          endpoint_path=api_utils.RAW_AUDIO_ENDPOINT_PATH,
          instances=instances,
      )

  def test_gcs_uri_endpoint_invalid_instances_type(self):
    instances = np.random.rand(2, 32000)
    with self.assertRaisesRegex(ValueError, "must be a list of strings"):
      api_utils.make_prediction(
          endpoint_path=api_utils.GCS_URI_ENDPOINT_PATH,
          instances=instances,
          gcs_bucket_name="bucket",
          gcs_creds=mock.MagicMock(),
      )

  def test_gcs_uri_endpoint_missing_bucket_name(self):
    instances = ["gs://bucket/file.wav"]
    with self.assertRaisesRegex(
        ValueError, "`gcs_bucket_name` must be specified"
    ):
      api_utils.make_prediction(
          endpoint_path=api_utils.GCS_URI_ENDPOINT_PATH,
          instances=instances,
          gcs_creds=mock.MagicMock(),
      )

  def test_gcs_uri_endpoint_missing_credentials(self):
    instances = ["gs://bucket/file.wav"]
    with self.assertRaisesRegex(ValueError, "`gcs_creds` must be specified"):
      api_utils.make_prediction(
          endpoint_path=api_utils.GCS_URI_ENDPOINT_PATH,
          instances=instances,
          gcs_bucket_name="bucket",
      )

  def test_invalid_endpoint_path(self):
    instances = np.random.rand(2, 32000)
    with self.assertRaisesRegex(
        ValueError, "Endpoint invalid_endpoint is not recognized."
    ):
      api_utils.make_prediction(
          endpoint_path="invalid_endpoint",
          instances=instances,
      )


if __name__ == "__main__":
  unittest.main()

"""Utils for calling HeAR on Vertex AI."""

import datetime
import time
from typing import Literal

import google.auth
import google.auth.transport.requests
from google.cloud.aiplatform.aiplatform import gapic
from google.protobuf import json_format
import numpy as np

from google.protobuf import struct_pb2

LOCATION = 'us-central1'
PROJECT_ID = '132886652110'
RAW_AUDIO_ENDPOINT_ID = '202'
GCS_URI_ENDPOINT_ID = '203'

CLIENT_OPTIONS = {'api_endpoint': f'{LOCATION}-aiplatform.googleapis.com'}

try:
  CLIENT = gapic.PredictionServiceClient(client_options=CLIENT_OPTIONS)
except google.auth.exceptions.DefaultCredentialsError as exc:
  # pylint: disable=line-too-long
  raise ValueError(
      'Note: you have not defined environment variable '
      '`GOOGLE_APPLICATION_CREDENTIALS`. That variable should point to the '
      'path of your service account key file, which you can create by running '
      '`gcloud auth application-default login` for your own identity or '
      '`gcloud auth application-default login --impersonate-service-account SERVICE_ACCT`'
      'for service accounts. This assumes that you have first installed '
      'https://cloud.google.com/sdk/docs/install) `gcloud` CLI and created a '
      'service account '
      '(see https://cloud.google.com/iam/docs/service-account-overview, '
      'https://cloud.google.com/iam/docs/service-accounts-create) '
      'identified by `SERVICE_ACCT` above.'
  ) from exc
  # pylint: enable=line-too-long


RAW_AUDIO_ENDPOINT_PATH = CLIENT.endpoint_path(
    project=PROJECT_ID,
    location=LOCATION,
    endpoint=RAW_AUDIO_ENDPOINT_ID,
)
GCS_URI_ENDPOINT_PATH = CLIENT.endpoint_path(
    project=PROJECT_ID,
    location=LOCATION,
    endpoint=GCS_URI_ENDPOINT_ID,
)


def initial_token_refresh(
    gcs_creds: google.auth.credentials.Credentials,
) -> None:
  """Obtains short lived credentials for your GCS bucket."""
  auth_req = google.auth.transport.requests.Request()
  gcs_creds.refresh(auth_req)
  if not gcs_creds.valid:
    raise ValueError('Unexpected error: GCS Credentials are invalid')
  assert isinstance(gcs_creds.valid, datetime.datetime)  # for pytype
  time_until_expiry = (
      gcs_creds.expiry - datetime.datetime.now()
  ).total_seconds() // 60
  print(
      'Token will expire at'
      f' {gcs_creds.expiry.strftime("%Y-%m-%d %H:%M:%S")} UTC'
      f' ({time_until_expiry} minutes)'
  )


def _get_prediction_instances(
    image_uris: list[str],
    gcs_bucket_name: str,
    gcs_creds: google.auth.credentials.Credentials,
) -> list[struct_pb2.Value]:
  """Gets a list of dicts to pass as Vertex PredictionService instances."""
  instances = []
  for image_uri in image_uris:
    instance_dict = {
        'bucket_name': gcs_bucket_name,
        'object_uri': image_uri,
        'bearer_token': gcs_creds.token,
    }
    instance = json_format.ParseDict(instance_dict, struct_pb2.Value())
    instances.append(instance)
  return instances


def make_prediction(
    endpoint_path: Literal[RAW_AUDIO_ENDPOINT_PATH, GCS_URI_ENDPOINT_PATH],
    instances: np.ndarray | list[str],
    gcs_bucket_name: str | None = None,
    gcs_creds: google.auth.credentials.Credentials | None = None,
    client: gapic.PredictionServiceClient = CLIENT,
) -> np.ndarray:
  """Makes prediction with HeAR.

  Args:
    endpoint_path: The endpoint to use for making the prediction.
    instances: The instances to use for making the prediction. When endpoint is
      `RAW_AUDIO_ENDPOINT_PATH`, `instances` must be a numpy array of shape
      [num_samples, num_timesteps], where num_timesteps = 32000. When endpoint
      is `GCS_URI_ENDPOINT_PATH`, `instances` must be a list of strings, each
      string corresponding to a path to a wav file in GCS.
    gcs_bucket_name: The name of the GCS bucket to use for making the prediction
      when endpoint is `GCS_URI_ENDPOINT_PATH`.
    gcs_creds: The credentials to use for making the prediction when endpoint is
      `GCS_URI_ENDPOINT_PATH`. These must be obtained by calling `gcs_creds,
      project = google.auth.default()` and `initial_token_refresh(gcs_creds)`.
    client: The client to use for making the prediction.

  Returns:
    The predictions from the model. Embeddings of shape [num_samples,
    embedding_dim], where embedding_dim is 512.

  Raises:
    ValueError: If the instances don't have the right type, if the endpoint is
    not recognized, or if the gcs_bucket_name or gcs_creds are not specified
    when endpoint is `GCS_URI_ENDPOINT_PATH`.
  """
  if endpoint_path == RAW_AUDIO_ENDPOINT_PATH:
    if not isinstance(instances, np.ndarray):
      raise ValueError(
          'For endpoint `RAW_AUDIO_ENDPOINT_PATH`, `instances` must be a numpy '
          f'array but was of type {type(instances)} with value {instances}'
      )
    instances = instances.astype(float)
    if instances.ndim != 2 or instances.shape[-1] != 32000:
      raise ValueError(
          'For endpoint `RAW_AUDIO_ENDPOINT_PATH`, `instances` must be a numpy '
          'array of shape [num_samples, num_timesteps], where num_timesteps = '
          f'32000, but got {instances.shape}.'
      )
    instances = instances.tolist()
  elif endpoint_path == GCS_URI_ENDPOINT_PATH:
    if not isinstance(instances, list) and not isinstance(instances[0], str):
      raise ValueError(
          'For endpoint `GCS_URI_ENDPOINT_PATH`, `instances` must be a list of '
          'strings.'
      )
    if gcs_bucket_name is None:
      raise ValueError(
          'For endpoint `GCS_URI_ENDPOINT_PATH`, `gcs_bucket_name` must be '
          'specified.'
      )
    if gcs_creds is None:
      raise ValueError(
          'For endpoint `GCS_URI_ENDPOINT_PATH`, `gcs_creds` must be specified.'
      )
    instances = _get_prediction_instances(
        image_uris=instances,
        gcs_bucket_name=gcs_bucket_name,
        gcs_creds=gcs_creds,
    )
  else:
    raise ValueError(f'Endpoint {endpoint_path} is not recognized.')
  response = client.predict(endpoint=endpoint_path, instances=instances)
  result = np.array(response.predictions)
  return result


def make_prediction_with_exponential_backoff(
    endpoint_path: Literal[RAW_AUDIO_ENDPOINT_PATH, GCS_URI_ENDPOINT_PATH],
    instances: np.ndarray | list[str],
    max_retries: int = 10,
    base_delay_secs: float = 1,
    max_delay_secs: float = 60,
    gcs_bucket_name: str | None = None,
    gcs_creds: google.auth.credentials.Credentials | None = None,
    client: gapic.PredictionServiceClient = CLIENT,
) -> np.ndarray:
  """Makes prediction with exponential backoff.

  Args:
    endpoint_path: The endpoint to use for making the prediction.
    instances: The instances to use for making the prediction. Array of shape
      [num_samples, num_timesteps], where num_timesteps = 32000.
    max_retries: The maximum number of retries to make.
    base_delay_secs: The base delay in seconds.
    max_delay_secs: The maximum delay in seconds.
    gcs_bucket_name: The name of the GCS bucket to use for making the prediction
      when endpoint is `GCS_URI_ENDPOINT_PATH`.
    gcs_creds: The credentials to use for making the prediction when endpoint is
      `GCS_URI_ENDPOINT_PATH`. These must be obtained by calling `gcs_creds,
      project = google.auth.default()` and `initial_token_refresh(gcs_creds)`.
    client: The client to use for making the prediction.

  Returns:
    The predictions from the model. Embeddings of shape [num_samples,
    embedding_dim], where embedding_dim is 512.

  Raises:
    ValueError: If the endpoint is not recognized,or if the query failed too
    many times and the maximum of retries is reached.
  """
  if endpoint_path not in {RAW_AUDIO_ENDPOINT_PATH, GCS_URI_ENDPOINT_PATH}:
    raise ValueError(
        f'Endpoint must be one of {RAW_AUDIO_ENDPOINT_PATH} or'
        f' {GCS_URI_ENDPOINT_PATH}, but got {endpoint_path}.'
    )

  retries = 0
  while retries < max_retries:
    try:
      result = make_prediction(
          endpoint_path=endpoint_path,
          instances=instances,
          client=client,
          gcs_bucket_name=gcs_bucket_name,
          gcs_creds=gcs_creds,
      )
      return result
    except Exception as e:  # pylint: disable=broad-except
      retries += 1
      if retries == max_retries:
        raise ValueError(f'Max retries reached. Last error: {e}') from e

      delay = min(max_delay_secs, base_delay_secs * (2 ** (retries - 1)))

      print(f'Attempt {retries} failed. Retrying in {delay} seconds...')
      time.sleep(delay)

  raise ValueError(
      'Unexpected error in `make_prediction_with_exponential_backoff`'
  )

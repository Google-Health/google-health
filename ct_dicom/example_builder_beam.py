"""Beam wrappers to create Examples from DICOM files.

These are meant to be used in conjunction with Beam stages in `dicomweb_beam.py`
to create Examples from DICOMs sourced from a CHC DICOM Store.
"""

import io

from absl import logging
import apache_beam as beam
from apache_beam import pvalue
import pydicom

import example_builder
from gcp import dicomweb_beam


class CreateCTExampleFn(beam.DoFn):
  """Beam wrapper to create CT Example."""

  # The tag to identify the `TaggedOutput` containing the error string emitted
  # in `process()`.
  ERROR_OUTPUT_TAG = 'errors'

  def __init__(self, dataset_name: str = 'adhoc') -> None:
    """Creates an instance.

    Args:
      dataset_name: The dataset name in all emitted TF Examples, stored under
        the key "volume/id".
    """
    super().__init__()
    self._dataset_name = dataset_name

  def process(self, series_scope_dicoms: dicomweb_beam.SeriesScopeDICOMs):
    """Creates a CT model-compatible TF Example from DICOMs in a Series.

    If successful, a key-value pair is emitted:
    - Key: `<Study Instance UID>/<Series Instance UID>` string.
    - Value: The prepared TF Example.

    In case of failure, a string formatted as a CSV row is emitted; it is
    routed to an output tagged 'errors'. The row is comma-separated, with column
    values enclosed within double-quotes. The column contents are:
    - Column 1: Study Instance UID
    - Column 2: Series Instance UID
    - Column 3 onwards: Stringified arguments passed to the Exception that was
        caught for the error.

    Args:
      series_scope_dicoms: DICOMs sharing the same value for the Series and
        Study Instance UID Attributes. The metadata includes the Study and
        Series Instance UID values referenced in this docstring.

    Yields:
      Prepared TF Example or error string.
    """
    try:
      dicoms = list(
          pydicom.filereader.dcmread(io.BytesIO(dicom_bytes))
          for dicom_bytes in series_scope_dicoms.dicoms
      )
      yield (
          series_scope_dicoms.key.encode('utf-8'),
          example_builder.create_ct_tfexample(dicoms, self._dataset_name),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Example creation failed %r', series_scope_dicoms.key)
      yield pvalue.TaggedOutput(
          self.ERROR_OUTPUT_TAG,
          dicomweb_beam.to_csv_row((
              series_scope_dicoms.metadata.study_instance_uid,
              series_scope_dicoms.metadata.series_instance_uid,
              *e.args,
          )),
      )

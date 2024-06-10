"""End-to-end Beam pipeline(s) for creating CT Examples.

These can be called in a main file using Beam Runners suitable for the target
runtime environment.
"""

import dataclasses
from typing import Optional, Tuple

import apache_beam as beam
from apache_beam.io import textio
from apache_beam.transforms import util as beam_util

import example_builder_beam
from gcp import dicomweb_beam

# Column name in the input CSV file corresponding to Study Instance UIDs.
_STUDY_INSTANCE_UID_COLUMN_NAME = 'study_instance_uid'


def _build_dicom_download_from_chc_dicomweb(
    root: beam.Pipeline,
    chc_dicom_store: dicomweb_beam.ChcDicomStore,
    study_instance_uid_filepath: Optional[str],
) -> Tuple[beam.PCollection, beam.PCollection]:
  """Builds pipeline fragment to retrieve DICOMs from DICOM Store."""
  if study_instance_uid_filepath is not None:
    study_instance_uids = (
        root
        | 'Collect Study Instance UIDs from CSV'
        >> textio.ReadFromCsv(
            study_instance_uid_filepath,
            usecols=[_STUDY_INSTANCE_UID_COLUMN_NAME],
        )
        | beam.Map(lambda x: getattr(x, _STUDY_INSTANCE_UID_COLUMN_NAME, None))
        | beam.Filter(lambda x: x is not None).with_output_types(str)
    )
  else:
    study_instance_uids = (
        root
        | 'Collect Study Instance UIDs from DICOMweb'
        >> beam.ParDo(dicomweb_beam.QueryStudyInstanceUidsFn(chc_dicom_store))
    )

  dicoms = (
      study_instance_uids
      | beam_util.Reshuffle()
      | 'Collect Series Instance UIDs'
      >> beam.ParDo(dicomweb_beam.QuerySeriesInstanceUidsFn(chc_dicom_store))
      # No reshuffling here, otherwise the DICOMweb API will be bombarded with
      # O(number of Studies) queries in a short duration. No reshuffling couples
      # Series Instance UID retrieval with DICOM download (a slower step),
      # helping spread the API calls over time.
      | 'Retrieve Series DICOMs'
      >> beam.ParDo(
          dicomweb_beam.DownloadMultipartDicomSeriesFn(chc_dicom_store)
      ).with_outputs(
          dicomweb_beam.DownloadMultipartDicomSeriesFn.ERROR_OUTPUT_TAG,
          main='values',
      )
  )

  return (
      dicoms.values,
      dicoms[dicomweb_beam.DownloadMultipartDicomSeriesFn.ERROR_OUTPUT_TAG],
  )


def _build_example_creation_from_dicom_bytes(
    dicoms: beam.PCollection,
) -> Tuple[beam.PCollection, beam.PCollection]:
  """Builds pipeline fragment to create Examples from downloaded DICOMs."""
  examples = dicoms | 'Create Examples' >> beam.ParDo(
      example_builder_beam.CreateCTExampleFn()
  ).with_outputs(
      example_builder_beam.CreateCTExampleFn.ERROR_OUTPUT_TAG,
      main='values',
  )
  return (
      examples.values,
      examples[example_builder_beam.CreateCTExampleFn.ERROR_OUTPUT_TAG],
  )


@dataclasses.dataclass(frozen=True)
class Outputs:
  """Container for PCollections returned by `build_for_chc_dicomweb_api()`.

  Attributes:
    example_key_values: Key-value pairs where the value is the created TF
      Example and the key is a unique string (formatted as "<Study Instance
      UID>/<Series Instance UID>"; all slices to create the Example have the
      same Study and Series Instance UID Attribute values) to identify the
      Example.
    error_csv_rows: CSV row-formatted error strings. Each "row" corresponds to
      an unique Study-Series Instance UID pair for which an error was
      encountered either while downloading a DICOM, parsing DICOM bytes, or
      creating an Example from parsed DICOMs.
  """

  example_key_values: beam.PCollection
  error_csv_rows: beam.PCollection


def build_for_chc_dicomweb_api(
    root: beam.Pipeline,
    chc_dicom_store: dicomweb_beam.ChcDicomStore,
    study_instance_uid_filepath: Optional[str] = None,
) -> Outputs:
  """Builds CT Example creation pipeline reading DICOMs from CHC DICOMweb API.

  If `study_instance_uid_filepath` is not set, the pipeline runs on all Study
  Instance UIDs within the CHC DICOM Store. Otherwise, it uses Study Instance
  UIDs listed in this file.

  The outputs include the created Examples (under the `example_key_values`
  attribute) and CSV-formatted error strings (under the `error_csv_rows`
  attribute).

  Each CSV-formatted string in the `error_csv_rows` output attribute captures
  the first error encountered while downloading, parsing, and creating an
  Example for a given Series Instance UID. It is a comma-separated,
  double-quoted sequence of entries:
  - Column 1: Study Instance UID associated with the Series (Column 2).
  - Column 2: The Series Instance UID for which Example creation failed.
  - Column 3 onwards: Arguments to the Exception caught.

  By default, this pipeline uses Application Default Credentials to authenticate
  with the CHC DICOMweb API. Additional runtime environments are supported via
  command-line flags that must be declared by calling `gcp.auth.define_flags()`
  before Abseil parses command-line flags at runtime.

  Args:
    root: The root (source) of the Beam pipeline to connect.
    chc_dicom_store: CHC DICOM Store to download DICOMs from.
    study_instance_uid_filepath: A CSV file containing input Study Instance UIDs
      to query from the DICOM Store (in lieu of querying all Study Instance UIDs
      from the CHC DICOM Store). The UIDs must be present in a column titled
      "study_instance_uid".

  Returns:
    PCollections containing the created Examples (`example_key_values`
    attribute) and any CSV-formatted error strings (`error_csv_rows` attribute).
  """
  dicoms, download_errors = _build_dicom_download_from_chc_dicomweb(
      root, chc_dicom_store, study_instance_uid_filepath
  )
  example_key_values, example_creation_errors = (
      _build_example_creation_from_dicom_bytes(dicoms)
  )
  errors = (
      download_errors,
      example_creation_errors,
  ) | 'Collect Errors' >> beam.Flatten()
  return Outputs(example_key_values, errors)

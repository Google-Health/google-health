# HeAR API documentation

This document describes how to obtain embeddings from HeAR, the health acoustic foundation model developed at Google Research described in this paper
https://arxiv.org/abs/2403.02522.


## What HeAR is

* HeAR is a neural network with a ViT-L architecture (https://arxiv.org/abs/2010.11929).
* It was trained using masked auto-encoding on a large corpus of health-related sounds
* It is made available via a Google Cloud API
* Its output is a low-dimensional embedding optimized for capturing the most salient parts of health-related sounds like coughs and breathing sounds. 


## What HeAR is not

* HeAR is not a diagnostic model
* HeAR is not a cough detector


## Step 1: Request access

Contact us at health_acoustic_representations@google.com to express interest.
You will need to fill this form
https://docs.google.com/forms/d/e/1FAIpQLSc0Riwp31Vq033_Prklyfp_rSp3OqdgC7xHMzoB98-sDSw5KA/viewform?usp=dialog
for us to learn about your use case. Your request might go through a
lightweight review process. Once your use case has been approved, you may
proceed with step 2.

## Step 2: Create a Google Cloud project

If you do not already have one, create a Google Cloud project at
 https://cloud.google.com.

## Step 3: Create a Service Account for querying the API in your Cloud project

Service accounts (https://cloud.google.com/iam/docs/service-account-overview)
are special types of identity used within Google Cloud to identify a group of
users and manage resource access. Instead of giving API access to a specific
user, we will give access to that account, and authorized users within your
organization will be able to query the API by impersonating this service account
(https://cloud.google.com/iam/docs/service-account-overview#impersonation). 

Once this service account is created, let us know so that we can give it
the permissions it needs to query the HeAR API.

Note that you can also identify as yourself. See example Colab Notebook.

## Step 4: Query the API(s)

### Online predictions

Online predictions are typically used for obtaining embeddings on a small number
of audio files. It works by sending the input you want to process as a JSON
payload to the API. Note that there are two alternatives for providing this
input, resulting in two API endpoints.

1. Send the raw content of the audio file as a JSON payload to the API. In that scenario, the JSON payload should have a single key named `"instances"`, its value being a list of lists. Each sublist represents a 2s audio clip, and should contain **exactly** 32000 floating point numbers. If your audio clips are longer, you should crop them. If shorter, you should pad them with zeros. The maximum number of elements in that list is 4 (that is, at most 4 lists containing exactly 32000 floats). Please note that the clips will be encoded as they are, and no further preprocessing including but not limited to cough detection will be applied on them before being processed by HeAR. Detection and cropping of the health sounds of interests is strongly advised before querying the API, since it is expected to result in more useful representations.

2. Send GCS bucket URIs in the JSON payload. In that scenario, the JSON payload should have a single key named `"instances"`, its value being a list of dicts. Each dict points to one or more audio files, and has the following keys: `bucket_name` (a string representing the GCS Bucket name (without leading slashes or "gs://" qualifier)), `object_uri` (a list of strings representing URIs of the target objects to read within the GCS Bucket specified by `bucket_name`. URIs should not include any leading slash), and `bearer_token` (a string representing the Bearer Token granting access to the objects (`object_uri`) within the GCS Bucket (`bucket_name`). Currently, only Access Tokens are supported. For further details, see: https://cloud.google.com/docs/authentication/token-types). Note that there is a maximum of 640 URIs (across all samples) using this method. The same remarks mentioned in the previous point regarding the preprocessing of the audio clips apply here.

Here is an example of the JSON payloads that you would need to pass to the REST API (e.g. if using the `gcloud` CLI) in both scenarios. If you are using Python (see code snippet at the end of this section), you don't need to care about this JSON, this is handled by the `google.cloud` library under the hood and we provide this only for illustrative purposes. Also see documentation https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions.

```json
// 1. Sending raw audio directly.
{
  "instances": [
    // There can be at most 4 elements in this list.
    [1.2923, -0.236, ...], // This list contains exactly 32000 samples.
    [0.09, 0.46, ...], //  This list contains exactly 32000 samples.
    ]
}

// 2. Sending GCS URIs
{
  "instances": [
    // There can be at most 640 elements in the concatenation of all `object_uri` lists.
    // For example, this JSON contains 4 URIs.
    {"bearer_token": "your-token1", "bucket_name": "your-bucket1", "object_uri": "path/1.wav"},
    {"bearer_token": "your-token1", "bucket_name": "your-bucket1", "object_uri": "path/2.wav"},
    {"bearer_token": "your-token2", "bucket_name": "your-bucket2", "object_uri": "path/3.wav"},
    {"bearer_token": "your-token2", "bucket_name": "your-bucket2", "object_uri": "path/4.wav"},
    ]
}
```

We provide two examples for querying the API: using Python and CURL.

#### Using Python

In order to use online predictions, you must first authenticate. This can be
done using

```bash
gcloud auth application-default login --impersonate-service-account your-service-account@<YOUR-PROJECT>.iam.gserviceaccount.com
```

which will create a local JSON file containing short-lived credentials. You must set the environment
variable `GOOGLE_APPLICATION_CREDENTIALS` to this value. More information here: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev.


Then, you can send queries to the API with the code provided in `hear_demo.ipynb`.

If you haven't set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable, you will get see this error 

```
DefaultCredentialsError: Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information.
```


Note that the online API has a 1.5 MB limit on the payload, which represents 4
clips of 2s sampled at 16kHz.

Alternatively, you can provide instead GCS bucket URIs and credentials using the code in `hear_demo.ipynb`.


#### Using bash

On the command line, you can do 

```bash
gcloud auth application-default login --impersonate-service-account your-service-account@<YOUR-PROJECT>.iam.gserviceaccount.com
```

You can then create some variables

```bash
ENDPOINT_ID="200"  # or "201" for the GCS URI endpoint.
PROJECT_ID="132886652110"
INPUT_DATA_FILE="path/to/your/json/file"
```

and finally use `curl`:

```bash
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-west1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-west1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
```

### If you have a lot of queries to run

In the case that you have a very large number of audio clips to embed (>1000),
calling this API sequentially will be too slow. We suggest that you use parallel
queries on our CPU endpoints instead, which will result in a more appropriate
throughput. See example in `hear_demo.ipynb`.

## General notes

Google does not keep a copy of any audio file sent.

Google monitors daily query volume and aggregates on a per-user and per-organization basis. Access can be revoked if a user or organization exceeds a reasonable query volume.

## HeAR Model Card

Model Name: HeAR (Health Acoustic Representations)

Model Type: Vision Transformer (Large), trained using a masked autoencoding objective

Developed by: Google Research

Description: HeAR is a Masked Auto Encoder (https://arxiv.org/abs/2111.06377), a transformer-based (https://arxiv.org/abs/1706.03762) neural network trained with
a self-supervised learning objective on a massive dataset (~175k hours) of 2
seconds audio clips. At training time, it tries to reconstruct masked
spectrogram patches from the visible patches. After having been trained, its
encoder can generate low-dimensional representations of 2s audio clips
containing salient health-related information. These representations, or
embeddings, can be used as inputs to other models trained for a variety of
supervised tasks related to health. In the preprint
(https://arxiv.org/abs/2403.02522), we evaluated the model using linear probes
on cough and spirometry inference tasks as well as health acoustic events
detection tasks.


### Key Features

* Self-Supervised Learning: Trained using a masked autoencoder approach, eliminating the need for expensive and time-consuming manual labeling of data.
* Large-Scale Training: Utilizes a dataset of over 300 million audio clips, enabling the model to learn robust and generalizable representations.
* Data Efficiency: Demonstrates high performance even with limited labeled training data for downstream tasks.
* Versatility: Exhibits strong performance across diverse health acoustic tasks.

### Potential Applications

* Can be a useful tool for AI research geared towards discovery of novel acoustic biomarkers in the following areas:
  * Respiratory diseases like COVID-19, tuberculosis, and COPD from cough and breath sounds.
  * Low-resource settings: Can potentially augment healthcare services in settings with limited resources by offering accessible screening and monitoring tools.

### Limitations

* Limited Sequence Length: Primarily trained on 2-second audio clips, potentially impacting performance on longer sequences.
* Model Size: Current model size is large, requiring further research for on-device deployment.
* Bias Considerations: Potential for biases based on demographics and recording device quality, necessitating further investigation and mitigation strategies.
* Clinical Validation: Acoustic biomarkers discovered using HeAR must go through adequate clinical validation to ensure their effectiveness and safety before any use in real-world healthcare settings.
* HeAR was trained using 2s audio clips of health-related sounds from a public non-copyrighted subset of Youtube. These clips come from a variety of sources but may by noisy or low-quality. 
* The model is only used to generate embeddings of the user-owned dataset. It does not generate any predictions or diagnosis on its own.

### Intended Use

* Research and development of health-related acoustic biomarkers.
* Exploration of novel applications in disease detection and health monitoring.

### Ethical Considerations
* Potential for bias and fairness issues requires careful consideration and mitigation strategies.
* Privacy concerns surrounding the collection and use of health-related audio data must be addressed.
* Transparency and explainability of model predictions are crucial for user trust and clinical adoption.
* Although Google does not store permanently any data sent to this model, it is the data owner's responsibility to ensure that Personally identifiable information (PII) and Protected Health Information (PHI) are removed prior to being sent to the model.

### Access and Availability
Individuals interested in using HeAR can contact health_acoustic_representations@google.com.

### Additional Information
This model card is based on the research paper "HeAR - Health Acoustic Representations" available on arXiv.
Further information about the HeAR system and its applications can be obtained by contacting the development team at health_acoustic_representations@google.com.

### Disclaimer
HeAR is intended to be a tool to accelerate research around acoustic biomarkers. Any technology built using HeAR should go through adequate validation and comply with applicable laws and regulations.
The developers are not responsible for any consequences arising from the use of HeAR.

We believe that HeAR represents a significant step forward in the field of health acoustics and has the potential to revolutionize healthcare by enabling accessible and non-invasive health monitoring and disease detection.

## License

See https://github.com/Google-Health/google-health/blob/master/LICENSE for
details.

## Contact us
Please reach out to us at health_acoustic_representations@google.com for issues such as, but not limited to:

* Seeking technical assistance
* Providing feedback
* Requesting permissions for publications
* Discussing clinical use cases
* Discussing enterprise requirements such as fitting within strict security perimeters of your organization, governing your data in GCS, training and serving custom models at scale on Vertex AI

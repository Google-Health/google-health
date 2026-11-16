<!--
Copyright (c) 2026, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-->

# ClinicalSense: An EIM-Grounded Video MCQ Benchmark for Clinical Perception

ClinicalSense is a benchmark for evaluating multimodal AI models'
video perception capabilities in clinical settings. The multi-stage
pipeline systematically generates diverse multiple-choice questions
(MCQs) evaluating various clinical perception domains.

## Setup

First, install dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

Set your API key for model inference execution:

```bash
export GOOGLE_API_KEY="your-api-key"
```

`ffmpeg` and `ffprobe` must also be on your `PATH` (they are used by
Stage 0, Stage 1 and `eval_mcq.py --audio_only`).

## Pipeline Overview

The pipeline consists of the following stages leading to
evaluation:

0. **Stage 0 (Clinical Event Discovery):** Scan full-length clinical
   videos for clinically relevant visual events and cut one short clip
   per event.
1. **Stage 1 (Prototyping):** Identify core actions and clinical
   anchors (evidence extraction).
   - **Stage 1B (Verification & Patching):** Decompose the extracted
     evidence into atomic claims, verify each claim against the clip,
     and correct or remove unsupported claims.
2. **Stage 2 (Question Generation):** Generate targeted STEM
   questions around specific clinical facts.
3. **Stage 3 (Distractor Generation):** Extract realistic,
   co-hyponym alternatives for clinical conditions.
4. **Stage 4 (Anti-Leak Guardrails):** Validate generated content
   against evidence constraints and verify alignment to avoid
   information leakage.
5. **Stage 5 (Quality & Deduplication):** Run final sanity checks
   and compile multiple-choice variations.
6. **Evaluation:** Assess model prediction using standard
   evaluation metrics.

### Stage 0: Clinical event discovery

`stage0_discover.py` turns full-length recordings into short event clips
and a manifest that Stage 1 reads. By default
(`--discovery_mode=coarse_to_fine`), it follows the coarse-to-fine scan
described in the paper:

1. **Language cues:** a text-only pass over the timestamped transcript
   proposes candidate windows. Transcripts can be provided with
   `--transcripts_dir` as `<video_id>.json` files, each a list of
   `{"start_sec", "end_sec", "speaker", "text"}` segments. Otherwise, the
   speech is first transcribed from the audio track
   (`--transcribe_audio`, on by default), which covers narrated
   instructional videos such as MedVidQA.
2. **Visual cues:** the muted full video is scanned at 5 FPS
   (`--coarse_fps`).
3. **Fusion:** the union of both sets of windows is merged when windows
   are at most 10 s apart (`--merge_gap_sec`) and share an EIM facet.
4. **Fine localization:** each merged window, padded by 5 s, is
   re-scanned as a muted clip at 10 FPS (`--fine_fps`), with the
   transcript excerpt as reference text, to localize individual events.
   Events that repeat an earlier event are removed (`--dedup_scope`).

Each event becomes a clip padded by 5 s (`--padding_sec`); clips shorter
than 5 s are skipped and clips longer than 60 s are truncated
(`--min_clip_duration_sec`, `--max_clip_duration_sec`). Every clip is
written as `<clip>.mp4`, `<clip>_noaudio.mp4` and `<clip>_audio.wav`
(16 kHz mono; not written for videos without an audio track).

`--discovery_mode=single_pass` instead makes a single call on the full
video with audio at 5 FPS (`--single_pass_fps`), pads events by 2 s and
does not truncate long clips. This is the procedure that was used to
build the released MedVidQA benchmark (v1).

```bash
python stage0_discover.py \
  --videos_dir=/path/to/videos/ \
  --transcripts_dir=/path/to/transcripts/ \
  --output_clips_dir=/path/to/clips/ \
  --output_manifest=/path/to/stage0_manifest.json

python stage1_prototype.py \
  --stage0_manifest=/path/to/stage0_manifest.json \
  --output_file=/path/to/stage1.json \
  --max_items=0
```

Large files are sent through the Gemini Files API. Scanning very long
recordings at 5 FPS can exceed the model's context window; split such
videos or lower `--coarse_fps` if requests fail.

### Stage 1B: Claim verification

`stage1b_verify.py` decomposes each Stage 1 observation into atomic
claims and verifies each claim against the clip with audio. Each claim
receives one initial verification pass. Claims that fail this initial
check, as well as all laterality claims, receive three additional
independent checks (`--recheck_runs`), and the final verdict is
determined by majority vote across those three runs; ties are broken by
the initial verdict. `stage1b_patch.py` then rewrites flagged claims
that have a suggested correction and removes flagged claims that have
none. If an observation cannot be decomposed or a claim cannot be verified
(e.g. because of API errors), the clip is left out of the output and is
retried when the run is resumed.

The released benchmarks were built with an earlier verification policy
in which only laterality claims were checked multiple times (three
self-consistency runs in total, decided by majority vote) and all other
claims received a single pass. Pass `--norecheck_failed_claims` to
reproduce it. `--laterality_runs` and `--consistency_runs` are
deprecated aliases of `--recheck_runs`.

`--fabrication_keyword_heuristic` (off by default) enables an optional
variant from a later revision of our internal pipeline that was not
used to build the released benchmarks: a rechecked claim that remains
unsupported, and whose evidence contains phrases such as "not visible"
or "does not", has its suggested correction cleared so that the claim
is removed rather than rewritten. It is a keyword heuristic with known
false positives.

## Running the full pipeline

Each stage reads the output of the previous one. The commands below run the
whole pipeline, from event discovery on full-length videos to model
evaluation. `/path/to/clips/` is the directory of clips written by Stage 0;
the later stages that need video read the clips from it.

```bash
# Stage 0: discover clinical events and cut one clip per event.
python stage0_discover.py \
  --videos_dir=/path/to/videos/ \
  --transcripts_dir=/path/to/transcripts/ \
  --output_clips_dir=/path/to/clips/ \
  --output_manifest=/path/to/stage0_manifest.json

# Stage 1: blind visual evidence extraction from the muted clips.
python stage1_prototype.py \
  --stage0_manifest=/path/to/stage0_manifest.json \
  --output_file=/path/to/stage1.json \
  --max_items=0

# Stage 1B: verify each evidence claim against the clip with audio.
python stage1b_verify.py \
  --stage1_file=/path/to/stage1.json \
  --clips_dir=/path/to/clips/ \
  --output_file=/path/to/stage1b.json

# Stage 1B-patch: correct or remove the claims flagged by Stage 1B.
python stage1b_patch.py \
  --stage1b_file=/path/to/stage1b.json \
  --output_file=/path/to/stage1b_patched.json

# Stage 2: generate one question per EIM facet.
python stage2_question_gen.py \
  --evidence_file=/path/to/stage1b_patched.json \
  --output_file=/path/to/stage2_questions.json

# Stage 3: generate distractors. The default of 9 distractors per question
# (10-option MCQs) is the setting used in the paper; see --num_distractors.
# The clips are used to ground the distractors of ABSENCE questions.
python stage3_distractors.py \
  --questions_file=/path/to/stage2_questions.json \
  --clips_dir=/path/to/clips/ \
  --output_file=/path/to/stage3_distractors.json

# Stage 4: distractor collision audit and text-only anti-leakage gate.
python stage4_antileak.py \
  --input_file=/path/to/stage3_distractors.json \
  --clips_dir=/path/to/clips/ \
  --output_file=/path/to/stage4_validated.json

# Stage 5: final quality gates and benchmark formatting.
python stage5_quality.py \
  --input_file=/path/to/stage4_validated.json \
  --output_file=/path/to/clinicalsense_benchmark.json

# Evaluation, by default on the clips without their audio track. Add
# --include_audio, --audio_only or --text_only to evaluate the other input
# conditions.
python eval_mcq.py \
  --benchmark_file=/path/to/clinicalsense_benchmark.json \
  --clips_dir=/path/to/clips/ \
  --output_file=/path/to/eval_results.json
```

Every stage uses the model from the paper by default; pass `--model_name` to
use another Gemini model.

## Tests

`./run.sh` installs the dependencies in a virtual environment and runs the
unit tests, which need no network access.

## License

This project is licensed under the BSD 3-Clause License. See the
root `LICENSE` file for more details.

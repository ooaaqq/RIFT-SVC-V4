# Dataset smoke test

Run this gate before downloading a complete corpus.  It is intentionally small
and read-only: use one archive, one HF shard, or at most 50 examples per
dataset.  A successful metadata request alone is not sufficient.

## Gate A: identify the artifact

Record the exact URL/repository, revision, download date, license text, and
SHA-256 of the downloaded archive or shard.  Reject an artifact whose name
contains `unit`, `hubert`, `contentvec`, `feature`, or `extract` unless the
dataset is explicitly being tested as an audio source and the files decode as
audio.

For Hugging Face datasets, inspect the file list before downloading:

```sh
huggingface-cli repo-files espnet/ace-kising-segments --repo-type dataset
huggingface-cli repo-files AaronZ345/GTSinger --repo-type dataset
```

The exact command may vary with the installed `huggingface_hub` version; the
required evidence is the remote file list and revision, not the command name.

## Gate B: decode a small sample

Copy a small sample into a temporary directory outside `sources/` and run:

```sh
find /tmp/rift-v4-dataset-sample -type f -print0 |
  xargs -0 -n1 ffprobe -v error \
    -show_entries format=duration:stream=codec_name,sample_rate,channels,bits_per_sample \
    -of default=noprint_wrappers=1
```

Every accepted sample must be decodable, non-empty, and contain a single audio
stream.  Record whether it is original human audio or synthesizer output.
Do not silently convert stereo or corrupted files during this test.

## Gate C: metadata and content checks

For the sample, count and record:

- unique speaker IDs and whether IDs are stable across shards;
- song IDs and whether multiple segments map back to one song;
- language and technique labels, when present;
- decoded duration and the number of files;
- sample rate, channels, bit depth, peak, and clipping ratio;
- the fraction of files that are speech, accompaniment, silence, or synthetic.

The sample is rejected when speaker/song identity cannot be reconstructed, when
the archive contains only extracted features, or when more than 5% of sampled
files fail to decode.

## Gate D: V4 compatibility

After the sample passes A-C, place only the sample under a temporary normalized
tree:

```text
<dataset>/<speaker>/<song>/<audio>.wav
```

Then run the existing manifest and QC commands without execution flags:

```sh
rift-v4-build-manifest \
  --source DATASET=/tmp/rift-v4-dataset-sample \
  --features-root /tmp/rift-v4-features \
  --output /tmp/rift-v4-manifest.pending.jsonl
rift-v4-qc \
  --manifest /tmp/rift-v4-manifest.pending.jsonl \
  --output /tmp/rift-v4-manifest.qc.jsonl
```

The result must include accepted entries, stable speaker/song fields, plausible
durations, and explicit exclusion reasons for rejected files.  Review at least
one accepted and one rejected item manually before scheduling a full download.

## Decision values

Use one of these values in `docs/dataset-availability.md`:

- `pass`: raw audio, metadata, license, and V4 QC are all confirmed;
- `conditional`: audio is usable but license, mirror provenance, or metadata
  still needs confirmation;
- `fail`: no raw audio, decode failure, missing identity metadata, or unsuitable
  content distribution.

Only `pass` datasets should be downloaded in full.  `conditional` datasets may
be retained as a separate experiment and must not enter the main V4 manifest by
default.

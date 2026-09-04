---
license: other
pipeline_tag: audio-to-audio
tags:
  - singing-voice-conversion
  - audio
  - rectified-flow
  - pytorch
---

# RIFT-SVC V4 Foundation

This repository contains the 130k-step EMA foundation checkpoint for RIFT-SVC
V4, a quality-first singing voice conversion model. It is not compatible with
RIFT-SVC V1-V3.

## Checkpoint

| File | State | SHA-256 |
| --- | --- | --- |
| `rift-v4-foundation-1024x16-step130000.pt` | step 130,000 EMA | `7ee24c3713382de6f392049105f40703b17a7a9ead2b9ae465d7e7ccc51d2b48` |

The release also includes the exact model configuration, speaker map, training
mel statistics, provenance summary, and checksums. It excludes optimizer, RNG,
and online training state and therefore cannot resume foundation training
exactly.

## Contract

- 1024-wide, 16-layer rectified-flow Transformer
- 44.1 kHz audio, hop length 512, 128 log-mel channels
- frozen dual-phase ContentVec conditioning, plus F0 and RMS
- dataset-channelwise mel normalization from `mel_stats.json`
- PC-NSF-HiFiGAN-compatible mel output

Download with:

```sh
hf download ooaaqq/RIFT-SVC-V4 \
  rift-v4-foundation-1024x16-step130000.pt \
  config.json speaker_map.json mel_stats.json
```

Use the checkpoint with the code and inference instructions in the project
repository. For a target singer, keep validation songs disjoint and reuse the
published foundation mel statistics.

## Training data

The foundation model used GTSinger, M4Singer, OpenSinger, Opencpop, Kiritan,
and low-weight synthetic ACE-Opencpop. Dataset-level sampling and audit details
are documented in the project repository.

## Limitations

- Output quality depends on ContentVec/F0 alignment and the downstream vocoder.
- Languages, techniques, and voices outside the training distribution may be
  unstable.
- The release has not been evaluated for identity misuse, impersonation, or
  production deployment.
- It is intended as a foundation for target-singer fine-tuning, not as a
  universal zero-shot voice converter.

## Terms and attribution

The weights are provided for research and non-commercial use. No additional
commercial-use grant is provided. Users are responsible for complying with the
terms of every source dataset, including their attribution and share-alike
requirements where applicable.

This work builds on RIFT-SVC, ContentVec, torchfcpe/RMVPE, and OpenVPI
SingingVocoders. Their original licenses and terms continue to apply.

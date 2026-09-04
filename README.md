# RIFT-SVC V4

RIFT-SVC V4 is a quality-first singing voice conversion training stack derived
from RIFT-SVC. It is intentionally incompatible with V1-V3 checkpoints.

The 130k-step EMA foundation checkpoint is available at
[Hugging Face](https://huggingface.co/ooaaqq/RIFT-SVC-V4).

## Design

- 1024-wide, 16-layer rectified-flow Transformer with QK-Norm, RoPE,
  AdaLN-Zero, and standard `1 / sqrt(head_dim)` attention scaling.
- Frozen dual-phase ContentVec features with corrected temporal alignment.
- 44.1 kHz, 128-bin mel features matched to OpenVPI PC-NSF-HiFiGAN.
- Speaker- and song-balanced sampling with bounded duration tempering.
- Song-disjoint validation, EMA checkpoints, and deterministic evaluation.
- Heun inference with a cosine time schedule.

## Install

Use an existing CUDA/PyTorch environment:

```sh
uv sync --extra features
```

## Inference

RIFT consumes precomputed, aligned ContentVec, F0, and RMS features and writes a
PC-NSF-compatible log-mel tensor:

```sh
rift-v4-infer \
  --checkpoint /models/rift-v4-foundation-1024x16-step130000.pt \
  --content /data/example.content.pt \
  --f0 /data/example.f0.pt \
  --rms /data/example.rms.pt \
  --speaker GTSinger:0042 \
  --output-mel /data/example.mel.pt
```

Decode the mel with the pinned official PC-NSF checkpoint:

```sh
rift-v4-pc-nsf \
  --checkout /models/SingingVocoders \
  --lock third_party/pc_nsf_hifigan.lock.json \
  --checkpoint /models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt \
  --mel /data/example.mel.pt \
  --f0 /data/example.f0.pt \
  --output /data/example.wav
```

## Training

Prepare an audited foundation manifest, then run:

```sh
scripts/prepare_training.sh
scripts/train.sh smoke
scripts/train.sh perf
scripts/train.sh rift
```

`smoke` verifies the formal BF16/Flash-SDPA/compiled selective-FP8/fused-AdamW
stack and the 12,288/16,384-frame memory envelope. `perf` compares eager BF16,
compiled BF16, and compiled FP8 throughput before the run configuration is
frozen.

For one target singer, keep songs disjoint between train and validation and
reuse the foundation model's mel statistics:

```sh
scripts/prepare_target.sh /data/target/incoming

export RIFT_MANIFEST=/data/target/manifests/training.content.jsonl
export RIFT_MEL_STATS=/data/foundation/manifests/mel-stats.json
export RIFT_RUN_DIR=/data/target/runs/rift
scripts/train.sh finetune /models/rift-v4-foundation-1024x16-step130000.pt
```

Monitor a run with `scripts/monitor_training.sh`.

## Documentation

- [V3 to V4 experiment history](docs/v3-to-v4-experiment-history.md)
- [Dataset layout](docs/dataset-layout.md)
- [Dataset availability](docs/dataset-availability.md)
- [Dataset audit](docs/dataset-audit.md)
- [Current V7 training protocol](docs/training-protocol-v7.md)
- [V6 pilot training protocol](docs/training-protocol-v6.md)
- [PC-NSF asset lock](third_party/pc_nsf_hifigan.lock.json)

## Terms

Code and model weights may have different terms. The published weights are for
research and non-commercial use; no additional commercial-use grant is
provided. Users are responsible for complying with the terms of every source
dataset and upstream dependency.

This project builds on RIFT-SVC, ContentVec, torchfcpe/RMVPE, and OpenVPI
SingingVocoders.

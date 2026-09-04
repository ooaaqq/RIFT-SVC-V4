# V7 foundation training protocol

V7 is a from-scratch run. Its model, data, sampler, optimizer groups, validation
panel, and random seed are frozen before step 0. Pilot checkpoints are not valid
initialization points.

The pinned cloud runtime is Python 3.12, PyTorch 2.12.1, CUDA 13.0, and TorchAO
0.18.0. Do not run a dependency sync that replaces the cloud CUDA build; install
the pinned TorchAO package without changing the preinstalled PyTorch stack.

## Fixed recipe

- RIFT 1024x16 with a 2,816-channel gated convolutional FFN; QK-Norm, RoPE,
  Pre-Norm, AdaLN-Zero, dual-phase ContentVec, explicit F0/voicing, RMS, and
  the V6 mel contract remain unchanged. The parameter-equivalent gated width
  reduces the model from about 377.1M to 313.5M parameters.
- The bounded dataset -> speaker -> song sampler remains unchanged.
- Each optimizer update is capped at 16,384 frames. The 256/384/512-frame
  buckets contain at most 64/42/32 crops respectively.
- AdamW uses 15k linear warmup, then constant learning rates: 1.5e-4 for the
  backbone and 2e-4 for the speaker/null embedding table.
- Linear and Conv matrix weights use weight decay 0.01. Their biases and the
  speaker/null embedding table use zero weight decay.
- Speaker dropout is 5%, precision is BF16, gradient clipping is 1.0, and EMA
  decay is 0.9999.
- CUDA training enforces Flash SDPA without a silent math-backend fallback,
  uses fused AdamW, enables TF32 for residual FP32 matrix operations, and
  compiles the model with `torch.compile(mode="default")`. BF16 remains the
  primary matrix precision.
- TorchAO 0.18.0 applies `rowwise_with_gw_hp` FP8 training to exactly 64 large
  Linear modules: attention QKV/output and FFN fused gate-up/down projections
  in all 16 blocks. Input projections, AdaLN/timestep conditioning, speaker
  embeddings, depthwise convolution, final modulation/output, loss, EMA, and
  optimizer master weights stay in their existing high-precision paths.
- The gate and value projections are already one GEMM. The depthwise
  convolution followed by SiLU and multiply remains in one compiled graph;
  the 5090 profiler, not source shape alone, is the acceptance test for the
  expected pointwise fusion.
- There is no preset cosine decay or minimum LR. After an observed endpoint
  plateau, resume explicitly with `scripts/train.sh decay CHECKPOINT`; this
  multiplies every parameter group's LR by 0.5 and records the multiplier in
  subsequent full checkpoints.

The optimizer constructor aborts unless every trainable tensor appears in
exactly one group. `run_metadata.json` records every parameter name, tensor
count, parameter count, base LR, and weight decay for each group.
Every checkpoint also records PyTorch, CUDA, and TorchAO versions. Resume is
rejected when these versions differ. FP8 conversion is performed before the
optimizer is built and is required to preserve canonical `nn.Linear` state
keys, so EMA release weights load into the ordinary BF16 inference model.

## Validation and checkpoint selection

The deterministic song-disjoint normalized flow validation remains a health
signal. Its best checkpoint is named `best-flow-health.pt`; it is not the model
selection result.

Routine checkpoint ranking uses EMA weights on a locked real-speaker,
song-disjoint panel with correct speaker conditioning, fixed crop positions and
noise, and 32-step Euler integration. A separate sealed shadow panel is created
without manual song selection. Its source manifest, mel statistics, speaker
mapping, selected feature tensors, crop positions, and Gaussian noise are
locked before the first comparison. Report full and active raw-log-mel MSE,
paired median, win rate, song-bootstrap confidence intervals, and speaker-macro
delta at both 512 and 768 frames.

The shadow panel is not evaluated at every checkpoint. It is used only to check
whether an apparent trajectory on the smaller routine panel generalizes. A
third, song-disjoint A-to-B conversion panel excludes every song unit in the
shadow panel and locks source/target pairs plus reference audio. Reconstruction
metrics cannot replace its target/source speaker similarity, content and F0
preservation, and blinded listening comparison.

Keep a final test panel out of LR decisions and checkpoint ranking. Evaluate it
only after the release candidate is selected.

## Retention and resume

- Every 2k: lightweight model + EMA audit checkpoint.
- Every 5k: full model + EMA + optimizer + RNG + sampler-position checkpoint.
- Every 5k: normalized song-disjoint health validation.
- Every 10k after the initial 50k: locked 32-step endpoint ranking panel.
- Every 25k: full 8/16/32/64 solver and conditioning audit.

When both intervals coincide, only the full checkpoint is written; model and
EMA are not duplicated in a second audit file. After deleting the old pilot,
the complete 500k trajectory is expected to require roughly 1.22 TB and fits
the available data volume.

Full checkpoints persist the manual LR multiplier. Resume reconstructs each
group's LR from its configured base LR, global step, and multiplier, preserving
the 4:3 speaker/backbone ratio after every manual decay.

## Launch gates

1. Freeze and hash the manifest, mel statistics, ContentVec provenance, config,
   panel definition, and code revision.
2. Review the sampler exposure audit and verify song-disjoint splits.
3. Review the complete optimizer group inventory. In particular,
   `modulation.weight` must decay, `modulation.bias` must not, and
   `model.speaker.weight` must be the only tensor in the speaker group.
4. Run the 12,288- and 16,384-frame formal-stack GPU smoke tests at 512 frames.
   A successful step proves BF16 autocast, Flash-only SDPA, compiled selective
   FP8, fused AdamW, backward, and optimizer update all work together.
5. Run `scripts/train.sh perf`. It compares eager BF16, compiled BF16, and
   compiled selective FP8 with the same 16,384-frame workload. Keep FP8 only
   when its post-warmup throughput and numerical checks pass on the 5090.
6. Start in a new empty run directory and preserve the full trajectory through
   at least the 300k frame-exposure comparison point.

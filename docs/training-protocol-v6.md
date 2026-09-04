# V6 training protocol

This is a from-scratch, incompatible run. It does not resume V5 checkpoints.

## Fixed model and optimization

- RIFT: 1024 hidden, 16 blocks, 16 heads of width 64, FFN 4x, kernel 31.
- QK-Norm, `1/sqrt(head_dim)`, RoPE, Pre-Norm, and AdaLN-Zero remain unchanged.
- Frozen pinned dual-phase ContentVec, explicit F0/voicing and RMS, no adapter.
- AdamW 2e-4, weight decay 0.01, 10k warmup, cosine to 2e-5 at 200k,
  BF16, gradient clip 1.0, and EMA 0.9999.
- 256/384/512-frame windows with probabilities 0.2/0.3/0.5 and a 12,288-frame
  batch budget. Crops remain 70% voiced-aware and 30% random.

The additional four blocks respond to the V5 observation that later FFNs and
the final attention block carried increasing residual load without numerical
collapse. Width, head statistics, initialization, optimizer, and LR are held
fixed so depth and sampling are the only material training changes.

## Sampling contract

Corpus probabilities are OpenSinger 43.5%, GTSinger 32%, M4Singer 20%,
ACE-Opencpop 2%, Opencpop 1.5%, and Kiritan 1%.

For dataset `d` with `N` speakers, raw speaker mass is
`r_s = accepted_frames_s ** 0.5`. It is normalized with capped-simplex
projection subject to:

```text
sum(q_s) = 1
0.5 / N <= q_s <= 2.0 / N
```

The projection fixes speakers that hit a bound and redistributes remaining
mass over the free speakers until every constraint holds. Songs independently
apply the same square-root weighting and `0.5/M <= q_song <= 2/M` projection.
Sampling then chooses a recording by available frames within the selected song.

Opencpop and ACE-Opencpop form one source family capped at 3.5%. Synthetic
voices do not enter the physical-speaker median. Any singleton real dataset
above three times the median real-speaker probability aborts preparation.

`scripts/prepare_training.sh` writes `records/sampling-audit.json` with expected
crops, audio hours, source hours, and repeat equivalents for every speaker,
song, and source-family song at 200k. This report is part of the frozen run
identity and must be reviewed before renting the training GPU.

## Validation and selection

Splits remain song-disjoint, with Opencpop and ACE derivatives sharing source
song assignments. Every validation song contributes up to two deterministic
recordings and has its own flow loss. Song losses are averaged to speaker loss;
speaker losses are then used for dataset and global summaries.

The primary checkpoint metric is EMA `real_speaker_macro_flow`. Every event also
records real-dataset macro, all-dataset macro, training-mixture-weighted loss,
online-model equivalents, and per-dataset loss. Complete per-speaker and
per-song online/EMA tables are stored in `validation/step-XXXXXXX.json` rather
than expanding the main log. Synthetic validation never selects `best.pt`.

The fixed audio panel remains required at 50k, 100k, 150k, and 200k. Preserve
all milestone candidates; neither the last checkpoint nor the lowest synthetic
or mixture-weighted loss is automatically the release model.

Model/EMA audit checkpoints are retained every 2k. Full optimizer/RNG resume
checkpoints are retained every 5k under `resume-step-*.pt`; lightweight files
are rejected by the resume path. `best.pt` is lightweight and `final.pt` is a
full checkpoint.

## Launch gates

1. Complete feature/audio integrity audit and sampling audit.
2. Review maximum singleton exposure and the most repeated speaker/song rows.
3. Run the configured 1024x16 GPU smoke test at 8,192 and 12,288 frames.
4. Start only in a fresh run directory; never inflate or resume the 12-layer run.
5. At 5k/10k/20k verify finite loss/gradients, attention entropy, AdaLN gates,
   absolute residual RMS, and RF loss by timestep bin.

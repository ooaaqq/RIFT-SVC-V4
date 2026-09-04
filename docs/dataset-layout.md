# V4 dataset layout

The training disk is organized into four lifetimes: immutable source audio,
manifests, reproducible feature caches, and retained model artifacts.  The
directory names below are the interface used by the command examples; source
roots may instead be mounted from object storage or a separate volume.

```text
/data/rift-v4/
  sources/                         # immutable, accepted input candidates
    ACE-Opencpop/<speaker>/<song>/*.wav
    GTSinger/<speaker>/<song>/*.wav
    M4Singer/<speaker>/<song>/*.wav
    OpenSinger/<speaker>/<song>/*.wav
    Kiritan/<speaker>/<song>/*.wav
    Opencpop/<speaker>/<song>/*.wav
  manifests/
    <corpus>.pending.jsonl          # discovery output
    <corpus>.qc.jsonl               # QC decisions and rejection reasons
    <corpus>.split.jsonl            # final whole-song split
    <corpus>.content.jsonl          # immutable ContentVec provenance
    training.content.jsonl          # audited merged training manifest
    mel-stats.json
  features/                        # mel, F0, RMS; safe to regenerate
    <dataset>/<source-id>.mel.pt
    <dataset>/<source-id>.f0.pt
    <dataset>/<source-id>.rms.pt
  raw-content/                     # immutable dual-phase ContentVec tensors
    <dataset>/<source-id>.content.pt
  pc-nsf/                          # disposable symlink staging/preprocessing
  runs/                            # checkpoints, logs, validation samples
  models/                          # pinned encoder and vocoder exports
```

Keep the original downloaded archives outside `sources/` (for example under
`/archives/rift-v4/`) and record their SHA-256 in the dataset manifest.  The
`sources/` tree should contain only normalized mono vocal recordings that are
actually eligible for indexing.  Do not split a song into independent random
files before manifest creation: the splitter keeps complete songs together.

## Storage budget

The budget is based on accepted duration, not compressed download size.

| item | approximate per hour | 295 h |
| --- | ---: | ---: |
| 44.1 kHz mono 16/24-bit WAV | 0.32–0.48 GB | 95–142 GB |
| mel + F0 + RMS tensors | 0.2 GB | 59 GB |
| dual-phase ContentVec float32 | 2.2 GB | 649 GB |
| manifests and logs | - | 10–30 GB |
| 1024x16 audit/full checkpoints | - | 450–600 GB |
| temporary extraction/PC-NSF workspace | 50–120 GB | 50–120 GB |

The repaired pool is expected to contain roughly 290–296 accepted hours. It
fits in about 1.5–1.7 TB including the full checkpoint policy and working
headroom; a 2 TB volume is sufficient after the previous pilot run is removed.
Do not create duplicate per-chunk WAVs. PC-NSF staging is
symlink-based and can be deleted after the vocoder run is verified.

ContentVec caches remain float32 for the first quality run.  Converting them to
float16 is a later storage optimization, not a prerequisite; it changes RIFT's
numerical conditioning and should be validated with listening samples before
adopting it.

## Recommended first pool

Use GTSinger, M4Singer, official OpenSinger, and original Opencpop as the main
real-recording pool. Kiritan is low-weight real data and ACE-Opencpop is capped
synthetic augmentation. Exclude paired speech, PopCS, KiSing, ACE-KiSing, and
the 16 kHz OpenSinger mirrors. GTSinger includes Control plus all six technique
groups; every variant of the same song must share one split.

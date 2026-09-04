# Dataset audit

Canonical snapshot: 2026-08-30, after waveform QC and song-disjoint splitting.
Counts come from the server manifests, not dataset descriptions or checkpoint
inference. Update this file whenever a canonical manifest or sampling policy
changes.

## Corpus summary

| Dataset | Raw entries / h | Accepted entries / h | Rejected | Speakers | Songs | Train / val / test | Main rejection reasons |
|---|---:|---:|---:|---:|---:|---:|---|
| ACE-Opencpop | 105,450 / 128.380 | 105,209 / 128.195 | 241 | 30 | 100 | 94,379 / 5,430 / 5,400 | duration 120; mostly silent 121; near silent 61 |
| GTSinger | 28,628 / 81.015 | 26,711 / 75.291 | 1,917 | 20 | 610 | 23,695 / 1,628 / 1,388 | duplicate audio 1,531; clipping 386 |
| Kiritan | 50 / 3.522 | 43 / 3.036 | 7 | 1 | 43 | 35 / 4 / 4 | mostly silent 7 |
| M4Singer | 20,896 / 29.696 | 20,889 / 29.689 | 7 | 20 | 419 | 18,927 / 1,032 / 930 | mostly silent 4; clipping 2; duration 1 |
| OpenSinger | 43,075 / 51.929 | 42,947 / 51.900 | 128 | 76 | 1,127 | 38,666 / 2,174 / 2,107 | duration 127; mostly silent 1; duplicate audio 1 |
| Opencpop | 3,756 / 5.226 | 3,752 / 5.218 | 4 | 1 | 100 | 3,365 / 181 / 206 | mostly silent 4 |
| **Total** | **201,855 / 299.768** | **199,551 / 293.329** | **2,304** | **148 namespaced** | - | **179,067 / 10,449 / 10,035** | - |

A rejected recording can have more than one technical reason, so reason counts
can exceed the rejected-entry count. `Songs` is the split key count within each
dataset; it is not summed because ACE-Opencpop and Opencpop deliberately share
the same 100 source-song identities.

## Sampling policy

The training sampler uses explicit corpus probabilities. Inside each dataset,
speaker mass starts at `sqrt(accepted frames)` and is projected onto the
bounded simplex `0.5/N <= q_s <= 2/N`. Songs use the same bounded square-root
projection within each speaker; recordings remain duration-weighted within a
song. The complete expected exposure is written to
`records/sampling-audit.json` before mel statistics or training begins.

| Dataset | Role | Draw probability | PC-NSF |
|---|---|---:|---:|---|
| ACE-Opencpop | low-weight synthetic augmentation | 2% | No |
| GTSinger | main real singing and techniques | 32% | Yes |
| Kiritan | low-weight real singing | 1% | Yes |
| M4Singer | main real singing | 20% | Yes |
| OpenSinger | main real singer diversity | 43.5% | Yes |
| Opencpop | main real Mandarin anchor | 1.5% | Yes |

The probabilities sum to 100%. Opencpop and ACE-Opencpop share one source
family capped at 3.5%. A singleton real dataset may not exceed three times the
median real-speaker probability. PC-NSF admission is read from
`config/datasets.json`; synthetic ACE audio is excluded.

## GTSinger groups

| Group | Raw | Accepted | Rejection summary |
|---|---:|---:|---|
| Control | 12,320 | 10,730 | duplicate audio 1,420; clipping 170 |
| Breathy | 2,169 | 2,157 | clipping 10; duplicate audio 2 |
| Falsetto | 3,988 | 3,965 | clipping 23 |
| Glissando | 2,142 | 2,099 | clipping 32; duplicate audio 11 |
| Mixed voice | 4,001 | 3,868 | duplicate audio 96; clipping 37 |
| Pharyngeal | 1,967 | 1,880 | clipping 85; duplicate audio 2 |
| Vibrato | 2,041 | 2,012 | clipping 29 |

Repeated control recordings distributed under multiple technique directories
are byte-identical. QC retains one deterministic copy and rejects the rest;
paired speech is never staged.

## Fixed integrity decisions

- The ACE release has 105,960 parquet rows. Exactly 510 are zero-frame WAVs;
  the canonical manifest correctly contains the 105,450 non-empty rows.
- ACE identities use `speaker/source_song/segment.wav`; the old
  `acesinger_N` song key was invalid and is no longer used.
- ACE-Opencpop and Opencpop share one split namespace. Official Opencpop test
  songs `2044`, `2086`, `2092`, `2093`, and `2100` are always test in both.
- GTSinger includes all seven singing groups and excludes paired speech. Its
  staged filename retains technique and group identity to prevent collisions.
- All accepted audio must have mel, F0, RMS, and one pinned dual-phase
  ContentVec tensor before the final training manifest passes audit.

After feature extraction, all accepted manifests are reconciled to the actual
mel tensor frame length. The final training manifest and merged mel statistics
both contain exactly `81,348,986` train frames; duration sampling and feature
normalization therefore use the same frame counts.

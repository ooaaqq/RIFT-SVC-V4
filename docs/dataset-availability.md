# Dataset availability checklist

Status checked on 2026-08-30. A public metadata page is not by itself proof
that the repository contains original WAV files; the final columns must be
rechecked after downloading a small sample and inspecting the archive.

Use [`docs/dataset-smoke-test.md`](dataset-smoke-test.md) as the required
pre-download gate.  Values below marked as pending are not yet `pass`.

| Dataset | Download address | License | Original audio available? | Audio format | Speakers | Songs / segments | Estimated hours | V4 decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| GTSinger | [HF official](https://huggingface.co/datasets/AaronZ345/GTSinger), [Google Drive](https://drive.google.com/drive/folders/1xcdvCxNAEEfJElt7sEP-xT8dMKxn1_Lz), [paper](https://arxiv.org/abs/2409.13832), [code](https://github.com/AaronZ345/GTSinger) | Dataset terms are in the [dataset license](https://github.com/AaronZ345/GTSinger/blob/master/dataset_license.md); accept before use | **Yes; complete official singing tree downloaded** | 48 kHz, mono, 24-bit WAV | 20 professional singers | 28,628 singing WAVs / 1,366 songs | 80.59 h singing + 16.16 h paired speech | **Main real pool**; include Control and all six technique groups, exclude paired speech |
| M4Singer | [official project](https://github.com/M4Singer/M4Singer), [Google Drive download](https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view?usp=share_link) | [Official agreement](https://github.com/M4Singer/M4Singer/blob/master/dataset_license.md), CC BY-NC-SA 4.0 (non-commercial) | **Yes; official archive downloaded, CRC checked, and fully decoded** | 44.1/48/96 kHz, mono WAV | 20 singers | 20,896 segments, 699 singer-song directories | 29.70 h decoded | **Main real pool** |
| OpenSinger | [official project](https://github.com/Multi-Singer/Multi-Singer.github.io), [paper](https://arxiv.org/abs/2112.10358), [male mirror](https://huggingface.co/datasets/CodecSR/opensinger_male), [female mirror](https://huggingface.co/datasets/CodecSR/opensinger_female) | Official project states CC BY-NC-SA; recheck the archive terms | **Yes; official archive downloaded, hash-verified, and fully scanned** | 44.1 kHz mono WAV; a 2,000-file sample was PCM 16-bit | 76 singers (28 male, 48 female) | 43,075 WAVs; 1,146 songs reported | 51.93 h | **Main real pool, including vocoder training**; reject the unrelated 16 kHz mirrors |
| ACE-Opencpop | [HF/ESPnet](https://huggingface.co/datasets/espnet/ace-opencpop-segments), [paper](https://arxiv.org/abs/2401.17619) | CC BY-NC 4.0 | **Yes; downloaded and decoded** | 48 kHz, mono, 16-bit PCM | 30 virtual vocalists | 105,450 non-empty rows; 105,209 passed V4 technical QC | 128.19 h accepted | **Low-weight synthetic augmentation only** |
| ACE-KiSing | [HF/ESPnet](https://huggingface.co/datasets/espnet/ace-kising-segments), [paper](https://arxiv.org/abs/2401.17619) | CC BY-NC 4.0 | Public shards exist; 18 shards are cached and decodable | 48 kHz, mono, 16-bit PCM | 34 labels in the cached portion, including `original` | Original KiSing songs plus eight added songs | 32.5 h full release; 24.50 h in the current 18-shard cache | **Hold for quality review**; no training exposure until listening and artifact audits pass |
| Opencpop (original) | [official page](https://wenet-e2e.github.io/opencpop/), [current ModelScope repository](https://modelscope.cn/datasets/wenet/opencpop) | Official terms accompany the archive | **Yes; official raw and segmented archives available** | 44.1 kHz studio mono WAV | 1 professional singer | 100 songs / 3,756 official segments | About 5.2 h | **Main real pool**; preserve official five-song test grouping and group ACE derivatives by source song |
| KiSing (original) | [ESPnet ACE paper/release references](https://huggingface.co/datasets/espnet/ace-kising-segments), original KiSing release to be located | Verify original terms | The ACE-KiSing package contains an `original` singer; standalone raw release is still pending | ACE package copy is 48 kHz mono PCM; verify the raw archive independently | 1 singer | 14 songs in KiSing-v1 | 0.7 h reported for v1; 0.9345 h of `original` segments in the current ACE package | **Do not add to the initial training pool**; cached copy has widespread clipping/DC-offset failures |
| Kiritan singing | [official distribution](https://zunko.jp/kiridev/login.php), [corpus paper](https://www.jstage.jst.go.jp/article/ast/42/3/42_E2074/_pdf) | Registration-gated research terms | **Yes; official archive downloaded and decoded** | 96 kHz, 24-bit studio recording | 1 professional singer | 50 WAVs; 43 accepted after QC | 3.52 h raw / 3.04 h accepted | **Low-weight real pool** |
| PopCS | [DiffSinger recipe](https://github.com/MoonInTheRiver/DiffSinger/blob/master/docs/README-SVS-popcs.md) | Not relevant to the current private selection decision | Raw corpus exists | 22.05 kHz | Single female vocalist in the commonly used release | About 5,498 pieces / 127 songs reported | About 5.9 h | **Excluded from the V4 acoustic target** |

## Interpretation

- GTSinger is the strongest verified real-singing source in this list: 80.59 h,
  20 singers, nine languages, and 1,366 songs. Its 16.16 h paired speech is a
  separate resource and is intentionally excluded from the main V4 sampler.
- ACE-Opencpop is accepted only as low-weight augmentation. ACE-KiSing and
  original KiSing are frozen behind a quality-review gate; availability alone
  does not authorize them for training.
- OpenSinger and M4Singer must be checked at the file level. A Hugging Face
  dataset named `*_unit` or `*_extract_unit` is not original audio and cannot
  be passed to the V4 manifest builder.
- The final `speaker`, `song/segment`, and hour values used for training must
  come from the accepted V4 manifest after QC, not from a paper headline or a
  mirror's repository size.

## Smoke-test results

- **GTSinger**: the complete official singing tree contains 28,628 WAVs and
  80.59 hours. The corrected staging preserves Control plus six technique
  groups and excludes `Paired_Speech_Group`.
- **ACE-Opencpop**: 128.19 accepted hours after technical QC; synthetic status
  remains a sampler constraint, not a file-integrity failure.
- **OpenSinger mirrors**: decoded successfully but rejected because they are
  16 kHz and contain only six speakers rather than the checkpoint's 76.
- **OpenSinger official archive**: the official Google Drive object is
  `OpenSinger.tar.gz`, file ID `1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK`, and
  contains 14,046,666,684 bytes. The assembled local and server copies matched
  SHA-256 `a94d9d87f756423cbfd3a5d92a5ee14e0d984a1c8b5385be1bc4c005adad4679`.
  All 43,075 WAVs are mono at 44.1 kHz and total 51.93 hours. A 2,000-file
  spectral audit found no obvious 12 kHz hard cutoff suggesting that the
  release had merely been upsampled from 24 kHz. The paper's 24 kHz statement
  and the rejected 16 kHz mirrors do not describe this verified archive.
- **M4Singer**: official archive CRC passed; 20,896 mono WAVs decode without
  errors and total 29.70 hours.
- **ACE-KiSing partial cache**: 18 decodable Parquet shards contain 22,499
  segments, 24.50 hours, and 34 singer labels. A cross-shard sample of 288
  segments was compared with 288 ACE-Opencpop segments. Median high-band
  energy and level statistics were similar, but three sampled ACE-KiSing
  failures all belonged to `singer=original`.
- **KiSing copy inside ACE-KiSing**: all 855 cached `original` segments were
  scanned. 461 contain at least one clipped sample; 413 (48.30%) exceed either
  0.1% clipped samples or absolute DC offset 0.001, and 345 (40.35%) exceed
  either 1% clipped samples or absolute DC offset 0.005. This proves the
  packaged copy is unsuitable without rejection/repair, but does not by itself
  prove that an independently obtained raw KiSing archive has the same defect.

## Necessity audit

- **Original OpenSinger: accepted main real source.** Relative to GTSinger +
  M4Singer + original Opencpop, it adds much more real-singer diversity than
  any other candidate. The verified official archive is 44.1 kHz and passed
  the bandwidth audit, so accepted files may be used for ContentVec, the
  acoustic model, and the 44.1 kHz singing vocoder. Keep its sampler weight
  explicit because its 51.93 hours would otherwise dominate smaller corpora.
- **Original KiSing: not necessary for the first V4 run.** It adds one singer
  and less than one hour, is already represented inside ACE-KiSing, and the
  cached copy has severe technical defects. A clean independent archive would
  be useful primarily as a small real-melisma validation set, not as a general
  pretraining source.
- **ACE-KiSing: optional targeted augmentation.** Its distinct value is
  melisma and limited language/style coverage, not general real-timbre
  diversity. Existing ACE-Opencpop already supplies ample synthetic singing,
  so ACE-KiSing should not delay the initial run. If later admitted, exclude
  failing `original` files, perform listening QC, group split by source song,
  and keep its sampler probability low.

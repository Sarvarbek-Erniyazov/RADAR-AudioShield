# Train/Test Leakage Audit — AudioShield

*Generated: 2026-06-27*

## Purpose

Cross-corpus OOD evaluation is scientifically valid only if the held-out test corpora share no **instance-level origin** with the training corpora — no shared utterances, source videos, speaker recordings, or near-duplicate audio segments. Acceptable domain similarity (same language, codec, TTS family, acoustic conditions) does **not** invalidate OOD testing and is explicitly not penalized here.

## Setup

- **Training corpora:** asvspoof5, diffssd, fakeorreal, vctk
- **Held-out OOD test corpora:** inthewild, replaydf, ai4t
- **Audit scope:** full dataset — all training files hashed against all OOD test files (no sampling).

## Checks performed

| Check | What it detects | Method | Result |
|---|---|---|---|
| 1. Metadata / path intersection | Shared video IDs, speaker IDs, utterance IDs, filename stems | Exact set intersection of all identifiers (incl. AI4T source-video IDs) | **0 shared** |
| 2a. Exact file hash | Byte-identical files | SHA-256 of raw bytes | **0 collisions** |
| 2b. Normalized-waveform hash | Resampled / re-encoded copies of the same audio | SHA-256 of 16 kHz mono peak-normalized waveform | **0 collisions** |
| 3. Embedding nearest-neighbour | Near-duplicates surviving heavier processing | Frozen-WavLM cosine NN, dual calibrated threshold | *Pending (second pass)* |

## Per-corpus results

| OOD corpus | Test files | Shared identifiers | Exact-hash collisions | Norm-waveform collisions |
|---|---|---|---|---|
| inthewild | 31,779 | 0 | 0 | 0 |
| replaydf | 52,320 | 0 | 0 | 0 |
| ai4t | 3,148 | 0 | 0 | 0 |

*Training files hashed: 531,075 (full set).*

## Finding

Across the **complete** dataset, Checks 1 and 2 find **no instance-level leakage**: no shared identifiers, no byte-identical files, and no resampled/re-encoded duplicates between any training corpus and any held-out OOD corpus. The normalized-waveform hash additionally rules out identical real-audio clips (e.g. shared LibriSpeech/LJSpeech sources between DiffSSD and any OOD corpus) appearing under different filenames.

## Scope and limitations (honest)

- **Checks 1–2 are complete and decisive** for exact and re-encoded duplication, on the full data.

- **Check 3 (embedding nearest-neighbour) is pending.** It detects near-duplicates that survive trimming, partial overlap, or re-synthesis — which hashing cannot catch. This requires frozen-WavLM embeddings of train and test sets and will be run as a second pass. Until then, this audit certifies absence of *exact and re-encoded* leakage, not yet *near-duplicate* leakage.

- **Checks 4–5 (speaker-embedding disjointness, fingerprint segment overlap)** are a planned third pass for corpora where these risks are highest (AI4T, In-the-Wild).

## Conclusion

The held-out OOD corpora show **no exact or re-encoded instance-level overlap** with training, on the full dataset. Reported OOD results are therefore not attributable to instance-level leakage of this kind. The embedding-based near-duplicate check (Check 3) is the remaining step to complete the certification.

## Reproduce

```bash
python scripts/leakage_audit.py --manifest-dir manifests \
  --train asvspoof5 diffssd fakeorreal vctk \
  --test inthewild replaydf ai4t --data-root .. --hash-max-files 0
```
# BioPhys-HyperRADAR DiffSSD Implementation

This folder contains a concrete research implementation for the proposed
BioPhys-HyperRADAR detector on `datasets/03_DiffSSD`.

## What was learned from DiffSSD

DiffSSD contains:

- `generated_speech/`: 70,000 synthetic utterances from 10 TTS methods.
- `train_val_test_splits.csv`: 94,226 rows with `filename`, `method_name`,
  `category`, `source`, `set`, and binary `target`.
- `target=1`: generated speech.
- `target=0`: real speech paths expected under `real_speech/`.

In the current local folder, the generated speech is present, but the real
speech files referenced by the split CSV are not present. The README says they
must be downloaded separately from LJ Speech and LibriSpeech and placed under:

- `datasets/03_DiffSSD/real_speech/ljspeech`
- `datasets/03_DiffSSD/real_speech/librispeech`

The implementation intentionally refuses normal detector training until both
classes are available. Training only on the current generated subset would not
evaluate bona fide vs spoof detection.

## Proposal assessment

The full proposal is strong as a research direction, but DiffSSD alone only
covers English clean generated speech plus metadata for generator method,
category, source, split, speaker, and OpenVoice accent. It does not directly
provide multilingual labels, replay captures, codec chains, RIR variants,
music/noise mixes, or paired media-state variants.

This implementation maps the proposal to what can be run now:

- Frozen/light SSL backbone: optional Hugging Face backbone via
  `--ssl-model-name`; otherwise a compact waveform CNN fallback is used.
- Learnable frontend: trainable log-mel filterbank initialized near mel
  features.
- Physiology branch: pause, breath-like low-energy/noise masks, energy
  modulation, and zero-crossing descriptors.
- Hyperbolic prototype memory: Poincare prototypes for bona fide/spoof and
  method families.
- Media-state-aware MoE: online media transforms create clean, codec proxy,
  resampled, RIR, replay, and noise-mixed states. A media-state head conditions
  expert routing.
- Multi-objective training: spoof BCE, method CE, media-state CE, hyperbolic
  prototype losses, bona fide compactness, energy regularization, and paired
  transform consistency.
- Evaluation: overall EER/ECE/accuracy plus breakdowns by method, category,
  source, speaker, accent, and predicted media state.

Feature extraction is online rather than a separate preprocessing stage:
each batch loads audio, resamples/crops it, creates paired media-transformed
views, then computes trainable STFT/log-mel features and physiology descriptors
inside the model forward pass.

## Files

- `biophys_hyperradar/dataset.py`: DiffSSD split parsing and PyTorch dataset.
- `biophys_hyperradar/models.py`: BioPhys-HyperRADAR architecture.
- `biophys_hyperradar/transforms.py`: RADAR-style online media transforms.
- `biophys_hyperradar/losses.py`: multi-objective loss.
- `biophys_hyperradar/metrics.py`: EER, ECE, grouped metrics.
- `scripts/inspect_diffssd.py`: local dataset audit without torch.
- `scripts/train.py`: training entry point.
- `scripts/evaluate.py`: grouped evaluation entry point.

## Environment

Create a clean environment if the global torch install is broken:

```powershell
cd E:\AI_voice_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r implementation\requirements.txt
```

For CPU-only PyTorch on Windows, use the official PyTorch install command if the
generic install does not resolve the correct wheel.

## Inspect the dataset

```powershell
cd E:\AI_voice_detection
python implementation\scripts\inspect_diffssd.py --dataset-root datasets\03_DiffSSD
```

Expected current result: 70,000 existing generated files and 24,226 missing real
files. This is useful because it confirms why full binary training is blocked
until `real_speech/` is added.

## Train

After adding LJ Speech and LibriSpeech into the expected `real_speech/` layout:

```powershell
cd E:\AI_voice_detection
python implementation\scripts\train.py `
  --dataset-root datasets\03_DiffSSD `
  --output-dir implementation\runs\diffssd_biophys_hyperradar `
  --epochs 100 `
  --early-stopping-patience 12 `
  --batch-size 32 `
  --num-workers 4

Training and evaluation show `tqdm` progress bars by default. Use
`--no-progress` for plain logs.

For harder generalization experiments, especially because DiffSSD keeps some
generator families only in the test split, reduce generator-ID auxiliary weight
and emphasize hard spoof examples:

```powershell
python implementation\scripts\train.py `
  --dataset-root datasets\03_DiffSSD `
  --output-dir implementation\runs\diffssd_biophys_hyperradar_focal `
  --epochs 100 `
  --early-stopping-patience 12 `
  --batch-size 64 `
  --num-workers 8 `
  --balanced-sampler `
  --focal-gamma 2.0 `
  --spoof-loss-weight 2.0 `
  --method-loss-weight 0.05 `
  --method-prototype-loss-weight 0.05 `
  --target-prototype-loss-weight 0.5 `
  --consistency-loss-weight 0.2
```

Evaluation supports `--score-mode logit`, `--score-mode prototype`, and
`--score-mode fused`. It reports both fixed-threshold accuracy and best
thresholds, because the raw scores can be poorly calibrated even when EER is
reasonable.
```

Optional SSL backbone:

```powershell
python implementation\scripts\train.py `
  --dataset-root datasets\03_DiffSSD `
  --ssl-model-name facebook/wav2vec2-xls-r-300m `
  --epochs 100 `
  --early-stopping-patience 12 `
  --batch-size 4
```

For pipeline debugging only, you can bypass the two-class guard:

```powershell
python implementation\scripts\train.py `
  --dataset-root datasets\03_DiffSSD `
  --allow-single-class-debug `
  --max-train-items 64 `
  --max-val-items 32 `
  --epochs 1 `
  --batch-size 4
```

That debug mode is not a valid detector experiment because the current existing
files are all spoof examples.

## Evaluate

```powershell
python implementation\scripts\evaluate.py `
  --dataset-root datasets\03_DiffSSD `
  --checkpoint implementation\runs\diffssd_biophys_hyperradar\best.pt `
  --split test `
  --output-json implementation\runs\eval_metrics.json `
  --output-csv implementation\runs\eval_records.csv
```

The JSON contains pooled metrics and grouped breakdowns.

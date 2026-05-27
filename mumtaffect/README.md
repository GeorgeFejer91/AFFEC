# mumtaffect — Enhanced MuMTAffect

Replication and enhancement of [MuMTAffect](https://github.com/itubrainlab/MuMTAffect) inside the AFFEC devkit.

## What's new vs. the original

| | Original MuMTAffect | This version |
|---|---|---|
| **EEG** | ✗ not included | ✅ 63-ch × 5-band power features (315-dim) |
| **Cursor tracking** | ✗ not included | ✅ velocity, acceleration, path length, click count |
| **AU features** | peaks, slopes, correlations | ✅ + mean velocity per AU channel |
| **GSR features** | SCR decomposition | ✅ + trial-level slope |
| **Data split** | random 15% test (subject leakage risk) | ✅ subject-disjoint GroupKFold |
| **CV strategy** | single train/val/test split | ✅ K-fold (default: 5-fold) |
| **JSON sidecar bug** | naive header read | ✅ robust resolution (AFFEC v2 fix) |
| **Model** | LSTM / GRU / Transformer variants | ✅ Transformer-Attention v2 + EEG/Cursor branches |

## Files

```
mumtaffect/
├── MUMT_V2_DESIGN.md      # Research/implementation plan for MUMT-v2
├── preprocess_v2.py       # Event-aware preprocessing → dataset_mumt_v2.pkl
├── dataset_v2.py          # MUMT-v2 feature selection, scaling, few-shot splits
├── event_features.py      # Native-rate event-window features per modality
├── profile_features.py    # Cognitive-profile and felt/perceived label features
├── audio_features.py      # Optional stimulus-audio context features
├── model_v2.py            # Event-token Transformer with felt/perceived heads
├── train_v2.py            # Subject-disjoint MUMT-v2 training scaffold
├── foundation_preprocess_v2.py # Full-run 50 ms self-supervised shards
├── foundation_dataset_v2.py    # Lazy shard loader for foundation pretraining
├── foundation_model_v2.py      # Cross-modal + next-step SSL Transformer
├── pretrain_v2.py              # Foundation pretraining CLI
├── pickle_generation.py   # Preprocessing → dataset_enhanced.pkl
├── dataset.py             # PyTorch Dataset (13-element tuple per sample)
├── model.py               # AFFECMultiTaskModel (Transformer-Attention v2 + EEG)
├── train.py               # Subject-disjoint K-fold CV training
└── requirements.txt       # Additional dependencies
```

## MUMT-v2 planning

See `MUMT_V2_DESIGN.md` for the proposed next version: event-aware preprocessing, EEG upgrades, planned audio context support, cognitive-profile conditioning, felt/perceived emotion heads, label-noise handling, and few-shot personalization for unseen users.

Generate the MUMT-v2 event-aware table without destructive cross-modality downsampling:

```bash
python -m mumtaffect.preprocess_v2 \
    --dataset_path data/raw \
    --output data/raw/dataset_mumt_v2.pkl
```

Smoke test:

```bash
python -m mumtaffect.preprocess_v2 \
    --dataset_path data/raw \
    --max_participants 1 \
    --output /tmp/dataset_mumt_v2_smoke.pkl
```

Optional EEG/audio context:

```bash
python -m mumtaffect.preprocess_v2 \
    --dataset_path data/raw \
    --include_eeg \
    --media_root /path/to/stimulus_media \
    --output data/raw/dataset_mumt_v2_eeg_audio.pkl
```

Generate full-run foundation-pretraining shards, including EEG:

```bash
conda run --no-capture-output -n autogluon_env python -u -m mumtaffect.foundation_preprocess_v2 \
    --dataset_path data/raw \
    --output_dir data/raw/mumt_v2_foundation_50ms_eeg \
    --include_eeg \
    --grid_hz 20 \
    --chunk_seconds 10 \
    --stride_seconds 5 \
    --overwrite
```

Smoke test the shard pipeline first:

```bash
conda run --no-capture-output -n autogluon_env python -u -m mumtaffect.foundation_preprocess_v2 \
    --dataset_path data/raw \
    --output_dir /tmp/mumt_foundation_smoke \
    --max_participants 1 \
    --max_runs 2 \
    --include_eeg \
    --eeg_max_channels 16 \
    --chunk_seconds 5 \
    --stride_seconds 5 \
    --max_chunks_per_run 8 \
    --overwrite
```

Pretrain with cross-modal reconstruction and 50 ms next-step prediction:

```bash
conda run --no-capture-output -n autogluon_env python -u -m mumtaffect.pretrain_v2 \
    --data_dir data/raw/mumt_v2_foundation_50ms_eeg \
    --out_dir mumtaffect/pretrain_results_v2/foundation_50ms_eeg \
    --epochs 20 \
    --batch_size 32 \
    --next_horizon 1 \
    --target_modalities all \
    --device auto
```

Add foundation embeddings to the supervised event-aware table after a checkpoint is saved:

```bash
conda run --no-capture-output -n autogluon_env python -u -m mumtaffect.foundation_transfer_v2 \
    --pickle data/raw/dataset_mumt_v2.pkl \
    --foundation_dir data/raw/mumt_v2_foundation_50ms_eeg \
    --checkpoint mumtaffect/pretrain_results_v2/foundation_50ms_eeg_all/foundation_model_best.pt \
    --output data/raw/dataset_mumt_v2_foundation.pkl \
    --event_flag video \
    --device auto
```

Train the first MUMT-v2 event-token model:

```bash
conda run --no-capture-output -n autogluon_env python -u -m mumtaffect.train_v2 \
    --pickle data/raw/dataset_mumt_v2.pkl \
    --folds 5 \
    --epochs 30 \
    --metric_interval 5 \
    --batch_log_interval 20 \
    --early_stop_patience 6 \
    --early_stop_monitor val_loss \
    --label_smoothing 0.05 \
    --dropout 0.30 \
    --modality_dropout 0.15 \
    --weight_decay 5e-4 \
    --device auto
```

Few-shot prototype evaluation for unseen users:

```bash
python -m mumtaffect.train_v2 \
    --pickle data/raw/dataset_mumt_v2.pkl \
    --few_shot_k 5 \
    --use_labelstyle \
    --metric_interval 5
```

## Quick start

### 1. Install dependencies
```bash
pip install torch tqdm scikit-learn scipy mne   # mne only needed for EEG
```

### 2. Generate the pickle (all 72 participants, skip EEG for speed)
```bash
python -m mumtaffect.pickle_generation \
    --dataset_path data/raw \
    --skip_eeg
# Output: data/raw/dataset_enhanced.pkl  (~2–4 GB)
```

With EEG (requires `mne`):
```bash
python -m mumtaffect.pickle_generation --dataset_path data/raw
```

Quick smoke-test (2 participants):
```bash
python -m mumtaffect.pickle_generation --dataset_path data/raw --max_participants 2 --skip_eeg
```

### 3. Train with subject-disjoint 5-fold CV
```bash
python -m mumtaffect.train \
    --pickle data/raw/dataset_enhanced.pkl \
    --folds 5 \
    --phase2_epochs 60 \
    --phase4_epochs 30
```

Results saved to `mumtaffect/grid_results/`:
- `cv_results.csv` — per-fold metrics for every config
- `results_aggregate.csv` — mean ± std across folds

Quick test (2 participants, 2 folds):
```bash
python -m mumtaffect.train \
    --pickle data/raw/dataset_enhanced.pkl \
    --folds 2 --max_participants 2 \
    --phase2_epochs 5 --phase4_epochs 3
```

## Model architecture

```
Eye seq ──────────────┐
Pupil seq ────────────┤  Modality Transformer Encoders
AU seq ───────────────┤  (per-modality, then fused via cross-modal Transformer)
GSR seq ──────────────┤
Cursor seq ───────────┘
                       ├─→ Task Attention (personality queries) ─→ Personality branch (OCEAN)
                       └─→ Task Attention (emotion queries)
                                │
EEG band power ────────────────►│ concat with trial-level features
AU/Eye/Shimmer stats ──────────►│
Cursor stats ──────────────────►│
                                 └─→ Emotion head v2 (self-attn MLP) ─→ 4×3-class output
```

**4-phase training schedule** (from original MuMTAffect):
1. Personality warm-up (freeze emotion heads, Adam lr=1e-4, until R²≥0.30)
2. Joint multitask (weighted emotion + personality loss, differential LRs)
3. Personality refinement (CosineAnnealingLR, until R²≥0.95)
4. Emotion fine-tuning (focal loss, ReduceLROnPlateau)

## Evaluation metrics

| Task | Metric |
|---|---|
| Emotion (4 targets) | Macro F1, Weighted F1, Accuracy per target |
| Personality (5 traits) | R² per trait, mean R² |
| Gender (optional) | Macro F1 |

## Data leakage fix (open GitHub issue #1)

The original MuMTAffect uses a random 15% test split, meaning participants can appear in both
train and test. This inflates scores.

This implementation uses **GroupKFold with participant as group**, guaranteeing that no
participant's trials appear in both training and test within any fold.

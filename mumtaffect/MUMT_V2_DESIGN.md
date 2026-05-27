# MUMT-v2 Design Notes

This document turns the current `mumtaffect` prototype into a concrete plan for a new MUMT version and a new paper. It is written around five goals:

1. preserve modality-specific signal quality instead of forcing every modality through the same downsampling path;
2. move from personality prediction to a broader cognitive-profile formulation;
3. support zero-shot and few-shot adaptation for unseen users;
4. explicitly separate felt and perceived emotion;
5. treat self-annotated labels as noisy, user-dependent, and temporally drifting.

The current code already gives a useful base:

- `mumtaffect/pickle_generation.py` supports eye/pupil, AU/videostream, GSR/Shimmer, cursor, participant metadata, and optional EEG band-power extraction.
- `mumtaffect/dataset.py` loads 400-frame sequence tensors plus trial-level features, EEG vectors, cursor vectors, stimulus emotion, Big Five, and gender.
- `mumtaffect/model.py` implements modality-specific Transformer encoders, fusion, task attention, an EEG feature encoder, cursor support, emotion heads, and personality/gender heads.
- `mumtaffect/train.py` uses subject-disjoint `GroupKFold`, per-fold scaling, and a multi-phase training schedule.

There is no audio extraction pipeline in the current `mumtaffect` code, and no local `.wav`, `.mp3`, `.flac`, or video files were found under `data/raw/`. Audio should therefore be treated as a planned extension that depends on locating the stimulus media or an external stimulus repository.

---

## 1. Current Limitations to Fix

### 1.1 Downsampling is too destructive

Current behavior:

- `FIXED_LENGTH = 400` in `pickle_generation.py` forces eye, pupil, AU, GSR, and cursor into the same frame count.
- `downsample_interpolate()` interpolates by row index rather than true event time.
- missing values are often filled as zero before model input.
- the model then performs another temporal compression before fusion.

Why this is a problem:

- GSR has slower dynamics and meaningful latency/recovery structure; aggressive uniform resampling can hide SCR onset, peak, rise time, and recovery time.
- AU signals contain short bursts and confidence failures; row-index interpolation can smooth expression events and ignore OpenFace confidence.
- pupil and gaze contain validity gaps/blinks; zero-filled invalid samples can become a false physiological pattern.
- EEG should not be treated as a 400-frame behavioral stream unless there is a separate raw EEG temporal encoder; band-power features should be extracted after EEG-specific filtering and windowing.
- audio, if added, is stimulus-level temporal context rather than a participant physiological stream.

MUMT-v2 decision:

- keep modality-native preprocessing first;
- derive event-aware tokens using trial events (`first_fix`, `scenario`, `video`, `last_frame_video`, rating stages);
- use masks and quality indicators instead of silently converting missingness to zero;
- perform fusion after modality-specific feature extraction, not before.

### 1.2 Personality prediction does not generalize to new users

Current results show personality `R²` collapses under subject-disjoint evaluation. This means the original formulation behaves closer to personality re-identification or known-user embedding than true unseen-user personality prediction.

MUMT-v2 decision:

- stop presenting Big Five prediction as the core personalization result;
- define a broader cognitive profile made of:
  - demographics: age, gender, handedness, education, English proficiency, glasses/contact-lens status;
  - environmental context: lux, temperature, prior exposure indicators if available;
  - stable self-report traits: Big Five as optional profile fields, not the only target;
  - physiological baselines: resting/first-fix pupil, gaze stability, tonic EDA, EEG baseline band power;
  - labeling style: per-user mean, variance, class thresholds, label entropy, and order drift, computed only from training users or from few-shot support trials for new users.

The paper should call this cognitive-profile conditioning, not personality prediction.

### 1.3 Few-shot adaptation is necessary

Subject-independent affect recognition is hard because physiological baselines and label thresholds vary heavily by participant. A pure zero-shot model is unlikely to fully solve this from 72 participants.

MUMT-v2 decision:

- evaluate both zero-shot and few-shot personalization;
- introduce a small support set for each unseen user (`k = 1, 3, 5, 10` trials);
- adapt only lightweight user modules, not the whole model;
- report adaptation curves per target and per modality configuration.

### 1.4 Felt and perceived emotions are different tasks

AFFEC’s strongest scientific value is that it contains both felt and perceived valence/arousal. Collapsing the discussion to generic valence/arousal wastes this advantage.

MUMT-v2 decision:

- predict all four labels separately:
  - `Felt_Arousal`;
  - `Felt_Valence`;
  - `Perceived_Arousal`;
  - `Perceived_Valence`.
- add explicit gap targets:
  - `Gap_Arousal = Felt_Arousal - Perceived_Arousal`;
  - `Gap_Valence = Felt_Valence - Perceived_Valence`.
- analyze whether physiological modalities mainly predict felt emotion, while stimulus/audio/facial context mainly predicts perceived emotion.

### 1.5 Self-annotated labels are noisy and user-dependent

Current binning maps 1--9 ratings into three hard classes. This throws away ordinal structure and ignores user rating tendencies.

MUMT-v2 decision:

- keep raw 1--9 labels for ordinal/regression experiments;
- compare hard 3-class classification against ordinal classification and continuous regression;
- estimate per-user label tendency only from training data or few-shot support trials;
- model trial order because users may recalibrate labels during the experiment.

---

## 2. Proposed MUMT-v2 Architecture

### 2.1 High-level structure

```text
Native modality streams
  ├─ Eye / gaze encoder
  ├─ Pupil encoder
  ├─ AU / face encoder
  ├─ GSR / EDA encoder
  ├─ Cursor encoder
  ├─ EEG encoder
  └─ Audio stimulus encoder

Event-aware tokenization
  ├─ baseline / first_fix tokens
  ├─ scenario tokens
  ├─ video tokens
  └─ rating-stage tokens

Profile and context conditioning
  ├─ CognitiveProfileEncoder
  ├─ LabelStyleEncoder
  ├─ StimulusContextEncoder
  └─ UserAdapter / few-shot profile update

Task heads
  ├─ Felt arousal
  ├─ Felt valence
  ├─ Perceived arousal
  ├─ Perceived valence
  ├─ Felt-perceived arousal gap
  └─ Felt-perceived valence gap
```

### 2.2 Modality encoders

#### Eye and gaze

Current code uses gaze columns from `GAZE_COLS` and extracts fixation-level features.

MUMT-v2 changes:

- keep native timestamps and compute event-window statistics;
- include validity masks for gaze samples;
- derive scanpath features:
  - fixation count/rate;
  - fixation duration;
  - fixation dispersion;
  - saccade amplitude and velocity;
  - gaze entropy over screen regions;
  - gaze-to-face/region-of-interest features if stimulus coordinates are available.

Expected contribution:

- perceived emotion and appraisal should benefit from gaze/attention patterns.

#### Pupil

Current code computes actual pupil size and summary statistics.

MUMT-v2 changes:

- compute baseline-corrected pupil dilation:
  - `pupil_delta_video = pupil_video - pupil_first_fix_baseline`;
  - slope, peak, recovery, and variability per event window;
  - blink rate and blink duration as separate behavioral features;
  - pupil validity ratio as a quality feature.

Expected contribution:

- arousal, cognitive effort, and stimulus engagement.

#### AU / videostream

Current code extracts AU intensity statistics, binary activation rates, slopes, peak counts, simple expression composites, and KMeans cluster proportions.

MUMT-v2 changes:

- use OpenFace confidence explicitly;
- compute AU features only on high-confidence frames or add confidence-weighted summaries;
- separate AU dynamics by event stage;
- add AU co-activation graph features:
  - smile: AU06 + AU12;
  - frown: AU04 + AU15;
  - surprise/attention: AU01/AU02/AU05;
  - blink: AU45;
  - mouth opening: AU25/AU26.

Expected contribution:

- perceived valence and stimulus interpretation; possibly felt-valence leakage if participants mimic/react facially.

#### GSR / EDA

Current code uses NeuroKit EDA processing and extracts SCR statistics, temperature, accelerometer magnitude, and slope.

MUMT-v2 changes:

- process GSR at native sampling rate before any resampling;
- extract tonic/phasic decomposition by event window;
- baseline-correct tonic EDA from first-fix/scenario to video;
- include SCR latency from video onset, SCR count, amplitude, rise time, and recovery;
- include movement/accelerometer quality as an artifact control.

Expected contribution:

- felt arousal more than perceived valence.

#### Cursor

Current code extracts path length, velocity, acceleration, spatial spread, and click count.

MUMT-v2 changes:

- distinguish cursor behavior during rating stages from stimulus-viewing behavior;
- use response latency and mouse trajectory during annotation as part of label uncertainty / cognitive profile;
- avoid mixing rating-stage behavior into physiological response tokens unless the research question explicitly includes annotation behavior.

Expected contribution:

- label confidence, indecision, response style, and few-shot profile estimation.

#### EEG

Current code already supports optional EEG features:

- `compute_eeg_features()` reads EDF with `mne`;
- extracts Welch PSD band power for 63 channels × 5 bands;
- merges features as `eeg_<channel>_<band>`.

Current limitations:

- no artifact rejection;
- no notch/bandpass filtering;
- no baseline correction;
- only video window band power;
- no hemispheric asymmetry;
- no event-stage comparison;
- no temporal EEG encoder.

MUMT-v2 EEG plan:

1. Minimal stable EEG feature pipeline:
   - load EDF with `mne`;
   - apply notch filter at local power-line frequency if needed;
   - bandpass filter, e.g. 1--45 Hz;
   - reject or mark bad channels/epochs using amplitude thresholds;
   - extract log band power per event window;
   - baseline-correct video-window power against first-fix/scenario;
   - add frontal asymmetry features where channel pairs are available.

2. EEG feature families:
   - absolute log band power: delta/theta/alpha/beta/gamma;
   - relative band power;
   - theta/beta ratio;
   - alpha asymmetry;
   - spectral entropy;
   - per-region aggregates: frontal, central, temporal, parietal, occipital.

3. EEG model options:
   - tabular EEG encoder for precomputed features;
   - lightweight temporal CNN for event-level EEG windows;
   - optional self-supervised EEG pretraining later, after the baseline is stable.

Expected contribution:

- felt arousal, cognitive engagement, attention, and individual baseline estimation.

#### Audio

Current code status:

- no audio extractor exists in `mumtaffect`;
- `requirements.txt` does not include audio libraries;
- no local audio/video media files were found under `data/raw/` during inspection.

Important distinction:

- audio is not a participant physiological response unless microphone recordings of the participant exist;
- stimulus audio is contextual information shared across participants;
- therefore audio should be handled as stimulus context, comparable to but more realistic than `Stim Emo`.

MUMT-v2 audio plan if stimulus media is available:

1. Resolve media source:
   - find original video/audio stimuli by `stim_file`;
   - extract audio track to `.wav`;
   - create a `stimulus_audio_features.csv` keyed by `stim_file`.

2. Handcrafted audio features:
   - pitch/F0 mean, std, range;
   - energy/intensity mean and dynamics;
   - speaking rate / pause ratio if speech segmentation is possible;
   - MFCC statistics;
   - spectral centroid, rolloff, bandwidth;
   - jitter/shimmer if voice quality is reliable.

3. Learned audio embeddings:
   - frozen wav2vec-style or HuBERT-style embeddings;
   - average pooled over the stimulus or over event-aligned windows;
   - keep embeddings frozen initially to avoid overfitting.

4. Experimental controls:
   - `NoContext`: physiology only;
   - `StimLabelContext`: current one-hot `Stim Emo`;
   - `AudioContext`: audio-derived context only;
   - `StimLabel+AudioContext`: upper-bound context setting.

Expected contribution:

- perceived emotion should benefit more from audio than felt emotion;
- if audio predicts felt emotion too strongly, analyze whether it reflects stimulus-induced affect or context leakage.

---

## 3. Cognitive Profile Formulation

### 3.1 Profile fields

Replace the narrow personality branch with a `CognitiveProfileEncoder`.

Candidate inputs:

- demographic profile:
  - age;
  - gender;
  - handedness;
  - education;
  - English proficiency;
  - glasses/contact-lens status.
- environment:
  - luminance/lux;
  - temperature;
  - prior exposure if available.
- stable trait profile:
  - Big Five scores as optional known metadata;
  - never claim unseen-user personality prediction unless evaluated without providing true Big Five at inference.
- physiological baseline profile:
  - baseline pupil;
  - tonic EDA;
  - baseline EEG band power;
  - gaze stability;
  - blink rate.
- label-style profile:
  - user mean and variance per target;
  - class threshold tendency;
  - response entropy;
  - trial-order drift;
  - discrepancy tendency between felt and perceived labels.

### 3.2 Prediction targets

Do not use cognitive profile as one single regression target. Use it in three ways:

1. `KnownProfile`: profile fields available at inference.
2. `EstimatedProfile`: profile estimated from baseline sensors and support trials.
3. `LatentUserAdapter`: learned user representation adapted from few-shot support examples.

The paper should report these separately.

---

## 4. Few-shot Personalization

### 4.1 Evaluation protocol

Use subject-disjoint folds. For each test user:

1. hold out the user from training;
2. split that user’s trials into support and query sets;
3. use `k` support trials for adaptation;
4. evaluate on remaining query trials.

Recommended `k` values:

- `k = 0`: zero-shot;
- `k = 1`: minimal calibration;
- `k = 3`: practical quick onboarding;
- `k = 5`: small calibration;
- `k = 10`: richer calibration.

Report mean ± std across users and folds.

### 4.2 Adaptation methods

Start simple and defensible:

1. User prototype:
   - compute support-set embedding mean;
   - condition query predictions on distance to support embeddings.

2. Label-style adapter:
   - estimate per-user label mean/variance from support labels;
   - adjust prediction thresholds or logits.

3. Profile adapter:
   - small MLP maps support examples to a user vector;
   - concatenate user vector to task heads.

4. Low-rank adapters:
   - freeze base encoders;
   - update only small LoRA/adapters in fusion and task heads.

5. Optional meta-learning:
   - MAML/Reptile-style training can be tested later;
   - only include it in the paper if simpler adapters are stable.

### 4.3 Key metrics

- zero-shot performance;
- few-shot improvement at each `k`;
- adaptation efficiency: improvement per support trial;
- per-target improvement;
- fairness of adaptation across gender/age groups if sample size allows.

---

## 5. Felt vs Perceived Modeling

### 5.1 Task setup

Use six outputs:

- `Felt_Arousal`;
- `Felt_Valence`;
- `Perceived_Arousal`;
- `Perceived_Valence`;
- `Gap_Arousal`;
- `Gap_Valence`.

For each label, compare:

- 3-class classification;
- ordinal classification;
- continuous regression.

### 5.2 Hypotheses

Candidate hypotheses for the paper:

1. physiological signals are stronger for felt arousal than perceived valence;
2. stimulus/audio context is stronger for perceived emotion than felt emotion;
3. cognitive-profile conditioning improves felt emotion more than perceived emotion;
4. few-shot adaptation improves felt emotion and label calibration more than stimulus-driven perceived emotion;
5. felt-perceived gap is predictable from profile, stimulus context, and physiological response mismatch.

### 5.3 Reporting

Report each target separately. Do not average away the distinction unless a compact summary is needed.

Minimum table:

| Model | Felt A Macro-F1 | Felt V Macro-F1 | Perceived A Macro-F1 | Perceived V Macro-F1 | Gap A MAE | Gap V MAE |
|---|---:|---:|---:|---:|---:|---:|

---

## 6. Label Preprocessing and Noise

### 6.1 Problems with current labels

- raw ratings are self-annotated and subjective;
- users have different thresholds for using high/medium/low values;
- users may become more consistent or change interpretation over time;
- hard 3-class binning loses ordinal information;
- felt and perceived labels have different noise sources.

### 6.2 MUMT-v2 label strategies

Compare at least three approaches:

1. Raw 3-class bins:
   - keep as backward-compatible baseline.

2. User-normalized labels:
   - subtract user mean or use user quantile bins;
   - only valid when user statistics are available from training/support data;
   - not valid for zero-shot users unless computed from support trials.

3. Ordinal targets:
   - preserve order of 1--9 ratings;
   - use ordinal regression or cumulative link loss;
   - evaluate with MAE, Spearman correlation, and adjacent accuracy.

4. Soft labels:
   - convert one rating into a distribution over nearby classes;
   - reduces penalty for adjacent-bin ambiguity.

5. Trial-order drift:
   - include trial index/run index as a nuisance variable or profile feature;
   - estimate whether early vs late trials show label calibration.

### 6.3 Leakage guardrails

Do not compute user-specific normalization using query/test labels. For unseen users:

- zero-shot: no user label statistics;
- few-shot: compute user label statistics only from support trials;
- report exactly which normalization setting is used.

---

## 7. Preprocessing Redesign

### 7.1 New dataset artifact

Create a new artifact instead of modifying `dataset_enhanced.pkl` in place:

- `dataset_mumt_v2.pkl`
- one row per trial;
- raw or event-windowed modality features;
- masks and quality fields;
- clear feature namespaces.

Suggested columns:

```text
keys:
  user, run, trial, stim_file

labels:
  felt_arousal_raw, felt_valence_raw
  perceived_arousal_raw, perceived_valence_raw
  felt_arousal_bin, felt_valence_bin
  perceived_arousal_bin, perceived_valence_bin
  gap_arousal, gap_valence

context:
  stim_emo
  audio_features
  audio_embedding

profile:
  age, gender, handedness, education, english_proficiency, glasses
  openness, conscientiousness, extraversion, agreeableness, neuroticism
  label_style_features

modalities:
  eye_event_features, eye_quality
  pupil_event_features, pupil_quality
  au_event_features, au_quality
  gsr_event_features, gsr_quality
  cursor_event_features, cursor_quality
  eeg_event_features, eeg_quality
```

### 7.2 Event windows

Use event markers as primary temporal structure:

- `first_fix`: baseline visual/physiological state;
- `scenario`: semantic context priming;
- `video`: stimulus response;
- `last_frame_video`: transition/post-stimulus state;
- `f_emotion_labelling`: felt rating behavior;
- `p_emotion_labelling`: perceived rating behavior.

For each modality, compute:

- baseline features;
- video features;
- video minus baseline deltas;
- temporal slope;
- quality/missingness.

### 7.3 Feature namespaces

Use explicit prefixes:

- `eye__`;
- `pupil__`;
- `au__`;
- `gsr__`;
- `cursor__`;
- `eeg__`;
- `audio__`;
- `profile__`;
- `labelstyle__`;
- `context__`.

This prevents accidental feature mixing and makes ablations auditable.

---

## 8. Model Redesign

### 8.1 Recommended minimal MUMT-v2 model

Start with a stable tabular/event-token hybrid before adding complex temporal encoders:

1. modality event-feature encoders:
   - one MLP per modality;
   - modality dropout;
   - missingness masks.

2. event-token Transformer:
   - tokens represent modality × event stage;
   - supports missing tokens naturally.

3. profile/context conditioning:
   - profile embedding;
   - audio/stimulus context embedding;
   - optional few-shot user adapter.

4. task heads:
   - four emotion heads;
   - two gap heads;
   - optional label-style head.

This is easier to debug than raw sequence Transformers over all modalities.

### 8.2 Later model variants

After the minimal model is stable:

- add raw temporal encoders for selected modalities;
- add EEG temporal CNN;
- add audio temporal Transformer;
- add graph-based AU co-activation;
- add cross-attention from physiology to stimulus/audio tokens.

---

## 9. Experimental Matrix

### 9.1 Core experiments

| Experiment | Purpose |
|---|---|
| Majority baseline | sanity-check class imbalance |
| Classical tabular baseline | test feature quality |
| Current MUMT reproduction | compare against original approach |
| Event-aware MUMT-v2 | measure preprocessing improvement |
| MUMT-v2 + cognitive profile | test profile conditioning |
| MUMT-v2 + EEG | test EEG contribution |
| MUMT-v2 + audio context | test stimulus audio contribution |
| MUMT-v2 + few-shot adapters | test new-user adaptation |

### 9.2 Ablations

Required:

- no profile;
- no label-style features;
- no stimulus emotion;
- no audio;
- no EEG;
- no GSR;
- no AU;
- no eye/pupil;
- no few-shot adaptation.

Important controls:

- random split vs subject-disjoint split as a leakage audit;
- known Big Five vs predicted/estimated profile;
- raw labels vs user-normalized labels vs ordinal labels.

### 9.3 Metrics

Classification:

- macro-F1;
- weighted-F1;
- accuracy;
- class-wise F1.

Ordinal/regression:

- MAE;
- RMSE;
- Spearman correlation;
- adjacent accuracy.

Few-shot:

- zero-shot vs `k`-shot curve;
- per-user improvement distribution;
- mean ± std across folds.

Profile:

- do not center the paper on Big Five `R²`;
- if reported, separate:
  - known-user profile reconstruction;
  - unseen-user profile estimation;
  - profile usefulness for emotion prediction.

---

## 10. Paper Framing

### 10.1 Candidate title

`MUMT-v2: Event-Aware Cognitive-Profile Conditioning and Few-Shot Personalization for Felt and Perceived Emotion Recognition`

### 10.2 Core claim

MUMT-v2 is not just a larger multimodal network. Its contribution is a cleaner formulation of AFFEC:

- modality-native feature extraction;
- event-aware fusion;
- cognitive-profile conditioning;
- few-shot new-user adaptation;
- explicit felt/perceived emotion modeling;
- label-noise-aware evaluation.

### 10.3 Contributions

Suggested contributions:

1. We introduce an event-aware multimodal preprocessing pipeline that preserves modality-specific physiology and produces auditable quality masks.
2. We reformulate personalization from Big Five prediction to cognitive-profile conditioning, combining demographics, baseline physiology, labeling style, and optional traits.
3. We evaluate zero-shot and few-shot adaptation protocols for unseen AFFEC users.
4. We model felt and perceived emotion as distinct but related tasks and add felt-perceived gap prediction.
5. We compare hard binning, user-normalized labels, and ordinal/continuous label formulations for noisy self-annotations.

### 10.4 Claim boundaries

Avoid:

- claiming personality prediction generalizes unless the protocol proves it;
- treating `Stim Emo` or audio as normal physiological sensors;
- claiming TCE implementation from Transformer modules;
- reporting only averaged valence/arousal metrics.

Use:

- “trait/profile-conditioned”;
- “event-aware”;
- “few-shot personalization”;
- “privileged stimulus context”;
- “felt/perceived distinction.”

---

## 11. Implementation Roadmap

### Phase 0: Audit and baselines

- Freeze current results as `MUMT-v1 reproduction`.
- Add majority-class and simple tabular baselines.
- Confirm participant counts and modality coverage.
- Document missing modalities per user/trial.

### Phase 1: Event-aware preprocessing

- Build `event_windows.py` utilities.
- Replace row-index downsampling with event-window feature extraction.
- Add quality/missingness masks.
- Save `dataset_mumt_v2.pkl`.

### Phase 2: EEG upgrade

- Improve `compute_eeg_features()`:
  - notch/bandpass;
  - artifact flags;
  - baseline correction;
  - regional aggregation;
  - asymmetry features.
- Add EEG quality metrics.

### Phase 2b: Full-run foundation pretraining

- Keep this separate from the supervised trial-window artifact.
- Generate full-run shards at a 20 Hz / 50 ms grid with explicit observed-value masks.
- Include eye, pupil, AU, GSR, cursor, event-context flags, and optional regional EEG mean/std streams.
- Use cross-modal reconstruction: mask one or more non-event modalities and reconstruct them from all remaining modalities.
- Use next-step prediction: predict the future grid state, default one step ahead = 50 ms.
- Report train/validation SSL loss before using the encoder for downstream emotion tasks.
- Treat event flags as context, not as a sensor target.

### Phase 3: Audio extension

- Locate stimulus media.
- Add `audio_features.py`.
- Precompute audio features/embeddings keyed by `stim_file`.
- Add `AudioContextEncoder`.
- Evaluate audio as stimulus context, not user physiology.

### Phase 4: Cognitive profile

- Add `profile_features.py`.
- Encode metadata, baselines, and label-style features.
- Separate known profile, estimated profile, and few-shot profile settings.

### Phase 5: MUMT-v2 model

- Implement event-token dataset.
- Implement modality MLP encoders and event Transformer.
- Add profile/context conditioning.
- Add four task heads plus gap heads.
- Add missing-modality dropout.

### Phase 6: Few-shot adaptation

- Implement support/query split per test user.
- Add user prototype adapter.
- Add label-style adapter.
- Optionally add low-rank adapters after prototypes work.

### Phase 7: Paper experiments

- Run subject-disjoint 5-fold CV.
- Run zero-shot and few-shot settings.
- Run ablations.
- Generate publication tables.

---

## 12. Immediate Code Tasks

1. Add `mumtaffect/event_features.py`.
2. Add `mumtaffect/profile_features.py`.
3. Add `mumtaffect/audio_features.py` once media location is known.
4. Add `mumtaffect/dataset_v2.py`.
5. Add `mumtaffect/model_v2.py`. ✅ Initial event-token Transformer added.
6. Add `mumtaffect/train_v2.py`. ✅ Initial subject-disjoint/few-shot scaffold added.
7. Add full-run foundation pretraining path. ✅ Initial shard generator, dataset, SSL model, and pretraining CLI added.
8. Keep old files intact for reproducibility.

This separation is important: MUMT-v2 should be a clean new implementation, not a patched version of the current pipeline.

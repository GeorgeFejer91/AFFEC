"""
Cognitive-profile and label utilities for MUMT-v2.

This module keeps leakage-sensitive label-style computation separate from
static preprocessing. Static metadata can be written into the v2 dataset,
while user label statistics should be fitted only on train/support splits.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


RAW_LABEL_COLS = {
    "felt_arousal": "felt_arousal_raw",
    "felt_valence": "felt_valence_raw",
    "perceived_arousal": "perceived_arousal_raw",
    "perceived_valence": "perceived_valence_raw",
}

SOURCE_LABEL_COLS = {
    "felt_arousal": "f_emotion_a",
    "felt_valence": "f_emotion_v",
    "perceived_arousal": "p_emotion_a",
    "perceived_valence": "p_emotion_v",
}

REACTION_TIME_COLS = {
    "felt_arousal": "f_emotion_at",
    "felt_valence": "f_emotion_vt",
    "perceived_arousal": "p_emotion_at",
    "perceived_valence": "p_emotion_vt",
}

PERSONALITY_COLS = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}


def rating_to_bin(value) -> float:
    """Map AFFEC 1--9 ratings to low/mid/high bins 0/1/2."""
    try:
        return float(np.clip((int(value) - 1) // 3, 0, 2))
    except Exception:
        return np.nan


def _float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clean_category(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    value = str(value).strip().strip('"').lower()
    return value if value else "unknown"


def _one_hot(prefix: str, value: object, categories: Iterable[str]) -> Dict[str, float]:
    clean = _clean_category(value)
    return {f"{prefix}__{category}": float(clean == category) for category in categories}


def participant_profile_features(participants: pd.DataFrame, user: str) -> Dict[str, float]:
    row = participants[participants["participant_id"] == user]
    if row.empty:
        return {}
    data = row.iloc[0]

    age = np.nan
    for col in data.index:
        if str(col).strip().strip('"').lower().startswith("age"):
            age = _float(data[col])
            break

    features: Dict[str, float] = {
        "profile__age": age,
        "profile__lux": _float(data.get("lux")),
        "profile__temperature": _float(data.get("tem")),
        "profile__prior_exposure": _float(data.get("ex")),
    }

    for name, source in PERSONALITY_COLS.items():
        features[f"profile__big5__{name}"] = _float(data.get(source))

    features.update(_one_hot("profile__gender", data.get("gender"), ("f", "m", "unknown")))
    features.update(_one_hot("profile__handedness", data.get("handedness"), ("l", "r", "unknown")))
    features.update(_one_hot("profile__education", data.get("education"), ("college", "bsc", "msc", "phd", "other", "unknown")))
    features.update(_one_hot("profile__english_prof", data.get("eng_prof"), ("n", "f", "unknown")))
    features.update(_one_hot("profile__glasses", data.get("glass"), ("yes", "no", "unknown")))
    return features


def label_features(label_row: pd.Series) -> Dict[str, float]:
    """Create raw, binned, response-time, and felt-perceived gap labels."""
    features: Dict[str, float] = {}
    for semantic_name, source_col in SOURCE_LABEL_COLS.items():
        raw_name = RAW_LABEL_COLS[semantic_name]
        raw_value = _float(label_row.get(source_col))
        features[raw_name] = raw_value
        features[raw_name.replace("_raw", "_bin")] = rating_to_bin(raw_value)

    for semantic_name, source_col in REACTION_TIME_COLS.items():
        features[f"label_rt__{semantic_name}"] = _float(label_row.get(source_col))

    features["gap_arousal_raw"] = features["felt_arousal_raw"] - features["perceived_arousal_raw"]
    features["gap_valence_raw"] = features["felt_valence_raw"] - features["perceived_valence_raw"]
    features["gap_arousal_abs"] = abs(features["gap_arousal_raw"])
    features["gap_valence_abs"] = abs(features["gap_valence_raw"])
    features["label__trial_index"] = _float(label_row.get("trial"))
    features["label__run_index"] = _float(label_row.get("run"))
    return features


def context_features(stim_emo: object) -> Dict[str, float]:
    categories = ("neutral", "sad", "happy", "angry", "fear", "disgust", "unknown")
    clean = _clean_category(stim_emo)
    return {
        "stim_emo": clean,
        **{f"context__stim_emo__{category}": float(clean == category) for category in categories},
    }


def fit_label_style(
    df: pd.DataFrame,
    user_col: str = "user",
    label_cols: Iterable[str] = tuple(RAW_LABEL_COLS.values()),
) -> Dict[str, Dict[str, float]]:
    """
    Fit per-user label-style features from a training/support dataframe.

    Do not call this on full data before cross-validation. For unseen users,
    fit it only on support trials.
    """
    style: Dict[str, Dict[str, float]] = {}
    if df.empty or user_col not in df.columns:
        return style

    for user, user_df in df.groupby(user_col):
        user_features: Dict[str, float] = {"labelstyle__n_support_trials": float(len(user_df))}
        for col in label_cols:
            if col not in user_df.columns:
                continue
            values = pd.to_numeric(user_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            prefix = f"labelstyle__{col}"
            if values.size == 0:
                user_features[f"{prefix}__mean"] = np.nan
                user_features[f"{prefix}__std"] = np.nan
                user_features[f"{prefix}__entropy"] = np.nan
                continue
            user_features[f"{prefix}__mean"] = float(np.mean(values))
            user_features[f"{prefix}__std"] = float(np.std(values))
            hist = np.bincount(np.clip(values.astype(int), 1, 9), minlength=10)[1:]
            probs = hist[hist > 0] / max(hist.sum(), 1)
            user_features[f"{prefix}__entropy"] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
        if {"felt_arousal_raw", "perceived_arousal_raw"}.issubset(user_df.columns):
            gap = pd.to_numeric(user_df["felt_arousal_raw"], errors="coerce") - pd.to_numeric(user_df["perceived_arousal_raw"], errors="coerce")
            user_features["labelstyle__gap_arousal__mean"] = float(np.nanmean(gap))
        if {"felt_valence_raw", "perceived_valence_raw"}.issubset(user_df.columns):
            gap = pd.to_numeric(user_df["felt_valence_raw"], errors="coerce") - pd.to_numeric(user_df["perceived_valence_raw"], errors="coerce")
            user_features["labelstyle__gap_valence__mean"] = float(np.nanmean(gap))
        style[str(user)] = user_features
    return style


def apply_label_style(df: pd.DataFrame, style_by_user: Mapping[str, Mapping[str, float]], user_col: str = "user") -> pd.DataFrame:
    """Append pre-fitted label-style columns to a dataframe."""
    rows = []
    for _, row in df.iterrows():
        values = dict(row)
        values.update(style_by_user.get(str(row.get(user_col)), {}))
        rows.append(values)
    return pd.DataFrame(rows)


def add_global_label_style_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure label-style columns exist when no user-specific support is available."""
    defaults = {
        "labelstyle__n_support_trials": 0.0,
        "labelstyle__gap_arousal__mean": 0.0,
        "labelstyle__gap_valence__mean": 0.0,
    }
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
    return df


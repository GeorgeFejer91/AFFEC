"""
Optional stimulus-audio features for MUMT-v2.

AFFEC physiological recordings do not currently expose an audio modality in
the `mumtaffect` pipeline. This module supports stimulus-level audio context
when the original stimulus media are available locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mov", ".avi")


def resolve_stimulus_media(stim_file: str, media_root: str | Path) -> Optional[Path]:
    """Resolve an AFFEC `stim_file` value against a media root."""
    if not stim_file or media_root is None:
        return None

    root = Path(media_root)
    stim_path = Path(str(stim_file))
    candidates = [
        root / stim_path,
        root / stim_path.name,
        root / stim_path.with_suffix(".wav"),
        root / stim_path.name.replace(stim_path.suffix, ".wav"),
    ]

    stem = stim_path.stem
    for ext in AUDIO_EXTENSIONS:
        candidates.append(root / f"{stem}{ext}")
        candidates.extend(root.glob(f"**/{stem}{ext}"))

    for candidate in candidates:
        if isinstance(candidate, Path) and candidate.exists():
            return candidate
    return None


def _load_audio(path: Path):
    """Load audio with librosa if available, otherwise scipy for WAV files."""
    try:
        import librosa

        waveform, sample_rate = librosa.load(str(path), sr=16000, mono=True)
        return waveform.astype(float), int(sample_rate)
    except Exception:
        pass

    if path.suffix.lower() != ".wav":
        return None, None

    try:
        from scipy.io import wavfile

        sample_rate, waveform = wavfile.read(str(path))
        waveform = waveform.astype(float)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        max_abs = np.max(np.abs(waveform)) if waveform.size else 0.0
        if max_abs > 0:
            waveform = waveform / max_abs
        return waveform, int(sample_rate)
    except Exception:
        return None, None


def extract_audio_features(path: str | Path, prefix: str = "audio") -> Dict[str, float]:
    """Extract lightweight stimulus-audio context features."""
    path = Path(path)
    waveform, sample_rate = _load_audio(path)
    if waveform is None or sample_rate is None or waveform.size == 0:
        return {}

    waveform = np.nan_to_num(waveform.astype(float))
    duration = waveform.size / float(sample_rate)
    features: Dict[str, float] = {
        f"{prefix}__duration_s": float(duration),
        f"{prefix}__rms_mean": float(np.sqrt(np.mean(waveform ** 2))),
        f"{prefix}__amplitude_mean_abs": float(np.mean(np.abs(waveform))),
        f"{prefix}__amplitude_std": float(np.std(waveform)),
        f"{prefix}__zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(waveform))).astype(float))) if waveform.size > 1 else 0.0,
    }

    frame_size = min(1024, waveform.size)
    if frame_size >= 8:
        spectrum = np.abs(np.fft.rfft(waveform[: frame_size * max(1, waveform.size // frame_size)].reshape(-1, frame_size), axis=1))
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
        denom = np.sum(spectrum, axis=1) + 1e-12
        centroid = np.sum(spectrum * freqs[None, :], axis=1) / denom
        bandwidth = np.sqrt(np.sum(spectrum * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / denom)
        features[f"{prefix}__spectral_centroid_mean"] = float(np.mean(centroid))
        features[f"{prefix}__spectral_centroid_std"] = float(np.std(centroid))
        features[f"{prefix}__spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features[f"{prefix}__energy_dynamic_range"] = float(np.max(denom) - np.min(denom))

    return features


def build_audio_feature_cache(stim_files: Iterable[str], media_root: str | Path) -> Dict[str, Dict[str, float]]:
    """Build features keyed by AFFEC `stim_file`."""
    cache: Dict[str, Dict[str, float]] = {}
    for stim_file in sorted(set(str(s) for s in stim_files if s)):
        media = resolve_stimulus_media(stim_file, media_root)
        cache[stim_file] = extract_audio_features(media) if media else {}
    return cache


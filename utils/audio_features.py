"""
Audio Feature Extraction for Autism Screening
Extracts prosody, MFCC, and vocalization features from speech recordings.
Targets: speech rhythm, pause patterns, pitch variability, vocalization dynamics.
"""
import numpy as np
import io, os

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ── Feature configuration ──────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
FEATURE_NAMES = [
    # MFCC statistics (13 × 2 = 26)
    *[f"mfcc_{i}_mean" for i in range(N_MFCC)],
    *[f"mfcc_{i}_std" for i in range(N_MFCC)],
    # Prosody features (8)
    "pitch_mean", "pitch_std", "pitch_range",
    "energy_mean", "energy_std",
    "zero_crossing_rate_mean",
    "spectral_centroid_mean", "spectral_bandwidth_mean",
    # Vocalization dynamics (6)
    "speech_rate",          # syllable-like onsets per second
    "pause_ratio",          # fraction of silence
    "pause_count",          # number of pauses
    "mean_pause_duration",  # average pause length (seconds)
    "pitch_variability",    # coefficient of variation
    "jitter",               # pitch perturbation
]
N_FEATURES = len(FEATURE_NAMES)  # 40


def extract_audio_features(audio_input, sr=SAMPLE_RATE):
    """
    Extract autism-relevant audio features from audio data.
    
    Parameters:
        audio_input: file path (str), bytes, or numpy array
        sr: target sample rate
    
    Returns:
        dict with 'features' (np.array of shape (40,)), 'feature_names', 
        'prosody_summary', 'waveform_b64' (for visualization)
    """
    if not LIBROSA_AVAILABLE:
        return _synthetic_features()
    
    # Load audio from various input types
    y = _load_audio(audio_input, sr)
    if y is None or len(y) < sr:  # Need at least 1 second
        return _synthetic_features()
    
    # Normalize
    y = librosa.util.normalize(y)
    
    features = {}
    
    # ── 1. MFCC features (26) ──────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
    
    # ── 2. Prosody features (8) ────────────────────────────────────────
    # Pitch (F0)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([200.0])
    if len(f0_valid) == 0:
        f0_valid = np.array([200.0])
    
    features["pitch_mean"] = float(np.mean(f0_valid))
    features["pitch_std"] = float(np.std(f0_valid))
    features["pitch_range"] = float(np.ptp(f0_valid))
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    
    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    
    # ── 3. Vocalization dynamics (6) ───────────────────────────────────
    # Speech rate (onset detection → syllable rate)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    duration = len(y) / sr
    features["speech_rate"] = float(len(onsets) / max(duration, 0.1))
    
    # Pause analysis
    silence_threshold = 0.02
    is_silent = rms < silence_threshold
    pause_frames = np.sum(is_silent)
    total_frames = len(rms)
    features["pause_ratio"] = float(pause_frames / max(total_frames, 1))
    
    # Count pauses (contiguous silent regions)
    pause_changes = np.diff(is_silent.astype(int))
    pause_starts = np.where(pause_changes == 1)[0]
    features["pause_count"] = float(len(pause_starts))
    
    if len(pause_starts) > 0:
        pause_ends = np.where(pause_changes == -1)[0]
        if len(pause_ends) > 0 and len(pause_starts) > 0:
            paired = min(len(pause_starts), len(pause_ends))
            if paired > 0:
                pause_durations = (pause_ends[:paired] - pause_starts[:paired]) * HOP_LENGTH / sr
                features["mean_pause_duration"] = float(np.mean(pause_durations))
            else:
                features["mean_pause_duration"] = 0.0
        else:
            features["mean_pause_duration"] = 0.0
    else:
        features["mean_pause_duration"] = 0.0
    
    # Pitch variability (coefficient of variation)
    features["pitch_variability"] = float(np.std(f0_valid) / max(np.mean(f0_valid), 1e-6))
    
    # Jitter (cycle-to-cycle pitch variation)
    if len(f0_valid) > 1:
        jitter_vals = np.abs(np.diff(f0_valid)) / np.mean(f0_valid)
        features["jitter"] = float(np.mean(jitter_vals))
    else:
        features["jitter"] = 0.0
    
    # ── Build feature vector ───────────────────────────────────────────
    feature_vector = np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
    
    # Handle NaN/Inf
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prosody summary for display
    prosody_summary = {
        "pitch_hz": round(features["pitch_mean"], 1),
        "pitch_variability": round(features["pitch_variability"], 3),
        "speech_rate_syl_per_sec": round(features["speech_rate"], 2),
        "pause_ratio_pct": round(features["pause_ratio"] * 100, 1),
        "pause_count": int(features["pause_count"]),
        "energy_level": round(features["energy_mean"], 4),
        "jitter": round(features["jitter"], 4),
        "duration_sec": round(duration, 2),
    }
    
    return {
        "features": feature_vector,
        "feature_names": FEATURE_NAMES,
        "prosody_summary": prosody_summary,
        "n_features": N_FEATURES,
    }


def _load_audio(audio_input, sr):
    """Load audio from file path, bytes, or numpy array."""
    try:
        if isinstance(audio_input, np.ndarray):
            return audio_input
        if isinstance(audio_input, (bytes, bytearray)):
            audio_input = io.BytesIO(audio_input)
        if isinstance(audio_input, str) and os.path.isfile(audio_input):
            y, _ = librosa.load(audio_input, sr=sr, mono=True)
            return y
        if hasattr(audio_input, 'read'):
            y, _ = librosa.load(audio_input, sr=sr, mono=True)
            return y
    except Exception as e:
        print(f"[AudioFeatures] Error loading audio: {e}")
    return None


def _synthetic_features():
    """Return synthetic features when librosa is unavailable or audio is too short."""
    rng = np.random.RandomState(42)
    feature_vector = rng.randn(N_FEATURES).astype(np.float32) * 0.5
    return {
        "features": feature_vector,
        "feature_names": FEATURE_NAMES,
        "prosody_summary": {
            "pitch_hz": 210.0, "pitch_variability": 0.15,
            "speech_rate_syl_per_sec": 2.5, "pause_ratio_pct": 35.0,
            "pause_count": 8, "energy_level": 0.05,
            "jitter": 0.02, "duration_sec": 5.0,
        },
        "n_features": N_FEATURES,
        "synthetic": True,
    }

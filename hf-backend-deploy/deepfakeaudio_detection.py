# -*- coding: utf-8 -*-
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import librosa
import torch
import torch.nn as nn


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, input_channels=3, n_mfcc=40, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, None)),
        )

        self.lstm = nn.LSTM(
            input_size=128 * 4, hidden_size=256,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )

        self.attention = nn.Linear(512, 1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).reshape(B, T, C * F)
        out, _ = self.lstm(out)
        attn = torch.softmax(self.attention(out), dim=1)
        context = (out * attn).sum(dim=1)
        return self.classifier(context)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and checkpoint path resolution
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
ORIGINAL_MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"
RETRAINED_MODEL_PATH = PROJECT_DIR / "models" / "best_model_retrained.pth"
COND_4B_MODEL_PATH = PROJECT_DIR / "models" / "phase4_experiments" / "cond_4b_finetune.pth"
WEBRTC_MODEL_PATH = PROJECT_DIR / "models" / "best_webrtc_model.pth"
MIC_MODEL_PATH = PROJECT_DIR / "models" / "best_mic_model.pth"
DEFAULT_MODEL_PATH = WEBRTC_MODEL_PATH if WEBRTC_MODEL_PATH.exists() else ORIGINAL_MODEL_PATH

MODEL_THRESHOLDS: dict[str, float] = {
    "original": 0.50,
    "production": 0.50,
    "retrained": 0.50,
    "webrtc": 0.620,
    "mic": 0.550,
    "cond_4b": 0.37,
    "condition_4b": 0.37,
    "4b": 0.37,
    "phase4_4b": 0.37,
    "cond_4b_finetune": 0.37,
    "experimental": 0.37,
}

_models: dict = {}


def get_active_model_name() -> str:
    default_mode = "webrtc" if WEBRTC_MODEL_PATH.exists() else "original"
    return os.getenv("ACTIVE_MODEL_CHECKPOINT", default_mode).strip().lower()


def get_model_threshold(model_tag: str, custom_threshold: float | None = None) -> float:
    if custom_threshold is not None and isinstance(custom_threshold, (int, float, str)):
        try:
            return float(custom_threshold)
        except (ValueError, TypeError):
            pass
    return MODEL_THRESHOLDS.get(str(model_tag).lower().strip(), 0.50)


def resolve_checkpoint_path(checkpoint: Path | str | None = None) -> tuple[Path, str]:
    if checkpoint is None or not isinstance(checkpoint, (str, Path)):
        checkpoint = get_active_model_name()

    chk_str = str(checkpoint).strip().lower()
    if chk_str in ("webrtc", "live", "default", "prod", "production", "none", ""):
        if WEBRTC_MODEL_PATH.exists():
            return WEBRTC_MODEL_PATH, "webrtc"
        return ORIGINAL_MODEL_PATH, "original"
    elif chk_str in ("mic",):
        if MIC_MODEL_PATH.exists():
            return MIC_MODEL_PATH, "mic"
        if WEBRTC_MODEL_PATH.exists():
            return WEBRTC_MODEL_PATH, "webrtc"
        return ORIGINAL_MODEL_PATH, "original"
    elif chk_str in ("original",):
        return ORIGINAL_MODEL_PATH, "original"
    elif chk_str in ("retrained", "balanced", "new", "test"):
        return RETRAINED_MODEL_PATH, "retrained"
    elif chk_str in ("cond_4b", "condition_4b", "4b", "phase4_4b", "cond_4b_finetune", "experimental"):
        return COND_4B_MODEL_PATH, "cond_4b"
    else:
        target_path = Path(checkpoint)
        if not target_path.is_absolute():
            target_path = PROJECT_DIR / target_path
        return target_path, target_path.stem


def get_device() -> torch.device:
    global device
    return device


def get_model(checkpoint_path: Path | str | None = None) -> AudioDeepfakeDetector:
    target_path, model_tag = resolve_checkpoint_path(checkpoint_path)
    cache_key = str(target_path.resolve())

    if cache_key in _models:
        return _models[cache_key]

    if not target_path.exists():
        fallback_path = BACKEND_DIR / target_path.name
        if fallback_path.exists():
            target_path = fallback_path
            cache_key = str(target_path.resolve())
        else:
            try:
                from download_model import download_model_if_missing
                download_model_if_missing(target_path)
            except Exception as dl_err:
                print(f"[Model Provisioning] Auto-download attempt failed: {dl_err}")

        if not target_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{target_path}'. Expected PyTorch checkpoint."
            )

    net = AudioDeepfakeDetector().to(device)
    try:
        raw_ckpt = torch.load(target_path, map_location=device, weights_only=False)
        if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["model_state_dict"]
            if "optimal_tau" in raw_ckpt and model_tag != "webrtc":
                MODEL_THRESHOLDS[model_tag] = float(raw_ckpt["optimal_tau"])
        else:
            state_dict = raw_ckpt

        incompatible = net.load_state_dict(state_dict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint at '{target_path}' is incompatible with AudioDeepfakeDetector.\n"
                f"Missing keys: {incompatible.missing_keys}\n"
                f"Unexpected keys: {incompatible.unexpected_keys}"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from '{target_path}': {e}") from e

    net.eval()
    _models[cache_key] = net
    return net


def __getattr__(name):
    if name == "model":
        return get_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


SAMPLE_RATE = 16000
DURATION = 3
MAX_LEN = SAMPLE_RATE * DURATION
N_MFCC = 40
N_MELS = 40
HOP_LENGTH = 512
N_FFT = 1024


def extract_features_webrtc(audio_array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Exact (3, 40, 94) feature pipeline used to train best_webrtc_model.pth and best_mic_model.pth."""
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0)
    wav = audio_array.astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8)

    if len(wav) < MAX_LEN:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    else:
        wav = wav[:MAX_LEN]

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfcc)

    def fix_len(arr, target=94):
        if arr.shape[1] < target:
            return np.pad(arr, ((0, 0), (0, target - arr.shape[1])))
        return arr[:, :target]

    return np.stack([fix_len(mfcc), fix_len(delta1), fix_len(log_mel)], axis=0).astype(np.float32)


def extract_features(audio_array, sr=SAMPLE_RATE, use_cmvn: bool = False):
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0)
    audio_array = audio_array.astype(np.float32)

    audio_array, _ = librosa.effects.trim(audio_array, top_db=25)
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))

    if len(audio_array) < MAX_LEN:
        audio_array = np.pad(audio_array, (0, MAX_LEN - len(audio_array)))
    else:
        audio_array = audio_array[:MAX_LEN]

    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    T = mfcc.shape[1]

    mel = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)[:N_MFCC, :T]

    features = np.stack([mfcc, delta, mel_db], axis=0)

    if not use_cmvn:
        for i in range(3):
            features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-8)
    else:
        for i in range(3):
            mean_f = features[i].mean(axis=1, keepdims=True)
            std_f = np.maximum(features[i].std(axis=1, keepdims=True), 1e-4)
            features[i] = (features[i] - mean_f) / std_f

    return features.astype(np.float32)


def generate_audio_windows(
    audio_array: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_sec: float = 3.0,
    stride_sec: float = 1.5,
    min_rms: float = 0.005,
) -> tuple[list[tuple[np.ndarray, float, float, float]], list[tuple[np.ndarray, float, float, float]]]:
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    total_samples = len(audio_array)

    if total_samples == 0:
        return [], []

    if total_samples <= window_samples:
        rms = float(np.sqrt(np.mean(audio_array ** 2)))
        win_item = (audio_array, 0.0, round(total_samples / sr, 2), rms)
        if rms >= min_rms or total_samples > 0:
            return [win_item], []
        return [], [win_item]

    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    final_start = total_samples - window_samples
    if final_start > starts[-1]:
        starts.append(final_start)

    active_windows, silent_windows = [], []
    for start_idx in starts:
        end_idx = start_idx + window_samples
        chunk = audio_array[start_idx:end_idx]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        win_item = (chunk, round(start_idx / sr, 2), round(end_idx / sr, 2), rms)
        if rms >= min_rms:
            active_windows.append(win_item)
        else:
            silent_windows.append(win_item)

    return active_windows, silent_windows


def predict_audio(
    audio_array: np.ndarray,
    sr: int = SAMPLE_RATE,
    model_version: Optional[str] = None,
    use_multiwindow: bool = True,
    window_sec: float = 3.0,
    stride_sec: float = 1.5,
    min_rms: float = 0.005,
    aggregation: str = "mean",
    use_cmvn: bool = False,
    threshold: Optional[float] = None,
) -> dict:
    target_path, model_tag = resolve_checkpoint_path(model_version)
    tau = get_model_threshold(model_tag, threshold)

    if audio_array is None or len(audio_array) == 0:
        return {
            "is_deepfake": False,
            "label": "REAL",
            "confidence": 0.0,
            "real_pct": 0.0,
            "fake_pct": 0.0,
            "model_version": model_tag,
            "threshold_used": tau,
            "windows_analyzed": 0,
            "total_windows": 0,
            "aggregation_method": aggregation,
            "mean_fake_pct": 0.0,
            "median_fake_pct": 0.0,
            "features": None,
        }

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0 if audio_array.shape[0] < audio_array.shape[1] else 1)
    audio_array = audio_array.astype(np.float32)

    feat_fn = (
        extract_features_webrtc
        if model_tag in ("webrtc", "mic")
        else (lambda a, s: extract_features(a, s, use_cmvn=use_cmvn))
    )

    total_samples = len(audio_array)
    window_samples = int(window_sec * sr)

    if not use_multiwindow or total_samples <= window_samples:
        feats = feat_fn(audio_array, sr)
        net = get_model(target_path)
        dev = get_device()
        tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(dev)

        with torch.inference_mode():
            logits = net(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        real_pct = float(probs[0].item() * 100.0)
        fake_pct = float(probs[1].item() * 100.0)
        is_deepfake = bool(fake_pct >= (tau * 100.0))
        label = "FAKE" if is_deepfake else "REAL"
        confidence = float(probs[1].item() if is_deepfake else probs[0].item())

        return {
            "is_deepfake": is_deepfake,
            "label": label,
            "confidence": round(confidence, 4),
            "real_pct": round(real_pct, 2),
            "fake_pct": round(fake_pct, 2),
            "model_version": model_tag,
            "threshold_used": tau,
            "windows_analyzed": 1,
            "total_windows": 1,
            "aggregation_method": "single_window",
            "mean_fake_pct": round(fake_pct, 2),
            "median_fake_pct": round(fake_pct, 2),
            "features": feats,
        }

    active_windows, silent_windows = generate_audio_windows(
        audio_array, sr=sr, window_sec=window_sec, stride_sec=stride_sec, min_rms=min_rms
    )
    total_windows_count = len(active_windows) + len(silent_windows)

    if not active_windows:
        fallback_chunk = audio_array[:window_samples]
        feats = feat_fn(fallback_chunk, sr)
        return {
            "is_deepfake": False,
            "label": "REAL",
            "confidence": 0.0,
            "real_pct": 50.0,
            "fake_pct": 50.0,
            "status": "Flagged: Audio Contains No Audible Speech (Silence)",
            "model_version": model_tag,
            "threshold_used": tau,
            "windows_analyzed": 0,
            "total_windows": total_windows_count,
            "aggregation_method": "silence_rejection",
            "mean_fake_pct": 50.0,
            "median_fake_pct": 50.0,
            "features": feats,
        }

    feats_list = [feat_fn(chunk, sr) for chunk, _start, _end, _rms in active_windows]

    net = get_model(target_path)
    dev = get_device()
    batch_tensor = torch.tensor(np.stack(feats_list), dtype=torch.float32).to(dev)

    with torch.inference_mode():
        logits = net(batch_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    p_real_all = probs[:, 0] * 100.0
    p_fake_all = probs[:, 1] * 100.0

    mean_real = float(np.mean(p_real_all))
    mean_fake = float(np.mean(p_fake_all))
    median_real = float(np.median(p_real_all))
    median_fake = float(np.median(p_fake_all))

    if aggregation == "median":
        agg_real, agg_fake = median_real, median_fake
    else:
        agg_real, agg_fake = mean_real, mean_fake

    is_deepfake = bool(agg_fake >= (tau * 100.0))
    label = "FAKE" if is_deepfake else "REAL"
    confidence = float(max(agg_real, agg_fake) / 100.0)

    return {
        "is_deepfake": is_deepfake,
        "label": label,
        "confidence": round(confidence, 4),
        "real_pct": round(agg_real, 2),
        "fake_pct": round(agg_fake, 2),
        "model_version": model_tag,
        "threshold_used": tau,
        "windows_analyzed": len(active_windows),
        "total_windows": total_windows_count,
        "aggregation_method": aggregation,
        "mean_real_pct": round(mean_real, 2),
        "mean_fake_pct": round(mean_fake, 2),
        "median_real_pct": round(median_real, 2),
        "median_fake_pct": round(median_fake, 2),
        "min_fake_pct": round(float(np.min(p_fake_all)), 2),
        "max_fake_pct": round(float(np.max(p_fake_all)), 2),
        "std_fake_pct": round(float(np.std(p_fake_all)), 2),
        "features": feats_list[0] if feats_list else None,
    }


def run_inference(audio_np, sr=16000):
    if audio_np is None:
        return None, 0.0, 0.0, None
    res = predict_audio(audio_np, sr)
    pred = 1 if res["is_deepfake"] else 0
    return pred, res["real_pct"], res["fake_pct"], res["features"]


def predict_file(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    res = predict_audio(audio, sr)
    return res["label"], res["real_pct"], res["fake_pct"]
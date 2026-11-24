from collections import deque
import io
import time
from typing import Dict, Optional
import numpy as np
import librosa

from deepfakeaudio_detection import predict_audio, get_model, SAMPLE_RATE

AUDIO_CONTAINER_SIGNATURES = {
    b"RIFF": "WAV",
    b"\x1a\x45\xdf\xa3": "WebM/Matroska",
    b"OggS": "OGG",
    b"ID3": "MP3",
    b"fLaC": "FLAC",
}


def is_valid_mp3_frame_header(header: bytes) -> tuple[bool, int, int, int]:
    if len(header) < 4:
        return False, 0, 0, 0
    b0, b1, b2, b3 = header[0], header[1], header[2], header[3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return False, 0, 0, 0
    version_bits = (b1 >> 3) & 0x03
    if version_bits == 1:
        return False, 0, 0, 0
    layer_bits = (b1 >> 1) & 0x03
    if layer_bits == 0:
        return False, 0, 0, 0
    bitrate_idx = (b2 >> 4) & 0x0F
    if bitrate_idx in (0x00, 0x0F):
        return False, 0, 0, 0
    sr_idx = (b2 >> 2) & 0x03
    if sr_idx == 3:
        return False, 0, 0, 0
    padding = (b2 >> 1) & 0x01
    if (b3 & 0x03) == 2:
        return False, 0, 0, 0

    if version_bits == 3:
        sr_table = [44100, 48000, 32000]
        sample_rate = sr_table[sr_idx]
        if layer_bits == 1:
            br_table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
            return True, int(144 * br_table[bitrate_idx] * 1000 / sample_rate) + padding, version_bits, layer_bits
        elif layer_bits == 2:
            br_table = [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
            return True, int(144 * br_table[bitrate_idx] * 1000 / sample_rate) + padding, version_bits, layer_bits
        elif layer_bits == 3:
            br_table = [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448]
            return True, int((12 * br_table[bitrate_idx] * 1000 / sample_rate) + padding) * 4, version_bits, layer_bits
    else:
        sr_table_m2 = [22050, 24000, 16000] if version_bits == 2 else [11025, 12000, 8000]
        sample_rate = sr_table_m2[sr_idx]
        if layer_bits == 1:
            br_table = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
            return True, int(72 * br_table[bitrate_idx] * 1000 / sample_rate) + padding, version_bits, layer_bits
        else:
            br_table = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
            return True, int(144 * br_table[bitrate_idx] * 1000 / sample_rate) + padding, version_bits, layer_bits

    return False, 0, 0, 0


def detect_container_format(audio_bytes: bytes) -> Optional[str]:
    if len(audio_bytes) >= 4:
        header4 = audio_bytes[:4]
        if header4.startswith(b"RIFF"):
            if len(audio_bytes) >= 12 and audio_bytes[8:12] == b"WAVE":
                return "WAV"
            elif len(audio_bytes) < 12:
                return "WAV"

        for sig, fmt in AUDIO_CONTAINER_SIGNATURES.items():
            if sig != b"RIFF" and header4.startswith(sig):
                return fmt

        v1, f_len, ver1, lay1 = is_valid_mp3_frame_header(header4)
        if v1 and f_len > 4 and len(audio_bytes) >= f_len + 4:
            v2, _, ver2, lay2 = is_valid_mp3_frame_header(audio_bytes[f_len : f_len + 4])
            if v2 and (ver1 == ver2) and (lay1 == lay2):
                return "MP3"
    return None


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_duration: float = 3.0,
        stride_duration: float = 1.0,
        max_buffer_duration: float = 6.0,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_duration)
        self.stride_samples = int(sample_rate * stride_duration)
        self.max_buffer_samples = int(sample_rate * max_buffer_duration)

        self.client_buffers: Dict[str, np.ndarray] = {}
        self.client_models: Dict[str, str] = {}
        self.client_prob_history: Dict[str, deque] = {}
        self.client_last_result: Dict[str, dict] = {}

        try:
            self._model = get_model()
        except Exception:
            self._model = None

    def set_client_model(self, client_id: str, model_version: str) -> None:
        self.client_models[client_id] = model_version

    def get_client_model(self, client_id: str) -> str:
        return self.client_models.get(client_id, "webrtc")

    def clear_client_buffer(self, client_id: str) -> None:
        self.client_buffers.pop(client_id, None)
        self.client_models.pop(client_id, None)
        self.client_prob_history.pop(client_id, None)
        self.client_last_result.pop(client_id, None)

    def decode_pcm16_bytes(self, audio_bytes: bytes) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)

        container = detect_container_format(audio_bytes)
        if container:
            raise ValueError(
                f"Received encoded {container} audio stream. WebSocket streaming expects raw PCM16."
            )

        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                f"Invalid PCM16 byte length ({len(audio_bytes)} bytes). Must be a multiple of 2."
            )

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0

    def decode_audio_file(self, file_bytes: bytes, filename: str = "") -> np.ndarray:
        if not file_bytes:
            raise ValueError("Empty audio file data.")
        try:
            bio = io.BytesIO(file_bytes)
            audio, _ = librosa.load(bio, sr=self.sample_rate, mono=True)
            if audio is None or len(audio) == 0:
                raise ValueError("Decoded audio contains no samples.")
            return audio.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file '{filename}': {e}") from e

    def extract_features(self, audio_signal: np.ndarray) -> dict:
        if len(audio_signal) == 0:
            return {"rms_energy": 0.0, "zcr": 0.0, "spectral_flux": 0.0}

        rms = float(np.sqrt(np.mean(audio_signal ** 2)))
        zero_crossings = np.nonzero(np.diff(audio_signal > 0))[0]
        zcr = float(len(zero_crossings) / len(audio_signal))
        fft_vals = np.abs(np.fft.rfft(audio_signal))
        spectral_flux = float(np.mean(fft_vals)) if len(fft_vals) > 0 else 0.0

        return {
            "rms_energy": round(rms, 4),
            "zcr": round(zcr, 4),
            "spectral_flux": round(spectral_flux, 4),
        }

    def process_client_stream(
        self, client_id: str, audio_bytes: bytes, model_version: Optional[str] = None
    ) -> dict:
        active_model = model_version or self.client_models.get(client_id, "webrtc")
        chunk = self.decode_pcm16_bytes(audio_bytes)
        telemetry = self.extract_features(chunk)

        current_buf = self.client_buffers.get(client_id, np.array([], dtype=np.float32))
        if len(chunk) > 0:
            current_buf = np.concatenate([current_buf, chunk])

        if len(current_buf) > self.max_buffer_samples:
            current_buf = current_buf[-self.max_buffer_samples :]

        self.client_buffers[client_id] = current_buf

        if len(current_buf) < self.window_samples:
            return {
                "status": f"Buffering ({len(current_buf)}/{self.window_samples})",
                "is_buffering": True,
                "confidence": 1.0,
                "is_anomaly": False,
                "label": "BUFFERING",
                "real_pct": 100.0,
                "fake_pct": 0.0,
                "model_version": active_model,
                "samples_buffered": len(current_buf),
                "samples_needed": self.window_samples,
                "features": telemetry,
            }

        window = current_buf[: self.window_samples]
        self.client_buffers[client_id] = current_buf[self.stride_samples :]

        # Speech Energy Gate (lowered to 0.008 so speaker-played AI voice is never skipped)
        window_rms = float(np.sqrt(np.mean(window ** 2)))
        if window_rms < 0.008:
            return {
                "status": "Authentic Human Voice",
                "is_anomaly": False,
                "confidence": 1.0,
                "label": "REAL",
                "real_pct": 100.0,
                "fake_pct": 0.0,
                "model_version": active_model,
                "is_buffering": False,
                "features": telemetry,
            }

        # Run model inference
        t0 = time.perf_counter()
        prediction = predict_audio(window, sr=self.sample_rate, model_version=active_model)
        proc_time = round((time.perf_counter() - t0) * 1000, 2)

        raw_fake_prob = prediction["fake_pct"] / 100.0
        tau = prediction.get("threshold_used", 0.620)

        # Lightweight 2-window smoothing so live AI playback triggers immediately
        hist = self.client_prob_history.setdefault(client_id, deque(maxlen=2))
        hist.append(raw_fake_prob)
        weights = [0.35, 0.65][-len(hist):]
        ema_prob = sum(w * p for w, p in zip(weights, hist)) / sum(weights)

        is_fake = bool(ema_prob >= tau or raw_fake_prob >= (tau + 0.08))
        ema_fake_pct = round(ema_prob * 100.0, 2)
        ema_real_pct = round(100.0 - ema_fake_pct, 2)
        status = "Flagged: Deepfake Detected" if is_fake else "Authentic Human Voice"

        print(
            f"[LIVE DETECT] client={client_id} | model={prediction.get('model_version')} "
            f"| rms={window_rms:.4f} | raw_fake={prediction['fake_pct']:.1f}% "
            f"| ema={ema_fake_pct:.1f}% | tau={tau:.3f} | verdict={'FAKE' if is_fake else 'REAL'}"
        )

        result = {
            "status": status,
            "is_anomaly": is_fake,
            "confidence": round(ema_prob if is_fake else (1.0 - ema_prob), 4),
            "label": "FAKE" if is_fake else "REAL",
            "real_pct": ema_real_pct,
            "fake_pct": ema_fake_pct,
            "raw_fake_pct": prediction["fake_pct"],
            "model_version": prediction.get("model_version", active_model),
            "threshold_used": tau,
            "processing_time_ms": proc_time,
            "is_buffering": False,
            "features": telemetry,
        }
        self.client_last_result[client_id] = result
        return result

    def process_audio_file(
        self,
        file_bytes: bytes,
        filename: str,
        model_version: Optional[str] = None,
        aggregation: str = "mean",
        threshold: Optional[float] = None,
    ) -> dict:
        audio_signal = self.decode_audio_file(file_bytes, filename)
        duration_sec = round(len(audio_signal) / self.sample_rate, 2)
        telemetry = self.extract_features(audio_signal)

        t0 = time.perf_counter()
        prediction = predict_audio(
            audio_signal,
            sr=self.sample_rate,
            model_version=model_version,
            aggregation=aggregation,
            threshold=threshold,
        )
        proc_time = round((time.perf_counter() - t0) * 1000, 2)

        is_fake = prediction["is_deepfake"]
        status = prediction.get("status") or (
            "Flagged: Deepfake Detected" if is_fake else "Authentic Human Voice"
        )

        return {
            "filename": filename,
            "duration_seconds": duration_sec,
            "status": status,
            "is_anomaly": is_fake,
            "confidence": prediction["confidence"],
            "label": prediction["label"],
            "real_pct": prediction["real_pct"],
            "fake_pct": prediction["fake_pct"],
            "processing_time_ms": proc_time,
            "model_version": prediction.get("model_version", "webrtc"),
            "threshold_used": prediction.get("threshold_used", 0.620),
            "windows_analyzed": prediction.get("windows_analyzed", 1),
            "total_windows": prediction.get("total_windows", 1),
            "aggregation_method": prediction.get("aggregation_method", "single_window"),
            "mean_fake_pct": prediction.get("mean_fake_pct", prediction["fake_pct"]),
            "median_fake_pct": prediction.get("median_fake_pct", prediction["fake_pct"]),
            "features": telemetry,
        }

    def detect_voice_authenticity(self, audio_signal: np.ndarray) -> dict:
        telemetry = self.extract_features(audio_signal)
        prediction = predict_audio(audio_signal, sr=self.sample_rate)
        is_fake = prediction["is_deepfake"]
        status = "Flagged: Deepfake Detected" if is_fake else "Authentic Human Voice"

        return {
            "status": status,
            "confidence": prediction["confidence"],
            "is_anomaly": is_fake,
            "label": prediction["label"],
            "real_pct": prediction["real_pct"],
            "fake_pct": prediction["fake_pct"],
            "features": telemetry,
        }


audio_processor = AudioProcessor()
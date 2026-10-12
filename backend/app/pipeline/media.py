"""Media decoding and speech segmentation for uploaded recordings (TRD §3.4, §7.4)."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SR = 16_000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".flv"}


class MediaError(ValueError):
    pass


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_audio(path: Path) -> np.ndarray:
    """Any container -> mono float32 at 16 kHz. Empty array if the file has no audio track."""
    cmd = [
        ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603 - fixed exe, no shell
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")
        if "does not contain any stream" in err or "matches no streams" in err:
            return np.zeros(0, dtype=np.float32)
        raise MediaError(f"Could not decode audio: {err.strip()[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def video_info(path: Path) -> tuple[bool, float]:
    """(has a video stream with frames, duration seconds)."""
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    ok, _ = cap.read()
    cap.release()
    if not ok or frames <= 0 or fps <= 0:
        return False, 0.0
    return True, frames / fps


def validate_media(path: Path, max_minutes: int) -> dict:
    """Check the upload is real media of acceptable length (TRD SEC-05)."""
    ext = path.suffix.lower()
    if ext not in AUDIO_EXTS | VIDEO_EXTS:
        raise MediaError(f"Unsupported file type '{ext}'")
    has_video, video_s = video_info(path) if ext in VIDEO_EXTS else (False, 0.0)
    audio = decode_audio(path)
    audio_s = audio.size / SR
    if audio_s < 0.5 and not has_video:
        raise MediaError("File contains no usable audio or video")
    duration = max(audio_s, video_s)
    if duration > max_minutes * 60:
        raise MediaError(f"Recording longer than {max_minutes} minutes")
    return {
        "has_video": has_video,
        "duration_s": duration,
        "audio_s": audio_s,
        "info": json.dumps({"has_video": has_video}),
    }


@dataclass
class Window:
    start_s: float
    end_s: float


def speech_segments(
    audio: np.ndarray,
    min_s: float = 1.0,
    max_s: float = 6.0,
    merge_gap_s: float = 0.35,
    top_db: float = 35.0,
) -> list[Window]:
    """Energy-based speech segmentation into 1-6 s utterance windows."""
    import librosa

    if audio.size < SR * 0.5:
        return []
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=1024, hop_length=256)
    merged: list[list[float]] = []
    for s, e in intervals / SR:
        if merged and s - merged[-1][1] <= merge_gap_s and e - merged[-1][0] <= max_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    windows: list[Window] = []
    for s, e in merged:
        while e - s > max_s:  # split long stretches into even chunks
            n = int(np.ceil((e - s) / max_s))
            step = (e - s) / n
            windows.append(Window(s, s + step))
            s += step
        if e - s >= min_s:
            windows.append(Window(s, e))
        elif windows and s - windows[-1].end_s < merge_gap_s:
            windows[-1].end_s = e  # attach a short tail to the previous window
    return windows


def fixed_windows(duration_s: float, size_s: float = 3.0) -> list[Window]:
    """Fallback for video without speech: consecutive fixed windows."""
    n = max(1, int(duration_s // size_s))
    return [Window(i * size_s, min(duration_s, (i + 1) * size_s)) for i in range(n)]


def words_in(words: list[dict], w: Window) -> list[dict]:
    return [x for x in words if w.start_s <= (x["start"] + x["end"]) / 2 < w.end_s]

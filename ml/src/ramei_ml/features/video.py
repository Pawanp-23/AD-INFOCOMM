"""Face pipeline for the visual branch (TRD §3.4, §4.3, §4.4).

For a video clip: sample N frames uniformly, detect the most confident face with MediaPipe,
crop with a margin, resize to 224x224, and record per-frame quality signals used by the
reliability module (detection confidence, face area ratio, blur, brightness, face presence).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ramei_ml.paths import DATA_DIR

DETECTOR_PATH = DATA_DIR / "models" / "blaze_face_short_range.tflite"
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_detector(path: Path = DETECTOR_PATH) -> Path:
    """Download the MediaPipe BlazeFace model (~230 KB, Apache-2.0) on first use."""
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(DETECTOR_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    return path


@dataclass(frozen=True)
class FaceConfig:
    n_frames: int = 16
    size: int = 224
    margin: float = 0.25  # expand the detected box by this fraction on each side
    min_confidence: float = 0.5
    jpeg_quality: int = 90
    # long side of the frame given to the detector; full-res frames are cropped
    detect_max_side: int = 640


class FaceDetector:
    """Thin wrapper over the MediaPipe Tasks face detector (one instance per process)."""

    def __init__(self, model_path: Path = DETECTOR_PATH, min_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_detector(model_path))),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
        """Best face as ((x, y, w, h), score) in rgb pixel coordinates, or None."""
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._detector.detect(image)
        if not result.detections:
            return None
        best = max(result.detections, key=lambda d: d.categories[0].score)
        b = best.bounding_box
        return (b.origin_x, b.origin_y, b.width, b.height), float(best.categories[0].score)

    def close(self) -> None:
        self._detector.close()


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — low values mean a blurry face."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frame_indices(n_total: int, n: int) -> np.ndarray:
    """n indices spread uniformly over the clip (centers of n equal segments)."""
    if n_total <= 0:
        return np.zeros(0, dtype=int)
    return np.clip(((np.arange(n) + 0.5) * n_total / n).astype(int), 0, n_total - 1)


def _crop(
    frame: np.ndarray, box: tuple[int, int, int, int], margin: float, size: int
) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    side = max(bw, bh) * (1 + 2 * margin)  # square crop centered on the face
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def faces_from_frames(
    frames: list[np.ndarray | None], detector: FaceDetector, cfg: FaceConfig = FaceConfig()
) -> tuple[np.ndarray, dict]:
    """Detect, crop and score faces in RGB frames (None = unreadable frame).

    Returns (crops uint8 [len(frames), size, size, 3] RGB, per-frame quality dict).
    Frames without a face are zeros with face_present = 0.
    """
    n = len(frames)
    crops = np.zeros((n, cfg.size, cfg.size, 3), dtype=np.uint8)
    conf, present, area, blur, bright = (np.zeros(n) for _ in range(5))
    for i, rgb in enumerate(frames):
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        scale = min(1.0, cfg.detect_max_side / max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
        found = detector.detect(small)
        if found is None:
            continue
        (x, y, bw, bh), score = found
        box = (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
        crop = _crop(rgb, box, cfg.margin, cfg.size)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crops[i], conf[i], present[i] = crop, score, 1.0
        area[i] = (box[2] * box[3]) / (w * h)
        blur[i], bright[i] = blur_score(gray), float(gray.mean() / 255.0)
    quality = {
        "face_present": present.tolist(),
        "det_conf": conf.round(4).tolist(),
        "face_area": area.round(5).tolist(),
        "blur": blur.round(2).tolist(),
        "brightness": bright.round(4).tolist(),
    }
    return crops, quality


def read_frames(
    video_path: Path, n: int, start_s: float | None = None, end_s: float | None = None
) -> tuple[list[np.ndarray | None], int, float]:
    """n RGB frames sampled uniformly from [start_s, end_s] (whole clip by default).

    Returns (frames, total frame count, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first = int(start_s * fps) if start_s is not None else 0
    last = min(n_total, int(end_s * fps)) if end_s is not None else n_total
    frames: list[np.ndarray | None] = []
    for idx in first + sample_frame_indices(max(0, last - first), n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    frames += [None] * (n - len(frames))
    return frames, n_total, fps


def extract_faces(
    video_path: Path,
    detector: FaceDetector,
    cfg: FaceConfig = FaceConfig(),
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (crops uint8 [N, size, size, 3] RGB, quality dict). Missing faces are zeros."""
    frames, n_total, fps = read_frames(video_path, cfg.n_frames, start_s, end_s)
    crops, quality = faces_from_frames(frames, detector, cfg)
    return crops, {"n_frames_total": n_total, "duration_s": round(n_total / fps, 3), **quality}


def save_faces(out_dir: Path, crops: np.ndarray, quality: dict, cfg: FaceConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(crops):
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
        )
        if ok:
            (out_dir / f"{i:02d}.jpg").write_bytes(buf.tobytes())
    (out_dir / "quality.json").write_text(json.dumps({**quality, "config": asdict(cfg)}))


def load_faces(face_dir: Path, n_frames: int = 16) -> tuple[np.ndarray, dict]:
    """Inverse of save_faces: (crops uint8 [N, H, W, 3] RGB, quality)."""
    quality = json.loads((face_dir / "quality.json").read_text())
    frames = []
    for i in range(n_frames):
        img = cv2.imread(str(face_dir / f"{i:02d}.jpg"))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames), quality

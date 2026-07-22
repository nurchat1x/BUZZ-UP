"""
Детекция сонливости через MediaPipe Face Landmarker (Tasks API) и EAR.

MediaPipe >= 0.10.30 убрал mp.solutions — используем mediapipe.tasks.vision.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Индексы landmarks для левого и правого глаза (p1..p6)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Скачивает модель при первом запуске, если файла нет."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _landmark_point(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def compute_ear(landmarks, eye_indices: tuple[int, ...], width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = (_landmark_point(landmarks[i], width, height) for i in eye_indices)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


@dataclass
class DrowsinessResult:
    status: str  # "Не Спит" | "Спит" | "Лицо не найдено"
    ear: float
    left_ear: float
    right_ear: float
    face_detected: bool
    closed_frames: int
    confidence: float


class DrowsinessDetector:
    """Детектор сонливости: MediaPipe Face Landmarker + EAR."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 30,
        draw_landmarks: bool = False,
    ):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.draw_landmarks = draw_landmarks
        self._closed_counter = 0
        self._drowsy = False
        self._frame_ts_ms = 0
        self._last_result = DrowsinessResult(
            status="Лицо не найдено",
            ear=0.0,
            left_ear=0.0,
            right_ear=0.0,
            face_detected=False,
            closed_frames=0,
            confidence=0.0,
        )

        model_path = ensure_model()
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    @property
    def last_result(self) -> DrowsinessResult:
        return self._last_result

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, DrowsinessResult]:
        height, width = frame_bgr.shape[:2]
        process_frame = frame_bgr
        scale_back = None

        if width > 640:
            scale = 640 / width
            new_size = (640, int(height * scale))
            process_frame = cv2.resize(frame_bgr, new_size)
            scale_back = (width, height)

        output, detection = self._process_frame_core(process_frame)

        if scale_back is not None:
            output = cv2.resize(output, scale_back)

        return output, detection

    def _process_frame_core(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, DrowsinessResult]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        output = frame_bgr.copy()

        if not result.face_landmarks:
            self._closed_counter = 0
            self._drowsy = False
            detection = DrowsinessResult(
                status="Лицо не найдено",
                ear=0.0,
                left_ear=0.0,
                right_ear=0.0,
                face_detected=False,
                closed_frames=0,
                confidence=0.0,
            )
            self._last_result = detection
            return output, detection

        face_landmarks = result.face_landmarks[0]
        left_ear = compute_ear(face_landmarks, LEFT_EYE, width, height)
        right_ear = compute_ear(face_landmarks, RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold:
            self._closed_counter += 1
        else:
            self._closed_counter = 0

        self._drowsy = self._closed_counter >= self.consecutive_frames
        status = "Спит" if self._drowsy else "Не Спит"

        if self.draw_landmarks:
            for lm in face_landmarks:
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(output, (x, y), 1, (0, 255, 0), -1)

        if self._drowsy:
            confidence = min(1.0, self._closed_counter / max(self.consecutive_frames, 1))
        else:
            margin = max(ear - self.ear_threshold, 0.0)
            confidence = min(1.0, 0.5 + margin * 2.0)

        detection = DrowsinessResult(
            status=status,
            ear=round(ear, 3),
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            face_detected=True,
            closed_frames=self._closed_counter,
            confidence=round(confidence, 2),
        )
        self._last_result = detection
        return output, detection

    def close(self) -> None:
        self._landmarker.close()


def probe_detector() -> tuple[bool, str | None]:
    """Проверка, что MediaPipe native libs загружаются (важно для Streamlit Cloud)."""
    detector = None
    try:
        detector = DrowsinessDetector()
        return True, None
    except OSError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    finally:
        if detector is not None:
            detector.close()

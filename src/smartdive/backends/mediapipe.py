"""MediaPipe Pose backend, remapped onto COCO-17.

MediaPipe is single-person and CPU-friendly. It is kept as a fallback so the
pipeline runs without torch, and as an independent detector for cross-checking
results, if two unrelated detectors agree on the takeoff velocity, the number
is probably real.
"""

from __future__ import annotations

import numpy as np

from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend
from smartdive.keypoints import NUM_KEYPOINTS

# BlazePose 33-landmark indices -> COCO-17 order.
_BLAZE_TO_COCO: tuple[int, ...] = (
    0,  # nose
    2,  # left eye
    5,  # right eye
    7,  # left ear
    8,  # right ear
    11,
    12,  # shoulders
    13,
    14,  # elbows
    15,
    16,  # wrists
    23,
    24,  # hips
    25,
    26,  # knees
    27,
    28,  # ankles
)


class MediaPipePoseBackend(PoseBackend):
    """Pose estimation via MediaPipe BlazePose."""

    name = "mediapipe"

    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "the 'mediapipe' backend needs mediapipe: pip install 'smartdive[mediapipe]'"
            ) from exc

        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return FrameDetections(index=index)

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark

        xy = np.zeros((NUM_KEYPOINTS, 2), dtype=float)
        conf = np.zeros(NUM_KEYPOINTS, dtype=float)
        for coco_i, blaze_i in enumerate(_BLAZE_TO_COCO):
            lm = landmarks[blaze_i]
            xy[coco_i] = (lm.x * w, lm.y * h)
            conf[coco_i] = lm.visibility

        visible = xy[conf >= 0.3]
        if len(visible) >= 2:
            x1, y1 = visible.min(axis=0)
            x2, y2 = visible.max(axis=0)
        else:  # pragma: no cover - degenerate frame
            x1 = y1 = 0.0
            x2, y2 = float(w), float(h)

        person = PersonDetection(
            box=np.array([x1, y1, x2, y2], dtype=float),
            score=float(conf.mean()),
            xy=xy,
            conf=conf,
        )
        return FrameDetections(index=index, people=[person])

    def close(self) -> None:
        self._pose.close()

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "model_complexity": self.model_complexity,
        }

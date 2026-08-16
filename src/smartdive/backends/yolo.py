"""Ultralytics YOLO pose backend."""

from __future__ import annotations

import numpy as np

from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend
from smartdive.keypoints import NUM_KEYPOINTS

#: Default checkpoint. The original project used ``yolov8n-pose`` (the smallest
#: model there is); ``yolo11m-pose`` costs a few ms more per frame and is
#: dramatically more reliable on motion-blurred, mid-flight bodies.
DEFAULT_WEIGHTS = "yolo11m-pose.pt"


class YoloPoseBackend(PoseBackend):
    """Pose estimation via an Ultralytics YOLO ``*-pose`` checkpoint."""

    name = "yolo"

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        imgsz: int = 1280,
        conf: float = 0.15,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "the 'yolo' backend needs ultralytics: pip install 'smartdive[yolo]'"
            ) from exc

        self.weights = weights
        # Inference resolution matters more than model size here: a diver is
        # ~400 px wide in a 1920 px frame, so the usual 640 px letterbox throws
        # away two thirds of the detail the keypoint head could have used.
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self._model = YOLO(weights)

    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        kwargs: dict[str, object] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self._model(frame, **kwargs)[0]
        people: list[PersonDetection] = []

        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                conf = result.keypoints.conf.cpu().numpy()
            else:
                conf = np.ones(xy.shape[:2], dtype=float)

            for box, score, kp, kc in zip(boxes, scores, xy, conf, strict=True):
                if kp.shape[0] != NUM_KEYPOINTS:
                    continue
                people.append(
                    PersonDetection(
                        box=box.astype(float),
                        score=float(score),
                        xy=kp.astype(float),
                        conf=kc.astype(float),
                    )
                )

        return FrameDetections(index=index, people=people)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "weights": self.weights,
            "imgsz": self.imgsz,
            "conf": self.conf,
        }

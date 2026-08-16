"""Pose backend registry."""

from __future__ import annotations

from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend

BACKENDS = ("yolo", "mediapipe")


def get_backend(name: str, **kwargs: object) -> PoseBackend:
    """Instantiate a pose backend by name.

    Backends are imported lazily so that installing only one of the optional
    dependency groups is enough to run the pipeline.
    """
    key = name.lower()
    if key == "yolo":
        from smartdive.backends.yolo import YoloPoseBackend

        return YoloPoseBackend(**kwargs)  # type: ignore[arg-type]
    if key == "mediapipe":
        from smartdive.backends.mediapipe import MediaPipePoseBackend

        return MediaPipePoseBackend(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")


__all__ = [
    "BACKENDS",
    "FrameDetections",
    "PersonDetection",
    "PoseBackend",
    "get_backend",
]

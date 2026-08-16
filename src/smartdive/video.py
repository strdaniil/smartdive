"""Streaming video I/O.

The original pipeline wrote every frame of the input to a JPEG on disk and then
re-read them one at a time for inference. That is slow, needs hundreds of
megabytes of scratch space, and quietly re-encodes the pixels the detector is
about to look at. Everything here streams instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata for a clip."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": round(self.duration, 3),
        }


def probe(path: str | Path) -> VideoInfo:
    """Read container metadata without decoding the whole file."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, bgr_frame)`` for every frame in the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


class VideoWriter:
    """Context-managed :class:`cv2.VideoWriter` with sane defaults."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], fourcc: str = "mp4v"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not self._writer.isOpened():
            raise OSError(f"cannot open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

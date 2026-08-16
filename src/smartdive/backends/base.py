"""Pose backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersonDetection:
    """One detected person in one frame, in pixel coordinates."""

    box: np.ndarray  # (4,) x1, y1, x2, y2
    score: float  # detector confidence for the person
    xy: np.ndarray  # (17, 2) COCO-17 keypoints
    conf: np.ndarray  # (17,) per-keypoint confidence

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def box_scale(self) -> float:
        x1, y1, x2, y2 = self.box
        return float(max(x2 - x1, y2 - y1))


@dataclass
class FrameDetections:
    """All people detected in a single frame."""

    index: int
    people: list[PersonDetection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.people)


class PoseBackend(ABC):
    """A pose estimator normalised to the COCO-17 schema.

    Backends are *stateless with respect to identity*: they report every person
    they can see and leave association across frames to
    :mod:`smartdive.tracking`. That split is what lets the same tracker and the
    same physics run on top of any detector.
    """

    #: Human-readable identifier used in reports and CLI flags.
    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        """Detect all people in a single BGR frame."""

    def close(self) -> None:  # noqa: B027 - an optional hook, not a requirement
        """Release any held resources. Backends that hold none need not override."""

    def __enter__(self) -> PoseBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def describe(self) -> dict[str, object]:
        """Metadata recorded in the analysis report for reproducibility."""
        return {"backend": self.name}

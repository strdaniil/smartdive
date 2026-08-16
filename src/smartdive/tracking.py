"""Multi-person tracking and automatic subject selection.

Two separate problems live here, and conflating them is what made the original
pipeline pick the wrong person:

1. **Association.** Link detections across frames into tracks. A constant
   velocity model plus Hungarian assignment handles the fast-moving subject and
   survives short detector dropouts (a diver at full extension over the water is
   routinely missed for several frames).
2. **Subject selection.** Decide which track is the athlete. The old code took
   ``argmax(confidence)`` *independently in every frame*, which reliably locks
   onto a well-lit bystander standing on the pool deck the moment the diver
   becomes a motion-blurred horizontal streak. Here the subject is instead the
   track that actually *moves like a diver*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from smartdive.backends.base import FrameDetections, PersonDetection


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two ``x1, y1, x2, y2`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    """A single person followed across frames."""

    track_id: int
    observations: dict[int, PersonDetection] = field(default_factory=dict)
    box: np.ndarray = field(default_factory=lambda: np.zeros(4))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_frame: int = -1
    misses: int = 0

    @classmethod
    def start(cls, track_id: int, frame_index: int, det: PersonDetection) -> Track:
        track = cls(track_id=track_id)
        track.box = det.box.copy()
        track.last_frame = frame_index
        track.observations[frame_index] = det
        return track

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def scale(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1], 1.0))

    @property
    def frames(self) -> np.ndarray:
        return np.array(sorted(self.observations))

    @property
    def span(self) -> int:
        f = self.frames
        return int(f[-1] - f[0] + 1) if len(f) else 0

    def predict_box(self, dt: int) -> np.ndarray:
        """Box position extrapolated ``dt`` frames forward at constant velocity."""
        box = self.box.copy()
        box[[0, 2]] += self.velocity[0] * dt
        box[[1, 3]] += self.velocity[1] * dt
        return box

    def update(self, frame_index: int, det: PersonDetection, alpha: float = 0.5) -> None:
        dt = max(frame_index - self.last_frame, 1)
        measured = (det.center - self.center) / dt
        # Exponential smoothing keeps the predictor stable through noisy frames
        # without lagging behind a genuine acceleration.
        self.velocity = (1 - alpha) * self.velocity + alpha * measured
        self.box = det.box.copy()
        self.last_frame = frame_index
        self.misses = 0
        self.observations[frame_index] = det


@dataclass
class TrackerConfig:
    """Association hyper-parameters."""

    #: How many consecutive frames a track may go unmatched before it is closed.
    max_age: int = 8
    #: Maximum matching cost that still counts as the same person. Costs are in
    #: units of body scale, so this is resolution independent.
    gate: float = 0.9
    #: Relative weight of the box-size-consistency term.
    scale_weight: float = 0.5
    #: Relative weight of the IoU-overlap reward.
    iou_weight: float = 0.5
    #: Tracks shorter than this are discarded as detector flicker.
    min_track_length: int = 4
    #: Mean detector confidence a track needs before it may be chosen as the
    #: athlete. Without this the ranking can be won by a low-confidence phantom
    #: that wanders across the frame and so scores well on displacement; a track
    #: the detector was never sure about is not a person who moved.
    min_subject_score: float = 0.4


class MultiPersonTracker:
    """Constant-velocity tracker with globally optimal per-frame assignment."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 1
        self._active: list[Track] = []
        self._closed: list[Track] = []

    def _cost(self, track: Track, det: PersonDetection) -> float:
        predicted = track.predict_box(max(track.misses, 1))
        pcx = np.array(
            [
                (predicted[0] + predicted[2]) / 2.0,
                (predicted[1] + predicted[3]) / 2.0,
            ]
        )
        scale = max(track.scale, det.box_scale, 1.0)
        position = float(np.linalg.norm(det.center - pcx) / scale)
        size = abs(float(np.log(max(det.box_scale, 1.0) / track.scale)))
        return (
            position
            + self.config.scale_weight * size
            - self.config.iou_weight * iou(predicted, det.box)
        )

    def update(self, detections: FrameDetections) -> None:
        """Consume one frame of detections."""
        cfg = self.config
        for track in self._active:
            track.misses += 1

        matched: set[int] = set()
        if self._active and detections.people:
            cost = np.array([[self._cost(t, d) for d in detections.people] for t in self._active])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] < cfg.gate:
                    self._active[r].update(detections.index, detections.people[c])
                    matched.add(c)

        for i, det in enumerate(detections.people):
            if i not in matched:
                self._active.append(Track.start(self._next_id, detections.index, det))
                self._next_id += 1

        still_active: list[Track] = []
        for track in self._active:
            (still_active if track.misses <= cfg.max_age else self._closed).append(track)
        self._active = still_active

    def finish(self) -> list[Track]:
        """Close the tracker and return every track worth keeping."""
        tracks = self._closed + self._active
        self._closed, self._active = [], []
        return [t for t in tracks if len(t.observations) >= self.config.min_track_length]


def track_all(
    detections: list[FrameDetections], config: TrackerConfig | None = None
) -> list[Track]:
    """Run the tracker over a whole sequence."""
    tracker = MultiPersonTracker(config)
    for frame in detections:
        tracker.update(frame)
    return tracker.finish()


@dataclass
class SubjectScore:
    """Why a particular track was (or was not) chosen as the athlete."""

    track_id: int
    n_observations: int
    span: int
    displacement_px: float
    displacement_bodies: float
    mean_score: float
    total: float

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "n_observations": self.n_observations,
            "span": self.span,
            "displacement_px": round(self.displacement_px, 1),
            "displacement_bodies": round(self.displacement_bodies, 2),
            "mean_detector_score": round(self.mean_score, 3),
            "score": round(self.total, 3),
        }


def score_tracks(tracks: list[Track]) -> list[SubjectScore]:
    """Rank tracks by how much they look like a person performing a dive.

    A diver is the person who *travels*: over the course of a start they cover
    several body lengths, while spectators and teammates on the blocks stay put.
    Displacement is measured in body lengths so the ranking does not depend on
    how far the camera is from the pool.
    """
    scores: list[SubjectScore] = []
    for track in tracks:
        centers = np.array([d.center for d in track.observations.values()])
        scales = np.array([d.box_scale for d in track.observations.values()])
        detector = float(np.mean([d.score for d in track.observations.values()]))

        displacement = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
        body = float(np.median(scales)) or 1.0
        bodies = displacement / body

        # Motion dominates; detector confidence and track length break ties.
        total = bodies + 0.35 * float(np.log1p(len(track.observations))) + 0.5 * detector
        scores.append(
            SubjectScore(
                track_id=track.track_id,
                n_observations=len(track.observations),
                span=track.span,
                displacement_px=displacement,
                displacement_bodies=bodies,
                mean_score=detector,
                total=total,
            )
        )
    scores.sort(key=lambda s: -s.total)
    return scores


def select_subject(
    tracks: list[Track], config: TrackerConfig | None = None
) -> tuple[Track, list[SubjectScore]]:
    """Pick the athlete's track, returning the ranking that justified it."""
    if not tracks:
        raise ValueError("no tracks: the detector found nobody in this video")
    cfg = config or TrackerConfig()

    eligible = [t for t in tracks if _mean_score(t) >= cfg.min_subject_score]
    scores = score_tracks(eligible or tracks)
    best_id = scores[0].track_id
    subject = next(t for t in tracks if t.track_id == best_id)

    if eligible and len(eligible) < len(tracks):
        # Keep the rejected tracks visible in the report so a wrong pick is
        # diagnosable without re-running anything.
        scores = scores + [s for s in score_tracks([t for t in tracks if t not in eligible])]
    return subject, scores


def _mean_score(track: Track) -> float:
    return float(np.mean([d.score for d in track.observations.values()]))

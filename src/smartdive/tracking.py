"""Multi-person tracking, fragment stitching, and automatic subject selection.

Three separate problems live here, and conflating any two of them is what makes
a pipeline measure the wrong person:

1. **Association.** Link detections across frames into tracks. The original
   ``smartdive`` did not attempt this at all; it took ``argmax(confidence)``
   independently in every frame, which locks onto a well-lit bystander the
   moment the diver becomes a motion-blurred horizontal streak.

2. **Stitching.** Even a good frame-to-frame tracker breaks at takeoff. In the
   space of two frames the athlete's bounding box rotates from vertical to
   horizontal, doubles in width, halves in height, and moves half its own width;
   the detector loses them for a few frames over the water. A tracker tolerant
   enough to survive that is also tolerant enough to swap two swimmers standing
   side by side on adjacent blocks. So the tracker stays strict and a separate
   pass reconnects fragments afterwards, where it has the whole clip to reason
   over instead of one frame of context.

3. **Subject selection.** Decide which track is the athlete. Ranking by total
   displacement, the previous version of this module, is fragile in exactly
   the case that matters: a diver whose track is broken into three pieces shows
   less displacement per piece than a swimmer walking the length of the deck.
   Peak speed *in body lengths per second* does not have that problem. Divers
   reach three to eight body lengths per second at takeoff; people walking reach
   one; people standing reach none. It is a property of an instant rather than
   of a whole track, so it survives fragmentation, and being expressed in body
   lengths it is invariant to how far away the camera is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from smartdive.backends.base import FrameDetections, PersonDetection
from smartdive.keypoints import LEFT_HIP, RIGHT_HIP, body_scale


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


def anchor(det: PersonDetection, min_conf: float = 0.3) -> np.ndarray:
    """A stable per-person reference point: the mid-hip, or the box centre.

    The mid-hip is used in preference to the box centre because the box centre
    moves when a limb extends. A diver's arms go from tucked to fully extended
    during takeoff, which shifts the box centre by a quarter of a body length
    without the athlete's mass having gone anywhere.
    """
    if det.conf[LEFT_HIP] >= min_conf and det.conf[RIGHT_HIP] >= min_conf:
        return (det.xy[LEFT_HIP] + det.xy[RIGHT_HIP]) / 2.0
    return det.center


def keypoint_distance(
    a_xy: np.ndarray,
    a_conf: np.ndarray,
    b_xy: np.ndarray,
    b_conf: np.ndarray,
    min_conf: float = 0.3,
) -> float:
    """Median displacement of the keypoints two detections share, in pixels.

    Returns NaN when they share too few confident keypoints to compare. This is
    a far better association cue than box overlap for a body that is changing
    shape: the ankles of a diver mid-somersault are still that diver's ankles,
    however little their bounding boxes overlap.
    """
    both = (np.asarray(a_conf) >= min_conf) & (np.asarray(b_conf) >= min_conf)
    if both.sum() < 4:
        return float("nan")
    return float(np.median(np.linalg.norm(np.asarray(a_xy)[both] - np.asarray(b_xy)[both], axis=1)))


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
    def anchor(self) -> np.ndarray:
        """Mid-hip of the most recent observation."""
        return anchor(self.observations[self.last_frame])

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

    @property
    def last(self) -> PersonDetection:
        return self.observations[self.last_frame]

    def body_scale_px(self, min_conf: float = 0.3) -> float:
        """Median widest-keypoint-separation over the track, in pixels."""
        values = [body_scale(d.xy, d.conf, min_conf) for d in self.observations.values()]
        values = [v for v in values if np.isfinite(v)]
        return float(np.median(values)) if values else float(np.median([self.scale]))

    def mean_score(self) -> float:
        return float(np.mean([d.score for d in self.observations.values()]))

    def predict(self, dt: int) -> np.ndarray:
        """Anchor position extrapolated ``dt`` frames forward at constant velocity."""
        return self.anchor + self.velocity * dt

    def predict_box(self, dt: int) -> np.ndarray:
        box = self.box.copy()
        box[[0, 2]] += self.velocity[0] * dt
        box[[1, 3]] += self.velocity[1] * dt
        return box

    def update(self, frame_index: int, det: PersonDetection, alpha: float = 0.6) -> None:
        dt = max(frame_index - self.last_frame, 1)
        measured = (anchor(det) - self.anchor) / dt
        # Exponential smoothing keeps the predictor stable through noisy frames
        # without lagging behind a genuine acceleration. Takeoff *is* a genuine
        # acceleration, so alpha is biased towards the new measurement.
        self.velocity = (1 - alpha) * self.velocity + alpha * measured
        self.box = det.box.copy()
        self.last_frame = frame_index
        self.misses = 0
        self.observations[frame_index] = det


@dataclass
class TrackerConfig:
    """Association, stitching and selection hyper-parameters.

    Every threshold here is in body lengths or in body lengths per second, never
    in raw pixels, so none of it needs retuning for a different resolution or a
    different camera distance.
    """

    #: How many consecutive frames a track may go unmatched before it is closed.
    max_age: int = 8
    #: Base matching gate, in body lengths. Costs are normalised by body scale.
    gate: float = 0.55
    #: How much of the track's own speed is added to the gate. A person moving
    #: half a body length per frame cannot be held to a fixed positional
    #: tolerance, because the constant-velocity prediction is only as good as
    #: the assumption that they did not accelerate, and at takeoff they did.
    gate_speed_factor: float = 0.6
    #: Weight of the keypoint-displacement term relative to anchor distance.
    keypoint_weight: float = 1.0
    #: Relative weight of the box-size-consistency term. Kept small: a diver's
    #: box legitimately changes shape by a factor of two through takeoff.
    scale_weight: float = 0.15
    #: Relative weight of the IoU-overlap reward.
    iou_weight: float = 0.35
    #: Tracks shorter than this are discarded as detector flicker.
    min_track_length: int = 3
    #: Mean detector confidence a track needs before it may be chosen as the
    #: athlete. Without this the ranking can be won by a low-confidence phantom.
    min_subject_score: float = 0.35
    #: A track whose median body scale is below this fraction of the clip's
    #: median is a fragment of a person, not a person, and cannot be the
    #: subject. Body-length-relative speeds explode for tiny scales.
    min_relative_body_scale: float = 0.45

    # --- stitching ---------------------------------------------------------
    #: Longest gap, in frames, that may be bridged when reconnecting fragments.
    stitch_max_gap: int = 12
    #: How far the extrapolated end of one fragment may miss the start of the
    #: next, in body lengths, before they are considered different people.
    stitch_tolerance_bodies: float = 0.9
    #: Fragments may overlap in time by this many frames and still be merged.
    #: A strict tracker routinely opens the new fragment a few frames before it
    #: gives up on the old one.
    stitch_max_overlap: int = 10
    #: Two fragments overlapping in time are the same person only if they are
    #: this close, in body lengths, on the frames they share.
    stitch_overlap_tolerance_bodies: float = 0.55


class MultiPersonTracker:
    """Constant-velocity tracker with globally optimal per-frame assignment."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 1
        self._active: list[Track] = []
        self._closed: list[Track] = []

    def _cost(self, track: Track, det: PersonDetection) -> float:
        """Match cost in body lengths. Lower is better; see :meth:`update`."""
        dt = max(track.misses, 1)
        scale = max(track.scale, det.box_scale, 1.0)

        position = float(np.linalg.norm(anchor(det) - track.predict(dt)) / scale)

        # Keypoint agreement, extrapolated the same way the anchor was. This is
        # what keeps identity through a shape change that destroys box overlap.
        shifted = track.last.xy + track.velocity * dt
        kp = keypoint_distance(shifted, track.last.conf, det.xy, det.conf)
        keypoint = position if not np.isfinite(kp) else kp / scale

        size = abs(float(np.log(max(det.box_scale, 1.0) / track.scale)))
        overlap = iou(track.predict_box(dt), det.box)

        cfg = self.config
        blended = (position + cfg.keypoint_weight * keypoint) / (1 + cfg.keypoint_weight)
        return blended + cfg.scale_weight * size - cfg.iou_weight * overlap

    def _gate(self, track: Track) -> float:
        """Matching gate, widened in proportion to how fast the track is moving."""
        speed = float(np.linalg.norm(track.velocity)) / max(track.scale, 1.0)
        return self.config.gate + self.config.gate_speed_factor * speed * max(track.misses, 1)

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
                if cost[r, c] < self._gate(self._active[r]):
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
    detections: list[FrameDetections],
    config: TrackerConfig | None = None,
    stitch: bool = True,
) -> list[Track]:
    """Run the tracker over a whole sequence, then reconnect its fragments."""
    cfg = config or TrackerConfig()
    tracker = MultiPersonTracker(cfg)
    for frame in detections:
        tracker.update(frame)
    tracks = tracker.finish()
    return stitch_tracks(tracks, cfg) if stitch else tracks


# --------------------------------------------------------------------------
# Fragment stitching
# --------------------------------------------------------------------------


def _endpoint_velocity(track: Track, at_end: bool, window: int = 4) -> np.ndarray:
    """Anchor velocity in px/frame at one end of a track, by least squares."""
    frames = track.frames
    if len(frames) < 2:
        return np.zeros(2)
    chosen = frames[-window:] if at_end else frames[:window]
    points = np.array([anchor(track.observations[int(f)]) for f in chosen])
    t = chosen.astype(float)
    if len(chosen) < 2 or np.ptp(t) == 0:
        return np.zeros(2)
    return np.array([np.polyfit(t, points[:, 0], 1)[0], np.polyfit(t, points[:, 1], 1)[0]])


def _stitch_cost(a: Track, b: Track, cfg: TrackerConfig) -> float:
    """How badly ``b`` fails to continue ``a``, in body lengths; inf if it cannot.

    ``a`` is the earlier fragment. Two regimes are handled: a clean gap, where
    ``a``'s extrapolation has to land on ``b``'s first observation, and a
    temporal overlap, where the two fragments have to agree on the frames they
    both saw. The overlap case exists because a strict tracker opens a new
    fragment while still holding the old one open for ``max_age`` frames.
    """
    a_last, b_first = int(a.frames[-1]), int(b.frames[0])
    scale = max(a.body_scale_px(), b.body_scale_px(), 1.0)

    shared = sorted(set(a.observations) & set(b.observations))
    if shared:
        if len(shared) > cfg.stitch_max_overlap:
            return float("inf")
        offsets = [
            np.linalg.norm(anchor(a.observations[f]) - anchor(b.observations[f])) for f in shared
        ]
        distance = float(np.median(offsets)) / scale
        return distance if distance <= cfg.stitch_overlap_tolerance_bodies else float("inf")

    gap = b_first - a_last
    if not 0 < gap <= cfg.stitch_max_gap:
        return float("inf")

    # Meet in the middle: extrapolate ``a`` forwards and ``b`` backwards, and
    # ask how far apart they are. Using both ends halves the extrapolation
    # distance, which matters because the athlete is accelerating.
    forward = anchor(a.observations[a_last]) + _endpoint_velocity(a, True) * gap
    backward = anchor(b.observations[b_first]) - _endpoint_velocity(b, False) * gap
    distance = float(np.linalg.norm(forward - backward)) / (2 * scale)

    # The tolerance grows with the gap: a constant-velocity extrapolation over
    # eight frames of a body that is accelerating cannot be held to the same
    # standard as one over two.
    budget = cfg.stitch_tolerance_bodies * (0.4 + 0.6 * gap / max(cfg.stitch_max_gap, 1))
    return distance if distance <= budget else float("inf")


def stitch_tracks(tracks: list[Track], config: TrackerConfig | None = None) -> list[Track]:
    """Greedily merge track fragments that describe the same person.

    Repeatedly joins the cheapest compatible pair until nothing is left to join.
    Merging is greedy rather than globally optimal on purpose: the cost is a
    hard-gated physical plausibility check, not a soft preference, so almost
    every pair it will accept is unambiguous, and a wrong merge is much more
    damaging than a missed one.
    """
    cfg = config or TrackerConfig()
    pool = {t.track_id: t for t in tracks}

    while True:
        best: tuple[float, int, int] | None = None
        ids = sorted(pool)
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                a, b = pool[i], pool[j]
                if a.frames[0] > b.frames[0]:
                    continue
                cost = _stitch_cost(a, b, cfg)
                if np.isfinite(cost) and (best is None or cost < best[0]):
                    best = (cost, i, j)
        if best is None:
            break

        _, i, j = best
        a, b = pool.pop(i), pool.pop(j)
        merged = Track(track_id=min(a.track_id, b.track_id))
        # On a shared frame the later fragment wins: it was opened because the
        # earlier one had started to disagree with the detector.
        merged.observations = {**a.observations, **b.observations}
        merged.last_frame = int(max(merged.observations))
        merged.box = merged.observations[merged.last_frame].box.copy()
        merged.velocity = _endpoint_velocity(merged, True)
        pool[merged.track_id] = merged

    return sorted(pool.values(), key=lambda t: t.track_id)


# --------------------------------------------------------------------------
# Subject selection
# --------------------------------------------------------------------------


@dataclass
class SubjectScore:
    """Why a particular track was (or was not) chosen as the athlete."""

    track_id: int
    n_observations: int
    span: int
    displacement_px: float
    displacement_bodies: float
    peak_speed_bodies_s: float
    body_scale_px: float
    mean_score: float
    total: float
    rejected: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "n_observations": self.n_observations,
            "span": self.span,
            "displacement_px": round(self.displacement_px, 1),
            "displacement_bodies": round(self.displacement_bodies, 2),
            "peak_speed_bodies_s": round(self.peak_speed_bodies_s, 2),
            "body_scale_px": round(self.body_scale_px, 1),
            "mean_detector_score": round(self.mean_score, 3),
            "score": round(self.total, 3),
            "rejected": self.rejected,
        }


def _speed_profile(track: Track, fps: float) -> tuple[float, float]:
    """``(peak speed, total displacement)`` for a track, in body lengths (per s).

    Peak speed is the 90th percentile of the per-frame anchor speed rather than
    the maximum, so that one badly localised frame cannot manufacture a diver
    out of a bystander.
    """
    frames = track.frames
    scale = track.body_scale_px()
    if len(frames) < 2 or not np.isfinite(scale) or scale <= 0:
        return 0.0, 0.0

    points = np.array([anchor(track.observations[int(f)]) for f in frames])
    displacement = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / scale

    steps = np.diff(frames)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    usable = steps <= 2
    if usable.sum() < 3:
        return 0.0, displacement
    speeds = deltas[usable] / steps[usable] * fps / scale
    return float(np.percentile(speeds, 90)), displacement


def score_tracks(tracks: list[Track], fps: float = 30.0) -> list[SubjectScore]:
    """Rank tracks by how much they look like a person performing a dive.

    The dominant term is peak speed in body lengths per second. A racing start
    puts the athlete's hips through three to eight body lengths per second at
    takeoff; walking the deck is about one; standing on a block is zero. Total
    displacement enters as a secondary term, because a single fast frame from a
    mis-associated detection should not outrank a body that actually travelled,
    and detector confidence and track length break the remaining ties.
    """
    scores: list[SubjectScore] = []
    for track in tracks:
        peak, displacement = _speed_profile(track, fps)
        scores.append(
            SubjectScore(
                track_id=track.track_id,
                n_observations=len(track.observations),
                span=track.span,
                displacement_px=displacement * track.body_scale_px(),
                displacement_bodies=displacement,
                peak_speed_bodies_s=peak,
                body_scale_px=track.body_scale_px(),
                mean_score=track.mean_score(),
                total=(
                    peak
                    + 0.8 * min(displacement, 6.0)
                    + 0.35 * float(np.log1p(len(track.observations)))
                    + 0.5 * track.mean_score()
                ),
            )
        )
    scores.sort(key=lambda s: -s.total)
    return scores


def select_subject(
    tracks: list[Track],
    config: TrackerConfig | None = None,
    fps: float = 30.0,
) -> tuple[Track, list[SubjectScore]]:
    """Pick the athlete's track, returning the ranking that justified it.

    Two eligibility filters run before the ranking, both of which exist to stop
    a *fragment* of a person from winning. A track the detector was never
    confident about is not a person who moved, and a track whose body scale is a
    fraction of everybody else's in the clip is a mis-detected limb, which,
    because speed is expressed in body lengths, would otherwise score
    spectacularly well.
    """
    if not tracks:
        raise ValueError("no tracks: the detector found nobody in this video")
    cfg = config or TrackerConfig()

    body_scales = np.array([t.body_scale_px() for t in tracks])
    typical = float(np.median(body_scales[np.isfinite(body_scales)])) if len(tracks) else 0.0

    eligible: list[Track] = []
    rejected: dict[int, str] = {}
    for track in tracks:
        if track.mean_score() < cfg.min_subject_score:
            rejected[track.track_id] = "detector confidence below threshold"
        elif typical > 0 and track.body_scale_px() < cfg.min_relative_body_scale * typical:
            rejected[track.track_id] = "body scale too small to be a whole person"
        else:
            eligible.append(track)

    ranked = score_tracks(eligible or tracks, fps)
    best_id = ranked[0].track_id
    subject = next(t for t in tracks if t.track_id == best_id)

    if rejected:
        # Keep the rejected tracks visible in the report so a wrong pick is
        # diagnosable without re-running anything.
        excluded = score_tracks([t for t in tracks if t.track_id in rejected], fps)
        for score in excluded:
            score.rejected = rejected[score.track_id]
        ranked = ranked + excluded
    return subject, ranked

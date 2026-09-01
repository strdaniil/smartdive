"""Second-pass pose detection on a crop that follows the athlete.

A whole-frame pose pass has to spend its input resolution on the whole frame. At
1280 px of network input on a 1920 px clip, an athlete 400 px tall is 270 px to
the network, and their wrist is a handful of pixels. That is survivable while
they are standing still on the block and it is not survivable at the moment they
are a motion-blurred horizontal streak over the water, which is precisely the
part of the clip the entire measurement depends on.

The symptom is not a wrong answer, it is a *missing* one. On the sample footage
the detector simply stops reporting the diver partway through the flight: on
``ginz_dive`` it loses them at frame 40 of a flight that ends at frame 50, so
half the flight is unmeasured and the reported flight time is half of what it
should be. No amount of downstream cleverness recovers a measurement that was
never made.

So once the athlete has been identified, this module goes back to the video and
looks again, at a crop around where the tracker says they are, resized so that
they fill the network's input. Same weights, same frame, several times the
effective resolution, and none of it spent on the far end of the pool. Two
further things fall out of it:

* **The crop is unambiguous.** The full-frame pass has to be told which of eight
  people is the subject; the crop mostly contains one person, and the one it
  contains is the right one.
* **It can look where the first pass did not.** The window is extended past both
  ends of the track, so frames the first pass dropped entirely, which is how
  a track ends in the first place, get a second chance from a predicted
  position rather than from a detection that does not exist.

The predicted position comes from the track itself: interpolated between
observations where there are any on both sides, extrapolated at constant
velocity beyond the last one. Extrapolation is bounded, because a prediction
several frames past the last sighting of a fast-moving body is not worth
cropping around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend
from smartdive.camera import CameraMotion
from smartdive.keypoints import NUM_KEYPOINTS
from smartdive.tracking import Track, iou


@dataclass
class RefineConfig:
    """How the follow-crop is built and how its detections are accepted."""

    #: Network input size for the crop pass. The crop is a few hundred pixels
    #: across, so this is an upscale: the athlete ends up filling an input the
    #: whole-frame pass could only give them a fifth of.
    imgsz: int = 960
    #: Crop padding as a fraction of the predicted box's larger side. Generous,
    #: because a bounding box drawn around a tucked body clips the limbs that
    #: same body is about to extend.
    padding: float = 0.55
    #: Smallest crop side in pixels. Stops a distant athlete from being cropped
    #: so tightly that the detector has no context to work with.
    min_crop_px: int = 224
    #: Frames to look beyond each end of the track. Water entry and the last
    #: frames on the block are both routinely outside it.
    extend_frames: int = 8
    #: Furthest a constant-velocity prediction is trusted past the last
    #: observation, in frames.
    max_extrapolate: int = 6
    #: Minimum overlap between a crop detection and the predicted box for it to
    #: be accepted as the athlete.
    min_iou: float = 0.35
    #: How far the accepted detection's centre may sit from the predicted one,
    #: in body lengths. Overlap alone is not enough on a crowded pool deck: two
    #: swimmers standing on adjacent blocks produce boxes that overlap a
    #: prediction comfortably while being different people, and the frames
    #: where this matters most are exactly the frames the first pass lost,
    #: where the prediction is an interpolation and there is no original
    #: detection to disagree with.
    max_centre_shift_bodies: float = 0.45
    #: A refined detection replaces the original only if the detector was at
    #: least this confident; otherwise the original is kept.
    min_score: float = 0.2
    #: Keypoints whose confidence is below this are not carried over, so a
    #: refined frame never *loses* a keypoint the first pass had.
    min_keypoint_conf: float = 0.25


@dataclass
class RefineReport:
    """What the second pass changed."""

    frames_attempted: int = 0
    frames_recovered: int = 0  # had no detection before
    frames_improved: int = 0  # had one, and the crop found more keypoints
    frames_unchanged: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_attempted": self.frames_attempted,
            "frames_recovered": self.frames_recovered,
            "frames_improved": self.frames_improved,
            "frames_unchanged": self.frames_unchanged,
        }


def _predicted_boxes(track: Track, lo: int, hi: int, cfg: RefineConfig) -> dict[int, np.ndarray]:
    """Where the athlete is expected to be, for every frame in ``[lo, hi]``."""
    frames = track.frames
    if not len(frames):
        return {}
    boxes = np.array([track.observations[int(f)].box for f in frames], dtype=float)

    out: dict[int, np.ndarray] = {}
    for f in range(lo, hi + 1):
        if f in track.observations:
            out[f] = track.observations[f].box.astype(float)
            continue
        if frames[0] <= f <= frames[-1]:
            out[f] = np.array([np.interp(f, frames, boxes[:, k]) for k in range(4)], dtype=float)
            continue
        # Outside the track: extrapolate at the velocity of the nearest end.
        at_end = f > frames[-1]
        gap = (f - frames[-1]) if at_end else (frames[0] - f)
        if gap > cfg.max_extrapolate or len(frames) < 2:
            continue
        edge = slice(-4, None) if at_end else slice(0, 4)
        t = frames[edge].astype(float)
        if len(t) < 2 or np.ptp(t) == 0:
            continue
        velocity = np.array([np.polyfit(t, boxes[edge, k], 1)[0] for k in range(4)])
        anchor_box = boxes[-1] if at_end else boxes[0]
        out[f] = anchor_box + velocity * (gap if at_end else -gap)
    return out


def _crop_window(box: np.ndarray, width: int, height: int, cfg: RefineConfig) -> tuple[int, ...]:
    """Square-ish crop around ``box``, padded and clipped to the frame."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1, cfg.min_crop_px)
    pad = side * cfg.padding
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0 + pad
    return (
        int(max(0, np.floor(cx - half))),
        int(max(0, np.floor(cy - half))),
        int(min(width, np.ceil(cx + half))),
        int(min(height, np.ceil(cy + half))),
    )


def _better(
    candidate: PersonDetection, existing: PersonDetection | None, cfg: RefineConfig
) -> bool:
    """Is the crop detection worth preferring to what the first pass had?"""
    if existing is None:
        return True
    new_kp = int((candidate.conf >= cfg.min_keypoint_conf).sum())
    old_kp = int((existing.conf >= cfg.min_keypoint_conf).sum())
    if new_kp != old_kp:
        return new_kp > old_kp
    return float(candidate.conf.sum()) > float(existing.conf.sum())


def refine_subject(
    video_path,
    subject: Track,
    backend: PoseBackend,
    motion: CameraMotion | None = None,
    n_frames: int | None = None,
    config: RefineConfig | None = None,
    progress: bool = False,
) -> tuple[dict[int, PersonDetection], RefineReport]:
    """Re-detect the athlete on a follow-crop, in stabilised coordinates.

    ``subject`` is expected in the same coordinate frame the rest of the
    pipeline works in, stabilised, if camera compensation is on, so the
    predicted boxes are mapped back into raw image space to cut the crop, and
    the resulting keypoints are mapped forward again on the way out.
    """
    from smartdive.video import frames as iter_frames

    cfg = config or RefineConfig()
    report = RefineReport()
    if not len(subject.frames):
        return {}, report

    lo = max(0, int(subject.frames[0]) - cfg.extend_frames)
    hi = int(subject.frames[-1]) + cfg.extend_frames
    if n_frames is not None:
        hi = min(hi, n_frames - 1)

    predicted = _predicted_boxes(subject, lo, hi, cfg)
    refined: dict[int, PersonDetection] = {}

    for index, frame in iter_frames(video_path):
        if index < lo:
            continue
        if index > hi:
            break
        box = predicted.get(index)
        if box is None:
            continue

        height, width = frame.shape[:2]
        image_box = _to_image(box, motion, index)
        x1, y1, x2, y2 = _crop_window(image_box, width, height, cfg)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue

        report.frames_attempted += 1
        detections = backend.detect(frame[y1:y2, x1:x2], index)
        if progress:
            print(f"  refine frame {index} ({len(detections.people)} candidates)", flush=True)
        if not detections.people:
            continue

        offset = np.array([x1, y1], dtype=float)
        predicted_centre = np.array(
            [(image_box[0] + image_box[2]) / 2.0, (image_box[1] + image_box[3]) / 2.0]
        )
        budget = cfg.max_centre_shift_bodies * max(
            image_box[2] - image_box[0], image_box[3] - image_box[1], 1.0
        )
        best: PersonDetection | None = None
        best_overlap = cfg.min_iou
        for person in detections.people:
            moved = PersonDetection(
                box=person.box + np.tile(offset, 2),
                score=person.score,
                xy=person.xy + offset,
                conf=person.conf.copy(),
            )
            if moved.xy.shape[0] != NUM_KEYPOINTS or moved.score < cfg.min_score:
                continue
            if np.linalg.norm(moved.center - predicted_centre) > budget:
                continue
            overlap = iou(moved.box, image_box)
            if overlap >= best_overlap:
                best, best_overlap = moved, overlap

        if best is None:
            continue

        candidate = _to_stabilised(best, motion, index)
        existing = subject.observations.get(index)
        if _better(candidate, existing, cfg):
            refined[index] = candidate
            if existing is None:
                report.frames_recovered += 1
            else:
                report.frames_improved += 1
        else:
            report.frames_unchanged += 1

    return refined, report


def _to_image(box: np.ndarray, motion: CameraMotion | None, index: int) -> np.ndarray:
    if motion is None:
        return np.asarray(box, dtype=float)
    corners = np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=float
    )
    warped = motion.unstabilise(index, corners)
    return np.array(
        [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()]
    )


def _to_stabilised(
    person: PersonDetection, motion: CameraMotion | None, index: int
) -> PersonDetection:
    if motion is None:
        return person
    corners = np.array(
        [
            [person.box[0], person.box[1]],
            [person.box[2], person.box[1]],
            [person.box[2], person.box[3]],
            [person.box[0], person.box[3]],
        ],
        dtype=float,
    )
    warped = motion.stabilise(index, corners)
    return PersonDetection(
        box=np.array(
            [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()]
        ),
        score=person.score,
        xy=motion.stabilise(index, person.xy),
        conf=person.conf,
    )


def reject_inconsistent(
    subject: Track,
    refined: dict[int, PersonDetection],
    max_jump_bodies: float = 0.55,
) -> dict[int, PersonDetection]:
    """Drop refined frames whose position disagrees with their own neighbours.

    A second opinion is only worth having if it is checked. A crop detection
    that lands on the wrong person is not merely useless, it is worse than the
    gap it filled, because a single frame displaced by half a body length is a
    spike large enough to move the noise estimate that everything downstream is
    gated on, measured on ``ginz_dive``, two such frames tripled it.

    So every refined frame is compared against a local prediction built from
    the frames either side of it, and rejected if it disagrees by more than
    half a body length. Only refined frames can be rejected: the pass is never
    allowed to leave the track worse than it found it.
    """
    if not refined:
        return refined

    merged = {**subject.observations, **refined}
    scale = max(subject.body_scale_px(), 1.0)
    budget = max_jump_bodies * scale
    frames = np.array(sorted(merged))
    anchors = np.array([_anchor(merged[int(f)]) for f in frames])

    kept: dict[int, PersonDetection] = {}
    for index in refined:
        position = int(np.searchsorted(frames, index))
        neighbours = [
            anchors[k]
            for k in (position - 2, position - 1, position + 1, position + 2)
            if 0 <= k < len(frames) and abs(frames[k] - index) <= 3 and int(frames[k]) != index
        ]
        if len(neighbours) < 2:
            kept[index] = refined[index]
            continue
        expected = np.median(np.array(neighbours), axis=0)
        if np.linalg.norm(anchors[position] - expected) <= budget:
            kept[index] = refined[index]
    return kept


def _anchor(det: PersonDetection) -> np.ndarray:
    from smartdive.tracking import anchor

    return anchor(det)


def apply_refinement(subject: Track, refined: dict[int, PersonDetection]) -> Track:
    """Return a copy of ``subject`` with the checked refined detections merged in."""
    refined = reject_inconsistent(subject, refined)
    merged = Track(track_id=subject.track_id)
    merged.observations = {**subject.observations, **refined}
    merged.last_frame = int(max(merged.observations)) if merged.observations else -1
    if merged.last_frame >= 0:
        merged.box = merged.observations[merged.last_frame].box.copy()
    merged.velocity = subject.velocity.copy()
    return merged


def refine_frame_detections(
    detections: list[FrameDetections], refined: dict[int, PersonDetection]
) -> list[FrameDetections]:
    """Splice refined athlete detections into a per-frame detection list.

    Used only for rendering and reporting, where it is convenient for the frame
    list and the subject track to agree about what was seen.
    """
    out: list[FrameDetections] = []
    for frame in detections:
        people = list(frame.people)
        if frame.index in refined:
            people.append(refined[frame.index])
        out.append(FrameDetections(index=frame.index, people=people))
    return out

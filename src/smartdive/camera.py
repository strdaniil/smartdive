"""Camera motion estimation and trajectory stabilisation.

Every metric in this project is derived from a *projectile* model, and a
projectile model is only valid in a frame of reference that is not itself
accelerating. Poolside footage is almost never shot from a locked-off tripod:
whoever is holding the phone pans to keep the athlete in shot, usually starting
the pan at the moment the athlete leaves the block. That is precisely the worst
possible time, because it puts a camera acceleration inside the flight window
and the fit has no way to tell it apart from the athlete's own motion.

The size of the problem is easy to underestimate. On the sample clips the
background drifts by 40 to 230 px over the clip, one to eight percent of the
athlete's own travel, which sounds negligible. It is not, for two reasons:

1. The *vertical* component lands directly on top of the quantity the physics
   depends on. Over a 0.35 s flight a parabola departs from its own chord by
   only ``g * T**2 / 8``, some 25 px at typical framing, so a 5 px vertical
   camera drift is a 20% error in the measured gravity, and therefore in any
   scale derived from it.
2. It is not white noise. A pan is smooth and monotone, so it does not average
   out over the window; it biases the fit in one direction.

So this module measures the camera and subtracts it. Sparse features are
tracked between consecutive frames with Lucas-Kanade optical flow, everything
inside a detected person's bounding box is discarded (people are exactly the
things that move independently), and a similarity transform is fitted to what is
left by RANSAC. Composing those per-frame transforms gives, for every frame, a
warp into the coordinate system of one reference frame, a synthetic locked-off
tripod, in which the athlete's flight really is a parabola.

A similarity transform (translation, rotation, uniform scale) is deliberately
chosen over a full homography. Hand-held pans over a few tenths of a second are
dominated by translation with a little roll and zoom, and a homography has eight
degrees of freedom to fit that with, enough to start absorbing part of the
athlete's motion through the perspective terms when the usable background is
concentrated on one side of the frame, which poolside it usually is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from smartdive.backends.base import FrameDetections


@dataclass
class CameraConfig:
    """Feature-tracking and model-fitting parameters for motion estimation."""

    #: Maximum corners to track per frame. More is steadier and slower.
    max_features: int = 900
    #: Shi-Tomasi quality cut, relative to the best corner in the frame.
    quality: float = 0.01
    #: Minimum spacing between tracked corners, in pixels.
    min_distance: int = 12
    #: Person boxes are dilated by this fraction of their size before masking,
    #: because a bounding box clips a swinging arm and the pixels just outside
    #: it are frequently still moving with the person.
    box_margin: float = 0.25
    #: RANSAC inlier threshold for the similarity fit, in pixels.
    ransac_px: float = 3.0
    #: Below this many surviving background features the frame-to-frame estimate
    #: is not trustworthy and identity is used instead.
    min_features: int = 24
    #: Longest side the frame is resized to before tracking. Camera motion is a
    #: low-frequency signal; tracking it at full 4K resolution costs time and
    #: buys nothing.
    work_size: int = 960


@dataclass
class CameraMotion:
    """Per-frame warps from image coordinates into a fixed reference frame.

    ``to_reference[i]`` is a 2x3 affine matrix mapping a point observed in frame
    ``i`` to where it would have appeared had the camera stayed where it was in
    ``reference_frame``.
    """

    reference_frame: int
    to_reference: np.ndarray  # (N, 2, 3)
    inliers: np.ndarray  # (N,) background features that agreed, per frame
    ok: np.ndarray  # (N,) bool, False where the estimate fell back to identity

    def __len__(self) -> int:
        return len(self.to_reference)

    def stabilise(self, frame_index: int, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 2)`` points from frame ``frame_index`` into reference space."""
        points = np.asarray(points, dtype=float)
        if not 0 <= frame_index < len(self.to_reference):
            return points.copy()
        matrix = self.to_reference[frame_index]
        flat = points.reshape(-1, 2)
        out = flat @ matrix[:, :2].T + matrix[:, 2]
        return out.reshape(points.shape)

    def unstabilise(self, frame_index: int, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 2)`` points from reference space back into frame ``frame_index``."""
        points = np.asarray(points, dtype=float)
        if not 0 <= frame_index < len(self.to_reference):
            return points.copy()
        matrix = self.to_reference[frame_index]
        inverse = cv2.invertAffineTransform(matrix)
        flat = points.reshape(-1, 2)
        out = flat @ inverse[:, :2].T + inverse[:, 2]
        return out.reshape(points.shape)

    def zoom(self, frame_index: int) -> float:
        """Uniform scale factor applied by the warp at ``frame_index``.

        A pixel length measured in frame ``frame_index`` is multiplied by this
        when it is carried into reference space, so any pixel-domain quantity
        compared across frames has to be divided by it.
        """
        if not 0 <= frame_index < len(self.to_reference):
            return 1.0
        m = self.to_reference[frame_index]
        return float(np.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])) or 1.0)

    def displacement_px(self) -> np.ndarray:
        """How far each frame's image centre has been moved by the warp."""
        return np.linalg.norm(self.to_reference[:, :, 2], axis=1)

    def to_dict(self) -> dict[str, object]:
        drift = float(np.nanmax(np.abs(self.to_reference[:, :, 2]))) if len(self) else 0.0
        return {
            "reference_frame": self.reference_frame,
            "frames": len(self),
            "max_drift_px": round(drift, 1),
            "median_inliers": int(np.median(self.inliers)) if len(self) else 0,
            "failed_frames": int((~self.ok).sum()),
        }


def identity_motion(n_frames: int, reference_frame: int = 0) -> CameraMotion:
    """A :class:`CameraMotion` that does nothing, for locked-off footage."""
    eye = np.tile(np.eye(2, 3, dtype=float), (max(n_frames, 0), 1, 1))
    return CameraMotion(
        reference_frame=reference_frame,
        to_reference=eye,
        inliers=np.zeros(max(n_frames, 0), dtype=int),
        ok=np.ones(max(n_frames, 0), dtype=bool),
    )


def _person_mask(
    shape: tuple[int, int],
    detections: FrameDetections | None,
    margin: float,
    scale: float = 1.0,
) -> np.ndarray:
    """255 everywhere except inside (dilated) person boxes.

    ``shape`` is the working-resolution frame shape and ``scale`` the factor the
    frame was resized by, since the boxes arrive in original image coordinates.
    """
    mask = np.full(shape, 255, dtype=np.uint8)
    if detections is None:
        return mask
    h, w = shape
    for person in detections.people:
        x1, y1, x2, y2 = np.asarray(person.box, dtype=float) * scale
        dx = (x2 - x1) * margin
        dy = (y2 - y1) * margin
        cv2.rectangle(
            mask,
            (max(0, int(x1 - dx)), max(0, int(y1 - dy))),
            (min(w - 1, int(x2 + dx)), min(h - 1, int(y2 + dy))),
            0,
            -1,
        )
    return mask


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Affine composition: apply ``b`` first, then ``a``."""
    return (np.vstack([a, [0, 0, 1]]) @ np.vstack([b, [0, 0, 1]]))[:2]


def estimate_motion(
    video_path: str | Path,
    detections: list[FrameDetections] | None = None,
    config: CameraConfig | None = None,
    reference_frame: int = 0,
) -> CameraMotion:
    """Measure how the camera moved, frame by frame, over a whole clip.

    Returns warps into the coordinate frame of ``reference_frame``. People are
    masked out using ``detections`` when they are available; without them the
    estimate still works on clips where the athlete occupies a small part of the
    frame, but RANSAC is doing all of the rejection on its own.
    """
    from smartdive.video import frames as iter_frames

    cfg = config or CameraConfig()

    pairwise: list[np.ndarray] = []  # frame i-1 -> frame i
    inliers: list[int] = []
    ok: list[bool] = []

    previous: np.ndarray | None = None
    previous_index = -1
    scale = 1.0

    for index, frame in iter_frames(video_path):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if max(grey.shape) > cfg.work_size:
            scale = cfg.work_size / max(grey.shape)
            grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if previous is None:
            pairwise.append(np.eye(2, 3))
            inliers.append(0)
            ok.append(True)
        else:
            det = None
            if detections is not None and 0 <= previous_index < len(detections):
                det = detections[previous_index]
            mask = _person_mask(previous.shape, det, cfg.box_margin, scale)
            matrix, n_inliers = _pairwise_similarity(previous, grey, mask, cfg)
            if matrix is None:
                pairwise.append(np.eye(2, 3))
                inliers.append(n_inliers)
                ok.append(False)
            else:
                # Undo the working-resolution downscale: a translation measured
                # on a half-size image is half the translation on the original.
                matrix = matrix.copy()
                matrix[:, 2] /= scale
                pairwise.append(matrix)
                inliers.append(n_inliers)
                ok.append(True)

        previous = grey
        previous_index = index

    n = len(pairwise)
    if n == 0:
        return identity_motion(0, reference_frame)

    # Chain the pairwise estimates into a warp from every frame to frame 0,
    # then re-base onto the requested reference frame.
    to_first = [np.eye(2, 3)]
    for i in range(1, n):
        # pairwise[i] maps frame i-1 into frame i, so its inverse carries frame
        # i back onto frame i-1.
        to_first.append(_compose(to_first[i - 1], cv2.invertAffineTransform(pairwise[i])))
    to_first_arr = np.array(to_first)

    reference_frame = int(np.clip(reference_frame, 0, n - 1))
    rebase = cv2.invertAffineTransform(to_first_arr[reference_frame])
    to_reference = np.array([_compose(rebase, m) for m in to_first_arr])

    return CameraMotion(
        reference_frame=reference_frame,
        to_reference=to_reference,
        inliers=np.array(inliers, dtype=int),
        ok=np.array(ok, dtype=bool),
    )


def _pairwise_similarity(
    previous: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    cfg: CameraConfig,
) -> tuple[np.ndarray | None, int]:
    """Similarity transform carrying ``previous`` onto ``current``."""
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=cfg.max_features,
        qualityLevel=cfg.quality,
        minDistance=cfg.min_distance,
        mask=mask,
    )
    if corners is None or len(corners) < cfg.min_features:
        return None, 0

    tracked, status, _err = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    if tracked is None:
        return None, 0

    # Forward-backward check: re-track the result and keep only the features
    # that land back where they started. Occlusion boundaries, which poolside
    # means the edge of every person, produce confident-looking flow that this
    # catches and a one-way check does not.
    back, status_b, _err_b = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None)
    good = (status.ravel() == 1) & (status_b.ravel() == 1)
    if back is not None:
        good &= np.linalg.norm((back - corners).reshape(-1, 2), axis=1) < 1.0
    if good.sum() < cfg.min_features:
        return None, int(good.sum())

    src = corners.reshape(-1, 2)[good]
    dst = tracked.reshape(-1, 2)[good]
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px, maxIters=2000
    )
    if matrix is None:
        return None, int(good.sum())
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else int(good.sum())
    if n_inliers < cfg.min_features:
        return None, n_inliers
    return matrix.astype(float), n_inliers


def stabilise_detections(
    detections: list[FrameDetections], motion: CameraMotion
) -> list[FrameDetections]:
    """Copy ``detections`` with every keypoint and box mapped into reference space."""
    import copy

    out: list[FrameDetections] = []
    for frame in detections:
        people = []
        for person in frame.people:
            moved = copy.copy(person)
            moved.xy = motion.stabilise(frame.index, person.xy)
            corners = np.array(
                [
                    [person.box[0], person.box[1]],
                    [person.box[2], person.box[1]],
                    [person.box[2], person.box[3]],
                    [person.box[0], person.box[3]],
                ]
            )
            warped = motion.stabilise(frame.index, corners)
            moved.box = np.array(
                [
                    warped[:, 0].min(),
                    warped[:, 1].min(),
                    warped[:, 0].max(),
                    warped[:, 1].max(),
                ]
            )
            people.append(moved)
        out.append(FrameDetections(index=frame.index, people=people))
    return out

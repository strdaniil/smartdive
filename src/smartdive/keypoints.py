"""Canonical keypoint schema and body-segment model.

Every pose backend is normalised to the COCO-17 layout so that the rest of the
pipeline never has to care which detector produced a frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- COCO-17 ordering -------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NAME_TO_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

#: Skeleton edges used for rendering.
SKELETON: tuple[tuple[int, int], ...] = (
    (LEFT_ANKLE, LEFT_KNEE),
    (LEFT_KNEE, LEFT_HIP),
    (RIGHT_ANKLE, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_EYE, RIGHT_EYE),
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
)

#: Left/right keypoint pairs. In a lateral view, how a racing start is almost
#: always filmed, the two members of each pair project to nearly the same
#: point, which is what makes contralateral substitution a safe way to complete
#: a half-detected skeleton. See :func:`smartdive.pose.mirror_missing`.
MIRROR: tuple[tuple[int, int], ...] = (
    (LEFT_EYE, RIGHT_EYE),
    (LEFT_EAR, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
)

#: Midpoint landmarks, resolved as the mean of two keypoints.
PAIR_LANDMARKS: dict[str, tuple[int, int]] = {
    "hips": (LEFT_HIP, RIGHT_HIP),
    "shoulders": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "knees": (LEFT_KNEE, RIGHT_KNEE),
    "ankles": (LEFT_ANKLE, RIGHT_ANKLE),
    "wrists": (LEFT_WRIST, RIGHT_WRIST),
    "elbows": (LEFT_ELBOW, RIGHT_ELBOW),
    "ears": (LEFT_EAR, RIGHT_EAR),
}

#: ``com`` is the whole-body centre of mass (see :mod:`smartdive.com`).
LANDMARKS: tuple[str, ...] = ("com", *sorted(PAIR_LANDMARKS), *KEYPOINT_NAMES)


@dataclass(frozen=True)
class Segment:
    """A rigid body segment for the centre-of-mass model."""

    name: str
    proximal: tuple[int, ...]
    distal: tuple[int, ...]
    com_ratio: float  # fraction of segment length from proximal to segment CoM
    mass_fraction: float


# Winter (2009) / Dempster segment inertial parameters, adapted to the joints
# COCO-17 actually gives us. Forearm and hand are merged (no hand keypoint);
# shank and foot are merged (no toe keypoint).
SEGMENTS: tuple[Segment, ...] = (
    Segment("trunk", (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP), 0.440, 0.4684),
    Segment("left_upper_arm", (LEFT_SHOULDER,), (LEFT_ELBOW,), 0.436, 0.0280),
    Segment("right_upper_arm", (RIGHT_SHOULDER,), (RIGHT_ELBOW,), 0.436, 0.0280),
    Segment("left_forearm_hand", (LEFT_ELBOW,), (LEFT_WRIST,), 0.682, 0.0220),
    Segment("right_forearm_hand", (RIGHT_ELBOW,), (RIGHT_WRIST,), 0.682, 0.0220),
    Segment("left_thigh", (LEFT_HIP,), (LEFT_KNEE,), 0.433, 0.1000),
    Segment("right_thigh", (RIGHT_HIP,), (RIGHT_KNEE,), 0.433, 0.1000),
    Segment("left_shank_foot", (LEFT_KNEE,), (LEFT_ANKLE,), 0.606, 0.0610),
    Segment("right_shank_foot", (RIGHT_KNEE,), (RIGHT_ANKLE,), 0.606, 0.0610),
)

#: Head + neck is placed at the centroid of the available head keypoints.
HEAD_KEYPOINTS: tuple[int, ...] = (LEFT_EAR, RIGHT_EAR, NOSE)
HEAD_MASS_FRACTION: float = 0.0826

TOTAL_MASS_FRACTION = HEAD_MASS_FRACTION + sum(s.mass_fraction for s in SEGMENTS)


def resolve_landmark(
    xy: np.ndarray,
    conf: np.ndarray,
    landmark: str,
    min_conf: float = 0.3,
) -> np.ndarray:
    """Return the ``(x, y)`` pixel position of ``landmark``, or ``(nan, nan)``.

    ``xy`` is ``(17, 2)`` and ``conf`` is ``(17,)``.
    """
    if landmark == "com":  # imported lazily to avoid a circular import
        from smartdive.com import center_of_mass

        return center_of_mass(xy, conf, min_conf=min_conf)

    if landmark in PAIR_LANDMARKS:
        i, j = PAIR_LANDMARKS[landmark]
        if conf[i] < min_conf or conf[j] < min_conf:
            return np.array([np.nan, np.nan])
        return (xy[i] + xy[j]) / 2.0

    if landmark in NAME_TO_INDEX:
        i = NAME_TO_INDEX[landmark]
        if conf[i] < min_conf:
            return np.array([np.nan, np.nan])
        return xy[i].astype(float).copy()

    raise ValueError(f"unknown landmark {landmark!r}; choose one of {', '.join(LANDMARKS)}")


def body_scale(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> float:
    """Largest distance between any two confident keypoints, in pixels.

    Used throughout the pipeline as a subject-relative length unit so that
    thresholds do not depend on video resolution or how far away the athlete is.
    """
    visible = xy[conf >= min_conf]
    if len(visible) < 4:
        return float("nan")
    d = np.linalg.norm(visible[:, None, :] - visible[None, :, :], axis=-1)
    return float(d.max())

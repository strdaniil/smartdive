"""Overlay rendering.

The overlay's job is to make the measurement checkable by eye. Three things it
draws are worth calling out.

**Everything is drawn back in the camera's frame.** The pipeline works in a
stabilised coordinate system in which the camera never moved (see
:mod:`smartdive.camera`), and a trajectory drawn from those coordinates straight
onto the original video would drift away from the athlete by however far the
camera panned. Every point is therefore mapped back through the per-frame warp
before it is drawn, so the trail sits on the body it describes.

**The model is drawn next to the measurement.** The green trail is what was
measured and the amber curve is the projectile the physics fitted. Where they
agree, the model explains the motion and the numbers mean something; where they
part company, they do not. That comparison is the whole point of the picture and
it is not available from a single line drawn on a video.

**Joint trails are optional and secondary.** Tracing the hips, shoulders and
head is what the original project drew, and it is genuinely useful for looking
at body rotation through the flight, but it is not what the metrics are
computed from, so it is drawn thin and behind the centre-of-mass trail rather
than in place of it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smartdive.analysis import DiveAnalysis
from smartdive.keypoints import NAME_TO_INDEX, PAIR_LANDMARKS, SKELETON
from smartdive.video import VideoWriter
from smartdive.video import frames as iter_frames

# BGR.
COLOR_MEASURED = (80, 220, 90)
COLOR_MODEL = (60, 200, 255)
COLOR_SKELETON = (235, 200, 80)
COLOR_JOINT = (255, 255, 255)
COLOR_PANEL = (28, 24, 20)
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (176, 176, 176)
COLOR_FLAG = (90, 120, 255)

#: Joints traced alongside the centre of mass, and the colour of each trail.
JOINT_TRAILS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("hips", (210, 130, 90)),
    ("shoulders", (200, 100, 200)),
    ("nose", (90, 190, 230)),
)


def _to_image(analysis: DiveAnalysis, index: int, points: np.ndarray) -> np.ndarray:
    """Map stabilised points back into frame ``index``'s own coordinates."""
    if not analysis.config.stabilise or not len(analysis.motion):
        return np.asarray(points, dtype=float)
    return analysis.motion.unstabilise(index, points)


def _draw_skeleton(
    frame: np.ndarray,
    xy: np.ndarray,
    conf: np.ndarray,
    min_conf: float,
    thickness: int,
) -> None:
    for a, b in SKELETON:
        if conf[a] >= min_conf and conf[b] >= min_conf:
            cv2.line(
                frame,
                tuple(np.round(xy[a]).astype(int)),
                tuple(np.round(xy[b]).astype(int)),
                COLOR_SKELETON,
                thickness,
                cv2.LINE_AA,
            )
    for i in range(len(xy)):
        if conf[i] >= min_conf:
            cv2.circle(
                frame,
                tuple(np.round(xy[i]).astype(int)),
                thickness + 1,
                COLOR_JOINT,
                -1,
                cv2.LINE_AA,
            )


def _polyline(frame: np.ndarray, points: np.ndarray, colour, thickness: int) -> None:
    """Draw a trail, breaking it wherever the trajectory has a gap."""
    good = ~np.isnan(points[:, 0])
    if not good.any():
        return
    idx = np.where(good)[0]
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, breaks):
        if len(run) > 1:
            cv2.polylines(
                frame,
                [np.round(points[run]).astype(np.int32)],
                False,
                colour,
                thickness,
                cv2.LINE_AA,
            )


def _joint_series(analysis: DiveAnalysis, name: str) -> np.ndarray:
    """``(N, 2)`` conditioned trajectory of a named joint, in stabilised space."""
    pose = analysis.pose
    present = pose.present()
    if name in PAIR_LANDMARKS:
        i, j = PAIR_LANDMARKS[name]
        out = (pose.xy[:, i, :] + pose.xy[:, j, :]) / 2.0
        out[~(present[:, i] & present[:, j])] = np.nan
        return out
    k = NAME_TO_INDEX[name]
    out = pose.xy[:, k, :].copy()
    out[~present[:, k]] = np.nan
    return out


def render_overlay(
    analysis: DiveAnalysis,
    output: str | Path,
    show_skeleton: bool = True,
    show_model: bool = True,
    show_panel: bool = True,
    show_joint_trails: bool = True,
    trail_length: int | None = None,
) -> Path:
    """Write an annotated copy of the source video."""
    info = analysis.video
    scale = max(0.6, min(info.width, info.height) / 720.0)
    thickness = max(2, round(2 * scale))

    traj = analysis.trajectory
    fit = analysis.fit
    min_conf = analysis.config.min_keypoint_conf

    trails = (
        [(name, colour, _joint_series(analysis, name)) for name, colour in JOINT_TRAILS]
        if show_joint_trails
        else []
    )

    model_points = None
    if show_model and fit is not None:
        model_points = fit.sample(180)

    output = Path(output)
    with VideoWriter(output, info.fps, (info.width, info.height)) as writer:
        for index, frame in iter_frames(info.path):
            det = analysis.subject.observations.get(index)
            if show_skeleton and det is not None:
                _draw_skeleton(
                    frame, _to_image(analysis, index, det.xy), det.conf, min_conf, thickness
                )

            lo = 0 if not trail_length else max(0, index + 1 - trail_length)
            for _name, colour, series in trails:
                _polyline(
                    frame,
                    _to_image(analysis, index, series[lo : index + 1]),
                    colour,
                    max(1, thickness - 1),
                )

            upto = _to_image(analysis, index, traj.xy[lo : index + 1])
            _polyline(frame, upto, COLOR_MEASURED, thickness)
            head = upto[~np.isnan(upto[:, 0])]
            if len(head):
                cv2.circle(
                    frame,
                    tuple(np.round(head[-1]).astype(int)),
                    thickness + 3,
                    COLOR_MEASURED,
                    -1,
                    cv2.LINE_AA,
                )

            if model_points is not None and fit is not None and index >= fit.start_frame:
                progress = min(
                    1.0,
                    (index - fit.start_frame) / max(fit.end_frame - fit.start_frame, 1),
                )
                end = max(2, int(progress * len(model_points)))
                drawn = _to_image(analysis, index, model_points[:end])
                cv2.polylines(
                    frame,
                    [np.round(drawn).astype(np.int32)],
                    False,
                    COLOR_MODEL,
                    thickness,
                    cv2.LINE_AA,
                )

            if fit is not None:
                if index == fit.start_frame:
                    _flag(frame, "TAKEOFF", scale)
                elif index == fit.end_frame:
                    _flag(frame, "ENTRY", scale)

            if show_panel:
                _panel(frame, _panel_lines(analysis, index), _phase(analysis, index), scale)

            writer.write(frame)

    return output


def _panel(
    frame: np.ndarray,
    lines: list[tuple[str, str]],
    title: str,
    scale: float,
) -> None:
    """Draw a translucent metric panel in the top-left corner."""
    pad = int(14 * scale)
    line_h = int(26 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55 * scale
    label_w = max([cv2.getTextSize(k, font, fs, 1)[0][0] for k, _ in lines] or [0])
    value_w = max([cv2.getTextSize(v, font, fs, 1)[0][0] for _, v in lines] or [0])
    width = label_w + value_w + 3 * pad
    height = (len(lines) + 1) * line_h + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + width, pad + height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + width, pad + height), (90, 86, 78), 1)

    y = pad + line_h
    cv2.putText(
        frame,
        title,
        (pad * 2, y),
        font,
        fs * 1.05,
        COLOR_MODEL,
        max(1, int(scale)),
        cv2.LINE_AA,
    )
    for key, value in lines:
        y += line_h
        cv2.putText(frame, key, (pad * 2, y), font, fs, COLOR_DIM, max(1, int(scale)), cv2.LINE_AA)
        cv2.putText(
            frame,
            value,
            (pad * 2 + label_w + pad, y),
            font,
            fs,
            COLOR_TEXT,
            max(1, int(scale)),
            cv2.LINE_AA,
        )


def _flag(frame: np.ndarray, text: str, scale: float) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 1.0 * scale
    (tw, _th), _ = cv2.getTextSize(text, font, fs, 2)
    cv2.putText(
        frame,
        text,
        (w - tw - int(30 * scale), h - int(30 * scale)),
        font,
        fs,
        COLOR_FLAG,
        max(2, int(2 * scale)),
        cv2.LINE_AA,
    )


def _phase(analysis: DiveAnalysis, index: int) -> str:
    fit = analysis.fit
    if fit is None:
        return "SmartDive - no flight detected"
    if index < fit.start_frame:
        return "SmartDive - on the block"
    if index <= fit.end_frame:
        return "SmartDive - flight"
    return "SmartDive - in the water"


def _panel_lines(analysis: DiveAnalysis, index: int) -> list[tuple[str, str]]:
    metrics = analysis.metrics
    fit = analysis.fit
    lines: list[tuple[str, str]] = [("frame", str(index))]

    if metrics is None or fit is None:
        lines.append(("status", "no ballistic window"))
        return lines

    if index < fit.start_frame:
        lines.append(("phase", f"takeoff at frame {fit.start_frame}"))
    elif index <= fit.end_frame:
        t = (index - fit.start_frame) / analysis.fps
        pos = fit.position(np.array([t]))[0]
        drop = (pos[1] - fit.y0) / metrics.px_per_metre
        lines += [
            ("airborne", f"{t:.2f} s"),
            ("height vs takeoff", f"{-drop:+.2f} m"),
        ]
    else:
        lines.append(("phase", f"entry at frame {fit.end_frame}"))

    lines += [
        ("flight time", f"{metrics.flight_time_s:.3f} s"),
        ("takeoff speed", f"{metrics.takeoff_speed_ms:.2f} m/s"),
        ("takeoff angle", f"{metrics.takeoff_angle_deg:+.1f} deg"),
        ("flight distance", f"{metrics.flight_distance_m:.2f} m"),
        ("scale", f"{metrics.px_per_metre:.0f} px/m ({metrics.calibration_method})"),
        ("fit R^2", f"{metrics.fit_r_squared:.4f}"),
    ]
    if analysis.time_base.corrected:
        lines.insert(1, ("capture rate", f"{analysis.fps:g} fps"))
    return lines


def render_comparison(
    top: str | Path,
    bottom: str | Path,
    output: str | Path,
    top_label: str = "before",
    bottom_label: str = "after",
    width: int = 960,
) -> Path:
    """Stack two rendered videos vertically with labels, for side-by-side proof."""
    caps = [cv2.VideoCapture(str(top)), cv2.VideoCapture(str(bottom))]
    try:
        if not all(c.isOpened() for c in caps):
            raise OSError("cannot open both videos for comparison")
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        heights = []
        for c in caps:
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            heights.append(round(h * width / w))

        output = Path(output)
        with VideoWriter(output, fps, (width, sum(heights))) as writer:
            while True:
                tiles = []
                for cap, h, label in zip(caps, heights, (top_label, bottom_label), strict=True):
                    ok, frame = cap.read()
                    if not ok:
                        tiles = []
                        break
                    tile = cv2.resize(frame, (width, h))
                    _banner(tile, label)
                    tiles.append(tile)
                if not tiles:
                    break
                writer.write(np.vstack(tiles))
    finally:
        for c in caps:
            c.release()
    return output


def _banner(tile: np.ndarray, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    h = tile.shape[0]
    cv2.rectangle(tile, (0, h - th - 20), (tw + 24, h), (25, 22, 20), -1)
    cv2.putText(tile, label, (12, h - 10), font, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

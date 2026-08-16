"""Overlay rendering.

Two things changed from the original overlay scripts. The trail is drawn with a
single ``polylines`` call per frame instead of re-drawing every past segment
individually, which turns an O(n^2) render into an O(n) one. And the overlay
shows the *model* next to the measurement, plus a live metric panel, so the
video is evidence rather than decoration.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smartdive.analysis import DiveAnalysis
from smartdive.keypoints import SKELETON
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


def render_overlay(
    analysis: DiveAnalysis,
    output: str | Path,
    show_skeleton: bool = True,
    show_model: bool = True,
    show_panel: bool = True,
    trail_length: int | None = None,
) -> Path:
    """Write an annotated copy of the source video.

    The green trail is what was measured; the amber curve is the fitted
    projectile. Where they agree, the physics model explains the motion.
    """
    info = analysis.video
    scale = max(0.6, min(info.width, info.height) / 720.0)
    thickness = max(2, round(2 * scale))

    traj = analysis.trajectory
    fit = analysis.fit
    min_conf = analysis.config.min_keypoint_conf

    model_points = None
    if show_model and fit is not None:
        model_points = np.round(fit.sample(180)).astype(np.int32)

    output = Path(output)
    with VideoWriter(output, info.fps, (info.width, info.height)) as writer:
        for index, frame in iter_frames(info.path):
            det = analysis.subject.observations.get(index)
            if show_skeleton and det is not None:
                _draw_skeleton(frame, det.xy, det.conf, min_conf, thickness)

            # Measured trail up to now, as one polyline per contiguous run.
            upto = traj.xy[: index + 1]
            if trail_length:
                upto = traj.xy[max(0, index + 1 - trail_length) : index + 1]
            good = ~np.isnan(upto[:, 0])
            if good.any():
                pts = np.round(upto[good]).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, COLOR_MEASURED, thickness, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), thickness + 3, COLOR_MEASURED, -1, cv2.LINE_AA)

            if model_points is not None and fit is not None and index >= fit.start_frame:
                progress = min(
                    1.0,
                    (index - fit.start_frame) / max(fit.end_frame - fit.start_frame, 1),
                )
                end = max(2, int(progress * len(model_points)))
                cv2.polylines(
                    frame, [model_points[:end]], False, COLOR_MODEL, thickness, cv2.LINE_AA
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
        t = (index - fit.start_frame) / analysis.video.fps
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

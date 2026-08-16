"""Render the side-by-side demos used in the README.

For each clip this produces:

* ``<clip>_original.mp4`` -- the original pipeline's overlay, reproduced from
  its own code path (argmax subject, no conditioning, cubic fit over the whole
  clip).
* ``<clip>_smartdive3.mp4`` -- the current overlay.
* ``<clip>_sidebyside.mp4`` and ``.gif`` -- the two stacked, for the README.

    python benchmarks/make_demos.py videos/adive.mp4 -o assets --cache .cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from smartdive.analysis import AnalysisConfig, analyze
from smartdive.legacy import LEGACY_POLY_DEGREE, legacy_metrics, legacy_trajectory
from smartdive.render import render_comparison, render_overlay
from smartdive.video import VideoWriter
from smartdive.video import frames as iter_frames

# The original scripts' colours: green measured trail, yellow/red cubic fit.
LEGACY_TRAIL = (0, 255, 0)
LEGACY_FIT = (0, 255, 255)
TEXT = (240, 240, 240)


def render_legacy_overlay(video: Path, trajectory: np.ndarray, fps: float, out: Path) -> Path:
    """Reproduce the original overlay_video.py / polyfit.py output."""
    metrics = legacy_metrics(trajectory, fps)
    fit_points = None
    if metrics is not None:
        valid = np.where(~np.isnan(trajectory[:, 0]))[0]
        t = np.linspace(valid[0], valid[-1], 200)
        fit_points = np.stack(
            [np.polyval(metrics.poly_x, t), np.polyval(metrics.poly_y, t)], axis=1
        ).astype(np.int32)

    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale = max(0.6, min(width, height) / 720.0)
    thickness = max(2, round(2 * scale))

    with VideoWriter(out, fps, (width, height)) as writer:
        for index, frame in iter_frames(video):
            upto = trajectory[: index + 1]
            good = ~np.isnan(upto[:, 0])
            if good.any():
                pts = np.round(upto[good]).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, LEGACY_TRAIL, thickness, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), thickness + 3, LEGACY_TRAIL, -1, cv2.LINE_AA)
            if fit_points is not None:
                cv2.polylines(frame, [fit_points], False, LEGACY_FIT, thickness, cv2.LINE_AA)

            label = [
                "original pipeline",
                f"argmax subject, degree-{LEGACY_POLY_DEGREE} fit over whole clip",
            ]
            if metrics is not None:
                label.append(
                    f"flight {metrics.flight_time_s:.2f} s, {metrics.horizontal_distance_px:.0f} px"
                )
            _caption(frame, label, scale)
            writer.write(frame)
    return out


def _caption(frame: np.ndarray, lines: list[str], scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55 * scale
    pad = int(14 * scale)
    lh = int(26 * scale)
    w = max(cv2.getTextSize(t, font, fs, 1)[0][0] for t in lines) + 2 * pad
    h = len(lines) * lh + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + w, pad + h), (28, 24, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    y = pad + lh - int(6 * scale)
    for text in lines:
        cv2.putText(frame, text, (pad * 2, y), font, fs, TEXT, max(1, int(scale)), cv2.LINE_AA)
        y += lh


def write_gif(
    video: Path, out: Path, width: int = 640, fps: float = 12.0, colors: int = 96
) -> Path | None:
    """Best-effort animated GIF for embedding in the README."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed, skipping GIF)")
        return None

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / fps))
    frames_out = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (width, int(h * width / w)))
            frames_out.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        index += 1
    cap.release()

    if not frames_out:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    # Quantise to a shared adaptive palette: a full-colour GIF of poolside
    # footage runs to tens of megabytes, which is not something to put in a
    # README.
    palette = frames_out[0].quantize(colors=colors, method=Image.MEDIANCUT)
    quantised = [f.quantize(colors=colors, palette=palette) for f in frames_out]
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("assets"))
    parser.add_argument("--landmark", default="hips")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument("--gif-colors", type=int, default=96)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig(landmark=args.landmark, bootstrap=0)

    for video in args.videos:
        print(f"[{video.name}]")
        result = analyze(video, config, cache=args.cache)
        stem = video.stem

        legacy_xy = legacy_trajectory(result.detections, landmark="hips")
        old = render_legacy_overlay(
            video, legacy_xy, result.video.fps, args.outdir / f"{stem}_original.mp4"
        )
        print(f"  {old}")

        new = render_overlay(result, args.outdir / f"{stem}_smartdive3.mp4")
        print(f"  {new}")

        combined = render_comparison(
            old,
            new,
            args.outdir / f"{stem}_sidebyside.mp4",
            top_label="original pipeline",
            bottom_label="SmartDive 3",
        )
        print(f"  {combined}")

        if not args.no_gif:
            gif = write_gif(
                combined,
                args.outdir / f"{stem}_sidebyside.gif",
                args.gif_width,
                colors=args.gif_colors,
            )
            if gif:
                size_mb = gif.stat().st_size / 1e6
                print(f"  {gif} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

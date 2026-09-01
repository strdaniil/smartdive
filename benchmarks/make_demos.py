"""Render the side-by-side demo assets used in the README.

For each clip this produces three files in ``assets/``:

* ``<clip>_original.mp4``, the legacy pipeline's overlay, drawn by the same
  renderer so the comparison is about the measurement and not the graphics;
* ``<clip>_smartdive3.mp4``, the current pipeline's overlay;
* ``<clip>_sidebyside.mp4``, the two stacked, which is the honest way to show
  the difference: the same frames, the same detections, two conclusions;
* ``<clip>_sidebyside.gif``, the same thing as a GIF for the README, if
  ``ffmpeg`` is on the path.

    python benchmarks/make_demos.py videos/adive.mp4 --cache .cache
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from smartdive.analysis import AnalysisConfig, analyze
from smartdive.legacy import legacy_metrics, legacy_trajectory
from smartdive.render import render_comparison, render_overlay
from smartdive.video import VideoWriter
from smartdive.video import frames as iter_frames

LEGACY_COLOR = (90, 220, 120)
PANEL = (28, 24, 20)
TEXT = (240, 240, 240)


def render_legacy(result, output: Path) -> Path:
    """Draw what the original pipeline saw, using the current renderer.

    Deliberately reproduces its *output*, not a caricature of it: the trail is
    the per-frame argmax landmark it actually tracked, and the caption is the
    metric it actually printed. What the picture shows, a short tangle of line
    somewhere other than the diver, with a confident number beside it, is
    what that code produced.
    """
    xy = legacy_trajectory(result.detections, landmark="hips")
    metrics = legacy_metrics(xy, result.video.fps)
    info = result.video
    scale = max(0.6, min(info.width, info.height) / 720.0)
    thickness = max(2, round(2 * scale))

    caption = [
        "original pipeline",
        "argmax subject, degree-3 fit over whole clip",
    ]
    if metrics is not None:
        caption.append(
            f"flight {metrics.flight_time_s:.2f} s, {metrics.horizontal_distance_px:.0f} px"
        )

    with VideoWriter(output, info.fps, (info.width, info.height)) as writer:
        for index, frame in iter_frames(info.path):
            upto = xy[: index + 1]
            good = ~np.isnan(upto[:, 0])
            if good.any():
                points = np.round(upto[good]).astype(np.int32)
                if len(points) > 1:
                    cv2.polylines(frame, [points], False, LEGACY_COLOR, thickness, cv2.LINE_AA)
                cv2.circle(frame, tuple(points[-1]), thickness + 3, LEGACY_COLOR, -1, cv2.LINE_AA)
            _caption(frame, caption, scale)
            writer.write(frame)
    return output


def _caption(frame: np.ndarray, lines: list[str], scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55 * scale
    pad = int(14 * scale)
    line_h = int(24 * scale)
    width = max(cv2.getTextSize(t, font, fs, 1)[0][0] for t in lines) + 2 * pad
    height = len(lines) * line_h + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + width, pad + height), PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    y = pad + line_h - int(6 * scale)
    for text in lines:
        cv2.putText(frame, text, (pad * 2, y), font, fs, TEXT, max(1, int(scale)), cv2.LINE_AA)
        y += line_h


def to_gif(video: Path, output: Path, width: int, fps: int = 12) -> Path | None:
    """Convert a rendered comparison to a GIF, via ffmpeg's two-pass palette.

    A single shared palette for the whole clip rather than per-frame dithering:
    the background is nearly static, so one palette keeps the pool from
    shimmering between frames and roughly halves the file.
    """
    if shutil.which("ffmpeg") is None:
        print("  (ffmpeg not found; skipping the GIF)", flush=True)
        return None
    chain = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", chain, "-loop", "0", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("assets"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--width", type=int, default=960, help="width of the stacked video")
    parser.add_argument("--gif-width", type=int, default=440, help="width of the README GIF")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig(bootstrap=0)

    for video in args.videos:
        print(f"[{video.name}] ...", flush=True)
        result = analyze(video, config, cache=args.cache)

        original = render_legacy(result, args.outdir / f"{video.stem}_original.mp4")
        current = render_overlay(result, args.outdir / f"{video.stem}_smartdive3.mp4")
        combined = render_comparison(
            original,
            current,
            args.outdir / f"{video.stem}_sidebyside.mp4",
            top_label="original pipeline",
            bottom_label="SmartDive 3",
            width=args.width,
        )
        print(f"  {original}\n  {current}\n  {combined}", flush=True)

        if not args.no_gif:
            gif = to_gif(combined, args.outdir / f"{video.stem}_sidebyside.gif", args.gif_width)
            if gif is not None:
                print(f"  {gif}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

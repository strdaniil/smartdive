# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

## Before opening a pull request

```bash
ruff check src tests benchmarks examples
ruff format src tests benchmarks examples
pytest
```

The test suite stubs the pose backend, so it runs in about twenty seconds
without model weights or footage. Please keep it that way: anything that needs
a real detector belongs in `benchmarks/`, not `tests/`.

The stub is not a placeholder. `tests/synthetic.py` generates an anatomically
proportioned COCO-17 skeleton at a chosen pixel stature, using the same Winter
segment ratios the calibration reads back out, so "generate a body at a known
size, analyse the clip, check the calibration recovers that size" is a real
round-trip test rather than a tautology. If you add a stage that depends on body
geometry, extend that generator rather than hard-coding keypoints in a test.

## Changing the analysis

Three rules make changes reviewable:

1. **Every threshold is subject-relative.** Express tolerances in body lengths
   or in units of the estimated noise, never in raw pixels. A constant tuned on
   1080p footage is a bug waiting for someone's 4K camera.
2. **Show the effect on real clips.** If you touch tracking, conditioning or
   phase detection, run

   ```bash
   python benchmarks/run_benchmark.py videos/*.mp4 videos/*.mov --cache .cache
   ```

   and put the before/after table in the pull request. Detections are cached, so
   after the first run this takes seconds.

   Watch the self-consistency block at the bottom of that table especially.
   `adive.mp4` and `adive2.mp4` are the same dive at different playback rates,
   so if a change makes them disagree *more*, it has made the analysis worse
   even when every individual number still looks plausible.

3. **Prefer a measurement to an assumption.** Most of the bugs this project has
   had were assumptions that nobody had written down: that the camera was
   locked off, that the container frame rate was the capture rate, that the
   widest keypoint separation meant the same thing in every pose. If a stage
   needs to know something about the footage, the first question is whether the
   footage can be asked.

## Reporting a bad result

Open an issue with the output of

```bash
smartdive analyze your_clip.mp4 --quiet > report.json
```

The JSON records the full track ranking, the camera-motion summary, the
recovered capture rate, the body measurement, the signal-quality numbers and
every warning, usually enough to tell a tracking failure from a phase-detection
failure from a calibration failure without the video.

If you can share the clip, `--overlay` is worth attaching too: a trail drawn on
the wrong person and a trail drawn on the right person with the wrong flight
window look nothing alike, and the difference is obvious in two seconds of
video and invisible in a table.

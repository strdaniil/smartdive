# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

## Before opening a pull request

```bash
ruff check src tests benchmarks
ruff format src tests benchmarks
pytest
```

The test suite stubs the pose backend, so it runs in a couple of seconds
without model weights or footage. Please keep it that way: anything that needs
a real detector belongs in `benchmarks/`, not `tests/`.

## Changing the analysis

Two rules make changes reviewable:

1. **Every threshold is subject-relative.** Express tolerances in body lengths
   or in units of the estimated noise, never in raw pixels. A constant tuned on
   1080p footage is a bug waiting for someone's 4K camera.
2. **Show the effect on real clips.** If you touch tracking, conditioning or
   phase detection, run

   ```bash
   python benchmarks/run_benchmark.py videos/*.mp4 --cache .cache
   ```

   and put the before/after table in the pull request. Detections are cached, so
   after the first run this takes seconds.

## Reporting a bad result

Open an issue with the output of

```bash
smartdive analyze your_clip.mp4 --quiet > report.json
```

The JSON records the full track ranking, the signal-quality numbers and every
warning, which is usually enough to tell a tracking failure from a phase
detection failure without the video.

# Benchmarks

Two scripts, both of which need real footage in `videos/` and the `[yolo]`
extra installed. Neither is part of the test suite — `pytest` stubs the backend
so CI stays fast and weight-free.

## `run_benchmark.py`

Runs the original pipeline (`smartdive.legacy`) and the current one over the
**same pose detections** and writes a comparison table, a JSON dump of every
number, and a per-clip figure.

```bash
python benchmarks/run_benchmark.py videos/*.mp4 -o benchmarks/results --cache .cache
```

`--cache` memoises detections per (clip, backend, options). The first run costs
a minute or so per clip; every rerun after that takes seconds, which is what
makes it usable as a regression check while changing the analysis.

## `make_demos.py`

Renders the side-by-side videos and the README GIF: the original pipeline's
overlay on top, the current one below.

```bash
python benchmarks/make_demos.py videos/adive.mp4 -o assets --cache .cache --gif-width 460
```

Rendered `.mp4` files are gitignored — they are large and regenerable. The
committed GIF and figures come from these commands.

## What the numbers mean

- **looking at the diver** — fraction of frames where per-frame `argmax`
  confidence, the original subject rule, picked the athlete the tracker follows.
- **path travelled** — straight-line displacement of the reported trajectory, in
  body lengths. A start covers two to four; a spectator covers none.
- **jitter** — RMS of the second difference. Real motion is smooth, so what a
  second difference leaves behind is measurement error. Read it alongside path
  travelled: large travel *and* large jitter means the trajectory is ricocheting
  between people, not following one.

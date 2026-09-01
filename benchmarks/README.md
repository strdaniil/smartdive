# Benchmarks

Two scripts, both of which need real footage in `videos/` and the `[yolo]`
extra installed. Neither is part of the test suite; `pytest` stubs the backend
so CI stays fast and weight-free.

## `run_benchmark.py`

Runs the original pipeline (`smartdive.legacy`) and the current one over the
**same pose detections** and writes a comparison table, a JSON dump of every
number, and a per-clip figure.

```bash
python benchmarks/run_benchmark.py videos/*.mp4 videos/*.mov \
    -o benchmarks/results --cache .cache
```

`--cache` memoises the pose detections, the camera-motion estimate and the
second detection pass, per (clip, backend, options). The first run costs a
couple of minutes per clip; every rerun after that takes seconds, which is what
makes it usable as a regression check while changing the analysis.

## `make_demos.py`

Renders the side-by-side videos and the README GIF: the original pipeline's
overlay on top, the current one below, drawn by the same renderer so the
comparison is about the measurement and not the graphics.

```bash
python benchmarks/make_demos.py videos/adive.mp4 -o assets --cache .cache --gif-width 440
```

The GIF needs `ffmpeg` on the path; without it the videos are still written and
the GIF is skipped. Rendered `.mp4` files are gitignored: they are large and
regenerable. The committed GIF and figures come from these commands.

## What the numbers mean

- **looking at the diver**: fraction of frames where per-frame `argmax`
  confidence, the original subject rule, picked the athlete the tracker follows.
- **path travelled**: straight-line displacement of the reported trajectory, in
  body lengths. A start covers two to four; a spectator covers none.
- **jitter**: RMS of the second difference. Real motion is smooth, so what a
  second difference leaves behind is measurement error. Read it alongside path
  travelled: large travel *and* large jitter means the trajectory is ricocheting
  between people, not following one.
- **flight phase**: the frames the current pipeline believes were airborne. The
  original had no such concept and always reported the whole clip.
- **at N fps capture**: appears when the clip's container frame rate was not
  its capture rate, and the real one was recovered from gravity. Two of the five
  sample clips are 120 fps footage in a 30 fps container.

## The self-consistency check

There is no tape measure in this footage, so there is no ground truth. There is
something almost as useful: `adive.mp4` and `adive2.mp4` are the **same dive**,
the second exported at four times slow motion. Nothing about the athlete differs
between them, so any disagreement in the reported metrics is the pipeline's own
error, and reaching agreement at all requires the slow-motion detection to
recover 120 fps from a container claiming 30.

`run_benchmark.py` prints that comparison automatically whenever both clips are
in the run. Speeds and distances currently agree to within 5%; flight time to
17%, which is the honest resolution of the phase detection at 30 fps.

Add a pair to `DUPLICATE_PAIRS` in `run_benchmark.py` if you have another clip
exported twice.

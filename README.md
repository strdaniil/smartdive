<h1 align="center">SmartDive</h1>

<p align="center">
  <b>Swim start analysis from ordinary poolside video, with the physics enforced.</b>
</p>

<p align="center">
  <a href="https://github.com/strdaniil/smartdive/actions/workflows/ci.yml"><img src="https://github.com/strdaniil/smartdive/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-261230.svg" alt="Code style: ruff"></a>
</p>

<p align="center">
  <img src="assets/demo.gif" width="82%" alt="SmartDive tracking a racing start">
</p>
<p align="center">
  <sub>SmartDive following the athlete's centre of mass through the airborne window,
  from takeoff to water entry.</sub>
</p>

Point it at a clip of a racing start. It finds the athlete, subtracts the
camera's own motion, works out which frames they were airborne, fits the
projectile model to their flight, and reports takeoff speed, takeoff angle,
flight time and flight distance in SI units, each with its calibration source
and an uncertainty next to it.

```bash
pip install "smartdive[yolo] @ git+https://github.com/strdaniil/smartdive"
smartdive analyze start.mp4 --overlay --stature-m 1.85
```

```
SmartDive 3.1.0  |  adive.mp4
1920x1080 @ 30 fps, 48 frames  |  analysed in 0.4 s
subject: track 1 of 6 (37 detections, 3.5 body lengths travelled, peak 7.4 body lengths/s)
signal: 60% coverage, jitter 23.9 -> 8.8 px, 8 frames interpolated, camera drift 61 px removed
refine: 0 frames recovered, 16 improved, 19 unchanged
scale:  429 px of stature from 10 segments (spread 4%)

Block time                 0.433 s
Flight time                0.400 s
Takeoff speed              4.53 m/s
Takeoff angle              +25.6 deg
  horizontal               4.08 m/s
  vertical                 +1.95 m/s
Flight distance            1.63 m
Peak height above takeoff  0.19 m
Entry angle                25.8 deg
Airborne frames            13 to 25
Front knee angle at set    113.4 deg
Scale                      207.5 px/m via stature
Fit quality                R^2 = 0.9893, RMS 11.0 px

95% CI on takeoff angle: +23.4 to +27.6 deg
```

## The model

A body in free flight obeys, exactly:

```
x(t) = x0 + vx*t
y(t) = y0 + vy*t + 0.5*g*t^2        (image y grows downward)
```

SmartDive fits that model to the athlete's flight, so the coefficients it
recovers *are* the takeoff velocity, the fit residual *is* the test of which
frames were airborne, and every distance comes out in metres with a stated
calibration and a bootstrap confidence interval. Nothing is reported in raw
pixels, and nothing is reported without an error bar.

## What a run produces

Each analysis writes a JSON report (metrics, the track ranking, signal quality,
camera motion, provenance and warnings), a per-frame CSV labelled `block` /
`flight` / `water`, an optional annotated video, and a diagnostic figure:

<p align="center">
  <img src="assets/adive_diagnostics.png" width="92%" alt="diagnostic figure">
</p>

The bottom-middle panel reads first: residuals inside a few percent of body
length mean the projectile model explains the motion and the metrics are sound;
structure in them means it does not. The top-right panel shows the water-entry
cue firing, and the bottom-right shows how much camera motion was removed.

## Results across the sample clips

Five starts filmed on ordinary phones, including two crowded pool decks and two
clips shot in slow motion. Reproduce with:

```bash
python benchmarks/run_benchmark.py videos/*.mp4 videos/*.mov --cache .cache
```

| clip | frames | flight window | flight time | takeoff speed | distance | capture |
|---|---|---|---|---|---|---|
| `adive.mp4` | 48 | 13-25 | 0.40 s | 4.4 m/s | 1.6 m | 30 fps |
| `adive_wide.mov` | 138 | 61-69 | 0.27 s | 5.0 m/s | 1.3 m | 30 fps |
| `adive2.mp4` | 192 | 47-103 | 0.47 s | 4.2 m/s | 1.7 m | 120 fps recovered |
| `ginz_dive.mp4` | 74 | 32-49 | 0.57 s | 4.3 m/s | 2.3 m | 30 fps |
| `test1.mp4` | 105 | 36-97 | 0.51 s | 3.9 m/s | 1.9 m | 120 fps recovered |

### How accurate is it?

There is no tape measure in this footage, so there is no ground truth. But two of
the clips are the *same dive*: `adive.mp4` in real time and `adive2.mp4` exported
at four times slow motion. Any disagreement between them is the tool's own error,
and reaching agreement at all requires the slow-motion detection to recover
120 fps from a container insisting on 30.

| quantity | `adive.mp4` | `adive2.mp4` | agreement |
|---|---|---|---|
| capture rate recovered | 30 fps | 120 fps | correct 4x ratio |
| flight time | 0.400 s | 0.467 s | 17% |
| takeoff speed | 4.39 m/s | 4.23 m/s | **4%** |
| flight distance | 1.57 m | 1.65 m | **5%** |

Speeds and distances agree to within 5%. The flight window agrees to about two
frames at 30 fps, which is where the 17% on flight time comes from and why the
timings are quoted with error bars.

## Camera and slow motion

Two things phone footage does that break a naive analysis, both handled
automatically.

**The camera is not bolted down.** Whoever is filming pans to keep the athlete in
shot, starting the pan the moment the athlete leaves the block, which puts a
camera acceleration inside the flight window where the fit cannot separate it
from the athlete's own motion. So the camera is measured and subtracted: sparse
features are tracked between frames, everything inside a detected person's box is
discarded, and a similarity transform is fitted by RANSAC to what is left.
Composing those warps gives a synthetic locked-off tripod.

**The container frame rate is not the capture rate.** A clip shot in a phone's
slow-motion mode carries the rate it will be played at, so times and speeds come
out wrong by that factor with nothing in the pixels to say so. Gravity provides
an independent clock. Having pinned the scale from the athlete's body, a body in
free fall shows apparent vertical acceleration `g * px_per_metre * k^2`, so
`k = sqrt(a_y / (g * px_per_metre))` and the true rate is `container_fps / k`.
On the sample footage this recovers 120 fps on both slow-motion clips and leaves
the three real-time clips alone. It is applied only when the fitted acceleration
is itself well determined, so ordinary 30 fps footage cannot wrongly declare
itself slow motion.

## Install

```bash
git clone https://github.com/strdaniil/smartdive
cd smartdive
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[yolo]"
```

`[yolo]` pulls in `ultralytics` and `torch`. For a CPU-only, torch-free setup,
use `pip install -e ".[mediapipe]"` and pass `--backend mediapipe`. Model weights
download on first run.

## Use

```bash
# Full analysis with an annotated video
smartdive analyze start.mp4 --overlay -o output/

# Exact metres: give it a length you measured in the frame
smartdive analyze start.mp4 --reference-px 512 --reference-m 2.5   # lane width

# Or the athlete's height, usually easier to find out
smartdive analyze start.mp4 --stature-m 1.83

# Cache detections and camera motion, so re-runs are instant
smartdive analyze start.mp4 --cache .cache

# Track something other than the centre of mass
smartdive analyze start.mp4 --landmark hips

# Override the phase detection
smartdive analyze start.mp4 --flight-window 14:26

# Escape hatches, if a clip defeats the automatic version
smartdive analyze start.mp4 --capture-fps 240 --no-depth-scale --no-refine
```

From Python:

```python
from smartdive import analyze

result = analyze("start.mp4", cache=".cache")

print(result.metrics.format_table())
print(result.metrics.takeoff_speed_ms)       # 4.53
print(result.fps)                             # capture rate, after slow-mo detection
print(result.uncertainty["angle_deg"])        # (23.4, 27.6)
print(result.quality())                       # coverage, jitter, identity errors, drift
for warning in result.warnings:
    print(warning)
```

## How it works

| stage | module | what it does |
|---|---|---|
| detection | `backends/` | YOLO or MediaPipe, normalised to COCO-17 |
| camera | `camera.py` | background similarity transform per frame, RANSAC |
| association | `tracking.py` | constant-velocity tracks, Hungarian assignment |
| stitching | `tracking.py` | reconnect fragments of the same person |
| subject choice | `tracking.py` | one decision per clip, by peak body-length speed |
| refinement | `refine.py` | second pose pass on a crop that follows the athlete |
| skeleton | `pose.py` | complete and condition all 17 keypoints, then the CoM |
| scale | `anthro.py` | stature from bone lengths; per-frame size for depth |
| phase | `physics.py` | visibility for entry, per-frame residual for takeoff |
| clock | `calibration.py` | recover the capture rate of slow-motion footage |
| metrics | `metrics.py` | SI kinematics plus bootstrap intervals |

A few details worth calling out; [`docs/method.md`](docs/method.md) has the rest.

**The subject is chosen once per clip, by peak speed in body lengths per second.**
Detections are associated into tracks, fragments of one person are stitched back
together, and the athlete is the one who briefly moves fastest relative to their
own size. Divers reach three to eight body lengths per second at takeoff; someone
walking the deck reaches one; someone standing on a block reaches none. That is
what lets it pick the diver out of a busy pool deck.

**The detector gets a second look at a crop that follows the athlete.** A
whole-frame pass spends its resolution on the whole frame, and the part that
needs it most, a motion-blurred body over the water, is where it fails. Once the
athlete is identified, the video is read again at a crop that follows them,
upscaled to fill the network's input, so the frames near water entry are actually
resolved.

**Keypoints are conditioned before the centre of mass.** The CoM is a weighted
sum over body segments; if a segment drops out of a frame the sum runs over a
different set of segments and moves even when the athlete does not. Conditioning
each of the 17 keypoints as its own series first, filling a missing one from its
contralateral partner, keeps the same segments contributing in every frame.

**The scale comes from bone lengths, not the bounding box.** A bounding box is
pose-dependent by a factor of three between a crouch and a streamline. Individual
segments are fixed fractions of stature whatever the athlete is doing, and
projection can only shorten them, so a high percentile of each segment's observed
length estimates its true length. See [Calibration](#calibration) below.

## Calibration

**Tell it the athlete's height.** One flag, and the single most useful thing you
can supply:

```bash
smartdive analyze start.mp4 --stature-m 1.85
```

Without it, a height of 1.78 m is assumed (roughly the mean for adult
competitive swimmers) and distances and speeds carry about 15% error. With it,
about 5%. Times and angles are unaffected either way, so a wrong height corrupts
only the speeds and leaves the flight time, takeoff angle and phase boundaries
exactly right.

A single camera cannot recover scale: a 1.8 m athlete far away and a 0.9 m doll
near the lens produce identical pixels, because projection collapses size and
distance into one number. Every scale estimate is therefore a choice of what to
assume. SmartDive assumes the athlete's proportions and, failing better
information, their height.

The athlete becomes a ruler through their bones, each a fixed fraction of
standing height (Winter, *Biomechanics and Motor Control of Human Movement*):

```
thigh (hip to knee)    0.245 of stature      upper arm   0.186
shank (knee to ankle)  0.246                  forearm     0.146
hip to ankle           0.491                  trunk       0.270
```

The chain for one clip, with real numbers from `adive.mp4`:

```
video       ->  left thigh measured at 94.5 px       (the only measurement)
Winter      ->  / 0.245  =  386 px of stature        (no height assumed yet)
your input  ->  / 1.85 m  =  207.5 px/m              (the assumption, one division)
                every distance / 207.5
```

The assumed height enters once, at the end. Steps one and two say "this athlete
is 386 pixels tall", a genuine measurement of your video that is the same number
whether the swimmer is 1.60 m or 2.00 m. Only the last division needs to know
which.

Two statistics measure the bones. **Within a segment, across frames, the 85th
percentile:** projection only ever shortens, so every frame underestimates and
the top of the distribution is the truth; the 85th percentile skips the
foreshortened frames without landing on whichever frame the detector blew out.
**Across segments, a weighted median with a trim:** ten segments give ten
stature estimates, combined by a median weighted towards the long leg segments,
with any segment more than 35% from consensus dropped. A mislocated wrist once
made a forearm read 1047 px, implying a seven-metre athlete; the median did not
notice.

Checked against a standing bystander in `adive.mp4`, whose eye-to-ankle height
can be measured directly, the estimate lands within 2%.

The three calibration routes, in order of preference:

| method | you supply | error on distances | flag |
|---|---|---|---|
| `reference` | a length you measured in the frame | your tape measure | `--reference-px 512 --reference-m 2.5` |
| `stature` | the athlete's height | ~5% | `--stature-m 1.85` |
| `stature_assumed` | nothing | ~15% | *(default, 1.78 m)* |

A reference is the best one-time investment: measure one lane width at the
diver's depth in a single frame and it removes the last assumption for that
camera position. Competition lanes are 2.5 m; many US pools are 7 ft (2.13 m),
so check rather than guess. A height in centimetres is caught, not used:
`--stature-m 185` is rejected with a message naming the likely mistake, because
a units slip here is a factor of a hundred and nothing downstream would notice.

Gravity is not used for scale. Over a 0.4 s flight at 30 fps a parabola departs
from its own chord by about 42 px, while keypoint noise is 10 to 16 px, so
refitting the same trajectory over windows a frame apart swings the gravity
scale by a factor of 2.3 where body scale holds to 3%. Gravity is computed and
reported as a cross-check only. Its one good job is the reverse: on
high-frame-rate footage, where the curvature is well determined, the same
equation recovers the clock instead of the ruler. Full detail in
[`docs/method.md`](docs/method.md).

## Known limits

Stated up front, because a measurement tool that hides its error bars is worse
than one that has none.

- **Entry timing** is accurate to about 3 frames (0.1 s at 30 fps). The
  visibility cue fires slightly before full immersion; the window is then
  extended forward while the trajectory still fits.
- **Takeoff timing** is accurate to about 2 frames at 30 fps, where the block
  push is compressed into a few frames. On high-frame-rate footage it is less
  sharp: the push resolves over dozens of frames and a parabola fits part of it,
  so expect takeoff a few hundredths early. The guard is that the athlete must
  already be past half their takeoff speed at the frame called takeoff;
  `--flight-window` is the fix when that is not enough.
- **Absolute distances** carry the calibration's uncertainty: roughly 15%
  without a reference length, 5% with `--stature-m`, your own accuracy with a
  measured reference.
- **Out-of-plane motion is invisible to one camera.** The depth correction fixes
  the scale of an athlete travelling away from the camera but cannot recover the
  velocity component pointing into the frame, so speeds from an obliquely filmed
  start are a lower bound and the report says so.
- **Single start per clip.** Trim first if the video contains several. Several
  *swimmers* in one clip are fine; that is the whole point.
- **The knee-angle band (125-150 deg)** is a general set-position heuristic, not
  clinical advice.

For the best numbers: film square-on to the lane from a fixed position, in slow
motion if the phone offers it, and pass `--stature-m`.

## Repository layout

```
src/smartdive/        the package; one module per pipeline stage (table above)
tests/                pytest suite, backend stubbed (no weights, no footage)
  synthetic.py        the proportioned synthetic athlete the stubs are built on
benchmarks/           reproduce the results table and render demos; needs footage
  run_benchmark.py    the results table, reproducibly
  make_demos.py       the overlay videos and the demo GIF
examples/quickstart.py  one clip, everything the pipeline knows about it
docs/method.md        why each stage is built the way it is
videos/               your clips (gitignored)
assets/               committed figures and the demo GIF
```

## Development

```bash
pip install -e ".[all]"
pytest                                   # ~110 tests, no weights needed
ruff check src tests benchmarks
python benchmarks/run_benchmark.py videos/*.mp4 videos/*.mov --cache .cache
python benchmarks/make_demos.py videos/adive.mp4 --cache .cache
```

The suite stubs the pose backend with an anatomically proportioned synthetic
athlete, so CI needs neither torch nor network access. Because the stub is
generated at a known pixel stature, "analyse it and see whether the calibration
finds that stature again" is a real test rather than a tautology. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT, see [LICENSE](LICENSE).

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and
[MediaPipe](https://github.com/google-ai-edge/mediapipe). Segment inertial
parameters and stature ratios from Winter, *Biomechanics and Motor Control of
Human Movement*.

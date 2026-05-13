# cross_embodiment

Overlay a YAM-style bimanual robot onto a video of a human performing a
manipulation task with handheld trackers. Built around the
[`YAMGripper_MatFolding`](https://huggingface.co/datasets/starpilot-ai/YAMGripper_MatFolding)
dataset: an egocentric scene camera (1024² fisheye) + Vive 6DoF poses for
two PIKA grippers per frame.

The end product, per frame, is the original scene with the human + their
hand-held grippers removed and a calibrated YAM end-effector rendered into
the same camera as the operator's gripper pose.

## Result

![pipeline preview](docs/pipeline.gif)

(Image generated from `artifacts/composite_pinhole.mp4`.)

## Pipeline

Each stage produces a per-frame artifact viewable in the included HTML
scrubber (`viewer/index.html`).

| # | Stage | Tool | Output |
|---|---|---|---|
| 1 | Original fisheye | — | `data/frames_scene_rgb_2/*.png` |
| 2 | Fisheye → pinhole undistort | OpenCV `cv2.fisheye` | `artifacts/pinhole/*.png` |
| 3 | Mask operator + arms | SAM2 (image predictor, box prompts) | `artifacts/pinhole_masks/*.png` |
| 4 | Inpaint masked region | LaMa (simple-lama-inpainting) | `artifacts/pinhole_inpainted/*.png` |
| 5 | Monocular depth | Depth Anything V2 (small) | `artifacts/depth_pinhole/*.png` |
| 6 | YAM gripper render | Mujoco (calibrated scene camera) | `artifacts/render/*.png` |
| 7 | Composite | luminance chroma-key | `artifacts/composite_pinhole/*.png` |

A click-based **scene-camera calibrator** (`viewer/calibrate.html`) lets you
mark the visible gripper position in a handful of frames, then a PnP solver
(`scripts/19_solve_pnp_auto.py`) recovers the camera intrinsics + extrinsics
plus the rigid transform between the two Vive trackers' local frames.

## Quickstart

Requires Python 3.11, CUDA-capable GPU, ffmpeg, [uv](https://docs.astral.sh/uv/).

```sh
# 1) Install
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe \
    opencv-python datasets huggingface_hub transformers scipy pyarrow av \
    mujoco accelerate einops sam2 hydra-core iopath matplotlib trimesh \
    simple-lama-inpainting

# 2) Fetch the dataset + decode frames
.venv/Scripts/python.exe scripts/01_fetch_data.py

# 3) Pull the YAM URDF and gripper meshes from i2rt
git clone --depth 1 https://github.com/i2rt-robotics/i2rt.git data/i2rt
cp data/i2rt/i2rt/robot_models/gripper/linear_4310/assets/*.stl \
   data/i2rt/i2rt/robot_models/arm/yam/assets/

# 4) Run the stages
.venv/Scripts/python.exe scripts/07_undistort.py        # fisheye -> pinhole
.venv/Scripts/python.exe scripts/12_mask_pinhole.py     # SAM2 mask
.venv/Scripts/python.exe scripts/14_inpaint_pinhole.py  # LaMa inpaint
.venv/Scripts/python.exe scripts/08_depth_pinhole.py    # depth

# 5) Calibrate (one-time per dataset)
.venv/Scripts/python.exe -m http.server 8765   # then open viewer/calibrate.html
#   click the base of each visible PIKA gripper across ~12 frames,
#   download JSON, save to data/calibration_clicks.json
.venv/Scripts/python.exe scripts/19_solve_pnp_auto.py

# 6a) Gripper-only render (sanity check the calibration)
.venv/Scripts/python.exe scripts/21_render_grippers.py
.venv/Scripts/python.exe scripts/15_composite_pinhole.py

# 6b) Full bimanual YAM render + IK across all frames
#   ROLL_OFFSET_DEG arg (default 90) adds a roll to the gripper around its
#   forward axis to give the IK headroom past YAM's tight wrist limits.
.venv/Scripts/python.exe scripts/32_render_full_episode.py 90

ffmpeg -y -framerate 30 -i artifacts/composite_pinhole/%06d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 artifacts/composite_pinhole.mp4
```

## Viewer

`python -m http.server 8765` from the project root, then:

* `http://localhost:8765/viewer/index.html` — 8-panel pipeline scrubber
  (fisheye, pinhole, mask, inpaint, depth, render, composite, calibration
  debug). Arrow keys / space to scrub.
* `http://localhost:8765/viewer/calibrate.html` — click-based scene-camera
  calibrator. Click the base of each visible PIKA gripper, download JSON.

## YAM full-body IK notes

* The canonical i2rt `yam.xml` is a *placeholder* for link6 — the real pos /
  quat / joint axis live in `data/i2rt/i2rt/robots/config/linear_4310.yml` and
  get spliced in by `combine_arm_and_gripper_xml()` at runtime. `scripts/32_…`
  inlines those values directly (`pos="2.4e-07 -0.042 0.040"`,
  `quat="0.5 -0.5 -0.5 -0.5"`, `axis="0 0 -1"`). Using the placeholder
  `quat=identity` produces a YAM whose wrist cannot reach most human hand
  orientations — symptom is j4/j5 pinned at their limits with large rotation
  residual.
* IK target: `R_link6 = R_vive @ Rx(180°) @ Rz(ROLL_OFFSET_DEG)` and
  `P_link6 = P_vive + R_vive @ (0, 0, -0.1092)`. The Rx(180°) matches script
  21's gfix mocap convention; the position offset puts the gripper geometry
  where script 21 places it. `ROLL_OFFSET_DEG=90` gives the IK the most
  headroom past YAM's ±90° j4/j5 wrist limits.
* The damped-LS IK warm-starts from the previous frame, then runs a multi-seed
  escape (8 alternate seeds) on any frame whose residual exceeds 30mm / 15°.
* Gripper finger widths are driven from `state[14]` (arm1) and `state[30]`
  (arm2) — normalized [0,1] scaled to the linear_4310 slide stroke of 4.75cm.

## Calibration design notes

* The dataset's `arm1_pose_*` and `arm2_pose_*` are in **separate per-Vive-tracker
  world frames** (offset ~37cm, rotated ~93°). The solver detects this and
  recovers the rigid transform between the two so a single calibrated scene
  camera can project both grippers.
* Auto-swap: the user's `arm1`/`arm2` click labels don't need to match the
  Vive convention — the solver tries all 4 assignments and picks the
  best-fitting subset.
* Click landmark = the midpoint between the two parallel-jaw fingertips at
  the base (where they emerge from the gripper body). The PIKA gripper
  geometry is read from the i2rt `linear_4310` Mujoco model.

## What's not in this repo

* The `data/i2rt/` submodule clone (fetch via the clone command above) — it's
  100s of MB of meshes that change rarely
* The 5GB of per-frame PNGs in `artifacts/` (regenerate via the pipeline)
* `data/hf/` (raw HuggingFace dataset snapshot — `01_fetch_data.py` pulls it)

## Status

This is research-quality code from a multi-day iteration session. The
calibration + per-stage pipeline works for the dataset, but several pieces
are eyeballed (YAM base placement, mask bbox prompts). The fisheye intrinsics
are estimated as an equidistant model; using true camera calibration would
tighten the pinhole crop.

License: see [LICENSE](LICENSE).

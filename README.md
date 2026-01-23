# ros2-audio-tone-recognition

Real-time prosody (tone) perception for robots that conditions manipulation behavior in ROS2. This repo publishes a calibrated `ToneSignal` and maps it to safe, stable `StyleParams` (speed, approach offset, compliance) that a planner/controller can utilize immediately.

## Why this exists

Robots operating around people need to adapt their motion style to human intent. The same instruction can imply different urgency, caution, or comfort depending on vocal tone. This project turns audio prosody into a low-latency control conditioning signal that is stable (hysteresis + hold time prevents flicker), safe (confidence gating + staleness fallback), and integration-ready (simple parameter interface into manipulation stacks).

## What it does

**Pipeline (Option 1: StyleParams → planner/controller):**

Audio → Tone perception → `/tone/signal` → Adapter → `/manip/style_params` → Manipulation stack

- `prosody2policy_tone` publishes `ToneSignal` (dummy now; scaffolded for real mic/VAD/model).
- `prosody2policy_adapter` converts `ToneSignal` → `StyleParams` with safety & hysteresis.
- `prosody2policy_bringup` launches everything with YAML configs.
- `tools/sim_adapter.py` provides a no-ROS2 simulator to validate behavior on macOS.

## Repository layout

```
ros2_ws/src/
	prosody2policy_msgs/         # ROS2 message definitions
	prosody2policy_tone/         # Tone perception node + scaffolding modules
	prosody2policy_adapter/      # Tone → Style adapter node + pure logic + tests
	prosody2policy_bringup/      # Launch + YAML configs
tools/
	sim_adapter.py               # Harness to simulate tone→style mapping
docs/
	RUNNING.md                   # Build/run instructions for Ubuntu ROS2
	ARCHITECTURE.md              # Topic graph + design notes
.github/workflows/
	ci.yml                       # CI: compile + unit tests + simulator smoke test
```

## Core ROS2 interfaces

### Topic: `/tone/signal`
**Message:** `prosody2policy_msgs/ToneSignal`

Conceptually:
- `arousal` (how “urgent/activated” the voice sounds)
- `valence` (how “positive/negative” it sounds)
- `confidence` (model certainty / calibration)
- (optional) embedding + speech probability + RMS (implementation-dependent)

### Topic: `/manip/style_params`
**Message:** `prosody2policy_msgs/StyleParams`

A compact conditioning interface intended for planners/controllers:
- `velocity_scale` (trajectory time scaling / speed)
- `accel_scale` (acceleration scaling)
- `approach_offset_m` (increase standoff distance in backoff mode)
- `stiffness_scale` (compliance / impedance gain scaling)
- `grip_force_cap` (cap max grasp force)
- `mode` (string label: `neutral`, `urgent`, `backoff`, `calm`)

## Robustness & safety design

The adapter is designed to be safe by default and stable under noisy perception:

- **Confidence gating:** If `confidence < min_conf`, output `neutral` `StyleParams`.
- **Staleness fallback:** If no tone messages arrive for `stale_s`, revert to `neutral`.
- **Hysteresis thresholds:** Separate enter/exit thresholds prevent rapid mode oscillations.
- **Minimum mode hold time:** A mode must persist for at least `min_mode_hold_s` before switching (prevents flicker and ensures predictable behavior).

These behaviors are configurable via YAML in `prosody2policy_bringup/config/adapter.yaml`.

## Quick demo (no ROS2 required)

If you’re on macOS (no ROS2 installed), you can still validate the mapping inner logic:

```bash
python tools/sim_adapter.py --scenario scripted --steps 50
```
You can also point the simulator at your ROS-style YAML config:
```bash
python tools/sim_adapter.py --scenario scripted --steps 50 \
	--config ros2_ws/src/prosody2policy_bringup/config/adapter.yaml
```

## Quickstart (Ubuntu / ROS2)

**Important:** The ROS2 commands below are intended to be run on Ubuntu 22.04 with ROS2 Humble installed (native Linux, or Docker).

### Build
```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Run (bringup)
```bash
ros2 launch prosody2policy_bringup demo.launch.py
```

### Inspect topics
```bash
ros2 topic echo /tone/signal
ros2 topic echo /manip/style_params
```

More detail: see `docs/RUNNING.md`.

## Configuration

### Adapter config
`ros2_ws/src/prosody2policy_bringup/config/adapter.yaml` controls:
- `min_conf` (confidence gate)
- `stale_s` (fallback to neutral after missing signals)
- `min_mode_hold_s` (anti-flicker hold time)
- hysteresis thresholds: `urgent_enter`, `urgent_exit`, `negative_enter`, `negative_exit`, `calm_arousal`
- parameter mapping defaults for velocity/accel/offset/stiffness

### Tone node config
`ros2_ws/src/prosody2policy_bringup/config/tone.yaml` controls:
- publish rate (`rate_hz`)
- embedding dimension (`emb_dim`)
- topic name

## Extending to real audio perception (planned + scaffolded)

The `prosody2policy_tone` package is structured like a real streaming perception component:
- `audio_stream.py` (mic/audio abstraction)
- `vad.py` (voice activity detection stub; can be replaced with WebRTC VAD)
- `streaming_estimator.py` (windowing + smoothing + confidence)
- `models/` (placeholder for encoder + regression/classifier heads)

**Recommended real implementation path:**
- Replace `audio_stream.py` with a Linux-friendly mic backend (e.g., sounddevice).
- Replace `vad.py` with WebRTC VAD for robust speech gating.
- Add a small prosody model (e.g., wav2vec2/HuBERT embeddings → lightweight head).
- Calibrate confidence (temperature scaling or isotonic regression) for safe gating.

## Integrating StyleParams into manipulation

This repo intentionally outputs a simple control-conditioning interface (`StyleParams`) so integration is straightforward:
- **Planner scaling:** apply `velocity_scale`/`accel_scale` to time-parameterization or controller limits.
- **Approach behavior:** use `approach_offset_m` to increase standoff distance in cautious/backoff mode.
- **Compliance:** scale impedance gains / stiffness using `stiffness_scale`.
- **Grasp safety:** cap gripper force via `grip_force_cap`.

This separation keeps tone perception modular and avoids entangling perception with the manipulation policy.

## Testing & CI

Pure logic is isolated in `prosody2policy_adapter/mode_logic.py` and validated with pytest.
CI runs Python compile checks, adapter unit tests, and a `tools/sim_adapter.py` smoke test (no ROS2 required).
See `.github/workflows/ci.yml`.

### Reproduce audio signal evaluation metrics 
```bash
python tools/replay_tone_csv.py --in tone.csv --out style.csv --hz 10
python tools/eval_adapter.py --tone tone.csv --style style.csv


## License

Apache-2.0

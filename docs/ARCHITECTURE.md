# Architecture: ROS2 Audio Tone → Manipulation StyleParams

## Goal
Convert streaming audio prosody (tone) into stable, safe manipulation style parameters that can modulate a planner/controller.

## Topic graph
Audio → Tone Perception Node → `/tone/signal` → Adapter Node → `/manip/style_params` → Manipulation stack

## ROS2 Topics
- `/tone/signal` (`prosody2policy_msgs/ToneSignal`)
  - arousal, valence, confidence, speech probability, RMS, optional embedding
- `/manip/style_params` (`prosody2policy_msgs/StyleParams`)
  - velocity_scale, accel_scale, approach_offset_m, stiffness_scale, grip_force_cap, mode

## Adapter robustness
- Confidence gating: if confidence < `min_conf`, output neutral style
- Stale timeout: if tone messages stop for `stale_s`, revert to neutral style
- Mode hysteresis: separate enter/exit thresholds prevent flicker
- Minimum hold time: `min_mode_hold_s` prevents rapid mode switching

## Extension points
- Replace dummy tone publisher with real streaming mic + VAD + model inference
- Integrate `/manip/style_params` into MoveIt/controller scaling and approach offsets

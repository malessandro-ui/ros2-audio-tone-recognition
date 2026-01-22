# Build & Run (Ubuntu / ROS2)

This repo is authored on macOS but intended to be built/run on Linux.
Recommended environment: **Ubuntu 22.04 + ROS2 Humble**.

## Build

```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch prosody2policy_bringup demo.launch.py
```

> **Note:** Use Ubuntu 22.04.


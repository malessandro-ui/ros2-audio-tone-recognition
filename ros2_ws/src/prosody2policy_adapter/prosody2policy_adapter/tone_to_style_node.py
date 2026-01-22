from __future__ import annotations
import time
import rclpy
from rclpy.node import Node
from prosody2policy_msgs.msg import ToneSignal, StyleParams

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

class ToneToStyleNode(Node):
    def __init__(self):
        super().__init__("tone_to_style_node")

        # Topics
        self.declare_parameter("tone_topic", "/tone/signal")
        self.declare_parameter("style_topic", "/manip/style_params")

        # Robustness
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("stale_s", 1.0)
        self.declare_parameter("min_mode_hold_s", 1.5)

        # Thresholds with hysteresis
        self.declare_parameter("urgent_enter", 0.40)
        self.declare_parameter("urgent_exit", 0.25)
        self.declare_parameter("negative_enter", -0.30)
        self.declare_parameter("negative_exit", -0.15)

        # Style values
        self.declare_parameter("vel_neutral", 1.00)
        self.declare_parameter("vel_urgent", 1.40)
        self.declare_parameter("vel_backoff", 0.70)
        self.declare_parameter("acc_neutral", 1.00)
        self.declare_parameter("acc_urgent", 1.30)
        self.declare_parameter("acc_backoff", 0.80)

        self.declare_parameter("offset_neutral_m", 0.00)
        self.declare_parameter("offset_backoff_m", 0.10)

        self.declare_parameter("stiff_neutral", 1.00)
        self.declare_parameter("stiff_calm", 0.85)
        self.declare_parameter("stiff_urgent", 1.10)

        self.declare_parameter("calm_arousal", -0.15)

        self.tone_topic = self.get_parameter("tone_topic").value
        self.style_topic = self.get_parameter("style_topic").value

        self.sub = self.create_subscription(ToneSignal, self.tone_topic, self.on_tone, 10)
        self.pub = self.create_publisher(StyleParams, self.style_topic, 10)

        self.last_tone_time = 0.0
        self.last_mode_change = 0.0
        self.mode = "neutral"  # neutral | urgent | backoff | calm

        self.timer = self.create_timer(0.1, self.tick)  # publish @10Hz
        self.latest: ToneSignal | None = None

        self.get_logger().info(f"Mapping {self.tone_topic} -> {self.style_topic}")

    def on_tone(self, msg: ToneSignal):
        self.latest = msg
        self.last_tone_time = time.time()

    def tick(self):
        now = time.time()
        stale_s = float(self.get_parameter("stale_s").value)
        min_conf = float(self.get_parameter("min_conf").value)
        hold_s = float(self.get_parameter("min_mode_hold_s").value)

        if (now - self.last_tone_time) > stale_s or self.latest is None:
            # stale fallback
            self.publish_style(now, "neutral", 1.0, 1.0, 0.0, 1.0, 0.0)
            return

        ar = float(self.latest.arousal)
        va = float(self.latest.valence)
        cf = float(self.latest.confidence)

        # confidence gating
        if cf < min_conf:
            target_mode = "neutral"
        else:
            target_mode = self.compute_mode(ar, va)

        # hold/hysteresis in time (prevents flicker)
        if target_mode != self.mode and (now - self.last_mode_change) >= hold_s:
            self.mode = target_mode
            self.last_mode_change = now

        # map mode to params
        vel, acc, offset, stiff, grip = self.params_for_mode(ar, va, cf, self.mode)
        self.publish_style(now, self.mode, vel, acc, offset, stiff, grip)

    def compute_mode(self, arousal: float, valence: float) -> str:
        urgent_enter = float(self.get_parameter("urgent_enter").value)
        urgent_exit  = float(self.get_parameter("urgent_exit").value)
        neg_enter    = float(self.get_parameter("negative_enter").value)
        neg_exit     = float(self.get_parameter("negative_exit").value)
        calm_a       = float(self.get_parameter("calm_arousal").value)

        # Hysteresis based on current mode
        if self.mode == "urgent":
            if arousal < urgent_exit:
                return "neutral"
            return "urgent"

        if self.mode == "backoff":
            if valence > neg_exit:
                return "neutral"
            return "backoff"

        # entering conditions
        if arousal > urgent_enter:
            return "urgent"
        if valence < neg_enter:
            return "backoff"
        if arousal < calm_a and valence > -0.1:
            return "calm"
        return "neutral"

    def params_for_mode(self, ar: float, va: float, cf: float, mode: str):
        vel_n = float(self.get_parameter("vel_neutral").value)
        vel_u = float(self.get_parameter("vel_urgent").value)
        vel_b = float(self.get_parameter("vel_backoff").value)
        acc_n = float(self.get_parameter("acc_neutral").value)
        acc_u = float(self.get_parameter("acc_urgent").value)
        acc_b = float(self.get_parameter("acc_backoff").value)

        off_n = float(self.get_parameter("offset_neutral_m").value)
        off_b = float(self.get_parameter("offset_backoff_m").value)

        stiff_n = float(self.get_parameter("stiff_neutral").value)
        stiff_c = float(self.get_parameter("stiff_calm").value)
        stiff_u = float(self.get_parameter("stiff_urgent").value)

        # default grip cap (placeholder for real grippers)
        grip = 1.0

        if mode == "urgent":
            return vel_u, acc_u, off_n, stiff_u, grip
        if mode == "backoff":
            return vel_b, acc_b, off_b, stiff_n, grip
        if mode == "calm":
            return 0.9, 0.9, off_n, stiff_c, grip
        return vel_n, acc_n, off_n, stiff_n, grip

    def publish_style(self, now: float, mode: str, vel: float, acc: float, offset: float, stiff: float, grip: float):
        msg = StyleParams()
        msg.stamp = self.get_clock().now().to_msg()
        msg.mode = mode
        msg.velocity_scale = float(clamp(vel, 0.1, 2.0))
        msg.accel_scale = float(clamp(acc, 0.1, 2.0))
        msg.approach_offset_m = float(clamp(offset, 0.0, 0.5))
        msg.stiffness_scale = float(clamp(stiff, 0.5, 1.5))
        msg.grip_force_cap = float(clamp(grip, 0.0, 1.0))
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ToneToStyleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

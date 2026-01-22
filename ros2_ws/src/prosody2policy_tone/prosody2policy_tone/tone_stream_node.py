from __future__ import annotations
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from prosody2policy_msgs.msg import ToneSignal

class ToneStreamNode(Node):
    """
    Dummy tone publisher (placeholder).
    Replace internals with mic + VAD + model inference later.
    """
    def __init__(self):
        super().__init__("tone_stream_node")
        self.declare_parameter("topic", "/tone/signal")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("emb_dim", 64)

        self.topic = self.get_parameter("topic").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.emb_dim = int(self.get_parameter("emb_dim").value)

        self.pub = self.create_publisher(ToneSignal, self.topic, 10)
        self.t0 = time.time()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(f"Publishing dummy ToneSignal on {self.topic} @ {self.rate_hz} Hz")

    def tick(self):
        t = time.time() - self.t0
        msg = ToneSignal()
        msg.stamp = self.get_clock().now().to_msg()

        # Dummy signals (smooth oscillations)
        msg.arousal = float(0.6 * math.sin(0.6 * t))
        msg.valence = float(0.6 * math.sin(0.4 * t + 1.0))
        msg.confidence = 0.8
        msg.vad_speech_prob = 1.0
        msg.rms = 0.02

        emb = np.zeros((self.emb_dim,), dtype=np.float32)
        emb[0] = msg.arousal
        emb[1] = msg.valence
        msg.embedding = [float(x) for x in emb.tolist()]

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ToneStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

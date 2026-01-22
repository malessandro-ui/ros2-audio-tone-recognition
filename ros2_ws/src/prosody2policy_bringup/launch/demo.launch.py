from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    tone_cfg = PathJoinSubstitution([FindPackageShare("prosody2policy_bringup"), "config", "tone.yaml"])
    adapter_cfg = PathJoinSubstitution([FindPackageShare("prosody2policy_bringup"), "config", "adapter.yaml"])

    return LaunchDescription([
        Node(
            package="prosody2policy_tone",
            executable="tone_stream_node",
            name="tone_stream_node",
            output="screen",
            parameters=[tone_cfg],
        ),
        Node(
            package="prosody2policy_adapter",
            executable="tone_to_style_node",
            name="tone_to_style_node",
            output="screen",
            parameters=[adapter_cfg],
        ),
    ])

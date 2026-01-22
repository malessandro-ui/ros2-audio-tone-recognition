from __future__ import annotations
from dataclasses import dataclass

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass(frozen=True)
class Thresholds:
    urgent_enter: float = 0.40
    urgent_exit: float = 0.25
    negative_enter: float = -0.30
    negative_exit: float = -0.15
    calm_arousal: float = -0.15

@dataclass(frozen=True)
class StyleMap:
    vel_neutral: float = 1.0
    vel_urgent: float = 1.4
    vel_backoff: float = 0.7
    acc_neutral: float = 1.0
    acc_urgent: float = 1.3
    acc_backoff: float = 0.8
    offset_neutral_m: float = 0.0
    offset_backoff_m: float = 0.10
    stiff_neutral: float = 1.0
    stiff_calm: float = 0.85
    stiff_urgent: float = 1.10
    grip_force_cap_default: float = 1.0

def compute_mode(current_mode: str, arousal: float, valence: float, th: Thresholds) -> str:
    """
    Mode machine with hysteresis.
    Modes: neutral | urgent | backoff | calm
    """
    if current_mode == "urgent":
        return "urgent" if arousal >= th.urgent_exit else "neutral"

    if current_mode == "backoff":
        return "backoff" if valence <= th.negative_exit else "neutral"

    # entering conditions from neutral/calm
    if arousal > th.urgent_enter:
        return "urgent"
    if valence < th.negative_enter:
        return "backoff"
    if arousal < th.calm_arousal and valence > -0.1:
        return "calm"
    return "neutral"

def params_for_mode(mode: str, sm: StyleMap) -> tuple[float, float, float, float, float]:
    """
    Returns (velocity_scale, accel_scale, approach_offset_m, stiffness_scale, grip_force_cap).
    """
    if mode == "urgent":
        return sm.vel_urgent, sm.acc_urgent, sm.offset_neutral_m, sm.stiff_urgent, sm.grip_force_cap_default
    if mode == "backoff":
        return sm.vel_backoff, sm.acc_backoff, sm.offset_backoff_m, sm.stiff_neutral, sm.grip_force_cap_default
    if mode == "calm":
        return 0.9, 0.9, sm.offset_neutral_m, sm.stiff_calm, sm.grip_force_cap_default
    return sm.vel_neutral, sm.acc_neutral, sm.offset_neutral_m, sm.stiff_neutral, sm.grip_force_cap_default

from prosody2policy_adapter.mode_logic import Thresholds, StyleMap, compute_mode, params_for_mode

def test_urgent_hysteresis():
    th = Thresholds(urgent_enter=0.4, urgent_exit=0.25)
    m = "neutral"
    m = compute_mode(m, arousal=0.41, valence=0.0, th=th)
    assert m == "urgent"
    m = compute_mode(m, arousal=0.30, valence=0.0, th=th)
    assert m == "urgent"
    m = compute_mode(m, arousal=0.20, valence=0.0, th=th)
    assert m == "neutral"

def test_backoff_hysteresis():
    th = Thresholds(negative_enter=-0.30, negative_exit=-0.15)
    m = "neutral"
    m = compute_mode(m, arousal=0.0, valence=-0.31, th=th)
    assert m == "backoff"
    m = compute_mode(m, arousal=0.0, valence=-0.20, th=th)
    assert m == "backoff"
    m = compute_mode(m, arousal=0.0, valence=-0.10, th=th)
    assert m == "neutral"

def test_params_mapping():
    sm = StyleMap()
    v, a, off, stiff, grip = params_for_mode("urgent", sm)
    assert v > 1.0 and a > 1.0
    assert off >= 0.0
    assert grip >= 0.0

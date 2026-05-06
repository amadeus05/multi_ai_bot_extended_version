import math


def build_entry_score(direction_prob: float, signal_gap: float, directional_proba_threshold: float) -> float:
    direction_prob = float(direction_prob)
    signal_gap = float(signal_gap)
    threshold = float(directional_proba_threshold)
    if not math.isfinite(direction_prob) or not math.isfinite(signal_gap) or not math.isfinite(threshold):
        return 0.0
    edge = max(0.0, direction_prob - threshold)
    return float(edge * 10.0 + signal_gap)


def resolve_directional_signal(
    p_long: float,
    p_short: float,
    directional_proba_threshold: float,
    min_signal_gap: float,
) -> tuple[int, float, float]:
    """
    Resolves directional signal based on model probabilities and thresholds.
    
    Returns:
        tuple: (signal_direction, direction_prob, signal_gap)
            signal_direction: 1 for LONG, -1 for SHORT, 0 for NEUTRAL
            direction_prob: probability of the chosen direction (or max prob if neutral)
            signal_gap: absolute difference between p_long and p_short
    """
    p_long = float(p_long)
    p_short = float(p_short)
    if not math.isfinite(p_long) or not math.isfinite(p_short):
        return 0, 0.5, 0.0

    signal_gap = abs(p_long - p_short)
    threshold = float(directional_proba_threshold)
    min_gap = max(0.0, float(min_signal_gap))

    long_edge = p_long - p_short
    short_edge = p_short - p_long
    if p_long > threshold and long_edge > min_gap:
        return 1, p_long, float(signal_gap)
    if p_short > threshold and short_edge > min_gap:
        return -1, p_short, float(signal_gap)
    return 0, max(p_long, p_short), float(signal_gap)

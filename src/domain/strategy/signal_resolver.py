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
    signal_gap = abs(float(p_long) - float(p_short))
    threshold = float(directional_proba_threshold)
    min_gap = float(min_signal_gap)

    if p_long >= threshold and (p_long - p_short) >= min_gap:
        return 1, float(p_long), float(signal_gap)
    if p_short >= threshold and (p_short - p_long) >= min_gap:
        return -1, float(p_short), float(signal_gap)
    return 0, max(float(p_long), float(p_short)), float(signal_gap)

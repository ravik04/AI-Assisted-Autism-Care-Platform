"""
Monitoring Agent — Time-series forecasting & anomaly detection.
================================================================
Stage 2 upgrade:
  • Exponential Weighted Moving Average (EWMA) for smoothed trend
  • Linear regression forecasting for next 3 sessions
  • CUSUM change-point detection for regression alerts
  • Z-score anomaly detection per modality
  • Velocity & acceleration metrics for trajectory characterization
  • Per-modality trend analysis with statistical significance
"""

import numpy as np
from scipy import stats as sp_stats


# ── Configuration ──────────────────────────────────────────────────────
EWMA_ALPHA = 0.4            # exponential smoothing factor
FORECAST_HORIZON = 3        # predict next N sessions
CUSUM_THRESHOLD = 0.25      # cumulative sum alert threshold
ANOMALY_Z_THRESHOLD = 2.0   # z-score threshold for anomaly flagging


def _ewma(values, alpha=EWMA_ALPHA):
    """Compute Exponential Weighted Moving Average."""
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _linear_forecast(values, horizon=FORECAST_HORIZON):
    """
    Fit a linear regression to score history and extrapolate forward.
    Returns forecasted values and (slope, intercept, r_squared).
    """
    n = len(values)
    if n < 2:
        return [], {
            "slope": 0.0,
            "intercept": round(float(values[0]) if values else 0.5, 4),
            "r_squared": 0.0,
            "p_value": 1.0,
            "std_err": 0.0,
        }

    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

    # Forecast
    future_x = np.arange(n, n + horizon, dtype=np.float64)
    forecast = np.clip(slope * future_x + intercept, 0.0, 1.0).tolist()

    return forecast, {
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_value ** 2), 4),
        "p_value": round(float(p_value), 4),
        "std_err": round(float(std_err), 6),
    }


def _cusum_detection(values, threshold=CUSUM_THRESHOLD):
    """
    Cumulative Sum (CUSUM) change-point detection.
    Detects sustained shifts in score trajectory.
    Returns list of change-point indices and direction.
    """
    if len(values) < 4:
        return []

    mean = np.mean(values)
    s_pos = 0.0
    s_neg = 0.0
    change_points = []

    for i, v in enumerate(values):
        s_pos = max(0, s_pos + (v - mean))
        s_neg = max(0, s_neg - (v - mean))
        if s_pos > threshold:
            change_points.append({"session": i, "direction": "increase"})
            s_pos = 0.0
        if s_neg > threshold:
            change_points.append({"session": i, "direction": "decrease"})
            s_neg = 0.0

    return change_points


def _compute_velocity_acceleration(values):
    """Compute velocity (rate of change) and acceleration (rate of rate of change)."""
    if len(values) < 2:
        return 0.0, 0.0
    velocity = float(values[-1] - values[-2])
    if len(values) < 3:
        return round(velocity, 4), 0.0
    prev_velocity = float(values[-2] - values[-3])
    acceleration = round(velocity - prev_velocity, 4)
    return round(velocity, 4), acceleration


def _classify_trajectory(slope, velocity, acceleration, n_sessions):
    """Rich trajectory classification based on statistical features."""
    if n_sessions < 2:
        return "insufficient_data"
    if n_sessions < 3:
        if velocity > 0.05:
            return "worsening"
        elif velocity < -0.05:
            return "improving"
        return "stable"

    # Multi-signal classification
    if slope > 0.02 and velocity > 0:
        if acceleration > 0.01:
            return "accelerating_worsening"
        return "worsening"
    elif slope < -0.02 and velocity < 0:
        if acceleration < -0.01:
            return "accelerating_improvement"
        return "improving"
    elif abs(slope) < 0.01:
        return "stable"
    elif slope > 0 and velocity <= 0:
        return "plateau_after_increase"
    elif slope < 0 and velocity >= 0:
        return "plateau_after_decrease"
    else:
        return "variable"


def _modality_trend_analysis(modality_history):
    """
    Enhanced per-modality trend analysis with statistical significance.
    """
    if not modality_history or len(modality_history) < 2:
        return {}

    all_mods = set()
    for h in modality_history:
        all_mods.update(h.keys())

    trends = {}
    for mod in all_mods:
        vals = [h.get(mod) for h in modality_history if h.get(mod) is not None]
        if len(vals) < 2:
            continue

        arr = np.array(vals, dtype=np.float64)
        n = len(arr)

        # EWMA smoothed trend
        smoothed = _ewma(arr.tolist())

        # Linear regression on this modality
        x = np.arange(n, dtype=np.float64)
        if n >= 3:
            slope, _, r_val, p_val, _ = sp_stats.linregress(x, arr)
        else:
            slope = float(arr[-1] - arr[0])
            r_val = 0
            p_val = 1.0

        # Trend direction with significance
        if p_val < 0.1:  # statistically suggestive
            if slope > 0.01:
                direction = "significant_increase"
            elif slope < -0.01:
                direction = "significant_decrease"
            else:
                direction = "stable"
        else:
            if arr[-1] > np.mean(arr) + 0.05:
                direction = "trending_up"
            elif arr[-1] < np.mean(arr) - 0.05:
                direction = "trending_down"
            else:
                direction = "stable"

        # Z-score anomaly detection for latest value
        if n >= 3:
            z_score = float((arr[-1] - np.mean(arr[:-1])) / max(np.std(arr[:-1]), 0.01))
            is_anomaly = abs(z_score) > ANOMALY_Z_THRESHOLD
        else:
            z_score = 0.0
            is_anomaly = False

        trends[mod] = {
            "current": round(float(arr[-1]), 4),
            "average": round(float(np.mean(arr)), 4),
            "smoothed": round(float(smoothed[-1]), 4),
            "slope": round(float(slope), 6),
            "trend": direction,
            "p_value": round(float(p_val), 4) if n >= 3 else None,
            "n_observations": n,
            "z_score": round(z_score, 2),
            "is_anomaly": is_anomaly,
        }

    return trends


def monitoring_agent(score_history, modality_history=None):
    """
    Analyse score trajectory using statistical forecasting,
    change-point detection, and per-modality anomaly detection.

    Returns backward-compatible keys plus new forecasting fields.
    """
    if not score_history:
        return {
            "trend": 0.0,
            "alert": "No data yet",
            "trajectory": "unknown",
            "sessions_count": 0,
        }

    scores = [float(s) for s in score_history]
    n = len(scores)

    # ── EWMA smoothed trend ────────────────────────────────────────
    smoothed = _ewma(scores)
    trend = round(smoothed[-1], 4)

    # ── Linear forecast ────────────────────────────────────────────
    forecast, regression = _linear_forecast(scores)

    # ── Velocity & acceleration ────────────────────────────────────
    velocity, acceleration = _compute_velocity_acceleration(scores)

    # ── Trajectory classification ──────────────────────────────────
    trajectory = _classify_trajectory(
        regression["slope"], velocity, acceleration, n
    )

    # ── CUSUM change-point detection ───────────────────────────────
    change_points = _cusum_detection(scores)

    # ── Alert level ────────────────────────────────────────────────
    if trajectory in ("accelerating_worsening",):
        alert = "URGENT: Accelerating deterioration detected — immediate clinical review"
    elif trajectory == "worsening" and trend > 0.6:
        alert = "WARNING: Worsening trajectory with elevated scores — consider clinical review"
    elif trajectory == "worsening":
        alert = "CAUTION: Increasing atypical patterns — closer monitoring recommended"
    elif trajectory in ("improving", "accelerating_improvement"):
        alert = "POSITIVE: Improving trajectory — continue current intervention"
    elif trend > 0.6:
        alert = "ELEVATED: Stable but elevated scores — clinical review recommended"
    elif change_points and change_points[-1]["direction"] == "increase":
        alert = "CHANGE DETECTED: Recent increase in risk scores — review latest data"
    elif trend < 0.3:
        alert = "LOW: Stable / low concern trajectory"
    else:
        alert = "Monitor developmental progression"

    # ── Per-modality trends ────────────────────────────────────────
    modality_trends = _modality_trend_analysis(modality_history)

    # ── Anomaly summary ────────────────────────────────────────────
    anomalies = [
        {"modality": mod, "z_score": t["z_score"]}
        for mod, t in modality_trends.items()
        if t.get("is_anomaly")
    ]

    result = {
        "trend": trend,
        "trajectory": trajectory,
        "alert": alert,
        "sessions_count": n,
        "velocity": velocity,
        "acceleration": acceleration,
        "forecast": {
            "next_sessions": [round(f, 4) for f in forecast],
            "horizon": FORECAST_HORIZON,
            "regression": regression,
        },
        "smoothed_history": [round(s, 4) for s in smoothed],
        "change_points": change_points,
        "anomalies": anomalies,
        "modality_trends": modality_trends,
    }

    return result
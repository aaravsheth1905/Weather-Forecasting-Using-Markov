"""
simulation.py — Weather Forecasting Using Markov
=================================
Monte Carlo simulation engine built on top of the Markov Chain transition matrix.

Monte Carlo simulation runs the Markov Chain N times and aggregates the results
to build a probability distribution over future weather states, including
confidence intervals and per-day forecasts.

Key difference from analytical Markov prediction:
- Analytical: exact probability (P^n)
- Monte Carlo: empirical distribution that captures simulation variance
  and makes complex multi-path statistics easy to compute

Author: Weather Forecasting Using Markov
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

from config import (
    WEATHER_STATES,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_NUM_SIMULATIONS,
    CONFIDENCE_LEVEL,
    STATE_ICONS,
)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Structured result container for Monte Carlo simulation output.

    Using a dataclass rather than a plain dict enforces a clear contract
    between the simulation engine and the visualization/UI layers.
    """
    start_state: str
    n_days: int
    n_simulations: int

    # Per-day probability distributions
    # day_probabilities[day_index] = {state: probability}
    day_probabilities: List[Dict[str, float]] = field(default_factory=list)

    # Most likely state per day
    most_likely_states: List[str] = field(default_factory=list)

    # Confidence score: fraction of simulations agreeing with modal forecast
    confidence_score: float = 0.0

    # Confidence interval (low, high) for the dominant state probability
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # All simulation paths (list of lists)
    all_paths: List[List[str]] = field(default_factory=list)

    # State frequency over entire forecast period
    overall_state_freq: Dict[str, float] = field(default_factory=dict)

    # Summary statistics
    summary: Dict[str, object] = field(default_factory=dict)


# ─── Single Path Simulation ───────────────────────────────────────────────────

def _simulate_single_path(
    start_state: str,
    n_steps: int,
    tm_dict: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> List[str]:
    """
    Run a single Markov Chain path using vectorized numpy sampling.

    Uses numpy's random generator rather than Python's random module for
    reproducibility (seeded RNG) and performance (vectorized operations).

    Args:
        start_state: Initial state.
        n_steps: Number of steps to simulate.
        tm_dict: Transition matrix as nested dict {from: {to: prob}}.
        rng: Seeded numpy random generator.

    Returns:
        List of state labels including the start state (length = n_steps + 1).
    """
    states_list = WEATHER_STATES
    path = [start_state]
    current = start_state

    for _ in range(n_steps):
        probs = tm_dict.get(current, {})
        if not probs or sum(probs.values()) == 0:
            # Absorbing state or unobserved: stay in current state
            path.append(current)
            continue

        # Build aligned probability array
        prob_vec = np.array([probs.get(s, 0.0) for s in states_list], dtype=float)

        # Renormalize (handles floating point drift)
        total = prob_vec.sum()
        if total > 0:
            prob_vec /= total
        else:
            prob_vec = np.ones(len(states_list)) / len(states_list)

        next_state = rng.choice(states_list, p=prob_vec)
        path.append(next_state)
        current = next_state

    return path


# ─── Monte Carlo Runner ───────────────────────────────────────────────────────

def run_monte_carlo(
    start_state: str,
    tm_df: pd.DataFrame,
    n_days: int = DEFAULT_FORECAST_DAYS,
    n_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: Optional[int] = 42,
    store_paths: bool = True,
) -> SimulationResult:
    """
    Run Monte Carlo simulation over the Markov transition matrix.

    Runs `n_simulations` independent Markov Chain paths, then aggregates
    to compute per-day probability distributions, confidence scores,
    and confidence intervals.

    Args:
        start_state: Starting weather state for all simulations.
        tm_df: Transition matrix DataFrame (rows = from, cols = to).
        n_days: Number of days to forecast.
        n_simulations: Number of Monte Carlo paths to run.
        seed: Random seed for reproducibility.
        store_paths: Whether to store all paths (memory-intensive for large N).

    Returns:
        SimulationResult dataclass with all computed statistics.
    """
    if start_state not in WEATHER_STATES:
        raise ValueError(
            f"Invalid start state '{start_state}'. "
            f"Choose from: {WEATHER_STATES}"
        )

    # Convert DataFrame to nested dict for fast lookup
    tm_dict = {
        row: {col: float(tm_df.loc[row, col]) for col in tm_df.columns}
        for row in tm_df.index
    }

    # Seeded RNG for reproducibility
    rng = np.random.default_rng(seed)

    # ── Run simulations ──────────────────────────────────────────────────────
    # Each path: [start_state, day_1_state, day_2_state, ..., day_N_state]
    all_paths: List[List[str]] = []

    for _ in range(n_simulations):
        path = _simulate_single_path(start_state, n_days, tm_dict, rng)
        all_paths.append(path)

    # ── Aggregate per-day distributions ──────────────────────────────────────
    # day 0 = start_state (known), days 1..n_days = forecast
    day_probabilities: List[Dict[str, float]] = []
    most_likely_states: List[str] = []

    for day_idx in range(1, n_days + 1):
        day_states = [path[day_idx] for path in all_paths]
        counts = Counter(day_states)
        total = sum(counts.values())

        probs = {
            state: round(counts.get(state, 0) / total, 4)
            for state in WEATHER_STATES
        }
        day_probabilities.append(probs)

        # Most likely state
        most_likely = max(probs, key=probs.get)
        most_likely_states.append(most_likely)

    # ── Confidence score ──────────────────────────────────────────────────────
    # Average probability assigned to the most likely state across all days
    confidence_score = round(
        np.mean([
            day_probabilities[i][most_likely_states[i]]
            for i in range(n_days)
        ]),
        4,
    )

    # ── Confidence interval for day 1 dominant state ─────────────────────────
    # Wilson score interval for binomial proportion
    # Formula: (p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
    from scipy.stats import norm as scipy_norm

    p_hat = day_probabilities[0][most_likely_states[0]]
    n = float(n_simulations)
    z = scipy_norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)  # z for 95% = 1.96

    center = p_hat + z ** 2 / (2 * n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    denom = 1 + z ** 2 / n

    ci_low = max(0.0, (center - margin) / denom)
    ci_high = min(1.0, (center + margin) / denom)
    confidence_interval = (round(float(ci_low), 4), round(float(ci_high), 4))

    # ── Overall state frequency across all simulations & all days ─────────────
    all_states_flat = [
        state
        for path in all_paths
        for state in path[1:]  # exclude start state
    ]
    total_obs = len(all_states_flat)
    state_counts = Counter(all_states_flat)
    overall_state_freq = {
        state: round(state_counts.get(state, 0) / total_obs, 4)
        for state in WEATHER_STATES
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    modal_forecast_sequence = " → ".join(
        [f"{STATE_ICONS.get(s, '')} {s}" for s in most_likely_states]
    )

    summary = {
        "start_state": start_state,
        "n_days": n_days,
        "n_simulations": n_simulations,
        "seed": seed,
        "confidence_score_pct": round(confidence_score * 100, 1),
        "ci_low_pct": round(ci_low * 100, 1),
        "ci_high_pct": round(ci_high * 100, 1),
        "modal_forecast": modal_forecast_sequence,
        "dominant_forecast_state": Counter(most_likely_states).most_common(1)[0][0],
    }

    return SimulationResult(
        start_state=start_state,
        n_days=n_days,
        n_simulations=n_simulations,
        day_probabilities=day_probabilities,
        most_likely_states=most_likely_states,
        confidence_score=confidence_score,
        confidence_interval=confidence_interval,
        all_paths=all_paths if store_paths else [],
        overall_state_freq=overall_state_freq,
        summary=summary,
    )


# ─── Forecast DataFrame ───────────────────────────────────────────────────────

def result_to_dataframe(result: SimulationResult) -> pd.DataFrame:
    """
    Convert SimulationResult to a structured DataFrame for export/display.

    Args:
        result: SimulationResult from run_monte_carlo().

    Returns:
        DataFrame with one row per forecast day.
    """
    rows = []
    for day_idx, (probs, state) in enumerate(
        zip(result.day_probabilities, result.most_likely_states), start=1
    ):
        rows.append({
            "Day": day_idx,
            "Predicted State": state,
            "Icon": STATE_ICONS.get(state, ""),
            "P(Sunny)": f"{probs.get('Sunny', 0) * 100:.1f}%",
            "P(Cloudy)": f"{probs.get('Cloudy', 0) * 100:.1f}%",
            "P(Rainy)": f"{probs.get('Rainy', 0) * 100:.1f}%",
            "Confidence": f"{probs.get(state, 0) * 100:.1f}%",
        })

    return pd.DataFrame(rows)


# ─── Sensitivity Analysis ─────────────────────────────────────────────────────

def sensitivity_by_start_state(
    tm_df: pd.DataFrame,
    n_days: int = 7,
    n_simulations: int = 500,
) -> Dict[str, SimulationResult]:
    """
    Run Monte Carlo for each possible starting state.

    Useful for the dashboard to show "what if today were Sunny/Cloudy/Rainy".

    Args:
        tm_df: Transition matrix.
        n_days: Forecast horizon.
        n_simulations: Simulations per starting state.

    Returns:
        Dict of {start_state: SimulationResult}
    """
    results = {}
    for state in WEATHER_STATES:
        results[state] = run_monte_carlo(
            start_state=state,
            tm_df=tm_df,
            n_days=n_days,
            n_simulations=n_simulations,
            seed=42,
            store_paths=False,
        )
    return results

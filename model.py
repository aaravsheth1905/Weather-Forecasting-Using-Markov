"""
model.py — WeatherSense AI
============================
Markov Chain model: transition counting, matrix normalization,
steady-state analysis, and matrix validation.

A first-order Markov Chain makes the Markov assumption: the probability of
the next state depends only on the current state, not on any prior history.
This is a reasonable approximation for short-range weather forecasting.

Author: WeatherSense AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import WEATHER_STATES, COLUMNS


# ─── Transition Counting ──────────────────────────────────────────────────────

def count_transitions(states: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Count state-to-state transitions in a sequence of weather states.

    This is a frequency table of observed transitions:
    "how many times did Cloudy → Rainy occur in the historical data?"

    Args:
        states: Ordered list of weather state labels.

    Returns:
        Nested dict: transition_counts[from_state][to_state] = count
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i in range(len(states) - 1):
        current = states[i]
        next_state = states[i + 1]
        counts[current][next_state] += 1

    return dict(counts)


# ─── Transition Matrix ────────────────────────────────────────────────────────

def build_transition_matrix(
    counts: Dict[str, Dict[str, int]],
    states: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw transition counts into a row-stochastic probability matrix.

    Each row sums to 1.0 (by the law of total probability — the chain must
    go *somewhere* from each state). Unobserved transitions get probability 0.

    Laplace smoothing (alpha=0) is used here — no smoothing by default since
    the dataset is small. This is a tunable parameter in a production system.

    Args:
        counts: Raw transition counts from count_transitions().
        states: Ordered list of state labels. Defaults to WEATHER_STATES.

    Returns:
        DataFrame where matrix[i][j] = P(next=j | current=i)
    """
    if states is None:
        states = WEATHER_STATES

    # Initialize with zeros for all state pairs
    matrix = {s: {t: 0.0 for t in states} for s in states}

    for from_state, transitions in counts.items():
        total = sum(transitions.values())
        if total == 0:
            continue
        for to_state, cnt in transitions.items():
            if from_state in matrix and to_state in matrix[from_state]:
                matrix[from_state][to_state] = round(cnt / total, 6)

    tm_df = pd.DataFrame(matrix).T  # rows = from, columns = to
    tm_df = tm_df[states]           # enforce consistent column ordering
    tm_df.index = pd.Index(states, name="From \\ To")

    return tm_df


# ─── Matrix Validation ────────────────────────────────────────────────────────

def validate_transition_matrix(tm_df: pd.DataFrame) -> Dict[str, object]:
    """
    Validate that the transition matrix is properly stochastic.

    A valid row-stochastic matrix satisfies:
    - All values in [0, 1]
    - Each row sums to 1.0 (within floating-point tolerance)

    Args:
        tm_df: Transition matrix DataFrame.

    Returns:
        Validation report dict with 'is_valid', 'row_sums', 'warnings'.
    """
    report = {"is_valid": True, "warnings": [], "row_sums": {}}

    for state in tm_df.index:
        row = tm_df.loc[state]
        row_sum = row.sum()
        report["row_sums"][state] = round(float(row_sum), 6)

        # Skip rows that are all zeros (state never observed)
        if row_sum == 0.0:
            report["warnings"].append(
                f"State '{state}' was never observed in training data. "
                "Predictions starting from this state may be unreliable."
            )
            continue

        if not np.isclose(row_sum, 1.0, atol=1e-4):
            report["is_valid"] = False
            report["warnings"].append(
                f"Row '{state}' sums to {row_sum:.6f}, not 1.0. "
                "Matrix normalization error."
            )

    # Check all values in [0, 1]
    vals = tm_df.values.flatten()
    if vals.min() < 0 or vals.max() > 1:
        report["is_valid"] = False
        report["warnings"].append(
            f"Transition probabilities outside [0, 1]: "
            f"min={vals.min():.4f}, max={vals.max():.4f}"
        )

    return report


# ─── Steady-State Distribution ────────────────────────────────────────────────

def compute_steady_state(tm_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the stationary (steady-state) distribution of the Markov Chain.

    The stationary distribution π satisfies: π = π · P
    This tells us the long-run proportion of time spent in each state —
    what the "average" weather looks like over a very long period.

    Solved using left eigenvectors of the transition matrix.

    Args:
        tm_df: Validated transition matrix DataFrame.

    Returns:
        Dict mapping state → steady-state probability.
    """
    states = list(tm_df.index)
    P = tm_df.values.astype(float)

    # Replace zero rows with uniform distribution to avoid singular matrix
    for i, row in enumerate(P):
        if row.sum() == 0:
            P[i] = np.ones(len(states)) / len(states)

    # Find left eigenvectors: solve π(P - I) = 0
    # Equivalent to right eigenvectors of P^T
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # The stationary distribution corresponds to eigenvalue = 1
    # Find the index closest to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = eigenvectors[:, idx].real

    # Normalize to sum to 1
    stationary = stationary / stationary.sum()

    return {state: round(float(prob), 4) for state, prob in zip(states, stationary)}


# ─── N-Step Transition ────────────────────────────────────────────────────────

def n_step_transition(tm_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute the n-step transition matrix: P^n

    P^n[i][j] = probability of being in state j after exactly n steps,
    starting from state i. Computed by matrix exponentiation.

    Args:
        tm_df: Base transition matrix.
        n: Number of steps.

    Returns:
        n-step transition matrix as DataFrame.
    """
    P = tm_df.values.astype(float)

    # Replace zero rows to avoid numerical issues
    for i, row in enumerate(P):
        if row.sum() == 0:
            P[i] = np.ones(len(P)) / len(P)

    P_n = np.linalg.matrix_power(P, n)

    return pd.DataFrame(
        P_n,
        index=tm_df.index,
        columns=tm_df.columns,
    ).round(6)


# ─── State Probability Vector ─────────────────────────────────────────────────

def state_probability_vector(
    initial_state: str,
    tm_df: pd.DataFrame,
    n: int,
) -> Dict[str, float]:
    """
    Compute the probability distribution over states after n steps,
    starting from a given initial state.

    This is the analytical Markov Chain prediction (no simulation needed).
    P(state_n = j | state_0 = initial_state) = (e_i · P^n)[j]

    Args:
        initial_state: Starting weather state.
        tm_df: Transition matrix.
        n: Number of steps ahead.

    Returns:
        Dict of {state: probability} after n steps.
    """
    states = list(tm_df.index)
    if initial_state not in states:
        raise ValueError(
            f"Unknown state '{initial_state}'. Valid states: {states}"
        )

    P_n = n_step_transition(tm_df, n).values
    state_idx = states.index(initial_state)
    probs = P_n[state_idx]

    return {state: round(float(p), 4) for state, p in zip(states, probs)}


# ─── Full Model Pipeline ──────────────────────────────────────────────────────

def build_markov_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Full Markov Chain model pipeline.

    Extracts state sequence → counts transitions → builds normalized matrix →
    validates → computes steady state.

    Args:
        df: Processed DataFrame with 'State' column.

    Returns:
        Tuple of:
        - transition_matrix (pd.DataFrame)
        - validation_report (dict)
        - steady_state (dict)
    """
    state_col = COLUMNS["state"]

    if state_col not in df.columns:
        raise ValueError(
            f"DataFrame missing '{state_col}' column. "
            "Run assign_weather_states() first."
        )

    states_seq = df[state_col].tolist()

    counts = count_transitions(states_seq)
    tm_df = build_transition_matrix(counts)
    validation = validate_transition_matrix(tm_df)
    steady_state = compute_steady_state(tm_df)

    return tm_df, validation, steady_state

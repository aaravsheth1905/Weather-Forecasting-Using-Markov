"""
utils.py — Weather Forecasting Using Markov
============================
Utility functions: data export, report generation, formatting helpers,
and a rule-based AI Weather Assistant (no API key required).

Author: Weather Forecasting Using Markov
"""

import io
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from config import (
    EXPORT_DATE_FORMAT,
    PREDICTION_CSV_NAME,
    REPORT_NAME,
    STATE_ICONS,
    COLORS,
    PROJECT_NAME,
    PROJECT_VERSION,
)


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def confidence_label(score: float) -> str:
    """Return a human-readable confidence label."""
    if score >= 0.80:
        return "🟢 High"
    elif score >= 0.60:
        return "🟡 Moderate"
    else:
        return "🔴 Low"


def state_badge(state: str) -> str:
    """Return state label with icon."""
    return f"{STATE_ICONS.get(state, '❓')} {state}"


# ─── CSV Export ───────────────────────────────────────────────────────────────

def forecast_to_csv_bytes(forecast_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    forecast_df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue()


def simulation_summary_to_csv_bytes(result) -> bytes:
    rows = []
    for day_idx, (probs, state) in enumerate(
        zip(result.day_probabilities, result.most_likely_states), start=1
    ):
        rows.append({
            "Day": day_idx,
            "Predicted State": state,
            "P(Sunny)": round(probs.get("Sunny", 0), 4),
            "P(Cloudy)": round(probs.get("Cloudy", 0), 4),
            "P(Rainy)": round(probs.get("Rainy", 0), 4),
            "Confidence": round(probs.get(state, 0), 4),
        })
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue()


# ─── Text Report ──────────────────────────────────────────────────────────────

def generate_forecast_report(result, summary: Dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        f"  {PROJECT_NAME} — Forecast Report",
        f"  Version {PROJECT_VERSION}",
        f"  Generated: {timestamp}",
        "=" * 60,
        "",
        "SIMULATION PARAMETERS",
        "-" * 40,
        f"  Starting State:     {state_badge(result.start_state)}",
        f"  Forecast Horizon:   {result.n_days} days",
        f"  Simulations Run:    {result.n_simulations:,}",
        f"  Confidence Score:   {pct(result.confidence_score)}",
        f"  Confidence Level:   {confidence_label(result.confidence_score)}",
        f"  95% CI (Day 1):     [{pct(result.confidence_interval[0])} – {pct(result.confidence_interval[1])}]",
        "",
        "DAY-BY-DAY FORECAST",
        "-" * 40,
    ]

    for day_idx, (probs, state) in enumerate(
        zip(result.day_probabilities, result.most_likely_states), start=1
    ):
        p_sunny = pct(probs.get("Sunny", 0))
        p_cloudy = pct(probs.get("Cloudy", 0))
        p_rainy = pct(probs.get("Rainy", 0))
        lines.append(
            f"  Day {day_idx:2d}:  {state_badge(state):20s} "
            f"[☀️ {p_sunny}  ⛅ {p_cloudy}  🌧️ {p_rainy}]"
        )

    lines += [
        "",
        "LONG-RUN DISTRIBUTION",
        "-" * 40,
    ]
    for state, freq in result.overall_state_freq.items():
        lines.append(f"  {state_badge(state):20s}  {pct(freq)}")

    lines += [
        "",
        "DATASET OVERVIEW",
        "-" * 40,
        f"  Total records:      {summary.get('n_rows', 'N/A')}",
        f"  Days in dataset:    {summary.get('n_days', 'N/A')}",
        f"  Temperature range:  {summary.get('temp_min', 'N/A')} – {summary.get('temp_max', 'N/A')} °C",
        f"  Humidity range:     {summary.get('humidity_min', 'N/A')} – {summary.get('humidity_max', 'N/A')} %",
        "",
        "=" * 60,
        "  Powered by Weather Forecasting Using Markov",
        "  Markov Chains + Monte Carlo Simulation",
        "=" * 60,
    ]

    return "\n".join(lines)


def report_to_bytes(report_text: str) -> bytes:
    return report_text.encode("utf-8")


# ─── AI Context Builder ───────────────────────────────────────────────────────

def build_ai_context(
    result=None,
    tm_df=None,
    steady_state: Optional[Dict] = None,
    summary: Optional[Dict] = None,
) -> str:
    context_parts = []

    if tm_df is not None:
        tm_str = tm_df.round(3).to_string()
        context_parts.append(f"TRANSITION MATRIX:\n{tm_str}")

    if steady_state:
        ss_str = ", ".join([f"{s}: {v:.1%}" for s, v in steady_state.items()])
        context_parts.append(f"STEADY-STATE DISTRIBUTION: {ss_str}")

    if result is not None:
        forecast_lines = []
        for i, (probs, state) in enumerate(
            zip(result.day_probabilities, result.most_likely_states), start=1
        ):
            forecast_lines.append(
                f"  Day {i}: {state} "
                f"(Sunny={probs.get('Sunny', 0):.1%}, "
                f"Cloudy={probs.get('Cloudy', 0):.1%}, "
                f"Rainy={probs.get('Rainy', 0):.1%})"
            )
        context_parts.append(
            f"CURRENT FORECAST (start={result.start_state}, "
            f"days={result.n_days}, sims={result.n_simulations:,}, "
            f"confidence={result.confidence_score:.1%}):\n" + "\n".join(forecast_lines)
        )

    if summary:
        context_parts.append(
            f"DATASET: {summary.get('n_rows')} hourly records, "
            f"{summary.get('n_days')} days, "
            f"dominant state: {summary.get('dominant_state')}"
        )

    return "\n\n".join(context_parts)


# ─── Rule-Based AI Weather Assistant (No API Key Required) ───────────────────

def ask_weather_assistant(
    user_question: str,
    context: str = "",
    conversation_history: Optional[list] = None,
    result=None,
    tm_df=None,
    steady_state: Optional[Dict] = None,
) -> str:
    """
    Rule-based assistant that answers questions about the weather model.
    Uses the actual transition matrix, forecast results, and steady-state
    data to give grounded, specific answers — no API key needed.
    """
    q = user_question.lower().strip()

    # Parse real values from context if available
    forecast_day1_state = None
    forecast_day1_prob = None
    start_state_val = None
    confidence_val = None

    if result:
        forecast_day1_state = result.most_likely_states[0] if result.most_likely_states else None
        if forecast_day1_state:
            forecast_day1_prob = result.day_probabilities[0].get(forecast_day1_state, 0)
        start_state_val = result.start_state
        confidence_val = result.confidence_score

    # ── Markov Chain questions ──
    if any(k in q for k in ["markov", "how does it work", "how does the model", "chain", "what is markov"]):
        return (
            "**Markov Chain — How it works:**\n\n"
            "A Markov Chain models a system that moves between states over time. "
            "The key rule (called the **Markov property**) is:\n\n"
            "> *The next state depends only on the current state — not on any history before it.*\n\n"
            "In this app:\n"
            "- States = ☀️ Sunny, ⛅ Cloudy, 🌧️ Rainy\n"
            "- We count how often each state transitions to each other state in the historical data\n"
            "- Those counts are normalized into **transition probabilities** (each row sums to 1.0)\n"
            "- For example: if it's Rainy today, there's an 81% chance it stays Rainy tomorrow\n\n"
            "The result is the **Transition Matrix** you can see on the Markov Chain dashboard."
        )

    # ── Monte Carlo questions ──
    if any(k in q for k in ["monte carlo", "simulation", "why simulate", "what does simulation"]):
        n_sims = result.n_simulations if result else 1000
        return (
            f"**Monte Carlo Simulation — What it does:**\n\n"
            f"Instead of calculating probabilities mathematically, Monte Carlo runs **{n_sims:,} random experiments**:\n\n"
            "1. Start from today's weather state\n"
            "2. At each step, randomly pick the next state using the transition probabilities\n"
            "3. Record the full path (e.g., Rainy → Rainy → Cloudy → Sunny)\n"
            f"4. Repeat {n_sims:,} times\n"
            "5. Count how often each state appears on each future day\n\n"
            "The result is a **probability distribution** — not just one prediction, but a full picture of possible futures. "
            "More simulations = more stable and accurate estimates."
        )

    # ── Why Rainy / Why prediction question ──
    if any(k in q for k in ["why rainy", "why is tomorrow", "why predicted", "why is it", "explain the prediction", "why rainy"]):
        if result and tm_df is not None:
            rainy_to_rainy = float(tm_df.loc["Rainy", "Rainy"]) if "Rainy" in tm_df.index else 0.81
            rainy_to_cloudy = float(tm_df.loc["Rainy", "Cloudy"]) if "Rainy" in tm_df.index else 0.19
            state = start_state_val or "Rainy"
            day1 = forecast_day1_state or "Rainy"
            prob = forecast_day1_prob or 0.79
            return (
                f"**Why is tomorrow predicted to be {day1}?**\n\n"
                f"Today's state is **{state}**. Looking at the transition matrix:\n\n"
                f"- P(Rainy → Rainy) = **{rainy_to_rainy:.1%}** ← very high persistence\n"
                f"- P(Rainy → Cloudy) = **{rainy_to_cloudy:.1%}**\n"
                f"- P(Rainy → Sunny) = 0.0%\n\n"
                f"Because Rainy has an **{rainy_to_rainy:.0%} self-transition probability**, "
                f"the model strongly predicts it stays Rainy. "
                f"Across 1,000 simulations, **{prob:.1%}** of paths had {day1} on Day 1 — "
                f"which is why that's the forecast."
            )
        return (
            "The prediction is driven by the **transition probabilities** learned from historical data. "
            "Rainy weather has a very high self-persistence probability (~81%), so once it's Rainy, "
            "the model predicts it will stay Rainy in the near term. Run a forecast first for specific numbers."
        )

    # ── Reliability / Confidence ──
    if any(k in q for k in ["reliable", "confidence", "how accurate", "trust", "how good"]):
        if confidence_val is not None:
            label = confidence_label(confidence_val)
            ci_low = result.confidence_interval[0] * 100
            ci_high = result.confidence_interval[1] * 100
            return (
                f"**Forecast Reliability:**\n\n"
                f"- Confidence score: **{confidence_val:.1%}** ({label})\n"
                f"- 95% Confidence Interval (Day 1): **{ci_low:.0f}% – {ci_high:.0f}%**\n\n"
                "This means if we ran this forecast 100 times with slightly different random seeds, "
                f"the Day 1 prediction would fall in that range 95% of the time.\n\n"
                "**Limitations to keep in mind:**\n"
                "- The dataset covers only 8 days — a larger dataset gives more reliable transitions\n"
                "- The model assumes weather only depends on the current state (Markov property)\n"
                "- No seasonal effects, temperature trends, or external factors are modeled"
            )
        return (
            "Reliability depends on the confidence score shown after running a simulation. "
            "Go to **🎲 Monte Carlo Simulation** and run it first — then ask again for specific numbers."
        )

    # ── Transition probability ──
    if any(k in q for k in ["transition", "probability", "matrix", "p(", "what is the probability"]):
        if tm_df is not None:
            lines = ["**Transition Probabilities (from historical data):**\n"]
            for from_state in tm_df.index:
                for to_state in tm_df.columns:
                    prob = float(tm_df.loc[from_state, to_state])
                    if prob > 0.001:
                        icon_f = STATE_ICONS.get(from_state, "")
                        icon_t = STATE_ICONS.get(to_state, "")
                        lines.append(f"- {icon_f} {from_state} → {icon_t} {to_state}: **{prob:.1%}**")
            lines.append("\nEach row sums to 100% — the chain must always go *somewhere*.")
            return "\n".join(lines)
        return (
            "The transition matrix shows the probability of moving from one weather state to another. "
            "For example, P(Rainy → Rainy) ≈ 81% in this dataset. "
            "See the **🔄 Markov Chain** dashboard for the full interactive heatmap."
        )

    # ── Steady state ──
    if any(k in q for k in ["steady state", "stationary", "long run", "long-run", "equilibrium", "steady-state", "steady"]):
        if steady_state:
            lines = ["**Steady-State (Long-Run) Distribution:**\n"]
            for state, prob in steady_state.items():
                icon = STATE_ICONS.get(state, "")
                lines.append(f"- {icon} {state}: **{prob:.1%}**")
            lines.append(
                "\nThis means: over a very long period, the weather will be Cloudy ~46% of the time, "
                "Rainy ~32%, and Sunny ~22% — regardless of what today's weather is. "
                "It's the 'climate fingerprint' of this dataset."
            )
            return "\n".join(lines)
        return (
            "The steady-state distribution tells you the long-run proportion of time spent in each state. "
            "It's computed from the eigenvectors of the transition matrix. "
            "See the **⚖️ Steady State** tab in the Markov Chain dashboard."
        )

    # ── Current forecast summary ──
    if any(k in q for k in ["forecast", "predict", "tomorrow", "next few days", "summary"]):
        if result:
            lines = [f"**Current Forecast (starting from {state_badge(result.start_state)}):**\n"]
            for i, (state, probs) in enumerate(
                zip(result.most_likely_states, result.day_probabilities), start=1
            ):
                prob = probs.get(state, 0)
                icon = STATE_ICONS.get(state, "")
                lines.append(f"- Day {i}: {icon} **{state}** ({prob:.1%} probability)")
            lines.append(f"\nOverall confidence: **{confidence_val:.1%}** {confidence_label(confidence_val)}")
            return "\n".join(lines)
        return (
            "No forecast has been run yet. Go to **🔮 Weather Forecast** or "
            "**🎲 Monte Carlo Simulation**, configure your parameters, and run the simulation. "
            "Then come back here and ask me about the results!"
        )

    # ── Dataset questions ──
    if any(k in q for k in ["dataset", "data", "csv", "how many", "records", "rows"]):
        return (
            "**About the Dataset:**\n\n"
            "- **192 rows** of hourly weather sensor readings\n"
            "- **8 days** of data (Day 2 through Day 9)\n"
            "- **6 features:** Temperature (°C), Relative Humidity (%), Wind Speed (km/h), "
            "Wind Direction (°), Day, Hour\n\n"
            "Weather states are derived from **Relative Humidity** thresholds:\n"
            "- 🌧️ Rainy: ≥ 78%\n"
            "- ⛅ Cloudy: 69–77%\n"
            "- ☀️ Sunny: < 69%\n\n"
            "State distribution: ⛅ Cloudy 46.9%, 🌧️ Rainy 30.7%, ☀️ Sunny 22.4%"
        )

    # ── Help / default ──
    if any(k in q for k in ["help", "what can you", "what questions", "hi", "hello"]):
        return (
            "Hi! I'm the **Weather Forecasting Using Markov Assistant**. I can explain:\n\n"
            "- 🔄 How the Markov Chain works\n"
            "- 🎲 What Monte Carlo Simulation does\n"
            "- 🔮 Why a specific state was predicted\n"
            "- 📊 What the transition probabilities mean\n"
            "- ⚖️ What the steady-state distribution tells us\n"
            "- 📈 How reliable the forecast is\n"
            "- 📂 About the dataset\n\n"
            "Try asking: *'Why is tomorrow predicted to be Rainy?'* or *'How does the Markov Chain work?'*"
        )

    # ── Fallback ──
    return (
        "Great question! Here's what I can help you with:\n\n"
        "- **'How does the Markov Chain work?'** — model explanation\n"
        "- **'Why is tomorrow predicted to be Rainy?'** — prediction reasoning\n"
        "- **'What does Monte Carlo simulation do?'** — simulation explanation\n"
        "- **'How reliable is this prediction?'** — confidence & CI\n"
        "- **'What are the transition probabilities?'** — matrix breakdown\n"
        "- **'What is the steady-state distribution?'** — long-run analysis\n\n"
        "Try rephrasing your question using one of those topics!"
    )


# ─── Streamlit CSS ────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0D0D1A 0%, #0F1628 50%, #0D0D1A 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1628 0%, #1A1A2E 100%);
    border-right: 1px solid #2E2E4E;
}
[data-testid="stMetric"] {
    background: rgba(30, 30, 60, 0.6);
    border: 1px solid #2E2E4E;
    border-radius: 12px;
    padding: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #1E90FF;
}
[data-testid="stMetricLabel"] {
    color: #9999BB !important;
    font-size: 0.85rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #E8E8F0 !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(30, 144, 255, 0.15) !important;
    color: #1E90FF !important;
    border-bottom: 2px solid #1E90FF !important;
}
[data-testid="stExpander"] {
    border: 1px solid #2E2E4E;
    border-radius: 10px;
    background: rgba(20, 20, 40, 0.5);
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1E90FF, #0066CC) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(30,144,255,0.3) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(30,144,255,0.45) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2E2E4E;
    border-radius: 10px;
    overflow: hidden;
}
.ai-chat-bubble-user {
    background: rgba(30, 144, 255, 0.15);
    border: 1px solid #1E90FF44;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #E8E8F0;
    font-size: 0.95rem;
}
.ai-chat-bubble-assistant {
    background: rgba(0, 206, 209, 0.08);
    border: 1px solid #00CED144;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #E8E8F0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.ai-assistant-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #00CED1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E8E8F0;
    border-bottom: 2px solid #1E90FF33;
    padding-bottom: 8px;
    margin-bottom: 20px;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D0D1A; }
::-webkit-scrollbar-thumb { background: #2E2E4E; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1E90FF; }
</style>
"""

"""
visualizations.py — Weather Forecasting Using Markov
======================================
All Plotly visualization functions for the dashboard.

Each function returns a Plotly Figure object — no Streamlit calls here.
This separation means charts can be:
- Rendered in Streamlit (st.plotly_chart)
- Saved as PNG/HTML (fig.write_image / fig.write_html)
- Tested independently

Design principle: consistent color palette, consistent template,
consistent font sizing. Every chart follows the same aesthetic.

Author: Weather Forecasting Using Markov
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from config import (
    COLORS,
    STATE_COLORS,
    WEATHER_STATES,
    PLOTLY_TEMPLATE,
    CHART_HEIGHT,
    CHART_HEIGHT_TALL,
    CHART_HEIGHT_HEATMAP,
    COLUMNS,
    STATE_ICONS,
)

# Shared layout defaults
_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    height=CHART_HEIGHT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"], size=13),
    margin=dict(l=40, r=30, t=50, b=40),
)


def _apply_defaults(fig: go.Figure, **overrides) -> go.Figure:
    """Apply consistent layout defaults to any figure."""
    layout = {**_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**layout)
    return fig


# ─── Dataset Exploration ──────────────────────────────────────────────────────

def temperature_trend(df: pd.DataFrame) -> go.Figure:
    """Hourly temperature trend with rolling mean overlay."""
    temp_col = COLUMNS["temperature"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[temp_col],
        mode="lines",
        name="Temperature",
        line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
        opacity=0.6,
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["Temp_Rolling3h"],
        mode="lines",
        name="3h Rolling Mean",
        line=dict(color=COLORS["accent"], width=2.5),
    ))

    _apply_defaults(
        fig,
        title="🌡️ Temperature Trend (°C)",
        xaxis_title="Hour Index",
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def humidity_trend(df: pd.DataFrame) -> go.Figure:
    """Hourly humidity with threshold lines for state classification."""
    hum_col = COLUMNS["humidity"]
    fig = go.Figure()

    # Background band for Rainy zone
    fig.add_hrect(
        y0=78, y1=df[hum_col].max() + 2,
        fillcolor=STATE_COLORS["Rainy"], opacity=0.08,
        layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=69, y1=78,
        fillcolor=STATE_COLORS["Cloudy"], opacity=0.08,
        layer="below", line_width=0,
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[hum_col],
        mode="lines",
        name="Humidity",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba(30,144,255,0.1)",
    ))

    # Threshold lines
    for threshold, label, color in [
        (78, "Rainy threshold", STATE_COLORS["Rainy"]),
        (69, "Cloudy threshold", STATE_COLORS["Cloudy"]),
    ]:
        fig.add_hline(
            y=threshold, line_dash="dash",
            line_color=color, opacity=0.8,
            annotation_text=label,
            annotation_position="right",
        )

    _apply_defaults(
        fig,
        title="💧 Relative Humidity Trend (%)",
        xaxis_title="Hour Index",
        yaxis_title="Relative Humidity (%)",
    )
    return fig


def wind_speed_trend(df: pd.DataFrame) -> go.Figure:
    """Wind speed over time with max annotation."""
    wind_col = COLUMNS["wind_speed"]
    max_idx = df[wind_col].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[wind_col],
        mode="lines",
        name="Wind Speed",
        line=dict(color=COLORS["secondary"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba(0,206,209,0.1)",
    ))

    # Annotate peak
    fig.add_annotation(
        x=max_idx,
        y=df.loc[max_idx, wind_col],
        text=f"Peak: {df.loc[max_idx, wind_col]:.1f} km/h",
        showarrow=True, arrowhead=2,
        font=dict(color=COLORS["accent"]),
        arrowcolor=COLORS["accent"],
    )

    _apply_defaults(
        fig,
        title="💨 Wind Speed Trend (km/h)",
        xaxis_title="Hour Index",
        yaxis_title="Wind Speed (km/h)",
    )
    return fig


def weather_state_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of weather state proportions."""
    state_col = COLUMNS["state"]
    counts = df[state_col].value_counts().reset_index()
    counts.columns = ["State", "Count"]

    colors = [STATE_COLORS.get(s, COLORS["primary"]) for s in counts["State"]]

    fig = go.Figure(go.Pie(
        labels=[f"{STATE_ICONS.get(s, '')} {s}" for s in counts["State"]],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["surface"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>hours",
        x=0.5, y=0.5,
        font=dict(size=16, color=COLORS["text_primary"]),
        showarrow=False,
    )

    _apply_defaults(fig, title="🌤️ Weather State Distribution", height=380)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric features."""
    numeric_cols = [
        COLUMNS["temperature"], COLUMNS["humidity"],
        COLUMNS["wind_speed"], COLUMNS["wind_direction"],
    ]
    corr = df[numeric_cols].corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))

    _apply_defaults(
        fig,
        title="🔗 Feature Correlation Heatmap",
        height=CHART_HEIGHT_HEATMAP,
    )
    return fig


def daily_weather_summary(daily_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of hours in each weather state per day."""
    fig = go.Figure()

    for state, color in STATE_COLORS.items():
        col = f"Hours_{state}"
        if col in daily_df.columns:
            fig.add_trace(go.Bar(
                name=f"{STATE_ICONS.get(state, '')} {state}",
                x=daily_df[COLUMNS["day"]].astype(str),
                y=daily_df[col],
                marker_color=color,
                opacity=0.85,
            ))

    fig.update_layout(barmode="stack")
    _apply_defaults(
        fig,
        title="📅 Daily Weather State Breakdown (hours)",
        xaxis_title="Day",
        yaxis_title="Hours",
    )
    return fig


# ─── Markov Chain ─────────────────────────────────────────────────────────────

def transition_matrix_heatmap(tm_df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap of the Markov transition probability matrix."""
    labels = [f"{STATE_ICONS.get(s, '')} {s}" for s in tm_df.index]

    fig = go.Figure(go.Heatmap(
        z=tm_df.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#0D0D1A"],
            [0.3, "#1E3A6E"],
            [0.6, COLORS["primary"]],
            [1.0, "#00CED1"],
        ],
        zmin=0, zmax=1,
        text=tm_df.values.round(3),
        texttemplate="%{text:.3f}",
        textfont=dict(size=14, color="white"),
        hovertemplate=(
            "<b>From:</b> %{y}<br>"
            "<b>To:</b> %{x}<br>"
            "<b>P =</b> %{z:.4f}<extra></extra>"
        ),
        colorbar=dict(title="P", tickformat=".2f"),
    ))

    _apply_defaults(
        fig,
        title="🔄 Transition Probability Matrix",
        height=CHART_HEIGHT_HEATMAP,
        xaxis_title="Next State",
        yaxis_title="Current State",
    )
    return fig


def markov_chain_diagram(tm_df: pd.DataFrame) -> go.Figure:
    """
    Circular state diagram with edge weights as transition probabilities.
    Uses Plotly scatter + annotations to draw the network.
    """
    import math

    states = list(tm_df.index)
    n = len(states)
    radius = 1.0

    # Node positions on a circle
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    positions = {s: (radius * math.cos(a), radius * math.sin(a)) for s, a in zip(states, angles)}

    fig = go.Figure()

    # Draw edges
    for from_state, (fx, fy) in positions.items():
        for to_state in states:
            prob = float(tm_df.loc[from_state, to_state])
            if prob < 0.01:
                continue

            tx, ty = positions[to_state]
            # Slightly offset for self-loops
            if from_state == to_state:
                continue  # handled separately

            # Edge line
            fig.add_trace(go.Scatter(
                x=[fx, tx], y=[fy, ty],
                mode="lines",
                line=dict(
                    width=max(1, prob * 8),
                    color=STATE_COLORS.get(from_state, COLORS["primary"]),
                ),
                opacity=0.6 + prob * 0.4,
                showlegend=False,
                hoverinfo="skip",
            ))

            # Probability label at midpoint
            mx, my = (fx + tx) / 2, (fy + ty) / 2
            fig.add_annotation(
                x=mx, y=my,
                text=f"{prob:.2f}",
                font=dict(size=11, color=COLORS["text_secondary"]),
                showarrow=False,
                bgcolor=COLORS["surface"],
            )

    # Draw nodes
    for state, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=70,
                color=STATE_COLORS.get(state, COLORS["primary"]),
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            text=[f"{STATE_ICONS.get(state, '')} {state}"],
            textfont=dict(size=13, color="white"),
            textposition="middle center",
            showlegend=False,
            hovertemplate=f"<b>{state}</b><br>Self-loop: {float(tm_df.loc[state, state]):.3f}<extra></extra>",
        ))

    fig.update_xaxes(visible=False, range=[-1.6, 1.6])
    fig.update_yaxes(visible=False, range=[-1.6, 1.6])

    _apply_defaults(
        fig,
        title="🔵 Markov Chain State Diagram",
        height=480,
    )
    return fig


# ─── Simulation / Forecast ────────────────────────────────────────────────────

def forecast_timeline(result) -> go.Figure:
    """
    Stacked area chart showing probability distribution per forecast day.
    """
    days = list(range(1, result.n_days + 1))

    fig = go.Figure()

    # Convert hex colors to rgba for fillcolor (required by newer Plotly versions)
    state_rgba = {
        "Sunny":  "rgba(255, 215,   0, 0.7)",
        "Cloudy": "rgba(135, 206, 235, 0.7)",
        "Rainy":  "rgba( 30, 144, 255, 0.7)",
    }

    for state in WEATHER_STATES:
        probs = [d.get(state, 0) * 100 for d in result.day_probabilities]
        fig.add_trace(go.Scatter(
            x=days,
            y=probs,
            name=f"{STATE_ICONS.get(state, '')} {state}",
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=STATE_COLORS.get(state)),
            fillcolor=state_rgba.get(state, "rgba(30,144,255,0.7)"),
            hovertemplate=f"<b>{state}</b><br>Day %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    # Overlay most likely state markers
    modal_probs = [
        result.day_probabilities[i][result.most_likely_states[i]] * 100
        for i in range(result.n_days)
    ]
    fig.add_trace(go.Scatter(
        x=days,
        y=modal_probs,
        mode="markers+text",
        marker=dict(size=10, color=COLORS["accent"], symbol="diamond"),
        text=[STATE_ICONS.get(s, "") for s in result.most_likely_states],
        textposition="top center",
        name="Most Likely",
        showlegend=True,
        hoverinfo="skip",
    ))

    _apply_defaults(
        fig,
        title="📊 Forecast Probability Timeline",
        xaxis_title="Forecast Day",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 101]),
        xaxis=dict(tickmode="linear", dtick=1),
        legend=dict(orientation="h", y=1.12),
        height=CHART_HEIGHT_TALL,
    )
    return fig


def simulation_distribution(result) -> go.Figure:
    """
    Bar chart of overall state frequency across all simulation paths.
    """
    freq = result.overall_state_freq
    states = list(freq.keys())
    values = [freq[s] * 100 for s in states]
    colors = [STATE_COLORS.get(s, COLORS["primary"]) for s in states]

    fig = go.Figure(go.Bar(
        x=[f"{STATE_ICONS.get(s, '')} {s}" for s in states],
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Frequency: %{y:.1f}%<extra></extra>",
    ))

    _apply_defaults(
        fig,
        title=f"🎲 Monte Carlo State Frequency ({result.n_simulations:,} simulations)",
        xaxis_title="Weather State",
        yaxis_title="Frequency (%)",
        yaxis=dict(range=[0, max(values) * 1.2]),
        height=380,
    )
    return fig


def simulation_path_sample(result, n_paths: int = 20) -> go.Figure:
    """
    Show a sample of Monte Carlo paths as a spaghetti plot.
    """
    state_to_num = {s: i for i, s in enumerate(WEATHER_STATES)}
    num_to_state = {i: s for i, s in enumerate(WEATHER_STATES)}

    days = list(range(result.n_days + 1))

    fig = go.Figure()

    # Draw sample paths (semi-transparent)
    sample_size = min(n_paths, len(result.all_paths))
    for i, path in enumerate(result.all_paths[:sample_size]):
        nums = [state_to_num[s] for s in path]
        fig.add_trace(go.Scatter(
            x=days, y=nums,
            mode="lines",
            line=dict(width=1, color=COLORS["primary"]),
            opacity=0.2,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Overlay most likely path
    modal_path = [result.start_state] + result.most_likely_states
    modal_nums = [state_to_num[s] for s in modal_path]
    fig.add_trace(go.Scatter(
        x=days,
        y=modal_nums,
        mode="lines+markers",
        name="Modal Forecast",
        line=dict(width=3, color=COLORS["accent"]),
        marker=dict(size=10),
        hovertemplate="Day %{x}: <b>%{text}</b><extra></extra>",
        text=modal_path,
    ))

    fig.update_yaxes(
        tickvals=list(range(len(WEATHER_STATES))),
        ticktext=[f"{STATE_ICONS.get(s, '')} {s}" for s in WEATHER_STATES],
    )

    _apply_defaults(
        fig,
        title=f"🌀 Simulation Paths Sample ({sample_size} of {result.n_simulations:,})",
        xaxis_title="Forecast Day",
        yaxis_title="Weather State",
        xaxis=dict(tickmode="linear", dtick=1),
        height=CHART_HEIGHT_TALL,
    )
    return fig


def steady_state_bar(steady_state: Dict[str, float]) -> go.Figure:
    """Bar chart of stationary distribution."""
    states = WEATHER_STATES
    values = [steady_state.get(s, 0) * 100 for s in states]
    colors = [STATE_COLORS.get(s, COLORS["primary"]) for s in states]

    fig = go.Figure(go.Bar(
        x=[f"{STATE_ICONS.get(s, '')} {s}" for s in states],
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))

    _apply_defaults(
        fig,
        title="⚖️ Long-Run Stationary Distribution",
        xaxis_title="Weather State",
        yaxis_title="Long-run Probability (%)",
        yaxis=dict(range=[0, max(values) * 1.25]),
        height=350,
    )
    return fig

"""
app.py — Weather Forecasting Using Markov
==========================
Main Streamlit application entry point.

Architecture:
  app.py (UI layer)
    ├── preprocessing.py  (data layer)
    ├── model.py          (Markov Chain layer)
    ├── simulation.py     (Monte Carlo layer)
    ├── visualizations.py (chart layer)
    └── utils.py          (helpers, export, AI)

Run locally:
    streamlit run app.py

Author: Weather Forecasting Using Markov
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Module imports ────────────────────────────────────────────────────────────
from config import (
    PROJECT_NAME, PROJECT_SUBTITLE, WEATHER_STATES, STATE_ICONS, STATE_COLORS,
    DEFAULT_FORECAST_DAYS, DEFAULT_NUM_SIMULATIONS,
    MIN_FORECAST_DAYS, MAX_FORECAST_DAYS, MIN_SIMULATIONS, MAX_SIMULATIONS,
    COLORS, DATA_PATH,
)
from preprocessing import run_preprocessing_pipeline
from model import build_markov_model, state_probability_vector
from simulation import run_monte_carlo, result_to_dataframe, sensitivity_by_start_state
from visualizations import (
    temperature_trend, humidity_trend, wind_speed_trend,
    weather_state_distribution, correlation_heatmap, daily_weather_summary,
    transition_matrix_heatmap, markov_chain_diagram, steady_state_bar,
    forecast_timeline, simulation_distribution, simulation_path_sample,
)
from utils import (
    CUSTOM_CSS, pct, confidence_label, state_badge,
    forecast_to_csv_bytes, simulation_summary_to_csv_bytes,
    generate_forecast_report, report_to_bytes,
    build_ai_context, ask_weather_assistant,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/yourusername/Weather Forecasting Using MarkovAI",
        "Report a bug": "https://github.com/yourusername/Weather Forecasting Using MarkovAI/issues",
        "About": f"**{PROJECT_NAME}** — {PROJECT_SUBTITLE}",
    },
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Data Loading ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_pipeline(path: str = DATA_PATH):
    return run_preprocessing_pipeline(path)


@st.cache_data(show_spinner="Building Markov model...")
def load_model(_df: pd.DataFrame):
    return build_markov_model(_df)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render sidebar navigation and return the selected page."""
    with st.sidebar:
        # Logo / Header
        st.markdown(f"""
        <div style="text-align:center; padding: 16px 0 24px;">
            <div style="font-size: 2.8rem;">🌦️</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #E8E8F0; 
                        letter-spacing: -0.02em; margin-top: 6px;">
                {PROJECT_NAME}
            </div>
            <div style="font-size: 0.72rem; color: #9999BB; margin-top: 4px; 
                        line-height: 1.4; padding: 0 8px;">
                Markov Chains &amp; Monte Carlo
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "🏠 Overview": "overview",
            "📊 Dataset Dashboard": "dataset",
            "📈 Data Visualization": "visualization",
            "🔄 Markov Chain": "markov",
            "🎲 Monte Carlo Simulation": "simulation",
            "🔮 Weather Forecast": "forecast",
            "🤖 AI Weather Assistant": "assistant",
            "💾 Download Center": "download",
        }

        selected_label = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Dataset info pill
        st.markdown("""
        <div style="background: rgba(30,144,255,0.1); border: 1px solid #1E90FF33; 
                    border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; color: #9999BB;">
            <b style="color:#1E90FF;">📂 Dataset</b><br>
            weather.csv · 192 rows · 8 days
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("v1.0.0 · Built with Streamlit + Anthropic")

    return pages[selected_label]


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview():
    st.markdown(f"""
    <div style="padding: 32px 0 24px;">
        <h1 style="font-size: 2.6rem; font-weight: 800; color: #E8E8F0; 
                   letter-spacing: -0.03em; margin: 0;">
            🌦️ {PROJECT_NAME}
        </h1>
        <p style="font-size: 1.15rem; color: #9999BB; margin-top: 8px; max-width: 680px;">
            {PROJECT_SUBTITLE}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hero metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weather States", "3", help="Sunny, Cloudy, Rainy")
    c2.metric("Algorithm", "Markov Chain", help="First-order discrete-time Markov Chain")
    c3.metric("Forecasting", "Monte Carlo", help="Probabilistic simulation over N paths")
    c4.metric("Dataset", "192 rows", help="8 days of hourly sensor data")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        with st.expander("📐 How It Works — Architecture", expanded=True):
            st.markdown("""
            **Weather Forecasting Using Markov** uses a two-stage probabilistic forecasting pipeline:

            **Stage 1 — Markov Chain Model**
            - Historical hourly data is classified into weather states: ☀️ Sunny, ⛅ Cloudy, 🌧️ Rainy
            - A **transition matrix** P is built from observed state-to-state frequencies
            - Each entry P[i][j] = probability of transitioning from state i to state j

            **Stage 2 — Monte Carlo Simulation**
            - The chain is simulated N times (default: 1,000) from a chosen start state
            - Each run produces one possible future weather path
            - Results are aggregated into per-day probability distributions

            **Output**
            - Most likely weather state per forecast day
            - Probability distribution across all states
            - 95% confidence interval
            - Overall confidence score
            """)

        with st.expander("🎓 About Markov Chains"):
            st.markdown("""
            A **Markov Chain** is a stochastic process satisfying the *Markov property*:
            the future state depends only on the current state, not the past.

            > *"The weather tomorrow depends only on today's weather, not last week's."*

            Formally: **P(X_{n+1} = j | X_n = i, X_{n-1}, ...) = P(X_{n+1} = j | X_n = i)**

            For weather forecasting, this is a reasonable approximation over short horizons
            (1–7 days), especially in climates with persistent patterns.
            """)

        with st.expander("🎲 About Monte Carlo Simulation"):
            st.markdown("""
            **Monte Carlo Simulation** uses repeated random sampling to compute numerical results.

            In this project:
            1. Start from a known weather state (today)
            2. At each step, sample the next state using the transition probabilities
            3. Repeat 1,000+ times to get a distribution of possible futures
            4. Aggregate to get P(state on day N) and confidence bounds

            More simulations = more stable probability estimates. The law of large numbers
            guarantees convergence to the true analytical probabilities as N → ∞.
            """)

    with col2:
        st.markdown("#### 🔬 Model Parameters")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Chain Order | First-order |
        | State Space | {Sunny, Cloudy, Rainy} |
        | Classification | Humidity-based |
        | Rainy threshold | ≥ 78% humidity |
        | Cloudy threshold | 69–77% humidity |
        | Sunny threshold | < 69% humidity |
        | Simulation depth | Up to 30 days |
        | Default simulations | 1,000 |
        | Confidence level | 95% |
        """)

        st.markdown("#### 🧱 Tech Stack")
        for tech, icon in [
            ("Python 3.11+", "🐍"), ("Pandas", "🐼"), ("NumPy", "🔢"),
            ("Plotly", "📊"), ("Streamlit", "⚡"), ("SciPy", "🔬"),
            ("Anthropic Claude API", "🤖"),
        ]:
            st.markdown(f"{icon} **{tech}**")


# ── Page: Dataset Dashboard ───────────────────────────────────────────────────

def page_dataset(df: pd.DataFrame, daily_df: pd.DataFrame, summary: dict):
    st.markdown('<p class="section-header">📊 Dataset Dashboard</p>', unsafe_allow_html=True)

    validation = summary.get("validation", {})
    if validation.get("warnings"):
        for w in validation["warnings"]:
            st.warning(w)

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{summary['n_rows']:,}")
    c2.metric("Days Observed", summary["n_days"])
    c3.metric("Features", summary["n_cols"])
    c4.metric("Missing Values", summary["missing_values"])
    c5.metric("Dominant State", state_badge(summary["dominant_state"]))

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗃️ Preview", "📋 Statistics", "🌤️ State Breakdown", "📅 Daily Summary"
    ])

    with tab1:
        st.markdown("**Hourly Weather Data**")
        st.dataframe(
            df.style.background_gradient(
                subset=["Temperature", "Relative Humidity", "Wind Speed"],
                cmap="Blues",
            ),
            use_container_width=True,
            height=380,
        )

    with tab2:
        cols_to_describe = [
            "Temperature", "Relative Humidity", "Wind Speed", "Wind Direction"
        ]
        stats = df[cols_to_describe].describe().round(3)
        st.dataframe(stats, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Temperature (°C)**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Min", summary["temp_min"])
            m2.metric("Mean", summary["temp_mean"])
            m3.metric("Max", summary["temp_max"])

        with col2:
            st.markdown("**Humidity (%)**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Min", summary["humidity_min"])
            m2.metric("Mean", summary["humidity_mean"])
            m3.metric("Max", summary["humidity_max"])

    with tab3:
        state_counts = summary.get("state_counts", {})
        total = sum(state_counts.values())

        col1, col2, col3 = st.columns(3)
        for col, state in zip([col1, col2, col3], WEATHER_STATES):
            cnt = state_counts.get(state, 0)
            pct_val = cnt / total * 100 if total > 0 else 0
            col.metric(
                f"{STATE_ICONS.get(state, '')} {state}",
                f"{cnt} hrs",
                f"{pct_val:.1f}%",
            )

        st.markdown("**Feature Information**")
        feature_info = pd.DataFrame({
            "Column": ["Day", "Hour", "Temperature", "Relative Humidity", "Wind Speed", "Wind Direction", "State"],
            "Type": ["int", "int", "float", "int", "float", "float", "categorical"],
            "Description": [
                "Day number (2–9)",
                "Hour of day (0–23)",
                "Air temperature in °C",
                "Relative humidity in %",
                "Wind speed in km/h",
                "Wind direction in degrees (0–360)",
                "Derived weather state: Sunny / Cloudy / Rainy",
            ],
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)

    with tab4:
        st.dataframe(daily_df, use_container_width=True, hide_index=True)


# ── Page: Data Visualization ──────────────────────────────────────────────────

def page_visualization(df: pd.DataFrame, daily_df: pd.DataFrame):
    st.markdown('<p class="section-header">📈 Data Visualization</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Weather Trends", "🌤️ State Distribution", "🔗 Correlations", "📅 Daily Analysis"
    ])

    with tab1:
        st.plotly_chart(temperature_trend(df), use_container_width=True)
        st.plotly_chart(humidity_trend(df), use_container_width=True)
        st.plotly_chart(wind_speed_trend(df), use_container_width=True)

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(weather_state_distribution(df), use_container_width=True)
        with col2:
            st.markdown("#### State Classification Logic")
            st.markdown("""
            Weather states are derived from **Relative Humidity**:

            | State | Humidity Range | Icon |
            |-------|---------------|------|
            | Rainy | ≥ 78% | 🌧️ |
            | Cloudy | 69% – 77% | ⛅ |
            | Sunny | < 69% | ☀️ |

            This threshold-based approach captures tropical coastal weather patterns
            where humidity is the strongest indicator of precipitation state.
            """)
            state_counts = df["State"].value_counts()
            for state in WEATHER_STATES:
                cnt = int(state_counts.get(state, 0))
                frac = cnt / len(df)
                st.progress(frac, text=f"{STATE_ICONS.get(state, '')} {state}: {cnt} hrs ({frac:.1%})")

    with tab3:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)
        with st.expander("📖 Reading the Heatmap"):
            st.markdown("""
            Values close to **+1** indicate strong positive correlation.
            Values close to **-1** indicate strong negative correlation.
            Values near **0** indicate no linear relationship.

            Key insight: Temperature and Humidity are often negatively correlated —
            higher temperatures tend to occur with lower relative humidity in this dataset.
            """)

    with tab4:
        st.plotly_chart(daily_weather_summary(daily_df), use_container_width=True)


# ── Page: Markov Chain ────────────────────────────────────────────────────────

def page_markov(tm_df: pd.DataFrame, validation: dict, steady_state: dict):
    st.markdown('<p class="section-header">🔄 Markov Chain Dashboard</p>', unsafe_allow_html=True)

    if validation.get("warnings"):
        for w in validation["warnings"]:
            st.warning(w)

    # Row sums validation
    col1, col2, col3 = st.columns(3)
    row_sums = validation.get("row_sums", {})
    for col, state in zip([col1, col2, col3], WEATHER_STATES):
        rs = row_sums.get(state, 0)
        col.metric(
            f"{STATE_ICONS.get(state, '')} {state} row sum",
            f"{rs:.4f}",
            "✓ Valid" if abs(rs - 1.0) < 0.001 else "⚠️ Check",
        )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Heatmap", "📋 Probability Table", "🔵 State Diagram", "⚖️ Steady State"
    ])

    with tab1:
        st.plotly_chart(transition_matrix_heatmap(tm_df), use_container_width=True)
        st.markdown("""
        **How to read this:** Each cell P[i→j] shows the probability of weather
        transitioning from state i (row) to state j (column) in one time step.
        Darker blue = higher probability.
        """)

    with tab2:
        st.markdown("**Transition Probability Matrix**")
        display_tm = tm_df.copy()
        display_tm.index.name = "From → To"

        styled = display_tm.style\
            .format("{:.4f}")\
            .background_gradient(cmap="Blues", axis=None)\
            .set_properties(**{"text-align": "center"})
        st.dataframe(styled, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Interpret a specific transition")
        col1, col2 = st.columns(2)
        with col1:
            from_state = st.selectbox("From state", WEATHER_STATES, key="tm_from")
        with col2:
            to_state = st.selectbox("To state", WEATHER_STATES, key="tm_to")

        prob = float(tm_df.loc[from_state, to_state])
        st.info(
            f"**P({STATE_ICONS.get(to_state,'')} {to_state} | {STATE_ICONS.get(from_state,'')} {from_state})** "
            f"= **{prob:.4f}** ({prob*100:.2f}%)\n\n"
            f"If weather is currently *{from_state}*, there is a **{prob*100:.1f}%** chance "
            f"the next observation will be *{to_state}*."
        )

    with tab3:
        st.plotly_chart(markov_chain_diagram(tm_df), use_container_width=True)
        st.markdown("""
        **Node size** represents the state. **Arrow width** is proportional to transition probability.
        Self-loops (staying in the same state) are visible as the dominant diagonal in the heatmap.
        """)

    with tab4:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(steady_state_bar(steady_state), use_container_width=True)
        with col2:
            st.markdown("#### What is the Steady State?")
            st.markdown("""
            The **stationary distribution** π is the long-run proportion of time the system
            spends in each state, regardless of where it starts:

            **π = π · P** (π is a left eigenvector of P with eigenvalue 1)

            After many time steps, the chain "forgets" where it started and converges
            to this distribution. It's the climate fingerprint of the dataset.
            """)
            for state, prob in steady_state.items():
                st.metric(
                    f"{STATE_ICONS.get(state,'')} {state} long-run probability",
                    f"{prob*100:.1f}%"
                )


# ── Page: Monte Carlo Simulation ──────────────────────────────────────────────

def page_simulation(tm_df: pd.DataFrame):
    st.markdown('<p class="section-header">🎲 Monte Carlo Simulation</p>', unsafe_allow_html=True)

    st.markdown("""
    Configure and run the Monte Carlo weather simulation. The model will simulate
    thousands of possible future weather paths using the learned Markov Chain.
    """)

    # Configuration panel
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            start_state = st.selectbox(
                "🌤️ Current Weather State",
                WEATHER_STATES,
                index=0,
                help="The state you are starting from (today's weather)",
            )
        with col2:
            n_days = st.slider(
                "📅 Forecast Days",
                min_value=MIN_FORECAST_DAYS,
                max_value=MAX_FORECAST_DAYS,
                value=DEFAULT_FORECAST_DAYS,
                help="Number of days to forecast ahead",
            )
        with col3:
            n_simulations = st.select_slider(
                "🔁 Simulations",
                options=[100, 250, 500, 1000, 2500, 5000, 10000],
                value=DEFAULT_NUM_SIMULATIONS,
                help="More simulations = more stable probabilities (slower)",
            )

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_button or "sim_result" in st.session_state:
        if run_button:
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                result = run_monte_carlo(
                    start_state=start_state,
                    tm_df=tm_df,
                    n_days=n_days,
                    n_simulations=n_simulations,
                    seed=42,
                    store_paths=True,
                )
            st.session_state["sim_result"] = result
            st.session_state["sim_params"] = {
                "start_state": start_state,
                "n_days": n_days,
                "n_simulations": n_simulations,
            }
        else:
            result = st.session_state["sim_result"]

        st.markdown("---")

        # Key result metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start State", state_badge(result.start_state))
        c2.metric("Forecast Days", result.n_days)
        c3.metric(
            "Confidence Score",
            f"{result.confidence_score*100:.1f}%",
            confidence_label(result.confidence_score),
        )
        c4.metric(
            "95% CI (Day 1)",
            f"{result.confidence_interval[0]*100:.0f}–{result.confidence_interval[1]*100:.0f}%",
        )

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Probability Timeline", "🌀 Simulation Paths", "🎲 State Frequency"])

        with tab1:
            st.plotly_chart(forecast_timeline(result), use_container_width=True)

        with tab2:
            if result.all_paths:
                st.plotly_chart(simulation_path_sample(result, n_paths=30), use_container_width=True)
            else:
                st.info("Enable path storage to see simulation spaghetti plot.")

        with tab3:
            st.plotly_chart(simulation_distribution(result), use_container_width=True)


# ── Page: Weather Forecast ────────────────────────────────────────────────────

def page_forecast(tm_df: pd.DataFrame):
    st.markdown('<p class="section-header">🔮 Weather Forecast Dashboard</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        start_state = st.selectbox("Today's Weather", WEATHER_STATES, key="fc_start")
        n_days = st.slider("Forecast Days", 1, MAX_FORECAST_DAYS, DEFAULT_FORECAST_DAYS, key="fc_days")
        run = st.button("Generate Forecast", type="primary", use_container_width=True)

    if run or "forecast_result" in st.session_state:
        if run:
            with st.spinner("Generating forecast..."):
                result = run_monte_carlo(
                    start_state=start_state,
                    tm_df=tm_df,
                    n_days=n_days,
                    n_simulations=2000,
                    seed=42,
                    store_paths=False,
                )
            st.session_state["forecast_result"] = result
        else:
            result = st.session_state["forecast_result"]

        with col2:
            st.markdown("#### 📅 Day-by-Day Forecast")
            forecast_df = result_to_dataframe(result)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Forecast cards
        st.markdown("#### 🌤️ Forecast at a Glance")
        cols = st.columns(min(n_days, 7))
        for i, (state, col) in enumerate(zip(result.most_likely_states[:7], cols)):
            prob = result.day_probabilities[i][state]
            with col:
                st.markdown(f"""
                <div style="background: rgba(30,30,60,0.6); border: 1px solid #2E2E4E;
                            border-radius: 12px; padding: 14px; text-align: center;
                            border-top: 3px solid {STATE_COLORS.get(state, '#1E90FF')};">
                    <div style="font-size: 1.8rem;">{STATE_ICONS.get(state, '')}</div>
                    <div style="color: #9999BB; font-size: 0.75rem; margin-top: 4px;">Day {i+1}</div>
                    <div style="color: #E8E8F0; font-weight: 700; font-size: 0.9rem;">{state}</div>
                    <div style="color: {STATE_COLORS.get(state, '#1E90FF')}; 
                                font-size: 0.85rem; margin-top: 2px;">{prob*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.plotly_chart(forecast_timeline(result), use_container_width=True)

        # Analytical check
        with st.expander("🔬 Analytical vs Monte Carlo Comparison"):
            analytical_day7 = state_probability_vector(start_state, tm_df, n=min(n_days, 7))
            mc_day7 = result.day_probabilities[min(n_days, 7) - 1]

            comparison_df = pd.DataFrame({
                "State": WEATHER_STATES,
                "Analytical P (Markov)": [f"{analytical_day7.get(s, 0)*100:.2f}%" for s in WEATHER_STATES],
                f"Monte Carlo P ({result.n_simulations:,} sims)": [f"{mc_day7.get(s, 0)*100:.2f}%" for s in WEATHER_STATES],
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.markdown("""
            As N (simulations) → ∞, Monte Carlo probabilities converge to the analytical Markov values.
            Differences here reflect simulation variance with a finite number of paths.
            """)


# ── Page: AI Weather Assistant ────────────────────────────────────────────────

def page_assistant(tm_df: pd.DataFrame, steady_state: dict, summary: dict):
    st.markdown('<p class="section-header">🤖 AI Weather Assistant</p>', unsafe_allow_html=True)

    st.markdown("""
    Ask the AI assistant anything about your weather forecast, the Markov Chain model,
    Monte Carlo simulation, or how to interpret the results.
    """)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []

    # Build context from current simulation results
    result = st.session_state.get("forecast_result") or st.session_state.get("sim_result")
    context = build_ai_context(
        result=result,
        tm_df=tm_df,
        steady_state=steady_state,
        summary=summary,
    )

    # Quick question buttons
    st.markdown("**💡 Suggested Questions**")
    q_cols = st.columns(3)
    quick_questions = [
        "Why is tomorrow predicted to be Rainy?",
        "How does the Markov Chain work?",
        "What does Monte Carlo simulation do?",
        "How reliable is this prediction?",
        "What is the steady-state distribution?",
        "Explain the transition probabilities.",
    ]

    for i, (col, q) in enumerate(zip(q_cols * 2, quick_questions)):
        with col:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state["pending_question"] = q

    st.markdown("---")

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"""
                <div class="ai-chat-bubble-user">
                    <div class="ai-assistant-header">👤 You</div>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-chat-bubble-assistant">
                    <div class="ai-assistant-header">🤖 Weather Forecasting Using Markov</div>
                    {content}
                </div>
                """, unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        default_q = st.session_state.pop("pending_question", "")
        user_input = st.text_input(
            "Ask anything about the weather forecast or model...",
            value=default_q,
            key="chat_input",
            placeholder="e.g., Why is it predicted to be rainy on day 3?",
            label_visibility="collapsed",
        )
    with col2:
        send = st.button("Send →", type="primary", use_container_width=True)

    if send and user_input.strip():
        with st.spinner("Thinking..."):
            response = ask_weather_assistant(
                user_question=user_input,
                context=context,
                conversation_history=st.session_state["ai_messages"],
                result=result,
                tm_df=tm_df,
                steady_state=steady_state,
            )

        # Update histories
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.session_state["ai_messages"].append({"role": "user", "content": user_input})
        st.session_state["ai_messages"].append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("🗑️ Clear Conversation", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.session_state["ai_messages"] = []
        st.rerun()


# ── Page: Download Center ─────────────────────────────────────────────────────

def page_download(df: pd.DataFrame, tm_df: pd.DataFrame, summary: dict):
    st.markdown('<p class="section-header">💾 Download Center</p>', unsafe_allow_html=True)

    result = st.session_state.get("forecast_result") or st.session_state.get("sim_result")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📂 Processed Dataset")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Processed Dataset (CSV)",
            data=csv_bytes,
            file_name="weather_forecasting_markov_processed_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("#### 🔄 Transition Matrix")
        tm_csv = tm_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Transition Matrix (CSV)",
            data=tm_csv,
            file_name="weather_forecasting_markov_transition_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 🔮 Forecast Results")
        if result:
            forecast_bytes = simulation_summary_to_csv_bytes(result)
            st.download_button(
                "⬇️ Download Forecast (CSV)",
                data=forecast_bytes,
                file_name="weather_forecasting_markov_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )

            report_text = generate_forecast_report(result, summary)
            report_bytes = report_to_bytes(report_text)
            st.download_button(
                "⬇️ Download Forecast Report (TXT)",
                data=report_bytes,
                file_name="weather_forecasting_markov_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info(
                "No forecast generated yet. Go to **🔮 Weather Forecast** or "
                "**🎲 Monte Carlo Simulation** and run a simulation first."
            )

    st.markdown("---")
    with st.expander("📖 Report Preview"):
        if result:
            report_text = generate_forecast_report(result, summary)
            st.code(report_text, language=None)
        else:
            st.markdown("Run a forecast first to preview the report.")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Load data pipeline (cached)
    try:
        df, daily_df, summary = load_pipeline()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

    # Build Markov model (cached)
    try:
        tm_df, validation, steady_state = load_model(df)
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        st.stop()

    # Render sidebar, get current page
    page = render_sidebar()

    # Route to page
    if page == "overview":
        page_overview()
    elif page == "dataset":
        page_dataset(df, daily_df, summary)
    elif page == "visualization":
        page_visualization(df, daily_df)
    elif page == "markov":
        page_markov(tm_df, validation, steady_state)
    elif page == "simulation":
        page_simulation(tm_df)
    elif page == "forecast":
        page_forecast(tm_df)
    elif page == "assistant":
        page_assistant(tm_df, steady_state, summary)
    elif page == "download":
        page_download(df, tm_df, summary)


if __name__ == "__main__":
    main()

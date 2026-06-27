# 🌦️ WeatherSense AI

**Intelligent Weather Forecasting using Markov Chains & Monte Carlo Simulation**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-purple.svg)](https://plotly.com)

> A production-quality, interactive weather forecasting platform that models weather state transitions using a **first-order Markov Chain** and generates probabilistic forecasts using **Monte Carlo Simulation**, with an integrated **AI Weather Assistant** powered by Claude.

---

## 🚀 Live Demo

**[→ Try WeatherSense AI Live](https://your-app.streamlit.app)**  
*(Deploy to Streamlit Community Cloud — see Deployment section below)*

---

## 📸 Screenshots

| Overview | Markov Chain Heatmap | Monte Carlo Forecast | AI Assistant |
|----------|---------------------|---------------------|--------------|
| ![Overview](assets/screenshot.png) | *See live demo* | *See live demo* | *See live demo* |

---

## ✨ Features

### 🏠 Project Overview
- Architecture explanation, tech stack, model parameters at a glance

### 📊 Dataset Dashboard
- Interactive preview of 192-row weather sensor dataset
- Summary statistics, missing value report, feature information
- Per-state hourly breakdown, daily aggregation

### 📈 Data Visualization (Interactive Plotly charts)
- Temperature, humidity, wind speed trends
- Weather state distribution (donut chart)
- Feature correlation heatmap
- Daily weather state breakdown

### 🔄 Markov Chain Dashboard
- Dynamically computed transition probability matrix
- Interactive heatmap + state diagram
- Row-stochastic validation
- Long-run stationary distribution (steady state)
- Transition probability interpreter

### 🎲 Monte Carlo Simulation
- Configurable: 100–10,000 simulations × 1–30 forecast days
- Per-day probability distributions
- Confidence score + 95% Wilson CI
- Simulation spaghetti plot (sample paths)

### 🔮 Weather Forecast Dashboard
- Day-by-day forecast cards
- Stacked area probability timeline
- Analytical (Markov P^n) vs Monte Carlo comparison

### 🤖 AI Weather Assistant
- Powered by **Claude Sonnet** (Anthropic API)
- Context-aware: sees the actual transition matrix, current forecast, simulation results
- Quick-question shortcuts
- Multi-turn conversation with memory

### 💾 Download Center
- Processed dataset CSV
- Transition matrix CSV
- Forecast results CSV
- Formatted text report

---

## 🏗️ Architecture

```
WeatherSenseAI/
│
├── app.py              ← Streamlit UI (page routing, sidebar, layout)
├── preprocessing.py    ← Data loading, cleaning, state classification
├── model.py            ← Markov Chain transition matrix + steady state
├── simulation.py       ← Monte Carlo simulation engine
├── visualizations.py   ← All Plotly chart functions
├── utils.py            ← Export, formatting, AI assistant integration
├── config.py           ← Central configuration (thresholds, colors, constants)
│
├── data/
│   └── weather.csv     ← Hourly weather sensor data (8 days, 192 rows)
│
├── notebooks/
│   └── STPA_Project.ipynb  ← Original research notebook
│
├── assets/
│   └── screenshot.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Data Flow

```
weather.csv
    │
    ▼
preprocessing.py  →  Hourly df with 'State' column + daily aggregation
    │
    ▼
model.py          →  Transition matrix P, validation, steady-state π
    │
    ▼
simulation.py     →  Monte Carlo paths → per-day P(state), confidence, CI
    │
    ▼
visualizations.py →  Plotly figures (heatmap, timeline, spaghetti plot...)
    │
    ▼
app.py + utils.py →  Streamlit dashboard + AI assistant + CSV exports
```

---

## 📐 Model Design

### Weather State Classification
States are derived from **Relative Humidity** (strongest precipitation predictor in the dataset):

| State | Humidity | Icon |
|-------|----------|------|
| Rainy | ≥ 78% | 🌧️ |
| Cloudy | 69–77% | ⛅ |
| Sunny | < 69% | ☀️ |

### Markov Chain (First-Order)
- **Markov property**: P(X_{n+1} | X_n, X_{n-1}, ...) = P(X_{n+1} | X_n)
- Transition matrix P is row-stochastic (each row sums to 1.0)
- Built from observed transition frequencies in historical data

### Monte Carlo Simulation
- N independent paths simulated from a chosen start state
- Each step: sample next state from transition probabilities using numpy RNG
- Aggregate N paths → per-day P(state) distribution
- Confidence: Wilson interval on day-1 modal state proportion

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Anthropic API key (for AI Weather Assistant)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/WeatherSenseAI.git
cd WeatherSenseAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
# On Windows: set ANTHROPIC_API_KEY=your-api-key-here

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "your-api-key-here"
   ```
5. Click **Deploy** — your app gets a public URL in ~2 minutes

Update `utils.py` to use `st.secrets["ANTHROPIC_API_KEY"]` for cloud deployment.

---

## 📊 Dataset

**Source:** Hourly weather sensor recordings  
**Period:** 8 days (Day 2 through Day 9)  
**Records:** 192 rows × 6 features  
**Location:** Tropical coastal environment

| Feature | Type | Description |
|---------|------|-------------|
| Day | int | Day number (2–9) |
| Hour | int | Hour of day (0–23) |
| Temperature | float | Air temperature (°C) |
| Relative Humidity | int | Relative humidity (%) |
| Wind Speed | float | Wind speed (km/h) |
| Wind Direction | float | Wind direction (degrees) |

---

## 🔭 Future Improvements

- [ ] Upload custom CSV datasets
- [ ] Second-order Markov Chain (memory = 2 steps)
- [ ] Rainfall amount regression (continuous prediction)
- [ ] Seasonal Markov models (separate matrices per season)
- [ ] Real-time weather API integration
- [ ] Export forecast as PDF report with charts
- [ ] Mobile-optimized responsive layout
- [ ] Model evaluation metrics (cross-validation on historical data)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**  
B.Tech Data Science — 3rd Year  
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourusername) · [Portfolio](https://yourportfolio.com)

---

## 🙏 Acknowledgments

Built with [Streamlit](https://streamlit.io), [Plotly](https://plotly.com), and [Anthropic Claude](https://anthropic.com).

---

*WeatherSense AI — Turning historical patterns into probabilistic forecasts.*

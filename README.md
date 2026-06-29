# 🌦️ Weather Forecasting Using Markov

**Intelligent Weather Forecasting using Markov Chains & Monte Carlo Simulation**

A Streamlit web application that forecasts weather states (Sunny, Cloudy, Rainy) using a first-order Markov Chain model trained on historical sensor data, with probabilistic forecasting via Monte Carlo Simulation.

---

## 📌 About the Project

This simulation model takes hourly weather sensor data and learns the probability of transitioning between weather states. It then simulates thousands of possible future weather paths to generate a day-by-day probabilistic forecast.

The app includes an interactive dashboard with charts, a Markov Chain heatmap, Monte Carlo simulation controls, a forecast viewer, an AI assistant (rule-based, no API key needed), and a download center.

---

## 📐 Model Design

### Weather State Classification

Weather states are derived from **Relative Humidity** readings:

| State | Humidity | Icon |
|-------|----------|------|
| Rainy | ≥ 78% | 🌧️ |
| Cloudy | 69–77% | ⛅ |
| Sunny | < 69% | ☀️ |

### Markov Chain (First-Order)

A first-order Markov Chain assumes the next weather state depends only on the current state, not on any history before it.

A **transition matrix** P is built by counting how often each state transitions to each other state in the historical data, then normalizing each row to sum to 1.0.

Example from this dataset:
- P(Rainy → Rainy) = 81% — Rainy weather is very persistent
- P(Sunny → Sunny) = 77% — Sunny weather is also persistent
- P(Cloudy → Rainy) = 13% — Cloudy can shift to Rainy

### Monte Carlo Simulation

The Monte Carlo engine runs N independent simulations (default 1,000) from a chosen start state. Each simulation follows the transition probabilities step by step to produce one possible future path. The N paths are then aggregated to compute:

- Per-day probability distribution over all states
- Most likely state per day
- Confidence score
- 95% Wilson confidence interval

---

## 🏗️ Architecture

```
Weather Forecasting Using Markov/
│
├── app.py              ← Streamlit UI (pages, sidebar, layout)
├── config.py           ← All constants, thresholds, colors
├── preprocessing.py    ← Data loading, cleaning, state classification
├── model.py            ← Markov Chain transition matrix + steady state
├── simulation.py       ← Monte Carlo simulation engine
├── visualizations.py   ← All Plotly chart functions
├── utils.py            ← Export helpers, report generation, AI assistant
│
├── data/
│   └── weather.csv     ← Hourly weather sensor data (8 days, 192 rows)
│
├── notebooks/
│   └── STPA_Project.ipynb  ← Original research notebook
│
├── requirements.txt
└── README.md
```

---

## 🔁 Data Flow

```
weather.csv
    │
    ▼
preprocessing.py
    → Clean column names
    → Classify each row into Sunny / Cloudy / Rainy (by humidity)
    → Engineer rolling mean features
    → Aggregate to daily summaries
    │
    ▼
model.py
    → Count state-to-state transitions
    → Normalize into transition matrix P
    → Validate (each row sums to 1.0)
    → Compute steady-state distribution
    │
    ▼
simulation.py
    → Run N Monte Carlo paths from chosen start state
    → Aggregate into per-day probability distributions
    → Compute confidence score and 95% CI
    │
    ▼
visualizations.py + app.py
    → Render interactive Plotly charts
    → Display forecast cards and tables
    → Export CSV / text reports
```

---

## 📊 Dataset

- **192 rows** of hourly weather sensor readings
- **8 days** of data (Day 2 through Day 9)
- **6 features:** Day, Hour, Temperature (°C), Relative Humidity (%), Wind Speed (km/h), Wind Direction (°)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/aaravsheth1905/Weather-Forecasting-Using-Markov.git
cd Weather-forecasting-using-markov

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**.
Below is the link:

https://weather-forecasting-using-markov.streamlit.app

---

## 🔭 Future Improvements

- Support for uploading custom CSV datasets
- Second-order Markov Chain (2-step memory)
- Seasonal transition matrices
- Real-time weather API integration
- PDF report export

---

## 👤 Author

Aarav Sheth — B.Tech Data Science, 3rd Year

---

*Weather Forecasting Using Markov — Turning historical patterns into probabilistic forecasts.*

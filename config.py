"""
config.py — WeatherSense AI
============================
Central configuration module. Single source of truth for all constants.

Author: WeatherSense AI
"""

import os

# ─── Project Metadata ────────────────────────────────────────────────────────

PROJECT_NAME = "WeatherSense AI"
PROJECT_SUBTITLE = "Intelligent Weather Forecasting using Markov Chains & Monte Carlo Simulation"
PROJECT_VERSION = "1.0.0"

# ─── Data Configuration ───────────────────────────────────────────────────────

# Look for weather.csv in data/ folder first, then root directory
def _find_data_path():
    candidates = [
        "data/weather.csv",
        "weather.csv",
        os.path.join(os.path.dirname(__file__), "data", "weather.csv"),
        os.path.join(os.path.dirname(__file__), "weather.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "data/weather.csv"  # default, will show clear error

DATA_PATH = _find_data_path()

COLUMNS = {
    "day": "Day",
    "hour": "Hour",
    "temperature": "Temperature",
    "humidity": "Relative Humidity",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "state": "State",
}

# ─── Weather State Classification ─────────────────────────────────────────────

WEATHER_STATES = ["Sunny", "Cloudy", "Rainy"]

HUMIDITY_THRESHOLDS = {
    "rainy_min": 78,
    "cloudy_min": 69,
}

STATE_ICONS = {
    "Sunny": "☀️",
    "Cloudy": "⛅",
    "Rainy": "🌧️",
}

# ─── Simulation Defaults ──────────────────────────────────────────────────────

DEFAULT_FORECAST_DAYS = 7
DEFAULT_NUM_SIMULATIONS = 1000
MIN_SIMULATIONS = 100
MAX_SIMULATIONS = 10000
MIN_FORECAST_DAYS = 1
MAX_FORECAST_DAYS = 30
CONFIDENCE_LEVEL = 0.95

# ─── UI & Theme ───────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#1E90FF",
    "primary_dark": "#0066CC",
    "secondary": "#00CED1",
    "accent": "#FFD700",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "surface": "#1E1E2E",
    "background": "#0D0D1A",
    "text_primary": "#E8E8F0",
    "text_secondary": "#9999BB",
    "border": "#2E2E4E",
}

STATE_COLORS = {
    "Sunny": "#FFD700",
    "Cloudy": "#87CEEB",
    "Rainy": "#1E90FF",
}

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 400
CHART_HEIGHT_TALL = 500
CHART_HEIGHT_HEATMAP = 450

# ─── Export ───────────────────────────────────────────────────────────────────

EXPORT_DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
PREDICTION_CSV_NAME = "weathersense_forecast.csv"
REPORT_NAME = "weathersense_report.txt"

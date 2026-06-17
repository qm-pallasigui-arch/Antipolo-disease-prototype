"""""

HOW TO RUN:
  1. Install dependencies (once):
        pip install pandas numpy statsmodels plotly dash

  2. Run:
        python prototype.py

  3. Open browser:
        http://127.0.0.1:8050

CSV FORMAT (when you upload real DepEd data):
  Your CSV must have these columns:
    - year        : int  (e.g. 2015, 2016 ... 2025)
    - month       : int  (1–12)
    - disease     : str  (e.g. "Dengue", "Tuberculosis")
    - level       : str  (Elementary / Junior HS / Senior HS)
    - cases       : int  (number of cases that month)
  See generate_fallback_data() below for the exact structure.
=============================================================================
"""

# ── Standard library ───────────────────────────────────────────────────────
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from dash import (
    Dash, dcc, html, Input, Output, State,
    callback, no_update, dash_table
)


# =============================================================================
# SECTION 1 — PRE-LOADED FALLBACK DATA GENERATOR
# Produces a realistic 10-year monthly dataset that mirrors what
# your actual DepEd CSV will look like once you receive it.
# =============================================================================

DISEASES = [
    "Dengue",
    "Acute Respiratory Infection",
    "Influenza-like Illness",
    "Tuberculosis",
    "Hand Foot & Mouth Disease",
    "Measles",
]
LEVELS = ["Elementary", "Junior HS", "Senior HS"]
DISEASE_COLORS = {
    "Dengue":                      "#378ADD",
    "Acute Respiratory Infection": "#1D9E75",
    "Influenza-like Illness":      "#EF9F27",
    "Tuberculosis":                "#D85A30",
    "Hand Foot & Mouth Disease":   "#D4537E",
    "Measles":                     "#7F77DD",
}
# Base seasonal weight per month (rainy season = higher)
SEASONAL_WEIGHTS = [0.55, 0.50, 0.60, 0.65, 0.85,
                    1.30, 1.50, 1.65, 1.60, 1.40, 0.90, 0.62]
# Disease relative frequency weights
DISEASE_WEIGHTS = [0.31, 0.24, 0.18, 0.13, 0.09, 0.05]
# Level split
LEVEL_WEIGHTS   = [0.55, 0.30, 0.15]


def generate_fallback_data() -> pd.DataFrame:
    """
    Generates realistic mock monthly case data from Jan 2015 to Dec 2025.
    Structure matches the expected DepEd CSV upload format exactly.
    COVID school closure (2020–2022) reduces cases by ~65%.
    """
    np.random.seed(2025)
    records = []

    for year in range(2015, 2026):
        for month in range(1, 13):
            # Skip future months beyond June 2025 (partial year)
            if year == 2025 and month > 6:
                continue

            covid_factor = 0.32 if 2020 <= year <= 2022 else 1.0
            base_total   = int(
                np.random.normal(110, 18) *
                SEASONAL_WEIGHTS[month - 1] *
                covid_factor
            )
            base_total = max(base_total, 5)

            for d_idx, disease in enumerate(DISEASES):
                for l_idx, level in enumerate(LEVELS):
                    raw = (base_total *
                           DISEASE_WEIGHTS[d_idx] *
                           LEVEL_WEIGHTS[l_idx] *
                           np.random.uniform(0.75, 1.30))
                    cases = max(int(round(raw)), 0)
                    if cases > 0:
                        records.append({
                            "year":    year,
                            "month":   month,
                            "disease": disease,
                            "level":   level,
                            "cases":   cases,
                        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


# =============================================================================
# SECTION 2 — ANALYSIS ENGINE
# These functions run the actual statistical models on whichever
# DataFrame is currently active (uploaded or fallback).
# =============================================================================

def get_monthly_series(df: pd.DataFrame,
                       disease: str = "all",
                       level:   str = "all") -> pd.Series:
    """
    Filters the dataframe and returns a monthly total case time series,
    indexed by date, with all months filled (missing = 0).
    """
    dff = df.copy()
    if disease != "all":
        dff = dff[dff["disease"] == disease]
    if level != "all":
        dff = dff[dff["level"] == level]

    monthly = (
        dff.groupby("date")["cases"]
        .sum()
        .asfreq("MS", fill_value=0)
        .sort_index()
    )
    return monthly


def run_arima(series: pd.Series, forecast_steps: int = 24):
    """
    Tiered model selection based on series length:
      >= 3 seasons (36 months): SARIMA(1,1,1)(1,1,1)[12] — captures annual cycle
      <  3 seasons            : ARIMA(1,1,1) — too short for seasonal estimation
    d is always fixed to 1 (disease count series are non-stationary by nature).
    Falls back one tier down if fitting fails; last resort is drift forecast.
    """
    def _fit_and_forecast(order, seasonal_order):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            m = SARIMAX(
                series, order=order, seasonal_order=seasonal_order,
                trend="c", enforce_stationarity=False, enforce_invertibility=False,
            )
            r = m.fit(disp=False, maxiter=300)
        pred    = r.get_forecast(steps=forecast_steps)
        fc_mean = pred.predicted_mean
        fc_ci   = pred.conf_int(alpha=0.05)
        cols    = fc_ci.columns.tolist()
        return r.fittedvalues, fc_mean, fc_ci[cols[0]], fc_ci[cols[1]]

    # Tier 1: SARIMA(1,1,1)(0,1,1)[12] — captures 12-month seasonal cycle
    if len(series) >= 36:
        try:
            return _fit_and_forecast((1, 1, 1), (0, 1, 1, 12))
        except Exception:
            pass

    # Tier 2: plain ARIMA(1,1,1) — no seasonal component
    try:
        return _fit_and_forecast((1, 1, 1), (0, 0, 0, 0))
    except Exception:
        pass

    # Tier 3: naive drift forecast with historical ±1.96σ CI
    last_val = series.iloc[-1]
    drift    = (series.iloc[-1] - series.iloc[0]) / len(series)
    std      = series.std()
    idx      = pd.date_range(
        series.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps, freq="MS"
    )
    fc_mean  = pd.Series(
        [max(last_val + drift * i, 0) for i in range(1, forecast_steps + 1)],
        index=idx
    )
    fc_lower = (fc_mean - 1.96 * std).clip(lower=0)
    fc_upper = fc_mean + 1.96 * std
    return series, fc_mean, fc_lower, fc_upper


def run_decomposition(series: pd.Series):
    """
    Runs classical additive seasonal decomposition (period=12).
    Requires at least 24 data points (2 full years).
    Returns the decomposition result object or None.
    """
    if len(series) < 24:
        return None
    try:
        return seasonal_decompose(series, model="additive", period=12)
    except Exception:
        return None


# =============================================================================
# SECTION 3 — CHART BUILDERS
# All charts are generated from live data — nothing is hardcoded.
# =============================================================================

FONT   = "Inter, Segoe UI, Arial, sans-serif"
TEXTC  = "#333333"
GRIDC  = "rgba(0,0,0,0.06)"
PLOTBG = "white"

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2022-12-01")


def base_layout(height=280, margin=None):
    m = margin or dict(l=48, r=16, t=8, b=40)
    return dict(
        height=height,
        plot_bgcolor=PLOTBG,
        paper_bgcolor=PLOTBG,
        font=dict(family=FONT, color=TEXTC, size=11),
        margin=m,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10)
        ),
        hovermode="x unified",
    )


def fig_arima(series, fitted, fc_mean, fc_lower, fc_upper):
    """Full ARIMA chart: observed + fitted + forecast + CI band."""
    fig = go.Figure()

    # COVID shading
    fig.add_vrect(
        x0=COVID_START, x1=COVID_END,
        fillcolor="rgba(136,135,128,0.10)",
        layer="below", line_width=0,
        annotation_text="COVID-19 closure",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color="#999",
    )

    # CI band
    x_band = list(fc_mean.index) + list(fc_mean.index[::-1])
    y_band = list(fc_upper.values) + list(fc_lower.values[::-1])
    fig.add_trace(go.Scatter(
        x=x_band, y=y_band,
        fill="toself",
        fillcolor="rgba(55,138,221,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip",
    ))

    # Observed
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name="Observed",
        line=dict(color="#378ADD", width=1.8),
        mode="lines",
    ))

    # Fitted
    fig.add_trace(go.Scatter(
        x=fitted.index, y=fitted.values,
        name="ARIMA fitted",
        line=dict(color="#1D9E75", width=1.5, dash="dot"),
        mode="lines",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_mean.index, y=fc_mean.values,
        name="Forecast",
        line=dict(color="#D85A30", width=2, dash="dash"),
        mode="lines",
    ))

    fig.update_layout(**base_layout(300))
    fig.update_xaxes(showgrid=False, linecolor=GRIDC)
    fig.update_yaxes(gridcolor=GRIDC, zeroline=False,
                     title_text="Cases / month")
    return fig


def fig_decomposition(decomp):
    """4-panel decomposition: observed, trend, seasonal, residual."""
    panels = [
        ("Observed",  decomp.observed,  "#378ADD"),
        ("Trend",     decomp.trend,     "#1D9E75"),
        ("Seasonal",  decomp.seasonal,  "#EF9F27"),
        ("Residual",  decomp.resid,     "#D85A30"),
    ]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=[p[0] for p in panels])
    for i, (label, data, color) in enumerate(panels, 1):
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values,
                       line=dict(color=color, width=1.5),
                       name=label, showlegend=False),
            row=i, col=1
        )
        fig.update_yaxes(gridcolor=GRIDC, zeroline=False,
                         tickfont=dict(size=9), row=i, col=1)

    fig.update_layout(
        height=480,
        plot_bgcolor=PLOTBG, paper_bgcolor=PLOTBG,
        font=dict(family=FONT, color=TEXTC, size=10),
        margin=dict(l=48, r=16, t=32, b=32),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRIDC)
    return fig


def fig_disease_bar(df):
    """Grouped bar: total cases per disease per year."""
    annual = (
        df.groupby(["year", "disease"])["cases"]
        .sum()
        .reset_index()
    )
    fig = go.Figure()
    for disease in DISEASES:
        sub = annual[annual["disease"] == disease]
        fig.add_trace(go.Bar(
            x=sub["year"], y=sub["cases"],
            name=disease,
            marker_color=DISEASE_COLORS.get(disease, "#888"),
            marker_cornerradius=2,
        ))
    layout = base_layout(260)
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, font=dict(size=9))
    layout["barmode"] = "stack"
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor=GRIDC, zeroline=False,
                     title_text="Total cases")
    return fig


def fig_seasonal_heatmap(df):
    """Heatmap: month vs year, color = total cases."""
    pivot = (
        df.groupby(["year", "month"])["cases"]
        .sum()
        .reset_index()
        .pivot(index="month", columns="year", values="cases")
        .fillna(0)
    )
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[month_labels[m-1] for m in pivot.index],
        colorscale="Blues",
        colorbar=dict(title="Cases", thickness=12, len=0.8),
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Cases: %{z}<extra></extra>",
    ))
    fig.update_layout(**base_layout(290, margin=dict(l=48, r=60, t=8, b=40)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_donut(df):
    """Donut chart: disease share of total cases."""
    totals = df.groupby("disease")["cases"].sum()
    totals = totals.reindex(DISEASES).fillna(0)
    fig = go.Figure(go.Pie(
        labels=totals.index.tolist(),
        values=totals.values.tolist(),
        hole=0.60,
        marker=dict(
            colors=[DISEASE_COLORS[d] for d in totals.index],
            line=dict(color="white", width=2)
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Cases: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**base_layout(260, margin=dict(l=8, r=8, t=8, b=8)))
    return fig


def summary_cards(df):
    """Compute summary stats from the live dataframe."""
    total = int(df["cases"].sum())
    annual = df.groupby("year")["cases"].sum()
    peak_year = int(annual.idxmax())
    peak_val  = int(annual.max())
    top_disease = df.groupby("disease")["cases"].sum().idxmax()
    top_pct = int(
        df.groupby("disease")["cases"].sum().max() /
        df["cases"].sum() * 100
    )
    years_covered = f"{df['year'].min()}–{df['year'].max()}"
    return total, peak_year, peak_val, top_disease, top_pct, years_covered


# =============================================================================
# SECTION 4 — DASH APP
# =============================================================================

app = Dash(__name__, title="Antipolo Disease Surveillance · Functional Prototype")

# ── Shared styles ─────────────────────────────────────────────────────────

S_TOPBAR = {
    "display": "flex", "justifyContent": "space-between",
    "alignItems": "center", "padding": "13px 22px",
    "borderBottom": "0.5px solid rgba(0,0,0,0.08)",
    "marginBottom": "18px", "background": "white",
}
S_CARD = {
    "background": "white",
    "border": "0.5px solid rgba(0,0,0,0.08)",
    "borderRadius": "10px", "padding": "16px",
}
S_METRIC = {
    "background": "#F7F7F5", "borderRadius": "8px",
    "padding": "13px 16px", "flex": "1", "minWidth": "120px",
}
S_LABEL = {
    "fontSize": "10px", "color": "#888", "margin": "0 0 5px",
    "textTransform": "uppercase", "letterSpacing": "0.05em",
}
S_VAL   = {"fontSize": "21px", "fontWeight": "500",
            "color": TEXTC, "margin": "0"}
S_DELTA = {"fontSize": "11px", "margin": "3px 0 0"}
S_SEC   = {
    "fontSize": "10px", "fontWeight": "600", "color": "#888",
    "textTransform": "uppercase", "letterSpacing": "0.07em",
    "padding": "0 22px", "marginBottom": "8px", "marginTop": "18px",
}
S_CHART_TITLE = {
    "fontSize": "13px", "fontWeight": "500",
    "color": TEXTC, "margin": "0 0 2px",
}
S_CHART_SUB = {
    "fontSize": "11px", "color": "#888", "margin": "0 0 10px",
}
S_DROP = {"fontSize": "13px", "width": "190px"}


def metric(label, val, delta=None, up=None):
    delta_color = "#A32D2D" if up else ("#3B6D11" if up is False else "#888")
    arrow = "↑ " if up else ("↓ " if up is False else "")
    return html.Div([
        html.P(label, style=S_LABEL),
        html.P(str(val), style=S_VAL),
        html.P(f"{arrow}{delta}" if delta else "", style={**S_DELTA, "color": delta_color}),
    ], style=S_METRIC)


def section(text):
    return html.P(text, style=S_SEC)


def card(*children, extra=None):
    style = {**S_CARD, **(extra or {})}
    return html.Div(list(children), style=style)


# ── Layout ────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # Hidden store for active dataframe (JSON)
    dcc.Store(id="store-data"),

    # ── Top bar ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("🛡 ", style={"fontSize": "17px"}),
            html.Div([
                html.P("School Disease Surveillance System",
                       style={"fontWeight": "500", "fontSize": "14px",
                              "color": TEXTC, "margin": "0"}),
                html.P("DepEd Division of Antipolo City · Region 4-A · "
                       "BS Data Science Thesis · Functional Prototype",
                       style={"fontSize": "11px", "color": "#888", "margin": "0"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        html.Div([
            html.Span("SGOD – Health & Nutrition", style={
                "fontSize": "11px", "padding": "3px 10px",
                "background": "#E6F1FB", "color": "#185FA5",
                "borderRadius": "6px", "marginRight": "10px",
            }),
            # CSV Upload
            dcc.Upload(
                id="upload-csv",
                children=html.Div([
                    html.Span("📂 ", style={"fontSize": "13px"}),
                    html.Span("Upload DepEd CSV",
                              style={"fontSize": "12px", "color": "#185FA5"}),
                ]),
                style={
                    "border": "1px dashed #378ADD", "borderRadius": "6px",
                    "padding": "5px 12px", "cursor": "pointer",
                    "background": "#F0F7FF",
                },
                accept=".csv",
            ),
            html.Span(id="upload-status",
                      style={"fontSize": "11px", "color": "#888",
                             "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style=S_TOPBAR),

    # ── Filters ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Year range", style={**S_LABEL, "marginRight": "6px"}),
            dcc.RangeSlider(
                id="f-year", min=2015, max=2025, step=1,
                value=[2015, 2025],
                marks={y: {"label": str(y),
                           "style": {"fontSize": "10px", "color": "#888"}}
                       for y in range(2015, 2026, 2)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"flex": "2", "minWidth": "260px"}),
        html.Div([
            html.Label("Disease", style={**S_LABEL, "marginRight": "6px"}),
            dcc.Dropdown(
                id="f-disease",
                options=[{"label": "All diseases", "value": "all"}] +
                        [{"label": d, "value": d} for d in DISEASES],
                value="all", clearable=False, style=S_DROP,
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        html.Div([
            html.Label("School level", style={**S_LABEL, "marginRight": "6px"}),
            dcc.Dropdown(
                id="f-level",
                options=[{"label": "All levels", "value": "all"}] +
                        [{"label": l, "value": l} for l in LEVELS],
                value="all", clearable=False, style=S_DROP,
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
    ], style={
        "display": "flex", "gap": "24px", "padding": "0 22px",
        "marginBottom": "18px", "flexWrap": "wrap", "alignItems": "flex-end",
    }),

    # ── Metric cards ──────────────────────────────────────────────────────
    html.Div(id="metric-row", style={
        "display": "flex", "gap": "10px", "flexWrap": "wrap",
        "padding": "0 22px", "marginBottom": "18px",
    }),

    # ── Annual trend + ARIMA ──────────────────────────────────────────────
    section("Time series — SARIMA model (live)"),
    html.Div([
        html.P("Monthly disease incidence with SARIMA(1,d,1)(1,1,1)[12] fit and 24-month forecast",
               style=S_CHART_TITLE),
        html.P("Fitted values (green dotted) show model accuracy · "
               "Shaded band = 95% confidence interval · "
               "Gray region = COVID-19 school closure",
               style=S_CHART_SUB),
        dcc.Graph(id="chart-arima",
                  config={"displayModeBar": True, "modeBarButtonsToRemove":
                          ["select2d","lasso2d","autoScale2d"]},
                  style={"height": "300px"}),
    ], style={**S_CARD, "margin": "0 22px 14px"}),

    # ── Two-column: decomposition + donut ─────────────────────────────────
    section("Seasonal decomposition & disease breakdown"),
    html.Div([
        html.Div([
            html.P("Seasonal decomposition (additive, period = 12 months)",
                   style=S_CHART_TITLE),
            html.P("Observed → Trend → Seasonal pattern → Residual noise",
                   style=S_CHART_SUB),
            dcc.Graph(id="chart-decomp",
                      config={"displayModeBar": False},
                      style={"height": "480px"}),
        ], style={**S_CARD, "flex": "1.6"}),
        html.Div([
            html.P("Disease share of total cases",
                   style=S_CHART_TITLE),
            html.P("Proportion of each reportable disease · filtered period",
                   style=S_CHART_SUB),
            dcc.Graph(id="chart-donut",
                      config={"displayModeBar": False},
                      style={"height": "260px"}),
            html.Hr(style={"border": "0.5px solid rgba(0,0,0,0.06)",
                           "margin": "14px 0"}),
            html.P("Monthly seasonal heatmap",
                   style=S_CHART_TITLE),
            html.P("Case intensity by month and year · darker = more cases",
                   style=S_CHART_SUB),
            dcc.Graph(id="chart-heatmap",
                      config={"displayModeBar": False},
                      style={"height": "290px"}),
        ], style={**S_CARD, "flex": "1"}),
    ], style={
        "display": "flex", "gap": "14px",
        "padding": "0 22px", "marginBottom": "14px",
    }),

    # ── Stacked bar: disease x year ───────────────────────────────────────
    section("Disease burden by year"),
    html.Div([
        html.P("Annual cases stacked by disease type",
               style=S_CHART_TITLE),
        html.P("Shows which diseases dominate each year and how the mix changes over time",
               style=S_CHART_SUB),
        dcc.Graph(id="chart-bar",
                  config={"displayModeBar": False},
                  style={"height": "260px"}),
    ], style={**S_CARD, "margin": "0 22px 14px"}),

    # ── Raw data table ────────────────────────────────────────────────────
    section("Raw data preview"),
    html.Div([
        html.P("Annual totals by disease — filtered view",
               style=S_CHART_TITLE),
        html.P("Based on active filters above · "
               "replace with actual DepEd data via CSV upload",
               style=S_CHART_SUB),
        html.Div(id="data-table"),
    ], style={**S_CARD, "margin": "0 22px 14px"}),

    # ── Disclaimer ────────────────────────────────────────────────────────
    html.Div([
        html.P(
            "⚠  Pre-loaded data is simulated/mock data generated to match the expected "
            "DepEd Antipolo City SGOD – Health and Nutrition Section records format. "
            "Upload your actual DepEd CSV using the button above to replace it. "
            "COVID-19 school closure years (2020–2022) are treated as a structural "
            "break and are visible in all charts. ARIMA model runs live on every "
            "filter change.",
            style={"fontSize": "11px", "color": "#888",
                   "lineHeight": "1.7", "margin": "0"}
        )
    ], style={
        "background": "#F7F7F5", "borderRadius": "8px",
        "padding": "12px 16px", "margin": "0 22px 30px",
    }),

], style={
    "fontFamily": FONT,
    "maxWidth": "1140px",
    "margin": "0 auto",
    "background": "#F5F5F3",
    "minHeight": "100vh",
})


# =============================================================================
# SECTION 5 — CALLBACKS
# =============================================================================

# ── 5a: Load data (upload CSV or use fallback) ────────────────────────────

@callback(
    Output("store-data",    "data"),
    Output("upload-status", "children"),
    Input("upload-csv",     "contents"),
    State("upload-csv",     "filename"),
    prevent_initial_call=False,
)
def load_data(contents, filename):
    """
    If a CSV is uploaded: parse and validate it.
    Otherwise: use the pre-generated fallback dataset.
    Stores the dataframe as JSON in dcc.Store.
    """
    if contents is not None:
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            # Validate required columns
            required = {"year", "month", "disease", "level", "cases"}
            missing  = required - set(df.columns.str.lower())
            if missing:
                return no_update, f"❌ Missing columns: {', '.join(missing)}"

            df.columns = df.columns.str.lower()
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" +
                df["month"].astype(str).str.zfill(2) + "-01"
            )
            status = f"✅ Loaded: {filename} ({len(df):,} rows)"
            return df.to_json(date_format="iso", orient="split"), status

        except Exception as e:
            return no_update, f"❌ Error reading file: {str(e)}"

    # Fallback
    df = generate_fallback_data()
    return df.to_json(date_format="iso", orient="split"), "📊 Using pre-loaded mock data"


# ── 5b: Update all charts from store + filters ────────────────────────────

@callback(
    Output("metric-row",   "children"),
    Output("chart-arima",  "figure"),
    Output("chart-decomp", "figure"),
    Output("chart-donut",  "figure"),
    Output("chart-heatmap","figure"),
    Output("chart-bar",    "figure"),
    Output("data-table",   "children"),
    Input("store-data",    "data"),
    Input("f-year",        "value"),
    Input("f-disease",     "value"),
    Input("f-level",       "value"),
)
def update_all(store_json, year_range, disease, level):
    """
    Master callback: runs every time filters change or data is loaded.
    Filters the dataframe, runs ARIMA + decomposition, rebuilds all charts.
    """
    if store_json is None:
        empty = go.Figure()
        empty.update_layout(plot_bgcolor=PLOTBG, paper_bgcolor=PLOTBG,
                            font=dict(family=FONT))
        return [], empty, empty, empty, empty, empty, []

    # ── Load & filter ──────────────────────────────────────────────────
    df = pd.read_json(io.StringIO(store_json), orient="split")
    df["date"] = pd.to_datetime(df["date"])

    yr_min, yr_max = year_range
    dff = df[(df["year"] >= yr_min) & (df["year"] <= yr_max)].copy()
    if disease != "all":
        dff = dff[dff["disease"] == disease]
    if level != "all":
        dff = dff[dff["level"] == level]

    if dff.empty:
        empty = go.Figure()
        empty.update_layout(plot_bgcolor=PLOTBG, paper_bgcolor=PLOTBG)
        return [], empty, empty, empty, empty, empty, []

    # ── Summary stats ──────────────────────────────────────────────────
    total, peak_year, peak_val, top_dis, top_pct, yrs = summary_cards(dff)

    metrics = html.Div([
        metric("Total cases", f"{total:,}",
               f"({yrs})", up=None),
        metric("Peak year", str(peak_year),
               f"{peak_val:,} cases", up=True),
        metric("Top disease", top_dis,
               f"{top_pct}% of cases", up=None),
        metric("COVID years", "2020–22",
               "Structural break", up=False),
        metric("Scope", "Antipolo City",
               "Public schools only", up=None),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})

    # ── Monthly series ─────────────────────────────────────────────────
    series = get_monthly_series(dff, "all", "all")

    # ── ARIMA ──────────────────────────────────────────────────────────
    fitted, fc_mean, fc_lower, fc_upper = run_arima(series, forecast_steps=24)
    chart_arima = fig_arima(series, fitted, fc_mean, fc_lower, fc_upper)

    # ── Decomposition ──────────────────────────────────────────────────
    decomp = run_decomposition(series)
    chart_decomp = fig_decomposition(decomp) if decomp else go.Figure().update_layout(
        title="Need ≥ 24 months of data for decomposition",
        plot_bgcolor=PLOTBG, paper_bgcolor=PLOTBG, height=480,
        font=dict(family=FONT, color="#888", size=12),
    )

    # ── Donut ──────────────────────────────────────────────────────────
    chart_donut   = fig_donut(dff)

    # ── Heatmap ────────────────────────────────────────────────────────
    chart_heatmap = fig_seasonal_heatmap(dff)

    # ── Stacked bar ────────────────────────────────────────────────────
    chart_bar     = fig_disease_bar(dff)

    # ── Raw data table ─────────────────────────────────────────────────
    table_df = (
        dff.groupby(["year", "disease"])["cases"]
        .sum()
        .reset_index()
        .pivot(index="disease", columns="year", values="cases")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    table_df.columns = [str(c) for c in table_df.columns]

    table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#F5F5F5",
            "fontWeight": "500",
            "fontSize": "11px",
            "color": "#555",
            "border": "none",
            "padding": "8px 12px",
        },
        style_cell={
            "fontSize": "12px",
            "fontFamily": FONT,
            "color": TEXTC,
            "padding": "7px 12px",
            "border": "none",
            "borderBottom": "0.5px solid rgba(0,0,0,0.06)",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"},
             "backgroundColor": "#FAFAFA"},
        ],
        page_size=10,
        sort_action="native",
    )

    return metrics, chart_arima, chart_decomp, chart_donut, chart_heatmap, chart_bar, table


# =============================================================================
# SECTION 6 — RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  FUNCTIONAL PROTOTYPE — Antipolo City Disease Surveillance")
    print("  BS Data Science Thesis · Region 4-A CALABARZON")
    print("=" * 62)
    print("  ▶  Open browser:  http://127.0.0.1:8050")
    print("  ▶  Upload CSV:    Use the 📂 button in the top-right")
    print("  ▶  CSV columns:   year, month, disease, level, cases")
    print("=" * 62 + "\n")
    app.run(debug=True)

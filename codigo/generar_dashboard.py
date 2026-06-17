from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "resultados"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_RESULTS = RESULTS_DIR / "resultados_modelo_oro_mejorado.xlsx"
HIST_FILE = DATA_DIR / "historicoauplata.xlsx"
OUTPUT_HTML = DASHBOARD_DIR / "index.html"

COLORS = {
    "base": "#f59e0b",
    "crisis_alta_incertidumbre": "#16a34a",
    "optimista": "#2563eb",
    "accent": "#f59e0b",
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "slate": "#334155",
}


def normalize_col(value: str) -> str:
    replacements = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return str(value).translate(replacements).lower().strip()


def find_col(columns: list[str], *tokens: str) -> str | None:
    for col in columns:
        norm = normalize_col(col)
        if any(token in norm for token in tokens):
            return col
    return None


def read_historical_gold() -> pd.DataFrame:
    raw = pd.read_excel(HIST_FILE, sheet_name="historicos", header=1)
    columns = list(raw.columns)
    date_col = find_col(columns, "fecha")
    open_col = find_col(columns, "apertura", "open")
    high_col = find_col(columns, "maximo", "high")
    low_col = find_col(columns, "minimo", "low")
    close_col = find_col(columns, "cierre", "close")
    rename = {
        date_col: "fecha",
        open_col: "apertura",
        high_col: "maximo",
        low_col: "minimo",
        close_col: "precio_oro",
    }
    rename = {k: v for k, v in rename.items() if k is not None}
    df = raw.rename(columns=rename).copy()
    keep = [col for col in ["fecha", "apertura", "maximo", "minimo", "precio_oro"] if col in df.columns]
    df = df[keep]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in keep:
        if col != "fecha":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["fecha", "precio_oro"]).sort_values("fecha").reset_index(drop=True)
    df["retorno_log"] = np.log(df["precio_oro"] / df["precio_oro"].shift(1))
    df["retorno_pct"] = df["precio_oro"].pct_change() * 100
    df["volatilidad_30d"] = df["retorno_log"].rolling(30).std() * np.sqrt(252) * 100
    df["ma_60"] = df["precio_oro"].rolling(60).mean()
    df["ma_200"] = df["precio_oro"].rolling(200).mean()
    return df


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "gold": read_historical_gold(),
        "cv": pd.read_excel(EXCEL_RESULTS, sheet_name="comparacion_modelos_cv"),
        "test": pd.read_excel(EXCEL_RESULTS, sheet_name="metricas_test_modelos"),
        "importance": pd.read_excel(EXCEL_RESULTS, sheet_name="importancia_variables"),
        "shap": pd.read_excel(EXCEL_RESULTS, sheet_name="shap_importancia"),
        "monthly": pd.read_excel(EXCEL_RESULTS, sheet_name="resumen_mensual_2026"),
        "scenario": pd.read_excel(EXCEL_RESULTS, sheet_name="resumen_escenarios_2035"),
        "dataset": pd.read_excel(EXCEL_RESULTS, sheet_name="dataset_modelado", nrows=1),
        "pred": pd.read_csv(RESULTS_DIR / "predicciones_test_modelo_mejorado.csv", parse_dates=["fecha", "fecha_objetivo"]),
        "forecast_2026": pd.read_csv(RESULTS_DIR / "pronostico_mayo_junio_julio_2026.csv", parse_dates=["fecha"]),
        "forecast_2035": pd.read_csv(RESULTS_DIR / "pronostico_hasta_2035.csv", parse_dates=["fecha"]),
        "single": pd.read_csv(RESULTS_DIR / "trayectoria_unica_monte_carlo_base.csv", parse_dates=["fecha"]),
    }
    return data


def fig_to_div(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=45, r=25, t=70, b=45),
        font=dict(family="Segoe UI, Arial", size=12),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
    )
    return pio.to_html(
        fig,
        include_plotlyjs=True if include_plotlyjs else False,
        full_html=False,
        config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:,.4f}")


def scenario_name(name: str) -> str:
    return {
        "base": "Base",
        "crisis_alta_incertidumbre": "Crisis / alta incertidumbre",
        "optimista": "Optimista",
    }.get(name, name)


def kpi_card(label: str, value: str, foot: str = "", tone: str = "gold") -> str:
    return f"""
    <article class="kpi {tone}">
      <span class="kpi-label">{label}</span>
      <strong>{value}</strong>
      <small>{foot}</small>
    </article>
    """


def fig_market_overview(gold: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32], subplot_titles=("Precio del oro y medias móviles", "Volatilidad anualizada 30 días"))
    if {"apertura", "maximo", "minimo"}.issubset(gold.columns):
        fig.add_trace(go.Candlestick(x=gold["fecha"], open=gold["apertura"], high=gold["maximo"], low=gold["minimo"], close=gold["precio_oro"], name="OHLC", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=gold["fecha"], y=gold["precio_oro"], name="Precio oro", line=dict(color=COLORS["accent"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=gold["fecha"], y=gold["ma_60"], name="MA 60", line=dict(color="#2563eb", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gold["fecha"], y=gold["ma_200"], name="MA 200", line=dict(color="#7c3aed", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gold["fecha"], y=gold["volatilidad_30d"], name="Vol 30d anualizada", fill="tozeroy", line=dict(color="#ef4444", width=1.4)), row=2, col=1)
    fig.update_layout(title="Panorama de mercado del oro", height=680, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="USD/oz", row=1, col=1)
    fig.update_yaxes(title_text="Vol. %", row=2, col=1)
    return fig


def fig_returns_distribution(gold: pd.DataFrame) -> go.Figure:
    returns = gold["retorno_pct"].dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribución de retornos diarios", "Retorno diario en el tiempo"))
    fig.add_trace(go.Histogram(x=returns, nbinsx=60, marker_color="#f59e0b", name="Retornos"), row=1, col=1)
    fig.add_vline(x=returns.mean(), line_color="#111827", line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scatter(x=gold["fecha"], y=gold["retorno_pct"], mode="lines", name="Retorno diario", line=dict(color="#334155", width=1)), row=1, col=2)
    fig.update_layout(title="Riesgo histórico: retornos y dispersión", height=440, showlegend=False)
    fig.update_xaxes(title_text="Retorno %", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Retorno %", row=1, col=2)
    return fig


def fig_model_leaderboard(cv: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    cvp = cv.copy().sort_values("cv_RMSE_mean", ascending=True)
    testp = test.copy().sort_values("RMSE", ascending=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validación temporal: menor RMSE es mejor", "Test final: contraste fuera de muestra"))
    fig.add_trace(go.Bar(y=cvp["family"], x=cvp["cv_RMSE_mean"], orientation="h", marker_color="#2563eb", text=cvp["cv_RMSE_mean"].round(2), name="CV RMSE"), row=1, col=1)
    fig.add_trace(go.Bar(y=testp["family"], x=testp["RMSE"], orientation="h", marker_color="#f59e0b", text=testp["RMSE"].round(2), name="Test RMSE"), row=1, col=2)
    fig.update_layout(title="Leaderboard de modelos", height=460, showlegend=False)
    fig.update_xaxes(title_text="RMSE")
    return fig


def fig_model_metrics_table(cv: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    merged = cv[["family", "optimized_by", "cv_MAE_mean", "cv_RMSE_mean", "cv_MAPE_mean", "cv_R2_mean"]].merge(
        test[["family", "MAE", "RMSE", "MAPE_pct", "R2"]], on="family", how="left"
    ).drop_duplicates("family")
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Modelo", "Optimización", "MAE CV", "RMSE CV", "MAPE CV", "R2 CV", "MAE Test", "RMSE Test", "MAPE Test", "R2 Test"], fill_color="#111827", font=dict(color="white"), align="left"),
        cells=dict(values=[
            merged["family"], merged["optimized_by"].fillna("manual"), merged["cv_MAE_mean"].round(3), merged["cv_RMSE_mean"].round(3), merged["cv_MAPE_mean"].round(3), merged["cv_R2_mean"].round(4), merged["MAE"].round(3), merged["RMSE"].round(3), merged["MAPE_pct"].round(3), merged["R2"].round(4)
        ], fill_color="#ffffff", align="left"))])
    fig.update_layout(title="Matriz de métricas por modelo", height=360)
    return fig


def fig_prediction_diagnostics(pred: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], subplot_titles=("Serie test: real vs predicho", "Dispersión real vs predicho", "Distribución del error"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=pred["fecha_objetivo"], y=pred["real_t1"], mode="lines", name="Real", line=dict(color="#111827")), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["fecha_objetivo"], y=pred["predicho_t1"], mode="lines", name="Predicho", line=dict(color="#dc2626")), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["real_t1"], y=pred["predicho_t1"], mode="markers", name="Real vs predicho", marker=dict(color="#2563eb", size=6, opacity=0.65)), row=2, col=1)
    min_val = min(pred["real_t1"].min(), pred["predicho_t1"].min())
    max_val = max(pred["real_t1"].max(), pred["predicho_t1"].max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Linea ideal", line=dict(color="#111827", dash="dash")), row=2, col=1)
    fig.add_trace(go.Histogram(x=pred["error"], nbinsx=45, name="Error", marker_color="#7c3aed"), row=2, col=2)
    fig.update_layout(title="Diagnóstico del desempeño predictivo", height=760)
    fig.update_yaxes(title_text="USD/oz", row=1, col=1)
    fig.update_xaxes(title_text="Real", row=2, col=1)
    fig.update_yaxes(title_text="Predicho", row=2, col=1)
    fig.update_xaxes(title_text="Error", row=2, col=2)
    return fig


def fig_error_profile(pred: pd.DataFrame) -> go.Figure:
    pred = pred.copy()
    pred["error_abs_ma20"] = pred["error_abs"].rolling(20).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Error diario", "Error absoluto y media móvil 20 días"))
    fig.add_trace(go.Bar(x=pred["fecha_objetivo"], y=pred["error"], name="Error", marker_color=np.where(pred["error"] >= 0, "#16a34a", "#dc2626")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["fecha_objetivo"], y=pred["error_abs"], mode="lines", name="Error absoluto", line=dict(color="#f59e0b", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred["fecha_objetivo"], y=pred["error_abs_ma20"], mode="lines", name="Media móvil 20d", line=dict(color="#111827", width=2)), row=2, col=1)
    fig.update_layout(title="Perfil de errores del modelo", height=600)
    fig.update_yaxes(title_text="USD/oz")
    return fig


def fig_feature_intelligence(importance: pd.DataFrame, shap: pd.DataFrame) -> go.Figure:
    imp = importance.head(18).sort_values("importance")
    shp = shap.head(18).sort_values("mean_abs_shap")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Importancia LightGBM", "Impacto medio absoluto SHAP"))
    fig.add_trace(go.Bar(y=imp["feature"], x=imp["importance"], orientation="h", marker_color="#0891b2", name="LightGBM"), row=1, col=1)
    fig.add_trace(go.Bar(y=shp["feature"], x=shp["mean_abs_shap"], orientation="h", marker_color="#84cc16", name="SHAP"), row=1, col=2)
    fig.update_layout(title="Inteligencia de variables", height=670, showlegend=False)
    return fig


def fig_variable_groups(dataset: pd.DataFrame) -> go.Figure:
    cols = [c for c in dataset.columns if c not in {"fecha", "target_precio_oro_t1", "target_logret_t1"}]
    groups = {
        "Oro técnico": [c for c in cols if "oro" in c or c == "precio_oro"],
        "Plata": [c for c in cols if "plata" in c or "ratio" in c],
        "NI 43-101": [c for c in cols if c.startswith("ni_")],
        "Macro": [c for c in cols if c.startswith("cpi") or c.startswith("fed") or c.startswith("sp500")],
        "Calendario": [c for c in cols if c in {"anio", "mes", "trimestre", "dia_mes", "dia_semana", "semana_anio", "mes_inicio", "mes_fin", "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"}],
    }
    fig = go.Figure(go.Pie(labels=list(groups.keys()), values=[len(v) for v in groups.values()], hole=0.55, marker=dict(colors=["#f59e0b", "#64748b", "#16a34a", "#2563eb", "#7c3aed"])))
    fig.update_layout(title="Composición de las 76 variables del modelo", height=420)
    return fig


def fig_forecast_2026(forecast: pd.DataFrame, monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bandas P10-P90 y mediana P50", "Promedio mensual por escenario"), specs=[[{}, {}]])
    for scenario, group in forecast.groupby("escenario"):
        color = COLORS.get(scenario, "#64748b")
        name = scenario_name(scenario)
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p10"], mode="lines", fill="tonexty", fillcolor="rgba(148,163,184,0.16)", line=dict(width=0), name=f"{name} P10-P90"), row=1, col=1)
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p50"], mode="lines", name=f"{name} P50", line=dict(color=color, width=2.4)), row=1, col=1)
    for scenario, group in monthly.groupby("escenario"):
        fig.add_trace(go.Bar(x=group["nombre_mes"], y=group["precio_promedio_usd_oz"], name=scenario_name(scenario), marker_color=COLORS.get(scenario)), row=1, col=2)
    fig.update_layout(title="Pronóstico de corto plazo: mayo, junio y julio de 2026", height=520, barmode="group")
    fig.update_yaxes(title_text="USD/oz")
    return fig


def fig_forecast_2035(forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for scenario, group in forecast.groupby("escenario"):
        color = COLORS.get(scenario, "#64748b")
        name = scenario_name(scenario)
        rgba = {
            "base": "rgba(245,158,11,0.13)",
            "crisis_alta_incertidumbre": "rgba(22,163,74,0.12)",
            "optimista": "rgba(37,99,235,0.12)",
        }.get(scenario, "rgba(100,116,139,0.12)")
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p10"], mode="lines", fill="tonexty", fillcolor=rgba, line=dict(width=0), name=f"{name} P10-P90"))
        fig.add_trace(go.Scatter(x=group["fecha"], y=group["precio_oro_p50"], mode="lines", name=f"{name} P50", line=dict(color=color, width=2.5)))
    fig.update_layout(title="Simulación Monte Carlo hasta 2035", height=620)
    fig.update_yaxes(title_text="USD/oz")
    return fig


def fig_single_path(single: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08, subplot_titles=("Trayectoria individual vs percentiles", "Choque diario aplicado al retorno base"))
    fig.add_trace(go.Scatter(x=single["fecha"], y=single["banda_p90_base"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=single["fecha"], y=single["banda_p10_base"], mode="lines", fill="tonexty", fillcolor="rgba(245,158,11,0.18)", line=dict(width=0), name="P10-P90"), row=1, col=1)
    fig.add_trace(go.Scatter(x=single["fecha"], y=single["mediana_escenario_base"], mode="lines", name="P50 base", line=dict(color="#f59e0b", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=single["fecha"], y=single["precio_oro_simulado"], mode="lines", name="Trayectoria única", line=dict(color="#dc2626", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=single["fecha"], y=single["choque_logret"], mode="lines", name="Choque log-retorno", line=dict(color="#7c3aed", width=1)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", row=2, col=1)
    fig.update_layout(title="Anatomía de una trayectoria Monte Carlo", height=700)
    fig.update_yaxes(title_text="USD/oz", row=1, col=1)
    fig.update_yaxes(title_text="Choque", row=2, col=1)
    return fig


def fig_scenario_summary(scenario: pd.DataFrame) -> go.Figure:
    plot = scenario.copy()
    plot["escenario_label"] = plot["escenario"].map(scenario_name)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Precio P50 inicial vs final", "Crecimiento P50 2026-2035"))
    fig.add_trace(go.Bar(x=plot["escenario_label"], y=plot["precio_inicio"], name="Inicio", marker_color="#94a3b8"), row=1, col=1)
    fig.add_trace(go.Bar(x=plot["escenario_label"], y=plot["precio_fin"], name="Fin", marker_color="#f59e0b"), row=1, col=1)
    fig.add_trace(go.Bar(x=plot["escenario_label"], y=plot["crecimiento_pct"], name="Crecimiento %", marker_color="#16a34a"), row=1, col=2)
    fig.update_layout(title="Resumen ejecutivo de escenarios", height=460, barmode="group")
    fig.update_yaxes(title_text="USD/oz", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    return fig


def build_dashboard(data: dict[str, pd.DataFrame]) -> str:
    gold = data["gold"]
    cv = data["cv"]
    test = data["test"]
    pred = data["pred"]
    monthly = data["monthly"]
    scenario = data["scenario"]
    single = data["single"]

    best_cv = cv.iloc[0]
    selected_test = test[test["family"] == best_cv["family"]].iloc[0]
    best_test_family = test.sort_values("RMSE").iloc[0]
    base_scenario = scenario[scenario["escenario"] == "base"].iloc[0]
    single_final = single.iloc[-1]
    last_gold = gold.iloc[-1]
    n_features = len([c for c in data["dataset"].columns if c not in {"fecha", "target_precio_oro_t1", "target_logret_t1"}])

    figs = [
        fig_market_overview(gold),
        fig_returns_distribution(gold),
        fig_model_leaderboard(cv, test),
        fig_model_metrics_table(cv, test),
        fig_prediction_diagnostics(pred),
        fig_error_profile(pred),
        fig_feature_intelligence(data["importance"], data["shap"]),
        fig_variable_groups(data["dataset"]),
        fig_forecast_2026(data["forecast_2026"], monthly),
        fig_forecast_2035(data["forecast_2035"]),
        fig_scenario_summary(scenario),
        fig_single_path(single),
    ]
    divs = [fig_to_div(fig, include_plotlyjs=(i == 0)) for i, fig in enumerate(figs)]

    monthly_table = html_table(monthly, 12)
    scenario_table = html_table(scenario, 10)
    top_features_table = html_table(data["importance"].head(12), 12)

    kpis = "".join([
        kpi_card("Modelo final", f"{best_cv['family']} + Optuna", "Seleccionado por menor RMSE CV", "blue"),
        kpi_card("RMSE validación", f"{best_cv['cv_RMSE_mean']:.2f}", "Promedio temporal", "gold"),
        kpi_card("MAPE test", f"{selected_test['MAPE_pct']:.2f}%", "Error porcentual fuera de muestra", "green"),
        kpi_card("R2 test", f"{selected_test['R2']:.4f}", "Ajuste en ventana final", "purple"),
        kpi_card("Variables", f"{n_features}", "Features usadas por LightGBM", "slate"),
        kpi_card("Último oro observado", f"{last_gold['precio_oro']:,.0f}", f"{last_gold['fecha']:%Y-%m-%d}", "gold"),
        kpi_card("P50 base 2035", f"{base_scenario['precio_fin']:,.0f}", "Mediana escenario base", "blue"),
        kpi_card("Trayectoria única 2035", f"{single_final['precio_oro_simulado']:,.0f}", "Una simulación individual", "red"),
    ])

    return rf"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predicción del precio del oro</title>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] }}, svg: {{ fontCache: 'global' }} }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      --bg:#eef2ef; --sidebar:#ffffff; --panel:#ffffff; --soft:#f7faf8; --text:#122018;
      --muted:#52645b; --line:#dfe7e2; --gold:#d6a73a; --blue:#1d4ed8; --green:#0f6b3a;
      --red:#b91c1c; --purple:#6d28d9; --slate:#334155; --cream:#f7f0dc;
    }}
    * {{ box-sizing:border-box; }}
    html, body {{ height:100%; overflow:hidden; }}
    body {{ margin:0; background:radial-gradient(circle at 16% 10%, rgba(214,167,58,.18), transparent 30%), linear-gradient(135deg,#f7faf8,#eef2ef 55%,#e6ece8); color:var(--text); font-family:Segoe UI, Arial, sans-serif; }}
    .layout {{ display:grid; grid-template-columns:292px 1fr; height:100vh; }}
    aside {{ background:rgba(255,255,255,.94); color:#122018; padding:24px 20px; border-right:1px solid rgba(15,77,47,.12); overflow:auto; box-shadow:12px 0 36px rgba(15,23,42,.08); }}
    aside h1 {{ font-size:19px; line-height:1.15; margin:0 0 8px; }}
    aside p {{ color:#52645b; font-size:13px; line-height:1.45; margin:0 0 16px; }}
    nav button {{ width:100%; display:block; color:#24362d; background:transparent; border:0; text-align:left; cursor:pointer; padding:9px 11px; border-radius:11px; margin:3px 0; font-size:13px; }}
    nav button:hover, nav button.active {{ background:#edf7ef; color:#0f4d2f; }}
    .main {{ min-width:0; height:100vh; display:flex; flex-direction:column; }}
    .stage {{ position:relative; flex:1; min-height:0; margin:22px 22px 12px; }}
    .slide {{ position:absolute; inset:0; overflow:auto; opacity:0; transform:translateX(24px) scale(.985); pointer-events:none; transition:opacity .24s ease, transform .24s ease; background:var(--panel); border:1px solid rgba(15,77,47,.12); border-radius:28px; padding:30px; box-shadow:0 22px 70px rgba(15,23,42,.13); }}
    .slide.active {{ opacity:1; transform:translateX(0) scale(1); pointer-events:auto; z-index:2; }}
    .cover {{ display:flex; flex-direction:column; min-height:100%; color:#122018; overflow:hidden; background:radial-gradient(circle at 86% 14%, rgba(214,167,58,.22), transparent 28%), linear-gradient(135deg,#ffffff 0%,#f8faf8 55%,#f3ead3 100%); }}
    .cover::before {{ content:""; position:absolute; inset:0; background:linear-gradient(115deg, rgba(15,77,47,.07), transparent 45%), repeating-linear-gradient(90deg, rgba(15,77,47,.045) 0 1px, transparent 1px 82px); pointer-events:none; }}
    .cover-content {{ position:relative; z-index:1; display:grid; grid-template-columns:1fr 180px; gap:28px; align-items:start; }}
    .logo-unal {{ width:150px; max-height:170px; object-fit:contain; justify-self:end; background:white; border:1px solid rgba(15,77,47,.12); border-radius:18px; padding:14px; box-shadow:0 16px 36px rgba(15,23,42,.10); }}
    .eyebrow {{ color:#0f6b3a; font-weight:800; letter-spacing:.16em; text-transform:uppercase; font-size:13px; }}
    .cover h2 {{ margin:16px 0 12px; color:#0f2419; font-size:clamp(36px,5.2vw,68px); line-height:1; max-width:980px; letter-spacing:-.052em; }}
    .subtitle {{ max-width:980px; color:#334155; font-size:clamp(18px,2vw,25px); line-height:1.35; margin:0 0 18px; }}
    .cover-intro {{ max-width:1040px; color:#334155; font-size:16px; line-height:1.58; margin:0 0 18px; }}
    .cover-meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; max-width:820px; }}
    .meta-card {{ background:rgba(255,255,255,.82); border:1px solid rgba(15,77,47,.12); border-radius:18px; padding:14px 16px; box-shadow:0 12px 30px rgba(15,23,42,.08); }}
    .meta-card span {{ display:block; color:#52645b; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .meta-card strong {{ display:block; margin-top:5px; font-size:18px; }}
    .summary-strip {{ position:relative; z-index:1; display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:18px; }}
    .summary-strip div {{ background:#10251a; color:white; border-radius:16px; padding:12px 14px; }}
    .summary-strip span {{ display:block; color:#f7d98d; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
    .summary-strip strong {{ display:block; margin-top:4px; }}
    .cover-footer {{ position:relative; z-index:1; margin-top:auto; display:flex; justify-content:space-between; gap:18px; align-items:end; color:#122018; border-top:1px solid rgba(15,77,47,.14); padding-top:20px; }}
    .cover-footer strong {{ display:block; font-size:20px; }}
    .cover-footer span {{ color:#52645b; }}
    .slide h2 {{ margin:0 0 10px; color:#0f4d2f; font-size:clamp(28px,3vw,44px); letter-spacing:-.035em; }}
    .slide h3 {{ margin:18px 0 8px; color:#143223; }}
    .lead {{ color:var(--muted); margin-top:0; line-height:1.6; font-size:17px; max-width:1080px; }}
    .tag {{ display:inline-flex; align-items:center; border-radius:999px; padding:6px 11px; background:#edf7ef; color:#0f6b3a; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:12px; }}
    .hero {{ background:radial-gradient(circle at top left, rgba(214,167,58,.25), transparent 28%), linear-gradient(135deg,#10251a,#1e293b); color:white; border-radius:24px; padding:26px; box-shadow:0 24px 80px rgba(0,0,0,.16); }}
    .hero h2 {{ color:white; margin:0; font-size:34px; letter-spacing:-.03em; }}
    .hero p {{ max-width:980px; color:#d7e2dc; line-height:1.6; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:20px; }}
    .kpi {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:16px; }}
    .kpi-label {{ color:#cbd5e1; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .kpi strong {{ display:block; color:white; font-size:25px; margin:7px 0 2px; }}
    .kpi small {{ color:#94a3b8; }}
    .kpi.gold strong {{ color:#fbbf24; }} .kpi.blue strong {{ color:#93c5fd; }} .kpi.green strong {{ color:#86efac; }} .kpi.red strong {{ color:#fca5a5; }} .kpi.purple strong {{ color:#c4b5fd; }}
    .two-col {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .insights {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin:12px 0 18px; }}
    .insight {{ background:var(--soft); border-left:5px solid var(--gold); border-radius:14px; padding:14px; color:#334155; line-height:1.45; }}
    .method-card {{ background:linear-gradient(180deg,#ffffff,#f8fbf8); border:1px solid var(--line); border-radius:18px; padding:18px; line-height:1.52; }}
    .method-card strong {{ color:#0f4d2f; }}
    .equation-card {{ background:#0f1f17; color:#f8fafc; border-radius:20px; padding:18px 20px; margin:12px 0; border:1px solid rgba(214,167,58,.28); overflow:auto; }}
    .equation-card p {{ color:#cbd8cf; margin:8px 0 0; }}
    .clean-list {{ margin:10px 0 0; padding-left:20px; line-height:1.58; }}
    .flow {{ counter-reset:step; display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
    .flow div {{ counter-increment:step; background:#f8faf8; border:1px solid var(--line); border-radius:16px; padding:14px; }}
    .flow div::before {{ content:counter(step); display:inline-grid; place-items:center; width:26px; height:26px; border-radius:50%; background:#0f6b3a; color:white; font-weight:700; margin-right:8px; }}
    .compact-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .compact-table th {{ background:#10251a; color:#fff; text-align:left; padding:10px; }}
    .compact-table td {{ border-bottom:1px solid var(--line); padding:10px; vertical-align:top; }}
    .reference-list {{ display:grid; gap:10px; margin-top:14px; }}
    .reference-list div {{ background:#f8faf8; border:1px solid var(--line); border-radius:16px; padding:14px 16px; color:#334155; line-height:1.5; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:13px; overflow:hidden; border-radius:12px; }}
    .data-table th {{ background:#111827; color:white; padding:9px; text-align:left; position:sticky; top:0; }}
    .data-table td {{ padding:8px 9px; border-bottom:1px solid #e5e7eb; }}
    .table-wrap {{ max-height:430px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    .downloads {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
    .downloads a {{ display:block; background:#0f172a; color:white; padding:14px 16px; border-radius:14px; text-decoration:none; }}
    .downloads a:hover {{ background:#1e293b; }}
    .controls {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 22px 18px; color:#122018; }}
    .control-group {{ display:flex; gap:10px; align-items:center; }}
    .controls button {{ border:1px solid rgba(15,77,47,.16); background:#ffffff; color:#122018; border-radius:999px; padding:10px 16px; cursor:pointer; font-weight:700; box-shadow:0 8px 20px rgba(15,23,42,.08); }}
    .controls button:hover {{ background:#edf7ef; }}
    .progress {{ flex:1; height:8px; background:rgba(15,77,47,.13); border-radius:999px; overflow:hidden; }}
    .progress span {{ display:block; height:100%; width:0; background:linear-gradient(90deg,#0f6b3a,#d6a73a); transition:width .2s ease; }}
    .counter {{ min-width:88px; text-align:center; color:#52645b; font-variant-numeric:tabular-nums; }}
    footer {{ color:#64746b; text-align:center; padding:14px; }}
    @media (max-width: 1100px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ display:none; }} .stage {{ margin:12px; }} .slide {{ padding:20px; border-radius:22px; }} .two-col, .grid-2, .grid-3, .cover-content {{ grid-template-columns:1fr; }} .logo-unal {{ justify-self:start; width:92px; }} .cover-footer {{ display:block; }} html, body {{ overflow:hidden; }} }}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>Modelo del precio del oro</h1>
    <p>Entregable académico para Economía de Minas, con resultados del modelo LightGBM + Optuna y escenarios Monte Carlo.</p>
    <nav id="nav">
      <button type="button" data-slide="0">Portada</button>
      <button type="button" data-slide="1">Resumen ejecutivo</button>
      <button type="button" data-slide="2">Contexto oro</button>
      <button type="button" data-slide="3">Objetivos</button>
      <button type="button" data-slide="4">Marco teórico</button>
      <button type="button" data-slide="5">Método tradicional</button>
      <button type="button" data-slide="6">XGBoost</button>
      <button type="button" data-slide="7">Hiperparámetros</button>
      <button type="button" data-slide="8">Ecuaciones</button>
      <button type="button" data-slide="9">Materiales y métodos</button>
      <button type="button" data-slide="10">Flujo metodológico</button>
      <button type="button" data-slide="11">Variables</button>
      <button type="button" data-slide="12">Métricas</button>
      <button type="button" data-slide="13">Mercado histórico</button>
      <button type="button" data-slide="14">Modelos</button>
      <button type="button" data-slide="15">Validación test</button>
      <button type="button" data-slide="16">Interpretabilidad</button>
      <button type="button" data-slide="17">Pronóstico 2026</button>
      <button type="button" data-slide="18">Monte Carlo 2035</button>
      <button type="button" data-slide="19">Trayectoria única</button>
      <button type="button" data-slide="20">Discusión</button>
      <button type="button" data-slide="21">Conclusiones</button>
      <button type="button" data-slide="22">Referencias</button>
      <button type="button" data-slide="23">Archivos</button>
    </nav>
  </aside>
  <main class="main">
    <div class="stage" id="stage">
    <section class="slide cover active" id="portada">
      <div class="cover-content">
        <div>
          <div class="eyebrow">Asignatura: Economía de Minas · Taller 4</div>
          <h2>Predicción del precio del oro: método tradicional vs Machine Learning</h2>
          <p class="subtitle">Entregable académico para contrastar un enfoque tendencial tradicional con modelos de Machine Learning aplicados a la proyección del commodity oro.</p>
          <p class="cover-intro">Este entregable presenta una comparación metodológica entre una regresión polinómica como referencia tradicional y modelos basados en árboles. La versión final del proyecto conserva la evaluación ampliada de algoritmos y selecciona LightGBM + Optuna por validación temporal, sin interpretar los escenarios como precios garantizados.</p>
          <div class="cover-meta">
            <div class="meta-card"><span>Autores</span><strong>Alejandro Ramírez Polo y Leidy Paola Patiño Muñoz</strong></div>
            <div class="meta-card"><span>Profesor</span><strong>Jheyson Bedoya</strong></div>
          </div>
        </div>
        <img class="logo-unal" src="assets/img/logo-unal.png" alt="Universidad Nacional de Colombia">
      </div>
      <div class="summary-strip">
        <div><span>Commodity</span><strong>Oro</strong></div>
        <div><span>Unidad</span><strong>USD/oz</strong></div>
        <div><span>Horizonte</span><strong>10 años</strong></div>
        <div><span>Método tradicional</span><strong>Regresión polinómica</strong></div>
        <div><span>Machine Learning</span><strong>XGBoost / LightGBM</strong></div>
        <div><span>Escenario central</span><strong>P50</strong></div>
      </div>
      <div class="cover-footer">
        <div><strong>Universidad Nacional de Colombia</strong><span>Facultad de Minas - Sede Medellín</span></div>
        <div><span>Junio de 2026</span></div>
      </div>
    </section>

    <section class="slide" id="resumen">
      <div class="hero">
        <span class="tag">Resumen ejecutivo</span>
        <h2>Modelo multivariado del precio del oro</h2>
      <p>Resultados de Machine Learning, validación temporal, importancia de variables, pronósticos por escenarios y simulación Monte Carlo. La selección principal se hizo por menor RMSE promedio de validación temporal.</p>
        <div class="kpi-grid">{kpis}</div>
      </div>
    </section>

    <section class="slide" id="introduccion">
      <span class="tag">Contexto</span>
      <h2>Contexto del commodity oro</h2>
      <p class="lead">El precio del oro es una variable relevante para la evaluación económica de proyectos mineros, la estimación de recursos y reservas, la planeación financiera y el análisis de riesgo. Su comportamiento depende de factores financieros, macroeconómicos y de mercado, entre ellos tasas de interés, inflación, dólar, volatilidad, precios de metales relacionados y expectativas de inversión.</p>
      <div class="grid-3">
        <div class="method-card"><strong>Commodity metálico</strong><br>El oro se negocia internacionalmente como activo financiero, reserva de valor e insumo relevante para análisis minero.</div>
        <div class="method-card"><strong>Horizonte histórico</strong><br>La base del proyecto cubre el precio del oro entre 2016 y 2026, junto con variables de mercado relacionadas.</div>
        <div class="method-card"><strong>Uso esperado</strong><br>Los resultados se interpretan como pronóstico condicional y distribución de escenarios posibles, no como precio exacto e invariable.</div>
      </div>
    </section>

    <section class="slide" id="objetivos">
      <span class="tag">Propósito</span>
      <h2>Objetivos</h2>
      <div class="grid-2">
        <div class="method-card"><strong>Objetivo general</strong><br>Construir, validar y documentar un modelo multivariado de Machine Learning para estimar el comportamiento del precio del oro y generar escenarios de simulación hasta 2035.</div>
        <div class="method-card"><strong>Objetivos específicos</strong>
          <ul class="clean-list">
            <li>Consolidar datos de oro, plata, NI 43-101 y variables macroeconómicas.</li>
            <li>Construir variables técnicas, temporales, macroeconómicas y mineras.</li>
            <li>Comparar modelos con validación temporal, evitando fuga de información.</li>
            <li>Generar pronósticos de corto plazo y escenarios Monte Carlo hasta 2035.</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="slide" id="marco-teorico">
      <span class="tag">Marco teórico</span>
      <h2>Predicción de series financieras</h2>
      <p class="lead">Las series financieras suelen presentar ruido, cambios de tendencia, volatilidad variable y dependencia temporal. Por ello, se modeló el log-retorno de un día adelante en lugar de predecir directamente el nivel del precio.</p>
      <div class="grid-3">
        <div class="method-card"><strong>Boosting</strong><br>LightGBM y XGBoost construyen árboles secuenciales que corrigen errores residuales.</div>
        <div class="method-card"><strong>Ensambles</strong><br>Random Forest y Extra Trees combinan múltiples árboles para capturar relaciones no lineales.</div>
        <div class="method-card"><strong>Optuna</strong><br>Busca hiperparámetros automáticamente minimizando el RMSE de validación temporal.</div>
      </div>
    </section>

    <section class="slide" id="metodo-tradicional">
      <span class="tag">Método tradicional</span>
      <h2>Regresión polinómica como referencia</h2>
      <p class="lead">El método tradicional ajusta una curva matemática al precio histórico del oro y luego extrapola esa tendencia hacia el futuro. Su ventaja es la simplicidad e interpretación directa; su limitación es que depende mucho de la forma de la curva y puede volverse rígido en horizontes largos.</p>
      <div class="grid-2">
        <div class="equation-card">\[P_{{trad}}(t)=\beta_0+\beta_1t+\beta_2t^2\]<p>Ajuste tendencial de segundo grado usado como referencia conceptual.</p></div>
        <div class="method-card"><strong>Alcance dentro de este entregable</strong><br>No se modifican los resultados numéricos del proyecto. La comparación final disponible en los archivos corresponde a familias de Machine Learning evaluadas con validación temporal; la regresión polinómica se presenta como base metodológica tradicional para interpretación.</div>
      </div>
    </section>

    <section class="slide" id="metodo-xgboost">
      <span class="tag">Machine Learning</span>
      <h2>Método XGBoost</h2>
      <p class="lead">XGBoost es un algoritmo supervisado basado en un ensamble secuencial de árboles de decisión. Cada árbol corrige parte del error cometido por los árboles anteriores. En este proyecto se comparó XGBoost con otras familias; la versión mejorada seleccionó LightGBM + Optuna por menor RMSE promedio de validación temporal.</p>
      <div class="grid-2">
        <div class="equation-card">\[y_t=\ln\left(\frac{{P_{{t+1}}^{{Au}}}}{{P_t^{{Au}}}}\right)\]\[\hat{{P}}_{{t+1}}^{{Au}}=P_t^{{Au}}\exp(\hat{{y}}_t)\]<p>El modelo predice primero el log-retorno y luego reconstruye el precio.</p></div>
        <div class="equation-card">\[\hat{{y}}_t=\sum_{{m=1}}^M \eta f_m(x_t)\]<p>\(f_m(x_t)\) es el árbol de decisión número \(m\), \(\eta\) es el learning rate y \(M\) el número total de árboles.</p></div>
      </div>
    </section>

    <section class="slide" id="hiperparametros">
      <span class="tag">Configuración</span>
      <h2>Hiperparámetros principales de XGBoost</h2>
      <p class="lead">Estos hiperparámetros corresponden a la configuración XGBoost registrada en el código del proyecto y buscan equilibrar capacidad predictiva con control del sobreajuste.</p>
      <div class="table-wrap">
        <table class="compact-table">
          <thead><tr><th>Hiperparámetro</th><th>Valor</th><th>Interpretación</th></tr></thead>
          <tbody>
            <tr><td><code>n_estimators</code></td><td>500</td><td>Número de árboles del ensamble.</td></tr>
            <tr><td><code>max_depth</code></td><td>4</td><td>Profundidad máxima de cada árbol.</td></tr>
            <tr><td><code>learning_rate</code></td><td>0.03</td><td>Aporte incremental de cada árbol.</td></tr>
            <tr><td><code>subsample</code></td><td>0.85</td><td>Porcentaje de datos usado por cada árbol.</td></tr>
            <tr><td><code>alpha</code></td><td>0.02</td><td>Regularización L1.</td></tr>
            <tr><td><code>lambda</code></td><td>1.40</td><td>Regularización L2.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="slide" id="ecuaciones">
      <span class="tag">Formulación</span>
      <h2>Ecuaciones principales</h2>
      <div class="grid-2">
        <div class="equation-card">\[r_{{t+1}}=\ln\left(\frac{{P_{{t+1}}}}{{P_t}}\right)\]<p>Log-retorno del precio de cierre del oro.</p></div>
        <div class="equation-card">\[\hat{{P}}_{{t+1}}=P_t e^{{\hat{{r}}_{{t+1}}}}\]<p>Reconstrucción del precio predicho.</p></div>
        <div class="equation-card">\[MA_k(t)=\frac{{1}}{{k}}\sum_{{i=0}}^{{k-1}}P_{{t-i}}\]<p>Media móvil simple de ventana \(k\).</p></div>
        <div class="equation-card">\[\sigma_k(t)=\sqrt{{\frac{{1}}{{k-1}}\sum_{{i=0}}^{{k-1}}\left(r_{{t-i}}-\bar{{r}}_k\right)^2}}\]<p>Volatilidad móvil de retornos recientes.</p></div>
      </div>
    </section>

    <section class="slide" id="materiales-metodos">
      <span class="tag">Materiales y métodos</span>
      <h2>Datos y fuentes</h2>
      <p class="lead">El archivo base fue <strong>historicoauplata.xlsx</strong>. Se usaron las hojas de históricos, plata y NI 43-101; además se consultaron series macroeconómicas externas desde FRED cuando estuvieron disponibles. Estas variables permiten representar tanto la dinámica histórica del oro como condiciones de mercado que pueden influir en su comportamiento futuro.</p>
      <div class="grid-3">
        <div class="method-card"><strong>Oro XAU/USD</strong><br>Serie principal de precio de cierre, variable objetivo y variables técnicas.</div>
        <div class="method-card"><strong>Plata</strong><br>Variable exógena para capturar relación entre metales preciosos.</div>
        <div class="method-card"><strong>NI 43-101</strong><br>Precios de recursos y reservas como referencia minera.</div>
        <div class="method-card"><strong>Macrofinancieras</strong><br>CPI, fed funds y S&amp;P 500 disponibles.</div>
        <div class="method-card"><strong>Calendario</strong><br>Año, mes, trimestre, día de semana y ciclos senoidales.</div>
        <div class="method-card"><strong>Base final</strong><br>76 variables explicativas después de limpieza y construcción de características.</div>
      </div>
    </section>

    <section class="slide" id="flujo">
      <span class="tag">Metodología</span>
      <h2>Flujo metodológico</h2>
      <div class="flow">
        <div>Lectura automática del archivo de datos.</div>
        <div>Detección de hoja histórica y columna objetivo.</div>
        <div>Limpieza de fechas, precios y valores faltantes.</div>
        <div>Integración de plata, NI 43-101 y variables macroeconómicas.</div>
        <div>Construcción de variables explicativas.</div>
        <div>Definición del objetivo como log-retorno de un día adelante.</div>
        <div>Separación temporal en entrenamiento y prueba.</div>
        <div>Validación temporal y selección por RMSE promedio.</div>
        <div>Pronóstico recursivo y simulación Monte Carlo.</div>
      </div>
    </section>

    <section class="slide" id="variables-metodo">
      <span class="tag">Variables explicativas</span>
      <h2>Construcción de características</h2>
      <div class="grid-2">
        <div class="method-card"><strong>Grupos usados</strong>
          <ul class="clean-list">
            <li>Nivel de precios: oro y plata.</li>
            <li>Rezagos del precio del oro.</li>
            <li>Retornos, diferencias, medias móviles y volatilidades.</li>
            <li>Relación oro-plata.</li>
            <li>Variables NI 43-101, macrofinancieras y calendario.</li>
          </ul>
        </div>
        <div>{divs[7]}</div>
      </div>
    </section>

    <section class="slide" id="metricas">
      <span class="tag">Evaluación</span>
      <h2>Métricas y Monte Carlo</h2>
      <div class="grid-2">
        <div class="equation-card">\[MAE=\frac{{1}}{{n}}\sum_{{i=1}}^n|y_i-\hat{{y}}_i|\]\[RMSE=\sqrt{{\frac{{1}}{{n}}\sum_{{i=1}}^n(y_i-\hat{{y}}_i)^2}}\]\[MAPE=\frac{{100}}{{n}}\sum_{{i=1}}^n\left|\frac{{y_i-\hat{{y}}_i}}{{y_i}}\right|\]<p>Métricas de error y ajuste fuera de muestra.</p></div>
        <div class="equation-card">\[r_t^{{MC}}=\hat{{r}}_t+\varepsilon_t\sigma_s+J_t\]\[P_t^{{MC}}=P_{{t-1}}^{{MC}}e^{{r_t^{{MC}}}}\]<p>Simulación con choques, volatilidad por escenario y saltos discretos.</p></div>
      </div>
    </section>

    <section class="slide" id="mercado">
      <span class="tag">Análisis de mercado</span>
      <h2>Mercado histórico y riesgo</h2>
      <p class="lead">Evolución del oro, medias móviles y volatilidad. Este bloque permite ubicar el régimen reciente antes de interpretar los pronósticos.</p>
      {divs[0]}
      {divs[1]}
    </section>

    <section class="slide" id="modelos">
      <span class="tag">Resultados</span>
      <h2>Selección y comparación de modelos</h2>
      <div class="insights">
        <div class="insight"><strong>Selección:</strong> LightGBM + Optuna fue escogido por menor RMSE en validación temporal.</div>
        <div class="insight"><strong>Contraste:</strong> CatBoost tuvo el menor RMSE de test, pero no se usa para seleccionar a posteriori.</div>
        <div class="insight"><strong>Criterio:</strong> se evita fuga de información y se respeta el orden cronológico.</div>
      </div>
      {divs[2]}
      {divs[3]}
    </section>

    <section class="slide" id="test">
      <span class="tag">Validación</span>
      <h2>Diagnóstico fuera de muestra</h2>
      <p class="lead">Comparación entre precios reales y predichos sobre la ventana final de prueba. Incluye dispersión y distribución de errores.</p>
      {divs[4]}
      {divs[5]}
    </section>

    <section class="slide" id="variables">
      <span class="tag">Interpretabilidad</span>
      <h2>Variables e interpretabilidad</h2>
      <p class="lead">El modelo usa variables técnicas del oro, plata, NI 43-101, macrofinancieras y calendario. La importancia muestra que volatilidad y retornos recientes dominan la decisión de los árboles.</p>
      <div class="two-col">
        <div>{divs[6]}</div>
        <div>{divs[7]}<h3>Top variables</h3><div class="table-wrap">{top_features_table}</div></div>
      </div>
    </section>

    <section class="slide" id="forecast2026">
      <span class="tag">Pronóstico</span>
      <h2>Pronóstico de corto plazo 2026</h2>
      <p class="lead">Mayo, junio y julio de 2026 por escenarios. La banda P10-P90 representa incertidumbre simulada y P50 la mediana.</p>
      {divs[8]}
      <h3>Resumen mensual</h3><div class="table-wrap">{monthly_table}</div>
    </section>

    <section class="slide" id="montecarlo">
      <span class="tag">Simulación</span>
      <h2>Monte Carlo hasta 2035</h2>
      <p class="lead">Escenarios de largo plazo. No son valores deterministas: muestran distribuciones posibles bajo supuestos de retorno, volatilidad y saltos.</p>
      <div class="insights">
        <div class="insight"><strong>P10:</strong> escenario bajo. El 10% de las simulaciones queda por debajo de ese valor.</div>
        <div class="insight"><strong>P50:</strong> escenario central o mediano. No garantiza el precio futuro; resume la trayectoria central de simulación.</div>
        <div class="insight"><strong>P90:</strong> escenario alto. El 90% de las simulaciones queda por debajo de ese valor.</div>
        <div class="insight"><strong>Banda P10-P90:</strong> representa aproximadamente el 80% central de los escenarios simulados.</div>
      </div>
      {divs[9]}
      {divs[10]}
      <h3>Tabla de escenarios</h3><div class="table-wrap">{scenario_table}</div>
    </section>

    <section class="slide" id="trayectoria">
      <span class="tag">Simulación individual</span>
      <h2>Una trayectoria Monte Carlo</h2>
      <p class="lead">La línea roja es una sola ruta posible. Se calcula con el retorno base de LightGBM, choques Monte Carlo y la fórmula acumulada de precios.</p>
      {divs[11]}
    </section>

    <section class="slide" id="discusion">
      <span class="tag">Discusión</span>
      <h2>Análisis de resultados</h2>
      <div class="grid-2">
        <div class="method-card"><strong>Lectura técnica</strong><br>El uso de log-retornos permitió estabilizar la variable objetivo y reducir el riesgo de fuga de información. La comparación temporal mostró que los modelos de boosting capturan relaciones no lineales entre variables técnicas, plata, precios NI y variables macroeconómicas.</div>
        <div class="method-card"><strong>Interpretación económica</strong><br>Las variables más importantes se relacionan con volatilidad reciente del oro, retornos de la plata y variaciones del S&amp;P 500. Esto sugiere una combinación entre dinámica propia del oro, relación con otro metal precioso y condiciones financieras generales.</div>
        <div class="method-card"><strong>Limitaciones</strong><br>Algunas series externas de FRED no estuvieron disponibles en la ejecución final por errores HTTP. Además, el horizonte largo acumula incertidumbre porque cada precio futuro depende del precio simulado anterior.</div>
        <div class="method-card"><strong>Uso recomendado</strong><br>Los resultados a 2035 deben verse como escenarios de sensibilidad y discusión económica, no como predicciones puntuales garantizadas.</div>
      </div>
    </section>

    <section class="slide" id="conclusiones">
      <span class="tag">Cierre</span>
      <h2>Conclusiones</h2>
      <div class="insights">
        <div class="insight">Se construyó un flujo reproducible de Machine Learning para el precio del oro, respetando el orden temporal.</div>
        <div class="insight">La versión inicial basada en XGBoost fue mejorada con comparación amplia de algoritmos y optimización de hiperparámetros.</div>
        <div class="insight">El modelo seleccionado por validación temporal fue LightGBM optimizado con Optuna, con MAPE de 1.1323% en test.</div>
        <div class="insight">La simulación Monte Carlo permitió generar bandas de incertidumbre y escenarios hasta 2035 útiles para análisis económico y sensibilidad en proyectos mineros.</div>
      </div>
    </section>

    <section class="slide" id="referencias">
      <span class="tag">Soporte académico</span>
      <h2>Referencias</h2>
      <p class="lead">Referencias principales usadas para sustentar los modelos, la predicción de series temporales y el contexto técnico del proyecto.</p>
      <div class="reference-list">
        <div>Chen, T. y Guestrin, C. (2016). <em>XGBoost: A scalable tree boosting system</em>. Proceedings of KDD.</div>
        <div>Ke, G. et al. (2017). <em>LightGBM: A highly efficient gradient boosting decision tree</em>. NeurIPS.</div>
        <div>Prokhorenkova, L. et al. (2018). <em>CatBoost: unbiased boosting with categorical features</em>. NeurIPS.</div>
        <div>Hyndman, R. y Athanasopoulos, G. (2021). <em>Forecasting: Principles and Practice</em>.</div>
        <div>Canadian Securities Administrators. (2014). <em>National Instrument 43-101 Standards of Disclosure for Mineral Projects</em>.</div>
        <div>Universidad Nacional de Colombia. Guía de identidad visual: <a href="https://identidad.unal.edu.co/guia-identidad-visual/" target="_blank" rel="noopener">identidad.unal.edu.co</a>.</div>
      </div>
    </section>

    <section class="slide" id="descargas">
      <span class="tag">Soporte</span>
      <h2>Archivos del entregable</h2>
      <div class="downloads">
        <a href="../pdf/informe_principal.pdf">Informe principal PDF</a>
        <a href="../pdf/lightgbm_optuna_detallado.pdf">PDF LightGBM + Optuna</a>
        <a href="../resultados/resultados_modelo_oro_mejorado.xlsx">Resultados Excel</a>
        <a href="../resultados/pronostico_hasta_2035.csv">Pronóstico hasta 2035 CSV</a>
        <a href="../resultados/trayectoria_unica_monte_carlo_base.csv">Trayectoria única CSV</a>
        <a href="../fuentes_latex_editables.zip">Fuentes LaTeX editables</a>
      </div>
      <footer>Universidad Nacional de Colombia - Facultad de Minas.</footer>
    </section>
    </div>
    <div class="controls">
      <div class="control-group">
        <button type="button" id="prev">Anterior</button>
        <button type="button" id="next">Siguiente</button>
      </div>
      <div class="progress" aria-hidden="true"><span id="progressBar"></span></div>
      <div class="counter" id="counter">1 / 24</div>
    </div>
  </main>
</div>
<script>
  const slides = Array.from(document.querySelectorAll('.slide'));
  const navButtons = Array.from(document.querySelectorAll('[data-slide]'));
  const progressBar = document.getElementById('progressBar');
  const counter = document.getElementById('counter');
  const prev = document.getElementById('prev');
  const next = document.getElementById('next');
  let current = 0;

  function showSlide(index) {{
    current = Math.max(0, Math.min(index, slides.length - 1));
    slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
    navButtons.forEach((button) => button.classList.toggle('active', Number(button.dataset.slide) === current));
    progressBar.style.width = (((current + 1) / slides.length) * 100) + '%';
    counter.textContent = (current + 1) + ' / ' + slides.length;
    prev.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    const activeSlide = slides[current];
    activeSlide.scrollTop = 0;
    if (window.Plotly) {{
      activeSlide.querySelectorAll('.js-plotly-plot').forEach((plot) => window.Plotly.Plots.resize(plot));
    }}
    if (window.MathJax && window.MathJax.typesetPromise) {{
      window.MathJax.typesetPromise([activeSlide]);
    }}
  }}

  prev.addEventListener('click', () => showSlide(current - 1));
  next.addEventListener('click', () => showSlide(current + 1));
  navButtons.forEach((button) => button.addEventListener('click', () => showSlide(Number(button.dataset.slide))));
  document.addEventListener('keydown', (event) => {{
    if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') showSlide(current + 1);
    if (event.key === 'ArrowLeft' || event.key === 'PageUp') showSlide(current - 1);
  }});
  window.addEventListener('resize', () => showSlide(current));
  showSlide(0);
</script>
</body>
</html>"""


def main() -> None:
    data = load_data()
    OUTPUT_HTML.write_text(build_dashboard(data), encoding="utf-8")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()

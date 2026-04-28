"""
Proyecto SIGMA — Dashboard Interactivo de Monitoreo del SSFV
Pontificia Universidad Javeriana Cali — Oficina de Recursos Físicos y Ambientales

Prototipo Fase 1: Simulación de condiciones climáticas de Cali, Valle del Cauca,
con perturbaciones manuales interactivas de sombra y polvo. El sistema correlaciona
daños físicos/ambientales en el panel con parámetros de Calidad de Energía (PQ).

Uso:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from datetime import datetime

from climate import CALI_PROFILE, get_hourly_irradiance, get_hourly_temperature
from diagnostics import AlertSeverity, DiagnosticEngine
from panel_model import ETA_NOMINAL_PCT, PANEL_SPECS, calculate_panel_state
from power_quality import (
    THD_LIMIT_CRITICAL_PCT,
    THD_LIMIT_WARNING_PCT,
    FP_LIMIT_WARNING,
    calculate_power_quality,
)


# ─── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="SIGMA — SSFV Javeriana Cali",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .sigma-header {
        color: #1a237e; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    .alert-box { padding: 12px 16px; border-radius: 6px; margin: 6px 0; }
    .alert-critical { background-color: #ffebee; border-left: 5px solid #b71c1c; }
    .alert-warning  { background-color: #fffde7; border-left: 5px solid #f9a825; }
    .alert-info     { background-color: #e3f2fd; border-left: 5px solid #1565c0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

_ENGINE = DiagnosticEngine()


# ─── Visualizaciones ───────────────────────────────────────────────────────────

def build_panel_figure(shadow_pct: float, dust_pct: float) -> go.Figure:
    """
    Construye la figura interactiva que representa visualmente el módulo FV.

    Modela un módulo de 60 celdas (6 filas × 10 columnas). Las celdas sombreadas
    se oscurecen progresivamente desde la esquina superior-izquierda, simulando
    el avance de una sombra sobre el campo solar. El polvo reduce el brillo
    global de todas las celdas de forma uniforme.

    Colorscale:
        Negro (#0d0d1a) → celda completamente sombreada.
        Azul oscuro     → baja producción.
        Azul claro      → producción media.
        Amarillo        → producción máxima.

    Args:
        shadow_pct: Porcentaje de área sombreada [0, 100].
        dust_pct:   Porcentaje de polvo acumulado [0, 100].

    Returns:
        Figura de Plotly con el mapa de calor del panel fotovoltaico.
    """
    ROWS, COLS = 6, 10
    total_cells = ROWS * COLS
    shaded_count = int(total_cells * shadow_pct / 100.0)
    # El polvo reduce la luminosidad de las celdas activas (máx. 25 %)
    active_brightness = 1.0 - (dust_pct / 100.0) * 0.25

    values = np.full((ROWS, COLS), active_brightness, dtype=float)

    # Sombra avanza columna a columna (de izquierda a derecha)
    cells_shaded = 0
    for col in range(COLS):
        for row in range(ROWS):
            if cells_shaded >= shaded_count:
                break
            values[row, col] = 0.07  # celda sombreada: casi oscura
            cells_shaded += 1
        if cells_shaded >= shaded_count:
            break

    colorscale = [
        [0.00, "#0d0d1a"],
        [0.10, "#1a1a40"],
        [0.30, "#1565c0"],
        [0.65, "#42a5f5"],
        [0.85, "#fdd835"],
        [1.00, "#fff9c4"],
    ]

    fig = go.Figure(go.Heatmap(
        z=values,
        colorscale=colorscale,
        zmin=0.0, zmax=1.0,
        showscale=False,
        xgap=2, ygap=2,
        hovertemplate="Celda [fila %{y}, col %{x}]<br>Estado: %{z:.0%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="Módulo Fotovoltaico — Vista de Celdas (6 × 10)",
            font=dict(size=14, color="#263238"),
        ),
        height=290,
        margin=dict(l=5, r=5, t=45, b=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#1c2833",
        paper_bgcolor="white",
    )
    return fig


@st.cache_data(show_spinner=False)
def build_daily_production_chart(shadow_pct: float, dust_pct: float) -> go.Figure:
    """
    Genera el gráfico de producción horaria diaria simulada.

    Compara la curva de producción ideal (sin perturbaciones) con la curva
    real (con los niveles actuales de sombra y polvo) para cuantificar el
    impacto energético de las perturbaciones a lo largo del día.

    El resultado está cacheado por Streamlit: solo se recalcula cuando
    cambian shadow_pct o dust_pct.

    Args:
        shadow_pct: Porcentaje de sombreado aplicado [0, 100].
        dust_pct:   Porcentaje de polvo aplicado [0, 100].

    Returns:
        Figura de Plotly con gráfico de área del perfil de producción diaria.
    """
    hours = list(range(24))
    ideal_power: list[float] = []
    real_power: list[float] = []

    for h in hours:
        irr = get_hourly_irradiance(float(h))
        ta = get_hourly_temperature(float(h))
        ideal_power.append(
            calculate_panel_state(irr, ta, 0.0, 0.0).power_w
        )
        real_power.append(
            calculate_panel_state(irr, ta, shadow_pct, dust_pct).power_w
        )

    labels = [f"{h:02d}:00" for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=ideal_power,
        fill="tozeroy", mode="lines",
        name="Producción ideal (sin perturbaciones)",
        line=dict(color="#1565c0", width=2),
        fillcolor="rgba(21, 101, 192, 0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=real_power,
        fill="tozeroy", mode="lines",
        name="Producción actual (con perturbaciones)",
        line=dict(color="#e65100", width=2.5, dash="dash"),
        fillcolor="rgba(230, 81, 0, 0.18)",
    ))
    fig.update_layout(
        title="Perfil de Producción Diaria Simulada — Cali, Valle del Cauca",
        xaxis_title="Hora del día",
        yaxis_title="Potencia (W)",
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font_size=11),
        margin=dict(l=45, r=10, t=65, b=40),
        hovermode="x unified",
        plot_bgcolor="#fafafa",
    )
    return fig


def build_harmonic_spectrum_chart(
    spectrum: dict[str, float],
    thd_pct: float,
) -> go.Figure:
    """
    Construye el gráfico de barras del espectro armónico de corriente.

    Cada barra representa la magnitud de un armónico como porcentaje del
    fundamental (60 Hz). El color indica el nivel de cumplimiento normativo:
        - Azul: dentro del límite (≤ 5 %).
        - Naranja: en zona de advertencia (5 % – 8 %).
        - Rojo: fuera de norma (> 8 %).

    Args:
        spectrum: Diccionario con magnitudes armónicas (% del fundamental).
        thd_pct:  THD total para mostrar en el título.

    Returns:
        Figura de Plotly con el espectro de armónicos.
    """
    harmonics = list(spectrum.keys())
    magnitudes = list(spectrum.values())

    colors = []
    for i, (h, m) in enumerate(zip(harmonics, magnitudes)):
        if i == 0:  # H1 siempre azul (fundamental = referencia)
            colors.append("#1565c0")
        elif m > THD_LIMIT_CRITICAL_PCT:
            colors.append("#b71c1c")
        elif m > THD_LIMIT_WARNING_PCT:
            colors.append("#f57f17")
        else:
            colors.append("#388e3c")

    fig = go.Figure(go.Bar(
        x=harmonics,
        y=magnitudes,
        marker_color=colors,
        text=[f"{m:.1f} %" for m in magnitudes],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f} % del fundamental<extra></extra>",
    ))

    # Línea del límite normativo IEEE 519
    fig.add_hline(
        y=THD_LIMIT_WARNING_PCT,
        line=dict(color="#f57f17", width=1.5, dash="dot"),
        annotation_text=f"Límite IEEE 519 ({THD_LIMIT_WARNING_PCT:.0f} %)",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_layout(
        title=f"Espectro Armónico de Corriente — THD = {thd_pct:.2f} %",
        yaxis_title="Magnitud (% del fundamental)",
        xaxis_title="Componente armónica",
        height=330,
        margin=dict(l=45, r=10, t=65, b=40),
        showlegend=False,
        plot_bgcolor="#fafafa",
    )
    # Excluir H1 (100 %) del cálculo del rango para que las barras armónicas sean legibles
    harmonic_max = max(magnitudes[1:], default=10.0)
    fig.update_yaxes(range=[0, max(harmonic_max * 1.35, THD_LIMIT_CRITICAL_PCT * 2)])
    return fig


def build_hourly_data_table(shadow_pct: float, dust_pct: float) -> pd.DataFrame:
    """
    Genera la tabla de datos horarios de producción y calidad de energía.

    Construye un DataFrame de pandas con la simulación completa del día,
    útil para análisis de tendencias y exportación de datos.

    Args:
        shadow_pct: Porcentaje de sombreado aplicado [0, 100].
        dust_pct:   Porcentaje de polvo aplicado [0, 100].

    Returns:
        DataFrame con columnas: Hora, Irradiancia, T° Ambiente, T° Celda,
        Potencia (W), Eficiencia (%), THD (%), Factor de Potencia.
    """
    rows = []
    for h in range(24):
        irr = get_hourly_irradiance(float(h))
        ta = get_hourly_temperature(float(h))
        state = calculate_panel_state(irr, ta, shadow_pct, dust_pct)
        pq = calculate_power_quality(
            shadow_pct, dust_pct, state.cell_temp_c, state.power_w
        )
        rows.append({
            "Hora": f"{h:02d}:00",
            "Irradiancia (W/m²)": round(irr, 1),
            "T° Ambiente (°C)": round(ta, 1),
            "T° Celda (°C)": round(state.cell_temp_c, 1),
            "Potencia (W)": round(state.power_w, 1),
            "Eficiencia (%)": round(state.efficiency_pct, 2),
            "THD (%)": round(pq.thd_pct, 2),
            "Factor de Potencia": round(pq.power_factor, 4),
        })
    return pd.DataFrame(rows)


def render_alerts(alerts: list) -> None:
    """
    Renderiza el panel de alertas con estilo HTML diferenciado por severidad.

    Muestra código, título, valores medidos, causa probable y recomendación
    para cada alerta activa. Si no hay alertas, muestra confirmación de estado
    normal del sistema.

    Args:
        alerts: Lista de objetos Alert del motor de diagnóstico.
    """
    if not alerts:
        st.success(
            "✅ Sistema operando dentro de parámetros normales. "
            "Sin alertas activas en este momento."
        )
        return

    for alert in alerts:
        icon = (
            "🔴" if alert.severity == AlertSeverity.CRITICAL
            else "⚠️" if alert.severity == AlertSeverity.WARNING
            else "ℹ️"
        )
        css_class = f"alert-{alert.severity.value}"
        html = (
            f'<div class="alert-box {css_class}">'
            f"<strong>{icon} [{alert.code}] {alert.title}</strong><br>"
            f"<span>{alert.message}</span><br>"
            f'<span style="color:#555"><em>📋 Causa probable:</em> {alert.cause}</span><br>'
            f'<span style="color:#333"><em>🔧 Recomendación:</em> {alert.recommendation}</span>'
            f"</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


# ─── Función principal ─────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto de entrada principal del dashboard SIGMA.

    Orquesta la renderización de todos los componentes del dashboard:
    encabezado institucional, panel de controles, visualización del módulo FV,
    métricas en tiempo real, gráficos de producción y PQ, tabla de datos
    horarios y sistema inteligente de alertas diagnósticas.
    """
    # ── Encabezado institucional ──────────────────────────────────────────────
    st.markdown(
        '<p class="sigma-header">'
        "Pontificia Universidad Javeriana Cali · "
        "Oficina de Recursos Físicos y Ambientales"
        "</p>",
        unsafe_allow_html=True,
    )
    st.title("☀️ SIGMA — Sistema Inteligente de Monitoreo, Diagnóstico y Alertas")
    st.caption(
        "Prototipo Fase 1  |  Simulación SSFV con condiciones climáticas de "
        "Cali, Valle del Cauca  |  Panel monocristalino 300 W"
    )
    st.divider()

    # ── Panel lateral de controles ────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        st.markdown("#### Parámetro Temporal")

        hour: int = st.slider(
            "🕐 Hora del día",
            min_value=0, max_value=23, value=12, step=1,
            format="%02d:00",
            help=(
                "Hora del día en formato 24 h. La irradiancia y la temperatura "
                "se calculan automáticamente según el perfil climático de Cali."
            ),
        )

        st.markdown("---")
        st.markdown("#### Perturbaciones Físicas")

        shadow_pct: float = float(st.slider(
            "🌑 Área sombreada (%)",
            min_value=0, max_value=100, value=0, step=5,
            help=(
                "Simula el porcentaje del panel cubierto por sombra. "
                "Por encima del 40 %, los bypass diodes se activan y el THD "
                "aumenta significativamente."
            ),
        ))

        dust_pct: float = float(st.slider(
            "💨 Acumulación de polvo (%)",
            min_value=0, max_value=100, value=0, step=5,
            help=(
                "Simula la suciedad acumulada en la superficie del módulo. "
                "Reduce la transmitancia óptica del vidrio frontal (soiling)."
            ),
        ))

        st.markdown("---")
        st.markdown("#### Perfil Climático Base — Cali")
        st.info(
            f"☀️ **Irradiancia pico:** {CALI_PROFILE.peak_irradiance_wm2:.0f} W/m²\n\n"
            f"🌡️ **T° media diaria:** {CALI_PROFILE.avg_temperature_c} °C\n\n"
            f"⏱️ **Horas Sol Pico:** {CALI_PROFILE.peak_sun_hours} h/día\n\n"
            f"📍 **Latitud:** {CALI_PROFILE.latitude_deg}°N  "
            f"(rango: {CALI_PROFILE.t_min_c}–{CALI_PROFILE.t_max_c} °C)"
        )
        st.markdown("---")
        st.markdown(
            f"**Módulo simulado:** Monocristalino {PANEL_SPECS.pmax_w:.0f} W STC"
        )
        st.caption(
            f"η nominal: {ETA_NOMINAL_PCT:.1f} %  |  "
            f"NOCT: {PANEL_SPECS.noct_c} °C  |  "
            f"γ_Pmax: {PANEL_SPECS.gamma_pmax*100:.2f} %/°C"
        )

    # ── Cálculos del instante seleccionado ────────────────────────────────────
    irr = get_hourly_irradiance(float(hour))
    ta = get_hourly_temperature(float(hour))
    panel_state = calculate_panel_state(irr, ta, shadow_pct, dust_pct)
    ideal_state = calculate_panel_state(irr, ta, 0.0, 0.0)
    pq = calculate_power_quality(
        shadow_pct, dust_pct, panel_state.cell_temp_c, panel_state.power_w
    )
    alerts = _ENGINE.run_diagnostics(panel_state, pq)

    # ── Fila 1: Panel visual | Métricas clave ─────────────────────────────────
    col_vis, col_met = st.columns([1.45, 1.0], gap="medium")

    with col_vis:
        st.plotly_chart(
            build_panel_figure(shadow_pct, dust_pct),
            use_container_width=True,
        )
        if shadow_pct == 0 and dust_pct == 0:
            st.caption(
                "💡 Use los controles del panel lateral para simular "
                "sombra o acumulación de polvo sobre el módulo."
            )
        else:
            loss_pct = (
                (1.0 - panel_state.power_w / ideal_state.power_w) * 100.0
                if ideal_state.power_w > 0 else 0.0
            )
            st.caption(
                f"🔻 Pérdida de potencia respecto al ideal: **{loss_pct:.1f} %**  "
                f"({ideal_state.power_w:.1f} W ideal → {panel_state.power_w:.1f} W actual)"
            )

    with col_met:
        st.markdown("### 📊 Estado en Tiempo Real")

        mc1, mc2 = st.columns(2)
        mc3, mc4 = st.columns(2)
        mc5, mc6 = st.columns(2)

        power_delta = panel_state.power_w - ideal_state.power_w
        mc1.metric(
            "⚡ Potencia",
            f"{panel_state.power_w:.1f} W",
            delta=f"{power_delta:+.1f} W vs ideal",
            delta_color="normal",
            help="Potencia DC generada por el módulo en el instante seleccionado.",
        )

        eff_delta = panel_state.efficiency_pct - ETA_NOMINAL_PCT
        mc2.metric(
            "📈 Eficiencia",
            f"{panel_state.efficiency_pct:.1f} %",
            delta=f"{eff_delta:+.1f} % vs nominal",
            delta_color="normal",
            help=f"Eficiencia sobre irradiancia incidente. Nominal STC: {ETA_NOMINAL_PCT:.1f} %.",
        )

        mc3.metric(
            "🌡️ T° Celda",
            f"{panel_state.cell_temp_c:.1f} °C",
            delta=f"Amb: {ta:.1f} °C",
            help="Temperatura de la celda FV calculada por modelo NOCT (IEC 61215).",
        )

        mc4.metric(
            "☀️ Irradiancia",
            f"{irr:.0f} W/m²",
            delta=f"Hora: {hour:02d}:00",
            help="Irradiancia solar estimada para la hora seleccionada en Cali.",
        )

        thd_delta = pq.thd_pct - THD_LIMIT_WARNING_PCT
        mc5.metric(
            "〰️ THD Corriente",
            f"{pq.thd_pct:.2f} %",
            delta=f"{thd_delta:+.2f} % vs límite",
            delta_color="inverse",
            help=(
                f"Distorsión Armónica Total. "
                f"Límite IEEE 519: {THD_LIMIT_WARNING_PCT} %."
            ),
        )

        fp_ok = pq.power_factor >= FP_LIMIT_WARNING
        mc6.metric(
            "⚡ Factor de Potencia",
            f"{pq.power_factor:.4f}",
            delta="OK" if fp_ok else "BAJO",
            delta_color="normal" if fp_ok else "inverse",
            help="Factor de Potencia verdadero. Mínimo CREG 024-2015: 0.92.",
        )

    st.divider()

    # ── Fila 2: Gráfico de producción | Espectro armónico ────────────────────
    col_prod, col_pq = st.columns(2, gap="medium")

    with col_prod:
        st.plotly_chart(
            build_daily_production_chart(shadow_pct, dust_pct),
            use_container_width=True,
        )
        # Energía diaria estimada (suma horaria ≈ integración de Potencia × 1h)
        daily_ideal_kwh = sum(
            calculate_panel_state(
                get_hourly_irradiance(float(h)),
                get_hourly_temperature(float(h)),
                0.0, 0.0,
            ).power_w
            for h in range(24)
        ) / 1000.0
        daily_real_kwh = sum(
            calculate_panel_state(
                get_hourly_irradiance(float(h)),
                get_hourly_temperature(float(h)),
                shadow_pct, dust_pct,
            ).power_w
            for h in range(24)
        ) / 1000.0
        energy_loss_pct = (
            (1.0 - daily_real_kwh / daily_ideal_kwh) * 100.0
            if daily_ideal_kwh > 0 else 0.0
        )
        st.info(
            f"📅 **Energía diaria estimada:** {daily_real_kwh:.3f} kWh/día  "
            f"(Ideal: {daily_ideal_kwh:.3f} kWh/día  |  "
            f"Pérdida: {energy_loss_pct:.1f} %)"
        )

    with col_pq:
        st.plotly_chart(
            build_harmonic_spectrum_chart(pq.harmonic_spectrum, pq.thd_pct),
            use_container_width=True,
        )
        st.info(
            f"📋 **THD Total:** {pq.thd_pct:.2f} %  |  "
            f"**FP:** {pq.power_factor:.4f}  |  "
            f"**Red:** {pq.frequency_hz:.0f} Hz (Colombia — NTC 1340)"
        )

    st.divider()

    # ── Fila 3: Tabla de datos horarios (expandible) ──────────────────────────
    with st.expander("📋 Ver tabla de datos horarios completa", expanded=False):
        df = build_hourly_data_table(shadow_pct, dust_pct)
        st.dataframe(
            df.style.format({
                "Irradiancia (W/m²)": "{:.1f}",
                "T° Ambiente (°C)": "{:.1f}",
                "T° Celda (°C)": "{:.1f}",
                "Potencia (W)": "{:.1f}",
                "Eficiencia (%)": "{:.2f}",
                "THD (%)": "{:.2f}",
                "Factor de Potencia": "{:.4f}",
            }).background_gradient(subset=["Potencia (W)"], cmap="YlOrRd"),
            use_container_width=True,
            height=350,
        )
        st.caption(
            "Energía total estimada: integración trapezoidal horaria (1 W durante 1 h = 1 Wh)."
        )

    st.divider()

    # ── Fila 4: Sistema de alertas diagnósticas ───────────────────────────────
    critical_n = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
    warning_n = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

    badge = ""
    if critical_n:
        badge += f" 🔴 {critical_n} crítica(s)"
    if warning_n:
        badge += f"  ⚠️ {warning_n} advertencia(s)"
    if not alerts:
        badge = "  ✅ Sin alertas"

    st.markdown(f"### 🔔 Sistema de Diagnóstico y Alertas{badge}")
    render_alerts(alerts)

    # ── Pie de página ──────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"SIGMA v0.1-Fase1  |  Pontificia Universidad Javeriana Cali  |  "
        f"Simulación generada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()

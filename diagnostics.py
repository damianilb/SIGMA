"""
Módulo de diagnóstico inteligente y generación de alertas — Proyecto SIGMA.

Implementa un motor de reglas que analiza el estado del panel solar y las
métricas de calidad de energía para detectar patrones anómalos y generar
alertas accionables con diagnóstico de causa probable y recomendaciones.

Alineado con:
    - IEC 62446: Requisitos de prueba y documentación para sistemas FV.
    - IEEE 1547-2018: Interconexión de recursos energéticos distribuidos.
    - IEEE 519-2022: Límites de armónicos de corriente.
    - RETIE / CREG 024-2015: Regulación eléctrica colombiana.
"""

from dataclasses import dataclass
from enum import Enum

from panel_model import PanelState, ETA_NOMINAL_PCT
from power_quality import (
    PowerQualityMetrics,
    THD_LIMIT_WARNING_PCT,
    THD_LIMIT_CRITICAL_PCT,
    FP_LIMIT_WARNING,
    FP_LIMIT_CRITICAL,
)


class AlertSeverity(Enum):
    """Niveles de severidad para las alertas del sistema SIGMA."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Alerta generada por el motor de diagnóstico.

    Atributos:
        severity:       Nivel de severidad (INFO, WARNING, CRITICAL).
        code:           Código único de identificación de la alerta.
        title:          Título corto y descriptivo.
        message:        Mensaje con la condición detectada y valores medidos.
        cause:          Causa probable identificada por el motor de reglas.
        recommendation: Acción recomendada para el operador/técnico.
    """

    severity: AlertSeverity
    code: str
    title: str
    message: str
    cause: str
    recommendation: str


# ─── Umbrales de detección del motor de reglas ─────────────────────────────────
SHADOW_WARNING_PCT: float = 20.0
SHADOW_CRITICAL_PCT: float = 40.0
DUST_WARNING_PCT: float = 30.0
DUST_CRITICAL_PCT: float = 60.0
TEMP_CELL_WARNING_C: float = 55.0
TEMP_CELL_CRITICAL_C: float = 70.0

# Umbral de eficiencia como fracción del nominal (ej. 0.65 = 65 % del nominal)
EFFICIENCY_WARNING_RATIO: float = 0.65
EFFICIENCY_CRITICAL_RATIO: float = 0.45

# Irradiancia mínima para evaluar eficiencia (evita alertas falsas en amanecer)
IRRADIANCE_MIN_FOR_EFFICIENCY_CHECK: float = 150.0


def _check_shadow_alerts(panel_state: PanelState) -> list[Alert]:
    """
    Evalúa el nivel de sombreado y genera alertas correspondientes.

    El sombreado parcial severo no solo reduce la potencia, sino que puede
    generar puntos calientes (hot-spots) que dañan irreversiblemente las celdas.

    Args:
        panel_state: Estado actual del panel solar.

    Returns:
        Lista de alertas relacionadas con sombreado (puede estar vacía).
    """
    alerts: list[Alert] = []
    s = panel_state.shadow_pct

    if s >= SHADOW_CRITICAL_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="SHAD-002",
            title="Sombreado Crítico Detectado",
            message=(
                f"El {s:.0f} % de la superficie del panel está sombreada. "
                "Riesgo de formación de puntos calientes (hot-spots)."
            ),
            cause=(
                "Sombreado parcial severo. Posible obstrucción estructural, "
                "vegetación, suciedad concentrada o microgrietas con efecto de "
                "bypass. Los diodos de bypass están activos en múltiples substrings."
            ),
            recommendation=(
                "Inspección física inmediata del módulo. Verificar obstrucciones "
                "en el campo fotovoltaico. Revisar integridad de celdas con "
                "cámara termográfica infrarroja."
            ),
        ))
    elif s >= SHADOW_WARNING_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="SHAD-001",
            title="Sombreado Parcial Detectado",
            message=f"El {s:.0f} % de la superficie del panel presenta sombra.",
            cause=(
                "Sombreado parcial. Posible sombra de estructuras cercanas, "
                "antenas, árboles o acumulación localizada de suciedad."
            ),
            recommendation=(
                "Verificar la fuente de sombreado. Considerar limpieza localizada "
                "o ajuste del ángulo de inclinación si el sombreado es recurrente."
            ),
        ))
    return alerts


def _check_dust_alerts(panel_state: PanelState) -> list[Alert]:
    """
    Evalúa el nivel de acumulación de polvo y genera alertas de mantenimiento.

    Args:
        panel_state: Estado actual del panel solar.

    Returns:
        Lista de alertas relacionadas con suciedad (puede estar vacía).
    """
    alerts: list[Alert] = []
    d = panel_state.dust_pct

    if d >= DUST_CRITICAL_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="DUST-002",
            title="Acumulación Severa de Polvo",
            message=(
                f"Nivel de polvo: {d:.0f} %. "
                "Pérdida de producción estimada superior al 15 %."
            ),
            cause=(
                "Acumulación prolongada de polvo, esmog o partículas orgánicas "
                "en la superficie del módulo. Frecuente en períodos secos en Cali "
                "(temporada seca: diciembre–enero y julio–agosto)."
            ),
            recommendation=(
                "Limpieza inmediata con agua desmineralizada y paño suave no abrasivo. "
                "Programar mantenimiento preventivo con frecuencia trimestral."
            ),
        ))
    elif d >= DUST_WARNING_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="DUST-001",
            title="Acumulación de Polvo Moderada",
            message=f"Nivel de polvo: {d:.0f} %. Se recomienda limpieza preventiva.",
            cause=(
                "Acumulación de polvo y partículas en ambiente urbano. "
                "Normal después de períodos sin lluvia en Cali."
            ),
            recommendation=(
                "Programar limpieza de módulos en las próximas 2 semanas. "
                "Evaluar sistema de limpieza automática si el ciclo es recurrente."
            ),
        ))
    return alerts


def _check_thermal_alerts(panel_state: PanelState) -> list[Alert]:
    """
    Evalúa la temperatura de la celda y genera alertas térmicas.

    La degradación térmica acelerada ocurre de forma continua por encima de
    55 °C, pudiendo causar delaminación, quiebre de soldaduras y degradación
    por potencial inducido (PID).

    Args:
        panel_state: Estado actual del panel solar.

    Returns:
        Lista de alertas térmicas (puede estar vacía).
    """
    alerts: list[Alert] = []
    tc = panel_state.cell_temp_c

    if tc >= TEMP_CELL_CRITICAL_C:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="THERM-002",
            title="Temperatura de Celda Crítica",
            message=(
                f"Temperatura de celda: {tc:.1f} °C "
                f"(límite seguro: {TEMP_CELL_WARNING_C} °C). "
                "Riesgo de degradación acelerada e irreversible."
            ),
            cause=(
                "Temperatura excesiva por alta irradiancia combinada con baja "
                "ventilación posterior, o por punto caliente generado por sombreado "
                "parcial que concentra disipación de energía en celdas activas."
            ),
            recommendation=(
                "Verificar libre circulación de aire en el sistema de montaje. "
                "Realizar inspección termográfica inmediata para detectar hot-spots. "
                "Revisar integridad del encapsulante (EVA)."
            ),
        ))
    elif tc >= TEMP_CELL_WARNING_C:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="THERM-001",
            title="Temperatura de Celda Elevada",
            message=(
                f"Temperatura de celda: {tc:.1f} °C. "
                f"Supera el umbral de operación normal ({TEMP_CELL_WARNING_C} °C)."
            ),
            cause=(
                "Alta temperatura ambiente combinada con irradiancia intensa. "
                "Típico en horas pico (11:00–15:00 h) en Cali."
            ),
            recommendation=(
                "Monitorear continuamente. Verificar libre circulación de aire "
                "detrás de los módulos y que no existan obstrucciones traseras."
            ),
        ))
    return alerts


def _check_power_quality_alerts(pq: PowerQualityMetrics) -> list[Alert]:
    """
    Evalúa las métricas de calidad de energía y genera alertas normativas.

    Compara los valores calculados contra los límites de la IEEE 519-2022
    y el RETIE/CREG de Colombia.

    Args:
        pq: Métricas de calidad de energía calculadas.

    Returns:
        Lista de alertas de calidad de energía (puede estar vacía).
    """
    alerts: list[Alert] = []

    # ── THD ────────────────────────────────────────────────────────────────────
    if pq.thd_pct >= THD_LIMIT_CRITICAL_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="PQ-THD-002",
            title="THD Crítico — Fuera de Norma",
            message=(
                f"THD = {pq.thd_pct:.2f} % "
                f"(límite normativo IEEE 519: {THD_LIMIT_CRITICAL_PCT} %). "
                "El sistema supera los límites del RETIE e IEEE 519-2022."
            ),
            cause=(
                "Distorsión armónica severa. Causa probable: sombreado parcial severo "
                "que provoca oscilación del algoritmo MPPT, microgrietas en celdas PV "
                "que generan curvas I-V con múltiples picos locales, o falla incipiente "
                "en el inversor fotovoltaico (condensadores de filtro degradados)."
            ),
            recommendation=(
                "Inspección física inmediata del módulo. Diagnóstico eléctrico del "
                "inversor FV. Medición in-situ con analizador de calidad de energía "
                "certificado. Documentar para reporte a la interventoría."
            ),
        ))
    elif pq.thd_pct >= THD_LIMIT_WARNING_PCT:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="PQ-THD-001",
            title="THD Elevado — Alerta de Calidad de Energía",
            message=(
                f"THD = {pq.thd_pct:.2f} % "
                f"(umbral de advertencia: {THD_LIMIT_WARNING_PCT} %). "
                "El valor supera el límite recomendado para sistemas de baja tensión."
            ),
            cause=(
                "Distorsión armónica por encima del umbral. Causa probable: "
                "sombreado parcial o acumulación de polvo que afecta la uniformidad "
                "de la curva I-V y la estabilidad del MPPT del inversor."
            ),
            recommendation=(
                "Revisar condiciones de operación del panel. Verificar el algoritmo "
                "MPPT del inversor. Registrar datos para análisis de tendencias a lo "
                "largo de la semana antes de escalar el reporte."
            ),
        ))

    # ── Factor de Potencia ─────────────────────────────────────────────────────
    if pq.power_factor < FP_LIMIT_CRITICAL:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="PQ-FP-002",
            title="Factor de Potencia Crítico",
            message=(
                f"FP = {pq.power_factor:.4f} "
                f"(mínimo CREG: {FP_LIMIT_WARNING}). "
                "Genera penalización tarifaria y puede afectar equipos conectados."
            ),
            cause=(
                "Factor de Potencia extremadamente bajo derivado del THD elevado. "
                "El contenido armónico de la corriente aumenta la potencia aparente "
                "sin aportar potencia activa útil."
            ),
            recommendation=(
                "Acción correctiva inmediata. Corregir la causa raíz del THD. "
                "Evaluar la necesidad de filtros de armónicos pasivos o activos "
                "en el punto de conexión a la red."
            ),
        ))
    elif pq.power_factor < FP_LIMIT_WARNING:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="PQ-FP-001",
            title="Factor de Potencia por Debajo del Umbral",
            message=(
                f"FP = {pq.power_factor:.4f} "
                f"(mínimo recomendado: {FP_LIMIT_WARNING}). "
                "Posible incumplimiento regulatorio CREG 024-2015."
            ),
            cause="Factor de Potencia reducido por el contenido armónico elevado.",
            recommendation=(
                "Monitorear evolución del FP. Investigar y corregir la causa del THD "
                "para restaurar el FP a niveles aceptables."
            ),
        ))
    return alerts


def _check_efficiency_alerts(panel_state: PanelState) -> list[Alert]:
    """
    Evalúa la eficiencia del panel y genera alertas si cae bajo umbrales críticos.

    Solo evalúa la eficiencia cuando hay irradiancia suficiente para un cálculo
    significativo (> 150 W/m²), evitando falsas alarmas en amanecer/atardecer.

    Una caída sostenida puede indicar degradación progresiva por LID
    (Light-Induced Degradation) o PID (Potential-Induced Degradation).

    Args:
        panel_state: Estado actual del panel solar.

    Returns:
        Lista de alertas de rendimiento (puede estar vacía).
    """
    alerts: list[Alert] = []

    if panel_state.irradiance_wm2 < IRRADIANCE_MIN_FOR_EFFICIENCY_CHECK:
        return alerts

    eff = panel_state.efficiency_pct
    eff_warning_threshold = ETA_NOMINAL_PCT * EFFICIENCY_WARNING_RATIO
    eff_critical_threshold = ETA_NOMINAL_PCT * EFFICIENCY_CRITICAL_RATIO

    if eff < eff_critical_threshold:
        alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            code="PERF-002",
            title="Eficiencia del Módulo en Nivel Crítico",
            message=(
                f"Eficiencia actual: {eff:.1f} % "
                f"(nominal STC: {ETA_NOMINAL_PCT:.1f} %). "
                f"Caída del {(1 - eff/ETA_NOMINAL_PCT)*100:.0f} % respecto al nominal."
            ),
            cause=(
                "Eficiencia muy por debajo del valor nominal. Posible degradación "
                "severa: microgrietas, deslaminación del encapsulante EVA, "
                "degradación por potencial inducido (PID) o suciedad extrema."
            ),
            recommendation=(
                "Realizar curva I-V del módulo con trazador certificado. "
                "Termografía infrarroja para detectar celdas inactivas. "
                "Evaluar reemplazo del módulo si la degradación es irreversible."
            ),
        ))
    elif eff < eff_warning_threshold:
        alerts.append(Alert(
            severity=AlertSeverity.WARNING,
            code="PERF-001",
            title="Rendimiento por Debajo del Nominal",
            message=(
                f"Eficiencia actual: {eff:.1f} % "
                f"(nominal STC: {ETA_NOMINAL_PCT:.1f} %). "
                f"Caída del {(1 - eff/ETA_NOMINAL_PCT)*100:.0f} % respecto al nominal."
            ),
            cause=(
                "Reducción de eficiencia por sombreado, polvo o degradación térmica. "
                "Verificar si la causa es transitoria (sombra/polvo) o permanente "
                "(degradación del módulo)."
            ),
            recommendation=(
                "Verificar y corregir condiciones de limpieza y sombreado. "
                "Registrar datos para análisis de tendencia de degradación anual."
            ),
        ))
    return alerts


class DiagnosticEngine:
    """
    Motor central de diagnóstico que consolida todas las reglas de alertas.

    Evalúa el estado del panel y las métricas PQ de forma integral, retornando
    una lista de alertas ordenadas por severidad para el panel de notificaciones
    del dashboard SIGMA.
    """

    def run_diagnostics(
        self,
        panel_state: PanelState,
        pq_metrics: PowerQualityMetrics,
    ) -> list[Alert]:
        """
        Ejecuta todas las reglas de diagnóstico y retorna las alertas activas.

        Cada regla es independiente y puede generar cero o más alertas.
        El resultado se ordena: CRITICAL primero, luego WARNING, luego INFO.

        Args:
            panel_state: Estado actual del panel solar.
            pq_metrics:  Métricas de calidad de energía actuales.

        Returns:
            Lista de alertas ordenadas por severidad descendente.
        """
        all_alerts: list[Alert] = []
        all_alerts.extend(_check_shadow_alerts(panel_state))
        all_alerts.extend(_check_dust_alerts(panel_state))
        all_alerts.extend(_check_thermal_alerts(panel_state))
        all_alerts.extend(_check_power_quality_alerts(pq_metrics))
        all_alerts.extend(_check_efficiency_alerts(panel_state))

        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2,
        }
        all_alerts.sort(key=lambda a: severity_order[a.severity])
        return all_alerts

"""
Módulo de cálculo de variables de Calidad de Energía (Power Quality – PQ).

Simula los efectos de las perturbaciones físicas del panel (sombreado, polvo,
temperatura) sobre los parámetros eléctricos de calidad de energía:
Distorsión Armónica Total (THD) y Factor de Potencia (FP).

Base teórica:
    - El sombreado parcial genera curvas I-V con múltiples picos de máxima
      potencia (MPPs). El MPPT del inversor oscila entre ellos, produciendo
      corriente discontinua e inyectando armónicos a la red.
    - IEC 61000-3-2: Límites de armónicos de corriente para equipos.
    - IEEE 519-2022 / RETIE: Estándares de calidad de energía.
    - CREG 024-2015: Resolución colombiana, FP mínimo de 0.9 en baja tensión.
"""

from dataclasses import dataclass

import numpy as np

from panel_model import PANEL_SPECS


@dataclass
class PowerQualityMetrics:
    """
    Métricas de calidad de energía calculadas para el sistema FV.

    Atributos:
        thd_pct:          Distorsión Armónica Total de corriente (%).
        power_factor:     Factor de Potencia verdadero (adimensional, [0, 1]).
        harmonic_spectrum: Magnitudes de armónicos individuales como % del
                           fundamental. Claves: 'H1 (60 Hz)', 'H3 (180 Hz)', etc.
        frequency_hz:     Frecuencia fundamental de la red eléctrica (Hz).
    """

    thd_pct: float
    power_factor: float
    harmonic_spectrum: dict[str, float]
    frequency_hz: float = 60.0  # Colombia opera a 60 Hz (NTC 1340)


# ─── Umbrales normativos de referencia ────────────────────────────────────────
# IEEE 519-2022 para sistemas de baja tensión (Isc/IL < 20)
THD_LIMIT_WARNING_PCT: float = 5.0   # umbral de advertencia
THD_LIMIT_CRITICAL_PCT: float = 8.0  # límite normativo máximo permitido

# CREG 024-2015 y RETIE Colombia
FP_LIMIT_WARNING: float = 0.92   # umbral de advertencia
FP_LIMIT_CRITICAL: float = 0.85  # límite mínimo aceptable


def calculate_thd(
    shadow_pct: float,
    dust_pct: float,
    cell_temp_c: float,
    power_w: float,
) -> float:
    """
    Calcula la Distorsión Armónica Total (THD) de corriente del sistema inversor.

    Modelo de contribuciones:
        - Base: THD mínimo del inversor en condiciones óptimas (~2.5 %).
        - Sombreado: Contribución dominante. La curva I-V con múltiples MPPs
          provoca oscilaciones del MPPT que se traducen en distorsión. La
          relación es super-lineal (ley de potencia 1.5) porque múltiples bypass
          diodes activos amplifican el efecto.
        - Polvo: Contribución menor; el soiling uniforme afecta poco la forma
          de onda ya que degrada todos los substrings por igual.
        - Temperatura: Alta temperatura aumenta la resistencia serie de la celda,
          distorsionando levemente la onda de corriente de salida.
        - Carga parcial: Los inversores operan con peor THD por debajo del 20 %
          de su potencia nominal (diseñados para rango nominal).

    Args:
        shadow_pct:  Porcentaje de sombreado [0, 100].
        dust_pct:    Porcentaje de polvo [0, 100].
        cell_temp_c: Temperatura de la celda PV en °C.
        power_w:     Potencia actualmente generada en W.

    Returns:
        THD en porcentaje [0.0, ~50.0].
    """
    THD_BASE: float = 2.5  # % mínimo para inversor moderno de alta eficiencia

    # El sombreado genera múltiples MPPs → oscilación MPPT → alta distorsión
    shadow_contribution = (shadow_pct / 100.0) ** 1.5 * 35.0

    # El polvo es uniforme, impacto menor en calidad de onda
    dust_contribution = (dust_pct / 100.0) * 4.0

    # Degradación térmica de la celda → mayor resistencia serie → leve distorsión
    temp_contribution = max(0.0, (cell_temp_c - 50.0) * 0.08)

    # Inversores con <20 % de carga nominal tienen THD significativamente mayor
    power_ratio = power_w / PANEL_SPECS.pmax_w
    low_load_penalty = 2.5 * max(0.0, 0.20 - power_ratio)

    thd = (
        THD_BASE
        + shadow_contribution
        + dust_contribution
        + temp_contribution
        + low_load_penalty
    )
    return float(np.clip(thd, 0.0, 50.0))


def calculate_power_factor(thd_pct: float) -> float:
    """
    Calcula el Factor de Potencia (FP) verdadero a partir del THD de corriente.

    Para inversores fotovoltaicos modernos con control de FP de desplazamiento
    unitario (cos φ ≈ 1.0), el FP verdadero está dominado por el contenido
    armónico de la corriente:

        FP_verdadero = FP_desplazamiento / √(1 + THD²)

    donde THD se expresa en por unidad (no en porcentaje) y
    FP_desplazamiento ≈ 0.99 para inversores de alta calidad.

    Args:
        thd_pct: Distorsión Armónica Total en porcentaje.

    Returns:
        Factor de Potencia verdadero [0.0, 1.0].
    """
    PF_DISPLACEMENT: float = 0.99  # FP de desplazamiento del inversor (práct. unitario)

    thd_pu = thd_pct / 100.0  # convertir a por unidad para la fórmula
    pf = PF_DISPLACEMENT / np.sqrt(1.0 + thd_pu**2)
    return float(np.clip(pf, 0.0, 1.0))


def estimate_harmonic_spectrum(thd_pct: float, shadow_pct: float) -> dict[str, float]:
    """
    Estima el espectro de armónicos de corriente a partir del THD total.

    Para inversores FV con modulación PWM, los armónicos impares dominan
    (3.°, 5.°, 7.°). El sombreado amplifica las componentes de baja
    frecuencia (3.° y 5.°) al crear distorsión de baja frecuencia por la
    oscilación del MPPT. Cada magnitud se expresa como porcentaje del fundamental.

    La distribución sigue proporciones típicas documentadas en la literatura PQ
    para inversores de string en sistemas FV residenciales/industriales.

    Args:
        thd_pct:    THD total en porcentaje.
        shadow_pct: Porcentaje de sombreado (amplifica armónicos de orden bajo).

    Returns:
        Diccionario con 'H1 (60 Hz)' = 100 % siempre (fundamental), y las
        magnitudes de H3, H5, H7, H9, H11 como % del fundamental.
    """
    # Factor de amplificación por sombreado: más sombra → más energía en H3 y H5
    shadow_amp = 1.0 + (shadow_pct / 100.0) * 0.6

    h3 = round(thd_pct * 0.38 * shadow_amp, 2)
    h5 = round(thd_pct * 0.30 * shadow_amp, 2)
    h7 = round(thd_pct * 0.18, 2)
    h9 = round(thd_pct * 0.09, 2)
    h11 = round(thd_pct * 0.05, 2)

    return {
        "H1 (60 Hz)": 100.0,
        "H3 (180 Hz)": h3,
        "H5 (300 Hz)": h5,
        "H7 (420 Hz)": h7,
        "H9 (540 Hz)": h9,
        "H11 (660 Hz)": h11,
    }


def calculate_power_quality(
    shadow_pct: float,
    dust_pct: float,
    cell_temp_c: float,
    power_w: float,
) -> PowerQualityMetrics:
    """
    Calcula todas las métricas de calidad de energía del sistema FV.

    Función de punto de entrada del módulo PQ. Orquesta el cálculo de THD,
    Factor de Potencia y espectro armónico, retornando un objeto unificado.

    Args:
        shadow_pct:  Porcentaje de sombreado [0, 100].
        dust_pct:    Porcentaje de polvo [0, 100].
        cell_temp_c: Temperatura de la celda PV en °C.
        power_w:     Potencia actualmente generada en W.

    Returns:
        Objeto PowerQualityMetrics con todas las métricas calculadas.
    """
    thd = calculate_thd(shadow_pct, dust_pct, cell_temp_c, power_w)
    fp = calculate_power_factor(thd)
    spectrum = estimate_harmonic_spectrum(thd, shadow_pct)

    return PowerQualityMetrics(
        thd_pct=round(thd, 2),
        power_factor=round(fp, 4),
        harmonic_spectrum=spectrum,
    )

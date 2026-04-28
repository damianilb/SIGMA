"""
Módulo de simulación climática para Cali, Valle del Cauca.

Modela el comportamiento típico de la irradiancia solar y la temperatura
ambiente basado en datos históricos del IDEAM y la NASA-SSE para la región
de Cali (latitud 3.45°N, longitud 76.53°W).

Valores de referencia:
    - Irradiancia pico promedio en día despejado: ~950 W/m²
    - Horas Sol Pico (HSP): 4.5 h/día
    - Temperatura promedio anual: 24 °C
    - Rango diario de temperatura: 20 °C (mín.) – 28 °C (máx.)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CaliClimateProfile:
    """
    Perfil climático de referencia para Cali, Colombia.

    Atributos:
        peak_irradiance_wm2: Irradiancia pico en día despejado (W/m²).
        avg_temperature_c:   Temperatura ambiente promedio diaria (°C).
        peak_sun_hours:      Horas Sol Pico por día (h/día).
        t_min_c:             Temperatura mínima diaria, típicamente al amanecer (°C).
        t_max_c:             Temperatura máxima diaria, típicamente a las 14:00 (°C).
        latitude_deg:        Latitud geográfica en grados decimales.
    """

    peak_irradiance_wm2: float = 950.0
    avg_temperature_c: float = 24.0
    peak_sun_hours: float = 4.5
    t_min_c: float = 20.0
    t_max_c: float = 28.0
    latitude_deg: float = 3.45


CALI_PROFILE = CaliClimateProfile()


def get_hourly_irradiance(hour: float, cloud_factor: float = 1.0) -> float:
    """
    Estima la irradiancia solar (W/m²) para una hora específica del día en Cali.

    Emplea una función gaussiana centrada al mediodía solar (12:00 h), ajustada
    para la latitud tropical de Cali. El sol emerge ~06:00 y se oculta ~18:00.

    Args:
        hour:         Hora del día en formato decimal [0.0, 24.0).
        cloud_factor: Factor de transmitancia por nubosidad [0.0, 1.0].
                      1.0 = cielo completamente despejado.

    Returns:
        Irradiancia estimada en W/m². Retorna 0.0 antes de las 6 h y después
        de las 18 h.
    """
    SOLAR_START: float = 6.0
    SOLAR_END: float = 18.0
    PEAK_HOUR: float = 12.0
    SIGMA: float = 2.8  # controla el ancho de la campana (~6 h efectivas)

    if hour < SOLAR_START or hour > SOLAR_END:
        return 0.0

    raw = CALI_PROFILE.peak_irradiance_wm2 * np.exp(
        -((hour - PEAK_HOUR) ** 2) / (2.0 * SIGMA**2)
    )
    return float(np.clip(raw * cloud_factor, 0.0, 1200.0))


def get_hourly_temperature(hour: float) -> float:
    """
    Estima la temperatura ambiente (°C) para una hora específica del día en Cali.

    Usa una onda sinusoidal desplazada: mínimo cerca de las 06:00 (amanecer)
    y máximo alrededor de las 14:00 h.

    Args:
        hour: Hora del día en formato decimal [0.0, 24.0).

    Returns:
        Temperatura ambiente estimada en °C.
    """
    T_MEAN: float = (CALI_PROFILE.t_min_c + CALI_PROFILE.t_max_c) / 2.0
    T_AMPLITUDE: float = (CALI_PROFILE.t_max_c - CALI_PROFILE.t_min_c) / 2.0
    PEAK_HOUR: float = 14.0  # hora del máximo térmico diario

    temperature = T_MEAN + T_AMPLITUDE * np.sin(
        2.0 * np.pi * (hour - PEAK_HOUR) / 24.0
    )
    return float(temperature)


def generate_daily_profile(cloud_factor: float = 1.0) -> list[dict]:
    """
    Genera el perfil climático horario completo para un día típico en Cali.

    Args:
        cloud_factor: Factor de nubosidad para todo el día [0.0, 1.0].

    Returns:
        Lista de 24 dicts con claves 'hour', 'irradiance_wm2', 'temperature_c'.
    """
    return [
        {
            "hour": h,
            "irradiance_wm2": get_hourly_irradiance(float(h), cloud_factor),
            "temperature_c": get_hourly_temperature(float(h)),
        }
        for h in range(24)
    ]

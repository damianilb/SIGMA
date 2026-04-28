"""
Módulo de modelado físico del panel solar fotovoltaico.

Implementa el modelo de un panel monocristalino de 300 W representativo
de los módulos del SSFV de la Pontificia Universidad Javeriana Cali.

Referencias del modelo:
    - IEC 61215: Módulos fotovoltaicos para uso terrestre.
    - King, D.L. et al. (2004). Sandia Array Performance Model.
    - Temperatura de celda NOCT (Normal Operating Cell Temperature).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanelSpecifications:
    """
    Especificaciones técnicas del módulo en Condiciones Estándar de Prueba (STC).

    STC: G_ref = 1000 W/m², T_celda = 25 °C, masa de aire AM = 1.5.

    Atributos:
        pmax_w:      Potencia máxima nominal (W).
        voc_v:       Tensión de circuito abierto (V).
        isc_a:       Corriente de cortocircuito (A).
        vmpp_v:      Tensión en el punto de máxima potencia (V).
        impp_a:      Corriente en el punto de máxima potencia (A).
        area_m2:     Área efectiva del módulo (m²).
        noct_c:      Temperatura Normal de Operación de la Celda (°C).
        gamma_pmax:  Coeficiente de temperatura de Pmax (fracción/°C).
                     Negativo: la potencia cae con el aumento de temperatura.
    """

    pmax_w: float = 300.0
    voc_v: float = 40.5
    isc_a: float = 9.38
    vmpp_v: float = 33.2
    impp_a: float = 9.05
    area_m2: float = 1.63
    noct_c: float = 45.0
    gamma_pmax: float = -0.0038  # -0.38 %/°C → fracción por grado


PANEL_SPECS = PanelSpecifications()

G_REF_WM2: float = 1000.0   # Irradiancia de referencia STC (W/m²)
TC_REF_C: float = 25.0       # Temperatura de celda de referencia STC (°C)

# Eficiencia nominal calculada a partir de las especificaciones STC
ETA_NOMINAL_PCT: float = (
    PANEL_SPECS.pmax_w / (G_REF_WM2 * PANEL_SPECS.area_m2) * 100.0
)


@dataclass
class PanelState:
    """
    Estado calculado del panel solar en un instante de tiempo.

    Atributos:
        irradiance_wm2:   Irradiancia solar incidente sobre el panel (W/m²).
        cell_temp_c:      Temperatura de la celda fotovoltaica calculada (°C).
        ambient_temp_c:   Temperatura ambiente en ese instante (°C).
        shadow_pct:       Porcentaje de superficie sombreada [0, 100].
        dust_pct:         Porcentaje de acumulación de polvo [0, 100].
        power_w:          Potencia generada calculada (W).
        efficiency_pct:   Eficiencia del módulo relativa a la irradiancia incidente (%).
        g_effective_wm2:  Irradiancia efectiva tras aplicar el factor de soiling (W/m²).
    """

    irradiance_wm2: float
    cell_temp_c: float
    ambient_temp_c: float
    shadow_pct: float
    dust_pct: float
    power_w: float
    efficiency_pct: float
    g_effective_wm2: float


def calculate_cell_temperature(
    ambient_temp_c: float,
    irradiance_wm2: float,
) -> float:
    """
    Calcula la temperatura de la celda PV usando el modelo NOCT.

    Fórmula IEC 61215:
        Tc = Ta + (NOCT − 20) × (G / 800)

    Args:
        ambient_temp_c: Temperatura ambiente en °C.
        irradiance_wm2: Irradiancia solar incidente en W/m².

    Returns:
        Temperatura de la celda fotovoltaica en °C.
    """
    if irradiance_wm2 <= 0.0:
        return ambient_temp_c

    tc = ambient_temp_c + (PANEL_SPECS.noct_c - 20.0) * (irradiance_wm2 / 800.0)
    return float(tc)


def calculate_shadow_factor(shadow_pct: float) -> float:
    """
    Calcula el factor multiplicativo de potencia por sombreado parcial.

    El sombreado en módulos con bypass diodes es no-lineal: al superar el
    umbral de activación (~15 % de sombra), los diodos de bypass desconectan
    substrings completos, produciendo pérdidas desproporcionadas al área
    sombreada. Se modela con una ley de potencia (exponente > 1) para capturar
    este efecto de cascada.

    Args:
        shadow_pct: Porcentaje de superficie sombreada [0.0, 100.0].

    Returns:
        Factor multiplicativo de potencia resultante [0.0, 1.0].
    """
    s: float = shadow_pct / 100.0
    # Exponente 1.7: super-lineal para reflejar el efecto de los bypass diodes
    return float(max(0.0, (1.0 - s) ** 1.7))


def calculate_dust_factor(dust_pct: float) -> float:
    """
    Calcula el factor de pérdida de irradiancia por acumulación de polvo (soiling).

    El polvo reduce la transmitancia óptica del vidrio frontal del módulo.
    Estudios en regiones tropicales reportan pérdidas de hasta 25 % en condiciones
    de suciedad severa sin lluvia. La relación es aproximadamente lineal con el
    nivel de acumulación.

    Args:
        dust_pct: Porcentaje de acumulación de polvo [0.0, 100.0].

    Returns:
        Factor de transmitancia óptica por soiling [0.0, 1.0].
    """
    MAX_SOILING_LOSS: float = 0.25  # 25 % de pérdida máxima en soiling extremo
    return 1.0 - (dust_pct / 100.0) * MAX_SOILING_LOSS


def calculate_panel_state(
    irradiance_wm2: float,
    ambient_temp_c: float,
    shadow_pct: float,
    dust_pct: float,
) -> PanelState:
    """
    Calcula el estado completo del panel solar dados los parámetros de entrada.

    Combina:
        1. Temperatura de celda por modelo NOCT.
        2. Corrección lineal de potencia por irradiancia.
        3. Corrección térmica de Pmax (coeficiente gamma).
        4. Factor de pérdida por polvo (soiling sobre irradiancia efectiva).
        5. Factor de pérdida por sombreado (bypass diodes, no-lineal).

    La eficiencia se calcula sobre la irradiancia total incidente (no sobre la
    irradiancia efectiva) para reflejar correctamente todas las pérdidas.

    Args:
        irradiance_wm2: Irradiancia solar incidente en W/m².
        ambient_temp_c: Temperatura ambiente en °C.
        shadow_pct:     Porcentaje de superficie sombreada [0.0, 100.0].
        dust_pct:       Porcentaje de acumulación de polvo [0.0, 100.0].

    Returns:
        Objeto PanelState con todos los parámetros calculados.
    """
    NULL_STATE = PanelState(
        irradiance_wm2=irradiance_wm2,
        cell_temp_c=ambient_temp_c,
        ambient_temp_c=ambient_temp_c,
        shadow_pct=shadow_pct,
        dust_pct=dust_pct,
        power_w=0.0,
        efficiency_pct=0.0,
        g_effective_wm2=0.0,
    )

    if irradiance_wm2 <= 0.0:
        return NULL_STATE

    tc = calculate_cell_temperature(ambient_temp_c, irradiance_wm2)
    dust_factor = calculate_dust_factor(dust_pct)
    shadow_factor = calculate_shadow_factor(shadow_pct)

    # Irradiancia efectiva después de la pérdida por soiling (polvo)
    g_effective = irradiance_wm2 * dust_factor

    # Potencia base escalada linealmente desde STC según irradiancia efectiva
    p_base = PANEL_SPECS.pmax_w * (g_effective / G_REF_WM2)

    # Corrección térmica: Pmax(T) = Pmax_stc × [1 + γ × (Tc − Tc_ref)]
    temp_correction = 1.0 + PANEL_SPECS.gamma_pmax * (tc - TC_REF_C)

    # Potencia final considerando todas las pérdidas
    power_w = max(0.0, p_base * temp_correction * shadow_factor)

    # Eficiencia sobre irradiancia total incidente (incluye pérdidas de sombra y polvo)
    efficiency_pct = (power_w / (irradiance_wm2 * PANEL_SPECS.area_m2)) * 100.0
    efficiency_pct = float(np.clip(efficiency_pct, 0.0, 100.0))

    return PanelState(
        irradiance_wm2=irradiance_wm2,
        cell_temp_c=float(tc),
        ambient_temp_c=ambient_temp_c,
        shadow_pct=shadow_pct,
        dust_pct=dust_pct,
        power_w=float(power_w),
        efficiency_pct=efficiency_pct,
        g_effective_wm2=float(g_effective),
    )

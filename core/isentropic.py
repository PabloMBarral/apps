"""Rendimientos isoentrópicos: turbinas, compresores, bombas, multietapa.

Funciones puras sobre :class:`core.fluids.StatePoint`, en unidades SI
internamente. Cada función devuelve un :class:`IsentropicResult` (o
:class:`PolytropicResult` para multietapa) con todos los estados
intermedios, los Δh isoentrópico y real, y un bloque didáctico de
fórmula LaTeX + sustitución + narrativa en español plano.

Los pasos didácticos se pueden pedir en cualquier sistema de unidades
con :func:`isentropic_steps` / :func:`multistage_steps` (Fase 1.5b); el
campo ``steps`` de los resultados se arma en sistema Técnico, como antes.

Los estados de salida se resuelven con
:func:`core.fluids.fluid_state_from_pair`, que valida el rango de la
ecuación de estado y explica los errores en castellano.

El módulo NO importa Streamlit.

Cita
----
Cengel, Y. A., & Boles, M. A. (2015). *Thermodynamics: An Engineering
Approach* (8th ed., §7-12 "Isentropic Efficiencies of Steady-Flow
Devices"). McGraw-Hill.

Barral, P. M. *Vademecum de Termodinámica* (vademecum-termo), §10.4
"Rendimientos isoentrópicos" y §6.3 "Compresión en dos etapas con
interenfriamiento". DOI: 10.5281/zenodo.20092635

Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014). "Pure and
Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source
Thermophysical Property Library CoolProp". *Industrial & Engineering
Chemistry Research*, 53(6), 2498-2508. DOI: 10.1021/ie4033999
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, Literal

import CoolProp.CoolProp as cp

from core.fluids import (
    FLUID_NAMES_ES,
    FluidState,
    PairCode,
    StatePoint,
    fluid_limits,
    fluid_state_from_pair,
    saturation_at_pressure,
    saturation_at_temperature,
)
from core.latex import (
    latex_is_wide,
    latex_number,
    latex_paren,
    latex_quantity,
    latex_value,
    text_quantity,
)
from core.state_report import states_table
from core.units_system import UnitSystem, convert_from_si, unit_label

DeviceKind = Literal["turbine", "compressor", "pump"]
Mode = Literal["direct", "inverse"]

#: Dispositivos de la página (incluye el compresor multietapa).
DeviceName = Literal["turbine", "compressor", "pump", "multistage"]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IsentropicSteps:
    """Pasos didácticos para renderizar en LaTeX + texto en español plano."""

    formula_latex: str
    substituted_latex: str
    narrative_es: str


@dataclass(frozen=True)
class IsentropicResult:
    """Resultado de un proceso isoentrópico de un dispositivo de flujo.

    ``mode`` y ``fluid`` se agregaron en la Fase 1.7 (con default, para
    no romper el código que construye el resultado a mano): permiten
    rearmar los pasos didácticos en otro sistema de unidades.
    """

    device: DeviceKind
    state_in: StatePoint
    state_out_isen: StatePoint
    state_out_real: StatePoint
    eta_s: float
    delta_h_isen_J_per_kg: float
    delta_h_real_J_per_kg: float
    steps: IsentropicSteps
    mode: Mode = "direct"
    fluid: str = ""

    @property
    def specific_work_J_per_kg(self) -> float:
        """Trabajo específico positivo: extraído (turbina) o consumido."""
        if self.device == "turbine":
            return -self.delta_h_real_J_per_kg
        return self.delta_h_real_J_per_kg


@dataclass(frozen=True)
class PolytropicStage:
    """Una etapa del compresor multietapa."""

    index: int  # 1-based
    state_in: StatePoint
    state_out_isen: StatePoint
    state_out_real: StatePoint
    delta_h_real_J_per_kg: float
    p_in_Pa: float
    p_out_Pa: float
    cooled_after: bool  # True si hay intercooler después de esta etapa


@dataclass(frozen=True)
class PolytropicResult:
    """Resultado del compresor multietapa, opcionalmente con intercooler."""

    fluid: str
    n_stages: int
    eta_s_per_stage: float
    pressure_ratio_per_stage: float
    intercool: bool
    t_intercool_K: float | None
    stages: list[PolytropicStage]
    total_delta_h_real_J_per_kg: float
    # Benchmark didáctico: misma compresión total con UNA sola etapa,
    # misma η_s. Permite mostrar el ahorro del intercooler.
    delta_h_single_stage_real_J_per_kg: float
    steps: IsentropicSteps

    @property
    def saving_pct(self) -> float:
        """Ahorro de trabajo frente a una sola etapa, en %."""
        single = self.delta_h_single_stage_real_J_per_kg
        if single == 0.0:
            return 0.0
        return 100.0 * (single - self.total_delta_h_real_J_per_kg) / single


# ---------------------------------------------------------------------
# Helpers de validación
# ---------------------------------------------------------------------


def _validate_eta_s(eta_s: float) -> None:
    if not (0.0 < eta_s <= 1.0):
        raise ValueError(
            f"El rendimiento isoentrópico debe estar en el intervalo "
            f"(0, 1]. Recibí η_s = {eta_s}. Un valor de 0 o negativo "
            f"no es físico y un valor mayor que 1 violaría el segundo "
            f"principio."
        )


def _validate_expansion(state_in: StatePoint, p_out_Pa: float) -> None:
    if p_out_Pa >= state_in.P_Pa:
        raise ValueError(
            f"Una turbina expande: la presión de salida debe ser "
            f"estrictamente menor que la de entrada. Recibí "
            f"P_out = {p_out_Pa / 1e5:.3f} bar ≥ P_in = "
            f"{state_in.P_Pa / 1e5:.3f} bar."
        )
    if p_out_Pa <= 0.0:
        raise ValueError(f"La presión de salida debe ser positiva. Recibí P_out = {p_out_Pa} Pa.")


def _validate_compression(state_in: StatePoint, p_out_Pa: float) -> None:
    if p_out_Pa <= state_in.P_Pa:
        raise ValueError(
            f"Un compresor (o bomba) comprime: la presión de salida "
            f"debe ser estrictamente mayor que la de entrada. Recibí "
            f"P_out = {p_out_Pa / 1e5:.3f} bar ≤ P_in = "
            f"{state_in.P_Pa / 1e5:.3f} bar."
        )


def _phase(fluid: str, state: StatePoint) -> str:
    """Fase según CoolProp (``'liquid'``, ``'gas'``, …) o ``''`` si falla."""
    try:
        return str(cp.PhaseSI("P", state.P_Pa, "T", state.T_K, fluid))
    except Exception:
        return ""


def _validate_liquid_inlet(fluid: str, state_in: StatePoint) -> None:
    """Verifica que ``state_in`` sea líquido (subenfriado o saturado).

    La convención ``x == -1`` de CoolProp solo dice "fuera de la
    región bifásica", lo cual incluye vapor sobrecalentado. Por eso
    además consultamos :func:`CoolProp.PhaseSI` para confirmar la fase.
    """
    # Caso fácil: x explícitamente saturado líquido.
    if state_in.x == 0.0:
        return
    # Caso inválido: mezcla bifásica o vapor.
    if 0.0 < state_in.x <= 1.0:
        raise ValueError(
            f"Una bomba opera con líquido. El estado de entrada tiene "
            f"título x = {state_in.x:.3f} (mezcla bifásica o vapor), "
            f"no es válido. El inlet debe ser líquido saturado (x = 0) "
            f"o subenfriado."
        )
    # x == -1 (fuera de la región bifásica): puede ser líquido
    # subenfriado, gas, supercrítico, etc. Consultamos PhaseSI.
    phase = _phase(fluid, state_in)
    liquid_phases = {"liquid", "supercritical_liquid"}
    if phase and phase not in liquid_phases:
        raise ValueError(
            f"Una bomba opera con líquido. El estado de entrada del "
            f"fluido '{fluid}' a T = {state_in.T_K - 273.15:.2f} °C y "
            f"P = {state_in.P_Pa / 1e5:.3f} bar está en fase '{phase}', "
            f"no en fase líquida. Revisá la presión o la temperatura "
            f"de entrada."
        )


def _validate_gas_inlet(fluid: str, state_in: StatePoint) -> None:
    """Un compresor trabaja con gas o vapor, no con líquido.

    Se admite vapor húmedo (0 < x ≤ 1) porque el cálculo es válido, pero
    un líquido saturado o comprimido a la entrada es un error conceptual:
    para elevar la presión de un líquido se usa una bomba.
    """
    if state_in.x == 0.0:
        is_liquid = True
    elif 0.0 < state_in.x <= 1.0:
        is_liquid = False
    else:
        is_liquid = _phase(fluid, state_in) in {"liquid", "supercritical_liquid"}
    if is_liquid:
        raise ValueError(
            f"Un compresor trabaja con gas o vapor, pero el estado de entrada "
            f"({state_in.T_K - 273.15:.2f} °C, {state_in.P_Pa / 1e5:.4g} bar) es "
            f"líquido. Para elevar la presión de un líquido se usa una bomba "
            f"(pestaña Bomba); si querías comprimir vapor, revisá la temperatura "
            f"o el título de entrada."
        )


def _validate_recovered_eta_s(eta_s: float, device: DeviceKind) -> None:
    """En el modo inverso, η_s computada debe caer en (0, 1]."""
    if eta_s > 1.0:
        raise ValueError(
            f"El rendimiento isoentrópico calculado dio η_s = {eta_s:.4f}, "
            f"que es mayor a 1. Físicamente imposible para una "
            f"{_device_label_es(device)}: significaría que el proceso "
            f"real extrae más trabajo (turbina) o consume menos (compresor/"
            f"bomba) que el ideal isoentrópico. Verificá los estados "
            f"ingresados — quizás están invertidos o las presiones son "
            f"inconsistentes."
        )
    if eta_s <= 0.0:
        raise ValueError(
            f"El rendimiento isoentrópico calculado dio η_s = {eta_s:.4f}, "
            f"no positivo. Esto suele indicar que el sentido del proceso "
            f"(expansión vs compresión) no coincide con el dispositivo "
            f"({_device_label_es(device)}) o que los estados son "
            f"incoherentes."
        )


def _device_label_es(device: DeviceKind) -> str:
    return {"turbine": "turbina", "compressor": "compresor", "pump": "bomba"}[device]


# ---------------------------------------------------------------------
# Helpers de cómputo
# ---------------------------------------------------------------------


def _isentropic_outlet(fluid: str, p_out_Pa: float, s_in_J_per_kg_K: float) -> StatePoint:
    """Estado 2s = (p_out, s_in), validado contra el rango de la ecuación de estado."""
    return fluid_state_from_pair(fluid, "PS", p=p_out_Pa, s=s_in_J_per_kg_K).to_state_point()


def _real_outlet_from_h(fluid: str, p_out_Pa: float, h_real_J_per_kg: float) -> StatePoint:
    """Estado 2 = (p_out, h_real), validado contra el rango de la ecuación de estado."""
    return fluid_state_from_pair(fluid, "PH", p=p_out_Pa, h=h_real_J_per_kg).to_state_point()


# ---------------------------------------------------------------------
# Turbina
# ---------------------------------------------------------------------


def turbine_direct(
    *,
    fluid: str,
    state_in: StatePoint,
    p_out_Pa: float,
    eta_s: float,
) -> IsentropicResult:
    """Modo directo: dado η_s y P_out, calcula el estado real de salida."""
    _validate_expansion(state_in, p_out_Pa)
    _validate_eta_s(eta_s)

    state_out_isen = _isentropic_outlet(fluid, p_out_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    # Turbina: trabajo positivo de salida = h_in - h_out. Real ≤ isoentrópico.
    delta_h_real = eta_s * delta_h_isen
    h_real = state_in.h_J_per_kg + delta_h_real
    state_out_real = _real_outlet_from_h(fluid, p_out_Pa, h_real)

    return IsentropicResult(
        device="turbine",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_turbine_direct(state_in, state_out_isen, state_out_real, eta_s),
        mode="direct",
        fluid=fluid,
    )


def turbine_inverse(
    *,
    fluid: str,
    state_in: StatePoint,
    state_out_real: StatePoint,
) -> IsentropicResult:
    """Modo inverso: dados estados de entrada y salida real, recupera η_s."""
    _validate_expansion(state_in, state_out_real.P_Pa)

    state_out_isen = _isentropic_outlet(fluid, state_out_real.P_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    delta_h_real = state_out_real.h_J_per_kg - state_in.h_J_per_kg
    if delta_h_isen == 0.0:
        raise ValueError(
            "El salto isoentrópico de entalpía es 0 (P_in = P_out o estado "
            "degenerado), no se puede definir η_s."
        )
    eta_s = delta_h_real / delta_h_isen
    _validate_recovered_eta_s(eta_s, "turbine")

    return IsentropicResult(
        device="turbine",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_turbine_inverse(state_in, state_out_isen, state_out_real, eta_s),
        mode="inverse",
        fluid=fluid,
    )


# ---------------------------------------------------------------------
# Compresor (gas)
# ---------------------------------------------------------------------


def compressor_direct(
    *,
    fluid: str,
    state_in: StatePoint,
    p_out_Pa: float,
    eta_s: float,
) -> IsentropicResult:
    _validate_compression(state_in, p_out_Pa)
    _validate_eta_s(eta_s)
    _validate_gas_inlet(fluid, state_in)

    state_out_isen = _isentropic_outlet(fluid, p_out_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    # Compresor: trabajo necesario h_out - h_in. Real ≥ isoentrópico.
    delta_h_real = delta_h_isen / eta_s
    h_real = state_in.h_J_per_kg + delta_h_real
    state_out_real = _real_outlet_from_h(fluid, p_out_Pa, h_real)

    return IsentropicResult(
        device="compressor",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_compressor_or_pump_direct(
            "compressor", state_in, state_out_isen, state_out_real, eta_s
        ),
        mode="direct",
        fluid=fluid,
    )


def compressor_inverse(
    *,
    fluid: str,
    state_in: StatePoint,
    state_out_real: StatePoint,
) -> IsentropicResult:
    _validate_compression(state_in, state_out_real.P_Pa)
    _validate_gas_inlet(fluid, state_in)

    state_out_isen = _isentropic_outlet(fluid, state_out_real.P_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    delta_h_real = state_out_real.h_J_per_kg - state_in.h_J_per_kg
    if delta_h_real == 0.0:
        raise ValueError("El salto real de entalpía es 0, no se puede definir η_s del compresor.")
    eta_s = delta_h_isen / delta_h_real
    _validate_recovered_eta_s(eta_s, "compressor")

    return IsentropicResult(
        device="compressor",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_compressor_or_pump_inverse(
            "compressor", state_in, state_out_isen, state_out_real, eta_s
        ),
        mode="inverse",
        fluid=fluid,
    )


# ---------------------------------------------------------------------
# Bomba (líquido)
# ---------------------------------------------------------------------


def pump_direct(
    *,
    fluid: str,
    state_in: StatePoint,
    p_out_Pa: float,
    eta_s: float,
) -> IsentropicResult:
    _validate_compression(state_in, p_out_Pa)
    _validate_eta_s(eta_s)
    _validate_liquid_inlet(fluid, state_in)

    state_out_isen = _isentropic_outlet(fluid, p_out_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    delta_h_real = delta_h_isen / eta_s
    h_real = state_in.h_J_per_kg + delta_h_real
    state_out_real = _real_outlet_from_h(fluid, p_out_Pa, h_real)

    return IsentropicResult(
        device="pump",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_compressor_or_pump_direct(
            "pump", state_in, state_out_isen, state_out_real, eta_s
        ),
        mode="direct",
        fluid=fluid,
    )


def pump_inverse(
    *,
    fluid: str,
    state_in: StatePoint,
    state_out_real: StatePoint,
) -> IsentropicResult:
    _validate_compression(state_in, state_out_real.P_Pa)
    _validate_liquid_inlet(fluid, state_in)

    state_out_isen = _isentropic_outlet(fluid, state_out_real.P_Pa, state_in.s_J_per_kg_K)
    delta_h_isen = state_out_isen.h_J_per_kg - state_in.h_J_per_kg
    delta_h_real = state_out_real.h_J_per_kg - state_in.h_J_per_kg
    if delta_h_real == 0.0:
        raise ValueError("El salto real de entalpía es 0, no se puede definir η_s de la bomba.")
    eta_s = delta_h_isen / delta_h_real
    _validate_recovered_eta_s(eta_s, "pump")

    return IsentropicResult(
        device="pump",
        state_in=state_in,
        state_out_isen=state_out_isen,
        state_out_real=state_out_real,
        eta_s=eta_s,
        delta_h_isen_J_per_kg=delta_h_isen,
        delta_h_real_J_per_kg=delta_h_real,
        steps=_build_steps_compressor_or_pump_inverse(
            "pump", state_in, state_out_isen, state_out_real, eta_s
        ),
        mode="inverse",
        fluid=fluid,
    )


@dataclass(frozen=True)
class PumpComparison:
    """Bomba: trabajo con la ecuación de estado vs. modelo de líquido incompresible."""

    v_in_m3_per_kg: float
    w_eos_J_per_kg: float
    w_incompressible_J_per_kg: float
    error_pct: float


def pump_incompressible_comparison(result: IsentropicResult) -> PumpComparison:
    """Compara el trabajo de la bomba con ``w ≈ v₁·(p₂ − p₁)/η_s``.

    Es la aproximación de líquido incompresible del vademecum §10.4 (y
    Cengel §7-13): el volumen específico no cambia al comprimir, así que
    w_s = ∫v dp ≈ v₁·Δp.
    """
    if result.device != "pump":
        raise ValueError("La comparación con el modelo incompresible es solo para bombas.")
    v_in = result.state_in.v_m3_per_kg
    if v_in is None:
        raise ValueError("El estado de entrada no trae volumen específico.")
    dp = result.state_out_real.P_Pa - result.state_in.P_Pa
    w_simple = v_in * dp / result.eta_s
    w_eos = result.delta_h_real_J_per_kg
    error = abs(w_eos - w_simple) / abs(w_eos) * 100.0 if w_eos != 0.0 else math.nan
    return PumpComparison(
        v_in_m3_per_kg=v_in,
        w_eos_J_per_kg=w_eos,
        w_incompressible_J_per_kg=w_simple,
        error_pct=error,
    )


# ---------------------------------------------------------------------
# Compresor multietapa (con intercooler opcional)
# ---------------------------------------------------------------------


def compressor_multistage(
    *,
    fluid: str,
    state_in: StatePoint,
    p_out_Pa: float,
    n_stages: int,
    eta_s_per_stage: float,
    intercool: bool,
    t_intercool_K: float | None = None,
) -> PolytropicResult:
    """Compresor multietapa con relación de presión equitativa por etapa.

    Parámetros
    ----------
    fluid : str
        Fluido aceptado por CoolProp.
    state_in : StatePoint
        Estado de entrada (gas).
    p_out_Pa : float
        Presión final de descarga, en Pa.
    n_stages : int
        Número de etapas (≥ 1). Cada etapa tiene la misma relación de
        compresión ``(p_out / p_in)^(1/n)`` — para dos etapas es la
        presión intermedia óptima p_x = √(p₁·p₂) del vademecum §6.3.
    eta_s_per_stage : float
        Rendimiento isoentrópico por etapa (mismo valor para todas).
    intercool : bool
        Si ``True``, entre etapas el fluido se enfría a temperatura
        ``t_intercool_K`` a la presión intermedia. La última etapa no
        tiene intercooler aguas abajo.
    t_intercool_K : float, opcional
        Temperatura objetivo del intercooler. Default = ``state_in.T_K``
        (cooling back to inlet temperature).

    Raises
    ------
    ValueError
        Además de las validaciones de cada etapa, si el intercooler
        enfría por debajo de la saturación a la presión intermedia (el
        fluido condensaría y la etapa siguiente comprimiría líquido).
    """
    if n_stages < 1:
        raise ValueError(f"n_stages debe ser ≥ 1. Recibí n_stages = {n_stages}.")
    _validate_compression(state_in, p_out_Pa)
    _validate_eta_s(eta_s_per_stage)
    if intercool:
        if t_intercool_K is None:
            t_intercool_K = state_in.T_K
        if t_intercool_K <= 0.0:
            raise ValueError(f"t_intercool_K debe ser positivo (kelvin). Recibí {t_intercool_K}.")

    pressure_ratio = (p_out_Pa / state_in.P_Pa) ** (1.0 / n_stages)

    stages: list[PolytropicStage] = []
    current = state_in
    total_delta_h = 0.0

    for i in range(n_stages):
        p_in_stage = current.P_Pa
        # Para la última etapa, garantizamos exactamente p_out_Pa
        # (evita acumulación numérica).
        p_out_stage = p_out_Pa if i == n_stages - 1 else p_in_stage * pressure_ratio
        stage_result = compressor_direct(
            fluid=fluid,
            state_in=current,
            p_out_Pa=p_out_stage,
            eta_s=eta_s_per_stage,
        )
        cooled_after = intercool and i < n_stages - 1
        stages.append(
            PolytropicStage(
                index=i + 1,
                state_in=stage_result.state_in,
                state_out_isen=stage_result.state_out_isen,
                state_out_real=stage_result.state_out_real,
                delta_h_real_J_per_kg=stage_result.delta_h_real_J_per_kg,
                p_in_Pa=p_in_stage,
                p_out_Pa=p_out_stage,
                cooled_after=cooled_after,
            )
        )
        total_delta_h += stage_result.delta_h_real_J_per_kg

        if cooled_after:
            assert t_intercool_K is not None
            current = _intercooled_state(fluid, t_intercool_K, p_out_stage, stage_index=i + 1)
        else:
            current = stage_result.state_out_real

    # Benchmark: misma compresión total en UNA sola etapa, misma η_s.
    single_stage = compressor_direct(
        fluid=fluid,
        state_in=state_in,
        p_out_Pa=p_out_Pa,
        eta_s=eta_s_per_stage,
    )

    return PolytropicResult(
        fluid=fluid,
        n_stages=n_stages,
        eta_s_per_stage=eta_s_per_stage,
        pressure_ratio_per_stage=pressure_ratio,
        intercool=intercool,
        t_intercool_K=t_intercool_K,
        stages=stages,
        total_delta_h_real_J_per_kg=total_delta_h,
        delta_h_single_stage_real_J_per_kg=single_stage.delta_h_real_J_per_kg,
        steps=_build_steps_multistage(
            n_stages=n_stages,
            eta_s_per_stage=eta_s_per_stage,
            pressure_ratio=pressure_ratio,
            intercool=intercool,
            t_intercool_K=t_intercool_K,
            total_delta_h=total_delta_h,
            single_stage_delta_h=single_stage.delta_h_real_J_per_kg,
        ),
    )


def _intercooled_state(fluid: str, t_K: float, p_Pa: float, *, stage_index: int) -> StatePoint:
    """Estado a la salida del intercooler; el fluido no puede condensar."""
    state = fluid_state_from_pair(fluid, "TP", t=t_K, p=p_Pa)
    if state.region in ("compressed_liquid", "saturated_liquid", "saturated_mixture"):
        sat = state.sat_at_P
        t_sat = f"{sat.vapor.T_K - 273.15:.2f} °C" if sat is not None else "—"
        raise ValueError(
            f"El intercooler después de la etapa {stage_index} enfría hasta "
            f"{t_K - 273.15:.2f} °C a {p_Pa / 1e5:.4g} bar, por debajo de la temperatura "
            f"de saturación a esa presión ({t_sat}): el fluido condensaría y la etapa "
            f"siguiente comprimiría líquido. Usá una temperatura de intercooler mayor "
            f"que la de saturación."
        )
    return state.to_state_point()


# ---------------------------------------------------------------------
# Valores iniciales por fluido y dispositivo
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceDefaults:
    """Datos iniciales (en SI) de un dispositivo para un fluido.

    Siempre son calculables (lo verifican los tests para cada fluido ×
    dispositivo). ``inlet`` son los kwargs de
    :func:`core.fluids.state_from_pair` para el par ``pair_in``.
    """

    pair_in: PairCode
    inlet: dict[str, float]
    p_out_Pa: float
    eta_s: float
    n_stages: int = 1
    t_intercool_K: float | None = None


def _p_sat(fluid: str, T_K: float) -> float:
    return saturation_at_temperature(fluid, T_K).P_sat_Pa


def _t_dew(fluid: str, P_Pa: float) -> float:
    return saturation_at_pressure(fluid, P_Pa).vapor.T_K


def suggested_device_inputs(fluid: str, device: DeviceName) -> DeviceDefaults:
    """Valores iniciales razonables para ``device`` con ``fluid``.

    - Agua: los estados de libro de la página (turbina 30 bar / 400 °C
      → 0,5 bar; bomba de condensado 0,1 bar → 150 bar) y compresión de
      vapor sobrecalentado (recompresión mecánica).
    - Aire: turbina de gas (10 bar / 900 °C → 1 atm), compresor de aire
      (1 → 8 bar) y bomba de aire líquido subenfriado (2 → 20 bar).
    - Refrigerantes, amoníaco y CO₂: compresión de vapor saturado del
      evaporador (−10 °C) a la presión del condensador (40 °C, o T_c − 5 K
      si la crítica es más baja, como en el CO₂), dos etapas desde −30 °C,
      y turbina/bomba de un ciclo orgánico.
    """
    if device not in ("turbine", "compressor", "pump", "multistage"):
        raise ValueError(f"Dispositivo '{device}' desconocido.")
    lim = fluid_limits(fluid)

    if fluid == "Water":
        if device == "turbine":
            return DeviceDefaults("TP", {"t": 673.15, "p": 3.0e6}, p_out_Pa=5.0e4, eta_s=0.90)
        if device == "pump":
            return DeviceDefaults("PX", {"p": 1.0e4, "x": 0.0}, p_out_Pa=1.5e7, eta_s=0.85)
        if device == "compressor":
            return DeviceDefaults("TP", {"t": 393.15, "p": 1.0e5}, p_out_Pa=3.0e5, eta_s=0.75)
        p_mid = math.sqrt(1.0e5 * 9.0e5)
        return DeviceDefaults(
            "TP",
            {"t": 393.15, "p": 1.0e5},
            p_out_Pa=9.0e5,
            eta_s=0.80,
            n_stages=2,
            t_intercool_K=_t_dew(fluid, p_mid) + 15.0,
        )

    if lim.T_crit_K < 273.15:  # gas permanente (aire)
        if device == "turbine":
            return DeviceDefaults("TP", {"t": 1173.15, "p": 1.0e6}, p_out_Pa=101_325.0, eta_s=0.88)
        if device == "pump":
            # Aire líquido subenfriado (T_sat a 2 bar ≈ −188 °C), como piden las
            # bombas criogénicas para no cavitar. Líquido saturado a 1 atm es el
            # estado de referencia de CoolProp (h = s = 0): con el ruido numérico
            # la tabla mostraba h₁ = −1.2×10⁻⁵ kJ/kg.
            return DeviceDefaults("TP", {"t": 78.15, "p": 2.0e5}, p_out_Pa=2.0e6, eta_s=0.75)
        if device == "compressor":
            return DeviceDefaults("TP", {"t": 298.15, "p": 1.0e5}, p_out_Pa=8.0e5, eta_s=0.80)
        return DeviceDefaults(
            "TP",
            {"t": 298.15, "p": 1.0e5},
            p_out_Pa=2.7e6,
            eta_s=0.85,
            n_stages=3,
            t_intercool_K=298.15,
        )

    # Refrigerantes, amoníaco y CO2.
    T_c, T_t = lim.T_crit_K, lim.T_triple_K
    T_cond = min(313.15, T_c - 5.0)
    if device == "compressor":
        T_evap = max(T_t + 10.0, min(263.15, T_c - 20.0))
        return DeviceDefaults(
            "PX", {"p": _p_sat(fluid, T_evap), "x": 1.0}, p_out_Pa=_p_sat(fluid, T_cond), eta_s=0.80
        )
    if device == "multistage":
        T_evap = max(T_t + 10.0, min(243.15, T_c - 40.0))
        p_in, p_out = _p_sat(fluid, T_evap), _p_sat(fluid, T_cond)
        p_mid = math.sqrt(p_in * p_out)
        return DeviceDefaults(
            "PX",
            {"p": p_in, "x": 1.0},
            p_out_Pa=p_out,
            eta_s=0.80,
            n_stages=2,
            # Como un enfriador flash: hasta casi vapor saturado a p_mid.
            t_intercool_K=max(T_evap, _t_dew(fluid, p_mid) + 1.0),
        )
    if fluid == "CarbonDioxide" and device == "turbine":
        # Expansión de CO2 supercrítico (ciclos Brayton de sCO2).
        return DeviceDefaults("TP", {"t": 423.15, "p": 1.0e7}, p_out_Pa=5.0e6, eta_s=0.85)
    # Ciclo orgánico: evaporación a T_hi, condensación a T_lo.
    T_hi = min(T_c - 15.0, 333.15)
    T_lo = min(293.15, T_hi - 10.0)
    p_hi, p_lo = _p_sat(fluid, T_hi), _p_sat(fluid, T_lo)
    if device == "pump":
        return DeviceDefaults("PX", {"p": p_lo, "x": 0.0}, p_out_Pa=p_hi, eta_s=0.75)
    T_in = min(_t_dew(fluid, p_hi) + 20.0, lim.T_max_K - 5.0)
    return DeviceDefaults("TP", {"t": T_in, "p": p_hi}, p_out_Pa=p_lo, eta_s=0.80)


# ---------------------------------------------------------------------
# Estados rotulados, tablas y exportación
# ---------------------------------------------------------------------


def isentropic_labeled_states(result: IsentropicResult) -> list[tuple[str, StatePoint]]:
    """Estados 1, 2s y 2 con su rótulo para tablas y diagramas."""
    return [
        ("1 (entrada)", result.state_in),
        ("2s (salida isoentrópica)", result.state_out_isen),
        ("2 (salida real)", result.state_out_real),
    ]


def multistage_labeled_states(result: PolytropicResult) -> list[tuple[str, StatePoint]]:
    """Entrada, salidas isoentrópica y real de cada etapa y salidas de los intercoolers."""
    rows: list[tuple[str, StatePoint]] = [("1 (entrada)", result.stages[0].state_in)]
    for k, stage in enumerate(result.stages):
        rows.append((f"Etapa {stage.index}: salida isoentrópica", stage.state_out_isen))
        rows.append((f"Etapa {stage.index}: salida real", stage.state_out_real))
        if stage.cooled_after and k + 1 < len(result.stages):
            rows.append(
                (f"Intercooler → entrada etapa {stage.index + 1}", result.stages[k + 1].state_in)
            )
    return rows


def to_fluid_states(
    fluid: str, labeled: list[tuple[str, StatePoint]]
) -> list[tuple[str, FluidState]]:
    """Completa cada estado (región, x, propiedades) a partir de su p y h."""
    return [
        (label, fluid_state_from_pair(fluid, "PH", p=s.P_Pa, h=s.h_J_per_kg))
        for label, s in labeled
    ]


_DEVICE_ES: dict[str, str] = {
    "turbine": "turbina",
    "compressor": "compresor",
    "pump": "bomba",
    "multistage": "compresor multietapa",
}


def _energy(value_si: float, system: UnitSystem) -> dict[str, Any]:
    return {
        "valor": convert_from_si(value_si, "specific_enthalpy", system),
        "unidad": unit_label("specific_enthalpy", system),
    }


def isentropic_to_dict(result: IsentropicResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    if not result.fluid:
        raise ValueError("El resultado no trae el fluido (construido a mano sin `fluid`).")
    states = to_fluid_states(result.fluid, isentropic_labeled_states(result))
    return {
        "dispositivo": _DEVICE_ES[result.device],
        "modo": "directo" if result.mode == "direct" else "inverso",
        "fluido": result.fluid,
        "fluido_es": FLUID_NAMES_ES.get(result.fluid, result.fluid),
        "sistema_de_unidades": system,
        "eta_s": result.eta_s,
        "trabajo_especifico": _energy(result.specific_work_J_per_kg, system),
        "delta_h_isoentropico": _energy(result.delta_h_isen_J_per_kg, system),
        "delta_h_real": _energy(result.delta_h_real_J_per_kg, system),
        "estados": states_table(states, system),
        "fuente": "CoolProp (HEOS; IAPWS-95 para el agua)",
    }


def multistage_to_dict(result: PolytropicResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado del compresor multietapa serializable a JSON."""
    states = to_fluid_states(result.fluid, multistage_labeled_states(result))
    t_ic = (
        None
        if result.t_intercool_K is None
        else {
            "valor": convert_from_si(result.t_intercool_K, "temperature", system),
            "unidad": unit_label("temperature", system),
        }
    )
    return {
        "dispositivo": _DEVICE_ES["multistage"],
        "fluido": result.fluid,
        "fluido_es": FLUID_NAMES_ES.get(result.fluid, result.fluid),
        "sistema_de_unidades": system,
        "etapas": result.n_stages,
        "eta_s_por_etapa": result.eta_s_per_stage,
        "relacion_de_compresion_por_etapa": result.pressure_ratio_per_stage,
        "intercooler": result.intercool,
        "temperatura_intercooler": t_ic,
        "trabajo_total": _energy(result.total_delta_h_real_J_per_kg, system),
        "trabajo_una_etapa": _energy(result.delta_h_single_stage_real_J_per_kg, system),
        "ahorro_vs_una_etapa_pct": result.saving_pct,
        "estados": states_table(states, system),
        "fuente": "CoolProp (HEOS; IAPWS-95 para el agua)",
    }


def summary_csv(summary: dict[str, Any]) -> str:
    """CSV de un resultado exportado: pares clave/valor y luego la tabla de estados."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for key, value in summary.items():
        if key == "estados":
            continue
        if isinstance(value, dict):
            writer.writerow([f"# {key}", value.get("valor"), value.get("unidad")])
        else:
            writer.writerow([f"# {key}", value])
    rows = summary.get("estados") or []
    if rows:
        dict_writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        dict_writer.writeheader()
        for row in rows:
            dict_writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buffer.getvalue()


# ---------------------------------------------------------------------
# Builders de pasos didácticos (LaTeX + narrativa), por sistema de unidades
# ---------------------------------------------------------------------

_EH = "specific_enthalpy"


def isentropic_steps(result: IsentropicResult, system: UnitSystem) -> IsentropicSteps:
    """Pasos didácticos de ``result`` con los valores en el sistema ``system``."""
    args = (result.state_in, result.state_out_isen, result.state_out_real, result.eta_s)
    if result.device == "turbine":
        if result.mode == "direct":
            return _build_steps_turbine_direct(*args, system=system)
        return _build_steps_turbine_inverse(*args, system=system)
    if result.mode == "direct":
        return _build_steps_compressor_or_pump_direct(result.device, *args, system=system)
    return _build_steps_compressor_or_pump_inverse(result.device, *args, system=system)


def multistage_steps(result: PolytropicResult, system: UnitSystem) -> IsentropicSteps:
    """Pasos didácticos del compresor multietapa en el sistema ``system``."""
    return _build_steps_multistage(
        n_stages=result.n_stages,
        eta_s_per_stage=result.eta_s_per_stage,
        pressure_ratio=result.pressure_ratio_per_stage,
        intercool=result.intercool,
        t_intercool_K=result.t_intercool_K,
        total_delta_h=result.total_delta_h_real_J_per_kg,
        single_stage_delta_h=result.delta_h_single_stage_real_J_per_kg,
        system=system,
    )


def _enthalpies(*states: StatePoint) -> tuple[float, ...]:
    """Entalpías de los estados, con el ruido de CoolProp alrededor de h = 0 en 0.

    En el estado de referencia del fluido (p. ej. aire líquido saturado a
    1 atm, h = 0 por convención NBP) CoolProp devuelve ~1e-8 J/kg: con 5
    cifras significativas el paso mostraría "h₁ = −1.2352×10⁻⁵ kJ/kg".
    Lo que está por debajo de 1e-5 veces la mayor entalpía del proceso es
    ruido numérico y se muestra como 0.
    """
    values = tuple(s.h_J_per_kg for s in states)
    scale = max(abs(v) for v in values)
    return tuple(0.0 if abs(v) < 1e-5 * scale else v for v in values)


def _build_steps_turbine_direct(
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
    *,
    system: UnitSystem = "Técnico",
) -> IsentropicSteps:
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"h_2 &= h_1 - \eta_s \cdot (h_1 - h_{2s}) \\"
        r"w_t &= h_1 - h_2"
        r"\end{aligned}"
    )
    h1, h2s, h2 = _enthalpies(state_in, state_out_isen, state_out_real)
    w_t = h1 - h2
    n1, n2s, n2 = (latex_value(h, _EH, system) for h in (h1, h2s, h2))
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {latex_quantity(h1, _EH, system)} \\"
        rf"h_{{2s}} &= {latex_quantity(h2s, _EH, system)} \\"
        + (
            rf"h_2 &= {n1} \\ &\quad - {latex_number(eta_s, 4)}\cdot({n1} \\ &\qquad "
            rf"- {latex_paren(n2s)}) \\"
            if latex_is_wide(n1, n2s)
            else rf"h_2 &= {n1} - {latex_number(eta_s, 4)}\cdot({n1} - {latex_paren(n2s)}) \\"
        )
        + rf"&= {latex_quantity(h2, _EH, system)} \\"
        rf"w_t &= {_diff_rows(n1, n2)} \\"
        rf"&= {latex_quantity(w_t, _EH, system)}"
        r"\end{aligned}"
    )
    narrative = (
        f"Turbina, modo directo. Se fija la presión de salida y se "
        f"asume η_s = {eta_s:.4f}. (1) El estado isoentrópico de salida "
        f"se obtiene imponiendo s₂s = s₁ y resolviendo con CoolProp para "
        f"P₂; eso da h₂s = {text_quantity(h2s, _EH, system)}. (2) El estado real "
        f"recupera solo una fracción η_s del salto ideal: h₂ = h₁ − η_s·(h₁ − h₂s). "
        f"(3) El trabajo específico extraído es w_t = h₁ − h₂ = "
        f"{text_quantity(w_t, _EH, system)}."
    )
    return IsentropicSteps(formula, substituted, narrative)


def _build_steps_turbine_inverse(
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
    *,
    system: UnitSystem = "Técnico",
) -> IsentropicSteps:
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"\eta_s &= \frac{h_1 - h_2}{h_1 - h_{2s}}"
        r"\end{aligned}"
    )
    h1, h2s, h2 = _enthalpies(state_in, state_out_isen, state_out_real)
    n1, n2s, n2 = (latex_value(h, _EH, system) for h in (h1, h2s, h2))
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {latex_quantity(h1, _EH, system)} \\"
        rf"h_2 &= {latex_quantity(h2, _EH, system)} \\"
        rf"h_{{2s}} &= {latex_quantity(h2s, _EH, system)} \\"
        rf"\eta_s &= \frac{{{n1} - {latex_paren(n2)}}}{{{n1} - {latex_paren(n2s)}}} \\"
        rf"&= {latex_number(eta_s, 4)}"
        r"\end{aligned}"
    )
    narrative = (
        f"Turbina, modo inverso. Dados los estados real de entrada y "
        f"salida, se calcula el estado isoentrópico de salida con "
        f"s₂s = s₁ y P₂ = P_out, obteniendo h₂s = {text_quantity(h2s, _EH, system)}. "
        f"El rendimiento isoentrópico es el cociente entre el salto "
        f"real de entalpía y el salto ideal: η_s = "
        f"(h₁ − h₂) / (h₁ − h₂s) = {eta_s:.4f}."
    )
    return IsentropicSteps(formula, substituted, narrative)


def _very_wide(a: str, b: str) -> bool:
    """Una resta ``a - b`` que no entra en un renglón de celular: dos ×10ⁿ y un negativo."""
    return (a + b).count(r"\times") == 2 and "-" in (a[:1], b[:1])


def _diff_rows(a: str, b: str) -> str:
    r"""``a - b`` para un renglón de ``aligned``; si es muy ancha, ``- b`` en otro renglón."""
    if _very_wide(a, b):
        return rf"{a} \\ &\quad - {latex_paren(b)}"
    return rf"{a} - {latex_paren(b)}"


def _compression_h2_row(n1: str, n2s: str, eta: str) -> str:
    r"""Renglón ``h_2 = h_1 + (h_2s − h_1)/η_s`` de un ``aligned``, angosto para un celular.

    Con números anchos (×10ⁿ o negativos) el término de la fracción va en otro
    renglón; si además el numerador tiene dos números con ×10ⁿ y alguno es
    negativo (un líquido orgánico en SI), ni la fracción entra: la resta se
    escribe como división, partida en dos renglones.
    """
    numerator = rf"{n2s} - {latex_paren(n1)}"
    if not latex_is_wide(n1, n2s):
        return rf"h_2 &= {n1} + \dfrac{{{numerator}}}{{{eta}}} \\"
    if _very_wide(n2s, n1):
        return rf"h_2 &= {n1} \\ &\quad + ({n2s} \\ &\qquad - {latex_paren(n1)})/{eta} \\"
    return rf"h_2 &= {n1} \\ &\quad + \dfrac{{{numerator}}}{{{eta}}} \\"


def _build_steps_compressor_or_pump_direct(
    device: DeviceKind,
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
    *,
    system: UnitSystem = "Técnico",
) -> IsentropicSteps:
    work_label = "w_c" if device == "compressor" else "w_p"
    label = _device_label_es(device).capitalize()
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"h_2 &= h_1 + \dfrac{h_{2s} - h_1}{\eta_s} \\"
        + work_label
        + r" &= h_2 - h_1"
        + r"\end{aligned}"
    )
    h1, h2s, h2 = _enthalpies(state_in, state_out_isen, state_out_real)
    w = h2 - h1
    n1, n2s, n2 = (latex_value(h, _EH, system) for h in (h1, h2s, h2))
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {latex_quantity(h1, _EH, system)} \\"
        rf"h_{{2s}} &= {latex_quantity(h2s, _EH, system)} \\"
        + _compression_h2_row(n1, n2s, latex_number(eta_s, 4))
        + rf"&= {latex_quantity(h2, _EH, system)} \\"
        + work_label
        + rf" &= {_diff_rows(n2, n1)} \\"
        + rf"&= {latex_quantity(w, _EH, system)}"
        + r"\end{aligned}"
    )
    narrative = (
        f"{label}, modo directo. Se fija la presión de salida y se "
        f"asume η_s = {eta_s:.4f}. (1) El estado isoentrópico de salida "
        f"se obtiene imponiendo s₂s = s₁ y resolviendo con CoolProp para "
        f"P₂; eso da h₂s = {text_quantity(h2s, _EH, system)}. (2) El estado real "
        f"consume más trabajo que el ideal: h₂ = h₁ + (h₂s − h₁)/η_s. "
        f"(3) El trabajo específico necesario es {work_label} = h₂ − h₁ "
        f"= {text_quantity(w, _EH, system)}."
    )
    return IsentropicSteps(formula, substituted, narrative)


def _build_steps_compressor_or_pump_inverse(
    device: DeviceKind,
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
    *,
    system: UnitSystem = "Técnico",
) -> IsentropicSteps:
    label = _device_label_es(device).capitalize()
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"\eta_s &= \frac{h_{2s} - h_1}{h_2 - h_1}"
        r"\end{aligned}"
    )
    h1, h2s, h2 = _enthalpies(state_in, state_out_isen, state_out_real)
    n1, n2s, n2 = (latex_value(h, _EH, system) for h in (h1, h2s, h2))
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {latex_quantity(h1, _EH, system)} \\"
        rf"h_2 &= {latex_quantity(h2, _EH, system)} \\"
        rf"h_{{2s}} &= {latex_quantity(h2s, _EH, system)} \\"
        rf"\eta_s &= \frac{{{n2s} - {latex_paren(n1)}}}{{{n2} - {latex_paren(n1)}}} \\"
        rf"&= {latex_number(eta_s, 4)}"
        r"\end{aligned}"
    )
    narrative = (
        f"{label}, modo inverso. Dados los estados real de entrada y "
        f"salida, se calcula el estado isoentrópico con s₂s = s₁ y "
        f"P₂ = P_out, obteniendo h₂s = {text_quantity(h2s, _EH, system)}. El "
        f"rendimiento isoentrópico es el cociente entre el trabajo ideal mínimo y "
        f"el real: η_s = (h₂s − h₁) / (h₂ − h₁) = {eta_s:.4f}."
    )
    return IsentropicSteps(formula, substituted, narrative)


def _build_steps_multistage(
    *,
    n_stages: int,
    eta_s_per_stage: float,
    pressure_ratio: float,
    intercool: bool,
    t_intercool_K: float | None,
    total_delta_h: float,
    single_stage_delta_h: float,
    system: UnitSystem = "Técnico",
) -> IsentropicSteps:
    saving_pct = (
        100.0 * (single_stage_delta_h - total_delta_h) / single_stage_delta_h
        if single_stage_delta_h != 0
        else 0.0
    )
    formula = (
        r"\begin{aligned}"
        r"\Pi &= \left(\dfrac{P_\text{out}}{P_\text{in}}\right)^{1/n} \\"
        r"h_{i,\text{out}} &= h_{i,\text{in}} "
        r"+ \dfrac{h_{i,\text{isen}} - h_{i,\text{in}}}{\eta_s} \\"
        r"T_{i+1,\text{in}} &= T_\text{ic} \quad \text{(intercooler)} \\"
        r"\Delta h_\text{total} &= \sum_{i=1}^{n} (h_{i,\text{out}} - h_{i,\text{in}})"
        r"\end{aligned}"
    )
    ic_descr = (
        f"con intercooler a T = {text_quantity(t_intercool_K, 'temperature', system)} entre etapas"
        if intercool and t_intercool_K is not None
        else "sin intercooler"
    )
    substituted = (
        r"\begin{aligned}"
        rf"n &= {n_stages},\quad \eta_s = {latex_number(eta_s_per_stage, 4)} \\"
        rf"\Pi &= {latex_number(pressure_ratio, 5)} \\"
        rf"\Delta h_\text{{total}} &= {latex_quantity(total_delta_h, _EH, system)} \\"
        rf"\Delta h_\text{{1 etapa}} &= {latex_quantity(single_stage_delta_h, _EH, system)}"
        r"\end{aligned}"
    )
    narrative = (
        f"Compresor multietapa de {n_stages} etapa(s), {ic_descr}. "
        f"(1) Se divide la relación de compresión total en partes "
        f"iguales: Π = (P_out / P_in)^(1/n) = {pressure_ratio:.4f}. "
        f"(2) Cada etapa se calcula como un compresor isoentrópico "
        f"con η_s = {eta_s_per_stage:.4f}. (3) Si hay intercooler, "
        f"entre etapas el fluido vuelve a la temperatura del intercooler "
        f"a la presión intermedia; eso reduce el trabajo de la etapa "
        f"siguiente porque comienza más frío. "
        f"(4) El Δh real total es {text_quantity(total_delta_h, _EH, system)}, "
        f"frente a {text_quantity(single_stage_delta_h, _EH, system)} que daría una "
        f"sola etapa con la misma η_s "
        f"(diferencia: {saving_pct:+.2f} %)."
    )
    return IsentropicSteps(formula, substituted, narrative)

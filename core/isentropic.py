"""Rendimientos isoentrópicos: turbinas, compresores, bombas, multietapa.

Funciones puras sobre :class:`core.fluids.StatePoint`, en unidades SI
internamente. Cada función devuelve un :class:`IsentropicResult` (o
:class:`PolytropicResult` para multietapa) con todos los estados
intermedios, los Δh isoentrópico y real, y un bloque didáctico de
fórmula LaTeX + sustitución + narrativa en español plano.

El módulo NO importa Streamlit.

Cita
----
Cengel, Y. A., & Boles, M. A. (2015). *Thermodynamics: An Engineering
Approach* (8th ed., §7-12 "Isentropic Efficiencies of Steady-Flow
Devices"). McGraw-Hill.

Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014). "Pure and
Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source
Thermophysical Property Library CoolProp". *Industrial & Engineering
Chemistry Research*, 53(6), 2498-2508. DOI: 10.1021/ie4033999
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import CoolProp.CoolProp as cp

from core.fluids import StatePoint, state_from_pair

DeviceKind = Literal["turbine", "compressor", "pump"]


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
    """Resultado de un proceso isoentrópico de un dispositivo de flujo."""

    device: DeviceKind
    state_in: StatePoint
    state_out_isen: StatePoint
    state_out_real: StatePoint
    eta_s: float
    delta_h_isen_J_per_kg: float
    delta_h_real_J_per_kg: float
    steps: IsentropicSteps


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
    try:
        phase = cp.PhaseSI("P", state_in.P_Pa, "T", state_in.T_K, fluid)
    except Exception:
        phase = ""
    liquid_phases = {"liquid", "supercritical_liquid"}
    if phase and phase not in liquid_phases:
        raise ValueError(
            f"Una bomba opera con líquido. El estado de entrada del "
            f"fluido '{fluid}' a T = {state_in.T_K - 273.15:.2f} °C y "
            f"P = {state_in.P_Pa / 1e5:.3f} bar está en fase '{phase}', "
            f"no en fase líquida. Revisá la presión o la temperatura "
            f"de entrada."
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
    return state_from_pair(fluid, "PS", p=p_out_Pa, s=s_in_J_per_kg_K)


def _real_outlet_from_h(fluid: str, p_out_Pa: float, h_real_J_per_kg: float) -> StatePoint:
    return state_from_pair(fluid, "PH", p=p_out_Pa, h=h_real_J_per_kg)


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
    )


def compressor_inverse(
    *,
    fluid: str,
    state_in: StatePoint,
    state_out_real: StatePoint,
) -> IsentropicResult:
    _validate_compression(state_in, state_out_real.P_Pa)

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
        compresión ``(p_out / p_in)^(1/n)``.
    eta_s_per_stage : float
        Rendimiento isoentrópico por etapa (mismo valor para todas).
    intercool : bool
        Si ``True``, entre etapas el fluido se enfría a temperatura
        ``t_intercool_K`` a la presión intermedia. La última etapa no
        tiene intercooler aguas abajo.
    t_intercool_K : float, opcional
        Temperatura objetivo del intercooler. Default = ``state_in.T_K``
        (cooling back to inlet temperature).
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
            current = state_from_pair(fluid, "TP", t=t_intercool_K, p=p_out_stage)
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


# ---------------------------------------------------------------------
# Builders de pasos didácticos (LaTeX + narrativa)
# ---------------------------------------------------------------------


def _build_steps_turbine_direct(
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
) -> IsentropicSteps:
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"h_2 &= h_1 - \eta_s \cdot (h_1 - h_{2s}) \\"
        r"w_t &= h_1 - h_2"
        r"\end{aligned}"
    )
    h1 = state_in.h_J_per_kg / 1e3
    h2s = state_out_isen.h_J_per_kg / 1e3
    h2 = state_out_real.h_J_per_kg / 1e3
    w_t = h1 - h2
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {h1:.4g}~\text{{kJ/kg}} \\"
        rf"h_{{2s}} &= {h2s:.4g}~\text{{kJ/kg}} \\"
        rf"h_2 &= {h1:.4g} - {eta_s:.4f}\cdot({h1:.4g} - {h2s:.4g}) "
        rf"= {h2:.4g}~\text{{kJ/kg}} \\"
        rf"w_t &= {h1:.4g} - {h2:.4g} = {w_t:.4g}~\text{{kJ/kg}}"
        r"\end{aligned}"
    )
    narrative = (
        f"Turbina, modo directo. Se fija la presión de salida y se "
        f"asume η_s = {eta_s:.4f}. (1) El estado isoentrópico de salida "
        f"se obtiene imponiendo s₂s = s₁ y resolviendo con CoolProp para "
        f"P₂; eso da h₂s = {h2s:.4g} kJ/kg. (2) El estado real recupera "
        f"solo una fracción η_s del salto ideal: h₂ = h₁ − η_s·(h₁ − h₂s). "
        f"(3) El trabajo específico extraído es w_t = h₁ − h₂ = "
        f"{w_t:.4g} kJ/kg."
    )
    return IsentropicSteps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
    )


def _build_steps_turbine_inverse(
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
) -> IsentropicSteps:
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"\eta_s &= \frac{h_1 - h_2}{h_1 - h_{2s}}"
        r"\end{aligned}"
    )
    h1 = state_in.h_J_per_kg / 1e3
    h2s = state_out_isen.h_J_per_kg / 1e3
    h2 = state_out_real.h_J_per_kg / 1e3
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {h1:.4g}~\text{{kJ/kg}},\quad "
        rf"h_2 = {h2:.4g}~\text{{kJ/kg}},\quad h_{{2s}} = {h2s:.4g}~\text{{kJ/kg}} \\"
        rf"\eta_s &= \frac{{{h1:.4g} - {h2:.4g}}}{{{h1:.4g} - {h2s:.4g}}}"
        rf" = {eta_s:.4f}"
        r"\end{aligned}"
    )
    narrative = (
        f"Turbina, modo inverso. Dados los estados real de entrada y "
        f"salida, se calcula el estado isoentrópico de salida con "
        f"s₂s = s₁ y P₂ = P_out, obteniendo h₂s = {h2s:.4g} kJ/kg. "
        f"El rendimiento isoentrópico es el cociente entre el salto "
        f"real de entalpía y el salto ideal: η_s = "
        f"(h₁ − h₂) / (h₁ − h₂s) = {eta_s:.4f}."
    )
    return IsentropicSteps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
    )


def _build_steps_compressor_or_pump_direct(
    device: DeviceKind,
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
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
    h1 = state_in.h_J_per_kg / 1e3
    h2s = state_out_isen.h_J_per_kg / 1e3
    h2 = state_out_real.h_J_per_kg / 1e3
    w = h2 - h1
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {h1:.4g}~\text{{kJ/kg}} \\"
        rf"h_{{2s}} &= {h2s:.4g}~\text{{kJ/kg}} \\"
        rf"h_2 &= {h1:.4g} + \dfrac{{{h2s:.4g} - {h1:.4g}}}{{{eta_s:.4f}}}"
        rf" = {h2:.4g}~\text{{kJ/kg}} \\"
        + work_label
        + rf" &= {h2:.4g} - {h1:.4g} = {w:.4g}~\text{{kJ/kg}}"
        + r"\end{aligned}"
    )
    narrative = (
        f"{label}, modo directo. Se fija la presión de salida y se "
        f"asume η_s = {eta_s:.4f}. (1) El estado isoentrópico de salida "
        f"se obtiene imponiendo s₂s = s₁ y resolviendo con CoolProp para "
        f"P₂; eso da h₂s = {h2s:.4g} kJ/kg. (2) El estado real consume "
        f"más trabajo que el ideal: h₂ = h₁ + (h₂s − h₁)/η_s. "
        f"(3) El trabajo específico necesario es {work_label} = h₂ − h₁ "
        f"= {w:.4g} kJ/kg."
    )
    return IsentropicSteps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
    )


def _build_steps_compressor_or_pump_inverse(
    device: DeviceKind,
    state_in: StatePoint,
    state_out_isen: StatePoint,
    state_out_real: StatePoint,
    eta_s: float,
) -> IsentropicSteps:
    label = _device_label_es(device).capitalize()
    formula = (
        r"\begin{aligned}"
        r"s_{2s} &= s_1 \\"
        r"h_{2s} &= h(P_2,\, s_{2s}) \\"
        r"\eta_s &= \frac{h_{2s} - h_1}{h_2 - h_1}"
        r"\end{aligned}"
    )
    h1 = state_in.h_J_per_kg / 1e3
    h2s = state_out_isen.h_J_per_kg / 1e3
    h2 = state_out_real.h_J_per_kg / 1e3
    substituted = (
        r"\begin{aligned}"
        rf"h_1 &= {h1:.4g}~\text{{kJ/kg}},\quad "
        rf"h_2 = {h2:.4g}~\text{{kJ/kg}},\quad h_{{2s}} = {h2s:.4g}~\text{{kJ/kg}} \\"
        rf"\eta_s &= \frac{{{h2s:.4g} - {h1:.4g}}}{{{h2:.4g} - {h1:.4g}}}"
        rf" = {eta_s:.4f}"
        r"\end{aligned}"
    )
    narrative = (
        f"{label}, modo inverso. Dados los estados real de entrada y "
        f"salida, se calcula el estado isoentrópico con s₂s = s₁ y "
        f"P₂ = P_out, obteniendo h₂s = {h2s:.4g} kJ/kg. El rendimiento "
        f"isoentrópico es el cociente entre el trabajo ideal mínimo y "
        f"el real: η_s = (h₂s − h₁) / (h₂ − h₁) = {eta_s:.4f}."
    )
    return IsentropicSteps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
    )


def _build_steps_multistage(
    *,
    n_stages: int,
    eta_s_per_stage: float,
    pressure_ratio: float,
    intercool: bool,
    t_intercool_K: float | None,
    total_delta_h: float,
    single_stage_delta_h: float,
) -> IsentropicSteps:
    saving_pct = (
        100.0 * (single_stage_delta_h - total_delta_h) / single_stage_delta_h
        if single_stage_delta_h != 0
        else 0.0
    )
    formula = (
        r"\begin{aligned}"
        r"\Pi &= \left(\dfrac{P_\text{out}}{P_\text{in}}\right)^{1/n} "
        r"\quad \text{(relación de compresión por etapa)} \\"
        r"\text{Por etapa } i: \quad &h_{i,\text{out}} = h_{i,\text{in}} "
        r"+ \dfrac{h_{i,\text{isen}} - h_{i,\text{in}}}{\eta_s} \\"
        r"\text{Si hay intercooler: } &T_{i+1,\text{in}} = T_\text{ic} "
        r"\text{ a } P_{i,\text{out}} \\"
        r"\Delta h_\text{total} &= \sum_{i=1}^{n} (h_{i,\text{out}} - h_{i,\text{in}})"
        r"\end{aligned}"
    )
    ic_descr = (
        f"con intercooler a T = {t_intercool_K - 273.15:.2f} °C entre etapas"
        if intercool and t_intercool_K is not None
        else "sin intercooler"
    )
    substituted = (
        r"\begin{aligned}"
        rf"n &= {n_stages},\quad \eta_s = {eta_s_per_stage:.4f},\quad "
        rf"\Pi = {pressure_ratio:.4f} \\"
        rf"\Delta h_\text{{total}} &= {total_delta_h / 1e3:.4g}~\text{{kJ/kg}} \\"
        rf"\Delta h_\text{{1 etapa}} &= {single_stage_delta_h / 1e3:.4g}~\text{{kJ/kg}}"
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
        f"(4) El Δh real total es {total_delta_h / 1e3:.4g} kJ/kg, "
        f"frente a {single_stage_delta_h / 1e3:.4g} kJ/kg que daría una "
        f"sola etapa con la misma η_s "
        f"(diferencia: {saving_pct:+.2f} %)."
    )
    return IsentropicSteps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
    )

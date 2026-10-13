"""Intercambiador de casco y tubos con el método de Kern (Fase 12.1).

Kern (1950), cap. 7; Kakaç y Liu (2002), cap. 8. Un casco de diámetro interior D_s con N_t
tubos (diámetro exterior d_o y calibre BWG, que da d_i; largo L) en arreglo triangular
(30°) o cuadrado (90°) de paso P_T, con N_p pasos por los tubos y bafles segmentados del
25 % de corte (el de Kern) separados B, N_b bafles.

- **Lado del casco (Kern)**: el área de flujo cruzado a_s = D_s·(P_T − d_o)·B/P_T, la
  velocidad másica G_s = ṁ/a_s, el diámetro equivalente D_e (con el área libre de la celda
  del arreglo y el perímetro mojado), Re_s = G_s·D_e/μ y
  h_o = 0,36·(k/D_e)·Re_s^0,55·Pr^⅓·(μ/μ_p)^0,14, para 2·10³ < Re_s < 10⁶.
  Δp_s = f·G_s²·D_s·(N_b + 1)/(2·ρ·D_e·φ), con φ = (μ/μ_p)^0,14 y
  f = exp(0,576 − 0,19·ln Re_s), el ajuste de Kakaç y Liu del gráfico de Kern
  (400 < Re_s ≤ 10⁶).
- **Lado de los tubos**: a_t = N_t·π·d_i²/(4·N_p), u_t = ṁ/(ρ·a_t), h_i con las
  correlaciones de la Fase 8.1 y Δp_t = (f·L·N_p/d_i + 4·N_p)·ρ·u_t²/2, con cuatro cargas
  de velocidad por paso en las vueltas (Kern; Kakaç y Liu).
- **U** limpio (U_C) y con el ensuciamiento pedido (U_D), referidos al área exterior
  A_o = N_t·π·d_o·L; con el ε-NTU de un paso de casco (Fase 8.2), Q̇ y las salidas.
- **El criterio de Kern** para un servicio dado: el U que hace falta,
  U_req = Q̇/(A_o·F·ΔT_ml), y el ensuciamiento que el equipo tolera,
  R_d = (U_C − U_req)/(U_C·U_req), contra el pedido R_d,req = R''_f,o + R''_f,i·d_o/d_i.

Las propiedades van a la temperatura media de cada corriente y μ_p a la de la pared (se
itera).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from core.heat_transfer.design_common import (
    TUBE_SIDE_CORRELATIONS,
    DesignStream,
    FluidProperties,
    Side,
    StreamProps,
    TubeFlow,
    check_single_phase,
    check_stream,
    overall_u_per_length,
    stream_name,
    stream_props,
    tube_flow,
    wall_viscosity,
)
from core.heat_transfer.exchangers import (
    Arrangement,
    ExchangerResult,
    OverallUResult,
    RatingInputs,
    SizingInputs,
    SizingTarget,
    Stream,
    solve_rating,
    solve_sizing,
)
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "BWG_WALL_IN",
    "LAYOUTS",
    "SHELL_PROBLEMS",
    "SHELL_TUBE_EXAMPLES",
    "TUBE_PASSES",
    "BaffleSweepPoint",
    "Layout",
    "ShellSide",
    "ShellTubeExample",
    "ShellTubeInputs",
    "ShellTubeResult",
    "baffle_sweep",
    "estimate_tube_count",
    "kern_equivalent_diameter",
    "kern_friction",
    "kern_shell_side",
    "shell_tube_notes",
    "shell_tube_to_dict",
    "solve_shell_tube",
    "tube_inner_diameter",
]

Layout = Literal["triangular", "square"]
ShellProblem = Literal["rating", "check"]

LAYOUTS: dict[Layout, str] = {
    "triangular": "Triangular (30°)",
    "square": "Cuadrado (90°)",
}
SHELL_PROBLEMS: dict[ShellProblem, str] = {
    "rating": "Verificación (las salidas de un equipo dado)",
    "check": "Servicio pedido (¿alcanza el equipo? El R_d de Kern)",
}
TUBE_PASSES: tuple[int, ...] = (1, 2, 4, 6, 8)

#: Espesor de pared de los tubos por calibre BWG, en pulgadas (TEMA; Kern, tabla 10).
BWG_WALL_IN: dict[int, float] = {
    10: 0.134,
    11: 0.120,
    12: 0.109,
    13: 0.095,
    14: 0.083,
    15: 0.072,
    16: 0.065,
    17: 0.058,
    18: 0.049,
    19: 0.042,
    20: 0.035,
}

#: CTP y CL de la estimación de la cantidad de tubos (Kakaç y Liu, 2002, cap. 8).
_CTP: dict[int, float] = {1: 0.93, 2: 0.90}
_CTP_MANY = 0.85
_CL: dict[Layout, float] = {"triangular": 0.87, "square": 1.0}

#: Rangos de Kern: el h_o (2·10³ < Re < 10⁶) y el ajuste de f (400 < Re ≤ 10⁶).
RE_KERN_H = (2.0e3, 1.0e6)
RE_KERN_F = (400.0, 1.0e6)

_SIDE_ES: dict[Side, str] = {"hot": "caliente", "cold": "frío"}
_IN = 0.0254


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _fmt(x: float) -> str:
    """Sin notación científica: 2621, 120, 38,3 o 0,0125."""
    if abs(x) >= 100.0:
        return f"{x:.0f}"
    return _num(x)


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


# ---------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------


def tube_inner_diameter(d_o_m: float, bwg: int) -> float:
    """d_i = d_o − 2·t con el espesor del calibre BWG (TEMA)."""
    if bwg not in BWG_WALL_IN:
        raise ValueError(f"Calibre BWG desconocido: {bwg} (van de 10 a 20).")
    return d_o_m - 2.0 * BWG_WALL_IN[bwg] * _IN


def _ctp(passes: int) -> float:
    return _CTP.get(passes, _CTP_MANY)


def estimate_tube_count(D_s_m: float, P_T_m: float, layout: Layout, passes: int) -> int:
    """N_t = 0,785·(CTP/CL)·D_s²/P_T² (Kakaç y Liu, 2002, cap. 8): CTP = 0,93, 0,90 o 0,85
    con 1, 2 o más pasos por los tubos (lo que ocupan las divisiones de los pasos), CL = 0,87
    triangular y 1 cuadrado (el área de la celda del arreglo es CL·P_T²)."""
    return int(0.785 * (_ctp(passes) / _CL[layout]) * D_s_m**2 / P_T_m**2)


def kern_equivalent_diameter(d_o_m: float, P_T_m: float, layout: Layout) -> float:
    """El diámetro equivalente del lado del casco (Kern, 1950, fig. 28):
    cuadrado, D_e = 4·(P_T² − π·d_o²/4)/(π·d_o); triangular,
    D_e = 4·(√3/4·P_T² − π·d_o²/8)/(π·d_o/2) (la mitad de un tubo en cada celda)."""
    if layout == "square":
        return 4.0 * (P_T_m**2 - math.pi * d_o_m**2 / 4.0) / (math.pi * d_o_m)
    return (
        4.0 * (math.sqrt(3.0) / 4.0 * P_T_m**2 - math.pi * d_o_m**2 / 8.0) / (math.pi * d_o_m / 2.0)
    )


def kern_friction(Re: float) -> float:
    """f del lado del casco: f = exp(0,576 − 0,19·ln Re), el ajuste de Kakaç y Liu (2002) del
    gráfico de Kern (1950, fig. 29) para 400 < Re ≤ 10⁶."""
    return math.exp(0.576 - 0.19 * math.log(Re))


@dataclass(frozen=True)
class ShellSide:
    """El lado del casco con Kern."""

    a_s_m2: float
    G_s: float
    D_e_m: float
    Re: float
    phi: float
    Nu: float
    h_W_per_m2K: float
    f: float
    crossings: int
    dp_Pa: float
    warnings: tuple[str, ...]


def kern_shell_side(
    props: StreamProps,
    m_dot_kg_s: float,
    D_s_m: float,
    B_m: float,
    P_T_m: float,
    d_o_m: float,
    layout: Layout,
    baffles: int,
    mu_wall_Pa_s: float | None,
) -> ShellSide:
    """h_o y Δp del lado del casco con el método de Kern (1950, cap. 7; Kakaç y Liu, 2002):
    h_o = 0,36·(k/D_e)·Re^0,55·Pr^⅓·φ y Δp = f·G_s²·D_s·(N_b + 1)/(2·ρ·D_e·φ)."""
    a_s = D_s_m * (P_T_m - d_o_m) * B_m / P_T_m
    G = m_dot_kg_s / a_s
    D_e = kern_equivalent_diameter(d_o_m, P_T_m, layout)
    Re = G * D_e / props.mu_Pa_s
    phi = (props.mu_Pa_s / mu_wall_Pa_s) ** 0.14 if mu_wall_Pa_s else 1.0
    Nu = 0.36 * Re**0.55 * props.Pr ** (1.0 / 3.0) * phi
    h = Nu * props.k_W_per_mK / D_e
    f = kern_friction(Re)
    crossings = baffles + 1
    dp = f * G**2 * D_s_m * crossings / (2.0 * props.rho_kg_per_m3 * D_e * phi)
    warnings: list[str] = []
    if not RE_KERN_H[0] <= Re <= RE_KERN_H[1]:
        warnings.append(f"Re_s = {_fmt(Re)} está fuera del rango de Kern para h_o (2·10³ a 10⁶).")
    if not RE_KERN_F[0] <= Re <= RE_KERN_F[1]:
        warnings.append(f"Re_s = {_fmt(Re)} está fuera del ajuste de f de Kakaç y Liu (400 a 10⁶).")
    return ShellSide(a_s, G, D_e, Re, phi, Nu, h, f, crossings, dp, tuple(warnings))


# ---------------------------------------------------------------------
# El intercambiador
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShellTubeInputs:
    """Un casco y tubos: las corrientes, la geometría y el problema.

    ``shell_side`` es la corriente que va por el casco. ``tube_count`` = None estima N_t con
    D_s (Kakaç y Liu). Con ``problem="check"``, ``target`` y ``target_value`` fijan el
    servicio pedido (una salida o Q̇).
    """

    hot: DesignStream
    cold: DesignStream
    D_s_m: float
    d_o_m: float
    d_i_m: float
    L_m: float
    P_T_m: float
    B_m: float
    baffles: int
    layout: Layout = "triangular"
    tube_passes: int = 2
    tube_count: int | None = None
    shell_side: Side = "hot"
    k_wall_W_per_mK: float = 45.0
    problem: ShellProblem = "rating"
    target: SizingTarget = "T_hot_out"
    target_value: float = 0.0
    tube_correlation: str = "auto"
    T0_K: float = 298.15


@dataclass(frozen=True)
class ShellTubeResult:
    """El casco y tubos resuelto."""

    inputs: ShellTubeInputs
    N_t: int
    shell_props: StreamProps
    tube_props: StreamProps
    shell: ShellSide
    tube: TubeFlow
    T_wall_tube_K: float
    T_wall_shell_K: float
    u_dirty: OverallUResult
    u_clean: OverallUResult
    exchanger: ExchangerResult
    warnings: tuple[str, ...]
    iterations: int

    @property
    def tube_stream(self) -> Side:
        return "cold" if self.inputs.shell_side == "hot" else "hot"

    @property
    def area_m2(self) -> float:
        """A_o = N_t·π·d_o·L."""
        i = self.inputs
        return self.N_t * math.pi * i.d_o_m * i.L_m

    @property
    def U_C_W_per_m2K(self) -> float:
        """U limpio, referido al área exterior."""
        return self.u_clean.UA_W_per_K / (math.pi * self.inputs.d_o_m)

    @property
    def U_D_W_per_m2K(self) -> float:
        """U con el ensuciamiento pedido, referido al área exterior."""
        return self.u_dirty.UA_W_per_K / (math.pi * self.inputs.d_o_m)

    @property
    def R_d_required(self) -> float:
        """El ensuciamiento pedido, referido al área exterior: R''_f,o + R''_f,i·d_o/d_i."""
        i = self.inputs
        shell_s, tube_s = _sides(i)
        return shell_s.R_f_m2K_per_W + tube_s.R_f_m2K_per_W * i.d_o_m / i.d_i_m

    @property
    def Q_W(self) -> float:
        return self.exchanger.Q_W

    @property
    def T_hot_out_K(self) -> float:
        return self.exchanger.hot.T_out_K

    @property
    def T_cold_out_K(self) -> float:
        return self.exchanger.cold.T_out_K

    @property
    def U_required_W_per_m2K(self) -> float | None:
        """Kern: el U que hace falta para el servicio con el área que hay,
        Q̇/(A_o·F·ΔT_ml) (solo con un servicio pedido)."""
        if self.inputs.problem != "check":
            return None
        ex = self.exchanger
        return ex.Q_W / (self.area_m2 * ex.F * ex.dT_lm_K)

    @property
    def R_d_available(self) -> float | None:
        """Kern: el ensuciamiento que el equipo tolera, (U_C − U_req)/(U_C·U_req)."""
        U = self.U_required_W_per_m2K
        if U is None:
            return None
        return (self.U_C_W_per_m2K - U) / (self.U_C_W_per_m2K * U)

    @property
    def area_required_m2(self) -> float | None:
        """El área que hace falta con U_D (servicio pedido)."""
        if self.inputs.problem != "check":
            return None
        return self.exchanger.A_m2

    @property
    def length_required_m(self) -> float | None:
        A = self.area_required_m2
        if A is None:
            return None
        return A / (self.N_t * math.pi * self.inputs.d_o_m)

    @property
    def overdesign(self) -> float | None:
        """El área que hay sobre la que hace falta, menos 1."""
        A = self.area_required_m2
        return None if A is None else self.area_m2 / A - 1.0

    @property
    def dp_tube_Pa(self) -> float:
        """Δp_t = (f·L·N_p/d_i + 4·N_p)·ρ·u²/2."""
        i, t = self.inputs, self.tube
        n = i.tube_passes
        dyn = self.tube_props.rho_kg_per_m3 * t.velocity_m_s**2 / 2.0
        return (t.f * i.L_m * n / i.d_i_m + 4.0 * n) * dyn

    @property
    def dp_tube_returns_Pa(self) -> float:
        i, t = self.inputs, self.tube
        return 4.0 * i.tube_passes * self.tube_props.rho_kg_per_m3 * t.velocity_m_s**2 / 2.0


def _sides(i: ShellTubeInputs) -> tuple[DesignStream, DesignStream]:
    """(la corriente del casco, la de los tubos)."""
    return (i.hot, i.cold) if i.shell_side == "hot" else (i.cold, i.hot)


def _validate(i: ShellTubeInputs) -> int:
    """Los datos; devuelve N_t (el dado o el estimado)."""
    check_stream(i.hot, "caliente")
    check_stream(i.cold, "frío")
    if i.hot.T_in_K <= i.cold.T_in_K:
        raise ValueError(
            f"El fluido caliente entra a {_degC(i.hot.T_in_K)} y el frío a "
            f"{_degC(i.cold.T_in_K)}: el caliente tiene que entrar más caliente que el frío."
        )
    if i.layout not in LAYOUTS:
        raise ValueError(f"Arreglo desconocido: {i.layout!r}.")
    if i.problem not in SHELL_PROBLEMS:
        raise ValueError(f"Problema desconocido: {i.problem!r}.")
    if i.tube_passes not in TUBE_PASSES:
        raise ValueError("Los pasos por los tubos son 1, 2, 4, 6 u 8.")
    if i.tube_correlation not in TUBE_SIDE_CORRELATIONS:
        raise ValueError(f"Correlación de tubo desconocida: {i.tube_correlation!r}.")
    if not (i.D_s_m > 0.0 and i.d_o_m > 0.0 and i.L_m > 0.0 and i.B_m > 0.0):
        raise ValueError(
            "El casco, los tubos, el largo y la separación de bafles tienen que ser positivos."
        )
    if not i.d_i_m > 0.0:
        raise ValueError(
            "El diámetro interior de los tubos no es positivo: el calibre deja la pared más "
            "gruesa que el tubo."
        )
    if i.d_i_m >= i.d_o_m:
        raise ValueError("El diámetro interior de los tubos tiene que ser menor que el exterior.")
    if i.P_T_m <= i.d_o_m:
        raise ValueError(
            f"El paso P_T = {_num(i.P_T_m * 1e3)} mm no supera al diámetro de los tubos "
            f"({_num(i.d_o_m * 1e3)} mm): los tubos se tocan."
        )
    if i.baffles < 1:
        raise ValueError("Hace falta al menos un bafle (el método de Kern es con bafles).")
    if not i.k_wall_W_per_mK > 0.0:
        raise ValueError("La conductividad de la pared de los tubos tiene que ser positiva.")
    N_max = int(0.785 / _CL[i.layout] * i.D_s_m**2 / i.P_T_m**2)
    N_t = (
        i.tube_count
        if i.tube_count is not None
        else estimate_tube_count(i.D_s_m, i.P_T_m, i.layout, i.tube_passes)
    )
    if N_t < 1:
        raise ValueError(
            f"En un casco de {_num(i.D_s_m * 1e3)} mm con paso de {_num(i.P_T_m * 1e3)} mm no "
            "entra ningún tubo: agrandá el casco."
        )
    if N_t > N_max:
        raise ValueError(
            f"{N_t} tubos no entran en un casco de {_num(i.D_s_m * 1e3)} mm con paso de "
            f"{_num(i.P_T_m * 1e3)} mm: como mucho unos {N_max}."
        )
    if N_t < i.tube_passes:
        raise ValueError("Hay más pasos por los tubos que tubos.")
    return N_t


def _arrangement(i: ShellTubeInputs) -> Arrangement:
    """Un paso por los tubos: contracorriente; dos o más: un paso de casco (Fase 8.2)."""
    if i.tube_passes == 1:
        return Arrangement("counter")
    return Arrangement("shell", 1, tubes="cold" if i.shell_side == "hot" else "hot")


def solve_shell_tube(inputs: ShellTubeInputs) -> ShellTubeResult:
    """El casco y tubos con Kern: los h de cada lado, el U, Q̇ y las salidas (verificación) o
    el R_d de Kern (servicio pedido).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno (geometría, salidas imposibles, cambio de fase).
    """
    i = inputs
    N_t = _validate(i)
    shell_s, tube_s = _sides(i)
    heating_tube = i.shell_side == "hot"
    arrangement = _arrangement(i)
    check = i.problem == "check"
    T_h_out = i.target_value if (check and i.target == "T_hot_out") else i.hot.T_in_K
    T_c_out = i.target_value if (check and i.target == "T_cold_out") else i.cold.T_in_K
    T_w_tube = T_w_shell = 0.5 * (i.hot.T_in_K + i.cold.T_in_K)
    A_o = N_t * math.pi * i.d_o_m * i.L_m
    per_tube = tube_s.m_dot_kg_s / (N_t / i.tube_passes)
    result: ExchangerResult | None = None
    iterations = 0
    for iterations in range(1, 201):  # noqa: B007 — se informa cuántas hicieron falta
        props_h = stream_props(i.hot, 0.5 * (i.hot.T_in_K + T_h_out), "caliente")
        props_c = stream_props(i.cold, 0.5 * (i.cold.T_in_K + T_c_out), "frío")
        props_s, props_t = (props_h, props_c) if i.shell_side == "hot" else (props_c, props_h)
        shell = kern_shell_side(
            props_s,
            shell_s.m_dot_kg_s,
            i.D_s_m,
            i.B_m,
            i.P_T_m,
            i.d_o_m,
            i.layout,
            i.baffles,
            wall_viscosity(shell_s, T_w_shell),
        )
        tube = tube_flow(
            props_t,
            per_tube,
            i.d_i_m,
            i.L_m,
            heating_tube,
            wall_viscosity(tube_s, T_w_tube),
            i.tube_correlation,
        )
        u = overall_u_per_length(
            tube.h_W_per_m2K,
            shell.h_W_per_m2K,
            i.d_i_m,
            i.d_o_m,
            i.k_wall_W_per_mK,
            tube_s.R_f_m2K_per_W,
            shell_s.R_f_m2K_per_W,
            "en los tubos",
            "en el casco",
        )
        U_D = u.UA_W_per_K / (math.pi * i.d_o_m)
        hot = Stream("caliente", i.hot.T_in_K, i.hot.m_dot_kg_s, props_h.cp_J_per_kgK)
        cold = Stream("frío", i.cold.T_in_K, i.cold.m_dot_kg_s, props_c.cp_J_per_kgK)
        if check:
            result = solve_sizing(
                SizingInputs(arrangement, hot, cold, U_D, i.target, i.target_value, T0_K=i.T0_K)
            )
        else:
            result = solve_rating(RatingInputs(arrangement, hot, cold, U_D, A_o, T0_K=i.T0_K))
        T_t = props_t.T_K
        T_s = props_s.T_K
        R_total = u.R_total_K_per_W
        new_w_tube = T_t + (T_s - T_t) * (1.0 / (tube.h_W_per_m2K * u.A_i_m2)) / R_total
        new_w_shell = T_s - (T_s - T_t) * (1.0 / (shell.h_W_per_m2K * u.A_o_m2)) / R_total
        new_h, new_c = result.hot.T_out_K, result.cold.T_out_K
        done = (
            abs(new_h - T_h_out) < 1e-9
            and abs(new_c - T_c_out) < 1e-9
            and abs(new_w_tube - T_w_tube) < 1e-6
            and abs(new_w_shell - T_w_shell) < 1e-6
        )
        T_h_out, T_c_out, T_w_tube, T_w_shell = new_h, new_c, new_w_tube, new_w_shell
        if done:
            break
    else:  # pragma: no cover — converge en pocas vueltas
        raise ValueError("Las temperaturas de salida no convergieron.")
    assert result is not None
    check_single_phase(i.hot, T_h_out, "caliente")
    check_single_phase(i.cold, T_c_out, "frío")
    u_clean = overall_u_per_length(
        tube.h_W_per_m2K,
        shell.h_W_per_m2K,
        i.d_i_m,
        i.d_o_m,
        i.k_wall_W_per_mK,
        0.0,
        0.0,
        "en los tubos",
        "en el casco",
    )
    warnings = [f"Casco: {w}" for w in shell.warnings]
    warnings += [f"Tubos: {w}" for w in tube.warnings]
    if not i.D_s_m / 5.0 <= i.B_m <= i.D_s_m:
        warnings.append(
            f"La separación de los bafles ({_num(i.B_m * 1e3)} mm) está fuera de lo habitual, "
            f"entre D_s/5 y D_s ({_num(i.D_s_m * 200)} a {_num(i.D_s_m * 1e3)} mm; Kern)."
        )
    warnings += list(result.warnings)
    return ShellTubeResult(
        inputs=i,
        N_t=N_t,
        shell_props=props_s,
        tube_props=props_t,
        shell=shell,
        tube=tube,
        T_wall_tube_K=T_w_tube,
        T_wall_shell_K=T_w_shell,
        u_dirty=u,
        u_clean=u_clean,
        exchanger=result,
        warnings=tuple(warnings),
        iterations=iterations,
    )


# ---------------------------------------------------------------------
# Barrido de bafles y notas
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BaffleSweepPoint:
    B_m: float
    baffles: int
    h_o_W_per_m2K: float
    dp_shell_Pa: float
    U_D_W_per_m2K: float


def baffle_sweep(inputs: ShellTubeInputs, n: int = 15) -> list[BaffleSweepPoint]:
    """h_o, Δp del casco y U contra la separación de los bafles, de D_s/5 a D_s, con
    N_b = L/B − 1: el compromiso del diseño (más bafles, más h_o y mucho más Δp)."""
    points: list[BaffleSweepPoint] = []
    lo, hi = inputs.D_s_m / 5.0, inputs.D_s_m
    for j in range(n):
        B = lo * (hi / lo) ** (j / (n - 1))
        baffles = max(1, round(inputs.L_m / B) - 1)
        try:
            r = solve_shell_tube(replace(inputs, B_m=B, baffles=baffles))
        except ValueError:
            continue
        points.append(
            BaffleSweepPoint(B, baffles, r.shell.h_W_per_m2K, r.shell.dp_Pa, r.U_D_W_per_m2K)
        )
    return points


#: Kern (1950) suele admitir unos 10 psi (69 kPa) por corriente en los líquidos.
DP_LIQUID_KERN_PA = 10.0 * 6894.757293168361


def _which(label: str, kind: str) -> str:
    rest = label[0].lower() + label[1:]
    return f"la del {rest}" if kind == "ensuciamiento" else f"la de la {rest}"


def shell_tube_notes(r: ShellTubeResult) -> list[str]:
    """Interpretación física del resultado (markdown)."""
    i = r.inputs
    notes: list[str] = []
    top = max(r.u_dirty.resistances, key=lambda x: x.R_K_per_W)
    notes.append(
        f"La resistencia mayor es {_which(top.label, top.kind)} "
        f"({_num(100.0 * r.u_dirty.share(top), 2)} % del total)."
    )
    if i.tube_count is None:
        notes.append(
            f"N_t = {r.N_t} sale de D_s con la estimación de Kakaç y Liu "
            "(CTP y CL: lo que ocupan las divisiones de los pasos y el arreglo)."
        )
    if i.problem == "check":
        Rd, Rreq = r.R_d_available, r.R_d_required
        assert Rd is not None
        if Rd < 0.0:
            notes.append(
                f"Ni limpio alcanza: hace falta U = {_fmt(r.U_required_W_per_m2K or 0.0)} "
                f"W/(m²·K) y limpio da {_fmt(r.U_C_W_per_m2K)}. Hace falta más área."
            )
        elif Rd >= Rreq:
            notes.append(
                f"El equipo sirve: tolera R_d = {_num(Rd)} m²·K/W de ensuciamiento y se piden "
                f"{_num(Rreq)} (Kern)."
            )
        else:
            notes.append(
                f"Limpio alcanza, pero tolera solo R_d = {_num(Rd)} m²·K/W y se piden "
                f"{_num(Rreq)}: cuando se ensucie no va a cumplir (Kern)."
            )
        over = r.overdesign
        if over is not None:
            notes.append(
                f"Con el ensuciamiento pedido hace falta {_fmt(r.area_required_m2 or 0.0)} m² "
                f"de los {_fmt(r.area_m2)} m² que hay ({'+' if over >= 0 else ''}"
                f"{_num(100.0 * over, 2)} % de sobrediseño)."
            )
    for where, dp in (("el casco", r.shell.dp_Pa), ("los tubos", r.dp_tube_Pa)):
        if dp > DP_LIQUID_KERN_PA:
            notes.append(
                f"La caída de presión en {where} es de {_fmt(dp / 1e3)} kPa: más que los "
                "10 psi (69 kPa) que Kern suele admitir por corriente en los líquidos."
            )
    return notes


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShellTubeExample:
    description: str
    inputs: ShellTubeInputs


_C = 273.15

#: El agua del ejemplo de Peters et al. a 30 °C, con μ a la pared a 40 °C (la que usa el libro).
_WATER_30 = FluidProperties(995.0, 4178.0, 0.000803, 0.615, mu_wall_Pa_s=0.000657, name="agua")
#: Kerosene a ~70 °C (propiedades ilustrativas), más viscoso junto a la pared fría.
_KEROSENE = FluidProperties(770.0, 2300.0, 0.0009, 0.135, mu_wall_Pa_s=0.0012, name="kerosene")

SHELL_TUBE_EXAMPLES: dict[str, ShellTubeExample] = {
    "Agua por el casco (Peters, Timmerhaus y West, ejemplo 14-6)": ShellTubeExample(
        '11 kg/s de agua por un casco de 0,584 m con tubos de 3/4" (19 mm) en arreglo '
        'cuadrado de 1" (25,4 mm), bafles cada 0,1524 m (22 bafles) y 3,66 m de largo. El '
        "libro compara los métodos del lado del casco con el agua a 30 °C y la pared a "
        "40 °C (esas propiedades van como dato): Kern da h_o = 3369 W/(m²·K) y Δp = 16,4 kPa "
        "leyendo j_H y f de los gráficos; Bell–Delaware, 3080 W/(m²·K) y 9,48 kPa. Lo de los "
        "tubos (agua caliente por un paso y la cantidad de tubos) es a elección.",
        ShellTubeInputs(
            hot=DesignStream(45.0 + _C, 30.0, fluid="Water", p_Pa=3.0e5),
            cold=DesignStream(25.0 + _C, 11.0, properties=_WATER_30),
            D_s_m=0.584,
            d_o_m=0.019,
            d_i_m=tube_inner_diameter(0.019, 16),
            L_m=3.66,
            P_T_m=0.0254,
            B_m=0.1524,
            baffles=22,
            layout="square",
            tube_passes=1,
            shell_side="cold",
            problem="rating",
        ),
    ),
    "Enfriador de kerosene con agua de río": ShellTubeExample(
        "5 kg/s de kerosene (propiedades a elección, a ~70 °C) entran al casco de 0,387 m a "
        "90 °C y se tienen que enfriar a 50 °C con 4 kg/s de agua de río que entra a 25 °C por "
        'los tubos de 3/4" BWG 16 (triangular de 1", 4 pasos, 6,1 m), con 0,0002 m²·K/W de '
        "ensuciamiento en el agua y 0,0009 en el kerosene (TEMA). ¿Alcanza el equipo? Probá "
        "acortar los tubos y mirá el R_d de Kern.",
        ShellTubeInputs(
            hot=DesignStream(90.0 + _C, 5.0, properties=_KEROSENE, R_f_m2K_per_W=0.0009),
            cold=DesignStream(25.0 + _C, 4.0, fluid="Water", p_Pa=3.0e5, R_f_m2K_per_W=0.0002),
            D_s_m=0.387,
            d_o_m=0.01905,
            d_i_m=tube_inner_diameter(0.01905, 16),
            L_m=6.1,
            P_T_m=0.0254,
            B_m=0.15,
            baffles=40,
            layout="triangular",
            tube_passes=4,
            shell_side="hot",
            problem="check",
            target="T_hot_out",
            target_value=50.0 + _C,
        ),
    ),
    "Intercambiador agua–agua de 1 paso por el casco y 2 por los tubos": ShellTubeExample(
        "20 kg/s de agua de proceso entran al casco de 0,686 m a 60 °C y se enfrían con "
        '40 kg/s de agua que entra a 20 °C por los tubos de 1" BWG 14 en arreglo cuadrado de '
        '1¼", con bafles cada 0,3 m y 6 m de largo. Verificación: las salidas que da el '
        "equipo con el ensuciamiento.",
        ShellTubeInputs(
            hot=DesignStream(60.0 + _C, 20.0, fluid="Water", p_Pa=3.0e5, R_f_m2K_per_W=0.0001),
            cold=DesignStream(20.0 + _C, 40.0, fluid="Water", p_Pa=3.0e5, R_f_m2K_per_W=0.0002),
            D_s_m=0.686,
            d_o_m=0.0254,
            d_i_m=tube_inner_diameter(0.0254, 14),
            L_m=6.0,
            P_T_m=0.03175,
            B_m=0.3,
            baffles=19,
            layout="square",
            tube_passes=2,
            shell_side="hot",
            problem="rating",
        ),
    ),
}


def _q(value: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value, kind, system), "unidad": unit_label(kind, system)}


def shell_tube_to_dict(r: ShellTubeResult, system: UnitSystem) -> dict[str, Any]:
    """El casco y tubos resuelto, serializable a JSON y CSV."""
    i = r.inputs
    shell_s, tube_s = _sides(i)
    s, t = r.shell, r.tube
    data: dict[str, Any] = {
        "problema": SHELL_PROBLEMS[i.problem],
        "por_el_casco": f"{stream_name(shell_s)} ({_SIDE_ES[i.shell_side]})",
        "por_los_tubos": f"{stream_name(tube_s)} ({_SIDE_ES[r.tube_stream]})",
        "D_s": _q(i.D_s_m, "small_length", system),
        "d_o": _q(i.d_o_m, "small_length", system),
        "d_i": _q(i.d_i_m, "small_length", system),
        "L": _q(i.L_m, "length", system),
        "P_T": _q(i.P_T_m, "small_length", system),
        "arreglo": LAYOUTS[i.layout],
        "N_t": r.N_t,
        "pasos_tubos": i.tube_passes,
        "B": _q(i.B_m, "small_length", system),
        "bafles": i.baffles,
        "casco": {
            "a_s": _q(s.a_s_m2, "area", system),
            "G_s": _q(s.G_s, "mass_flux", system),
            "D_e": _q(s.D_e_m, "small_length", system),
            "Re": s.Re,
            "phi": s.phi,
            "Nu": s.Nu,
            "h_o": _q(s.h_W_per_m2K, "heat_transfer_coefficient", system),
            "f": s.f,
            "Delta_p": _q(s.dp_Pa, "pressure_loss", system),
        },
        "tubos": {
            "velocidad": _q(t.velocity_m_s, "speed", system),
            "Re": t.Re,
            "correlacion": t.correlation,
            "Nu": t.Nu,
            "h_i": _q(t.h_W_per_m2K, "heat_transfer_coefficient", system),
            "f": t.f,
            "Delta_p": _q(r.dp_tube_Pa, "pressure_loss", system),
        },
        "U_C": _q(r.U_C_W_per_m2K, "heat_transfer_coefficient", system),
        "U_D": _q(r.U_D_W_per_m2K, "heat_transfer_coefficient", system),
        "area": _q(r.area_m2, "area", system),
        "Q": _q(r.Q_W, "heat_rate", system),
        "T_caliente_salida": _q(r.T_hot_out_K, "temperature", system),
        "T_frio_salida": _q(r.T_cold_out_K, "temperature", system),
        "F": r.exchanger.F,
        "Delta_T_ml": _q(r.exchanger.dT_lm_K, "temperature_difference", system),
        "avisos": list(r.warnings),
        "notas": shell_tube_notes(r),
    }
    if i.problem == "check":
        data["U_requerido"] = _q(r.U_required_W_per_m2K or 0.0, "heat_transfer_coefficient", system)
        data["R_d_disponible"] = _q(r.R_d_available or 0.0, "fouling_resistance", system)
        data["R_d_pedido"] = _q(r.R_d_required, "fouling_resistance", system)
        data["area_requerida"] = _q(r.area_required_m2 or 0.0, "area", system)
    return data

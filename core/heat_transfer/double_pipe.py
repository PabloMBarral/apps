"""Intercambiador de doble tubo (Fase 12.1).

Incropera et al. (2007), §8.7 (el anular) y §11.3; Çengel y Ghajar (2015), §11-2; Kern
(1950), cap. 6. Un tubo interior (diámetros d_i y d_o, pared de conductividad k) dentro de
un caño de diámetro interior D: una corriente va por el tubo y la otra por el anular, en
contracorriente o en paralelo; el equipo se arma con horquillas de dos ramas.

- **h_i**: las correlaciones de tubo de la Fase 8.1 con Re = ρ·u·d_i/μ.
- **h_o**: el anular con el diámetro hidráulico D_h = D − d_o. Laminar, con la tabla de
  Kays y Perkins (Incropera, tabla 8.2; Çengel y Ghajar, tabla 8-4): el Nu de la pared
  interior con la exterior aislada, interpolado en d_o/D. Turbulento, con las mismas
  correlaciones del tubo y D_h (Incropera, §8.7).
- **U** con el ensuciamiento de cada lado y la pared, referido al área exterior del tubo,
  A = π·d_o·L (con d_o = d_i la pared es delgada y no cuenta, como en los libros).
- **Dimensionamiento**: la salida de una corriente (o Q̇) da el área y el largo,
  L = A/(π·d_o), y las horquillas. **Verificación**: las horquillas dan el largo y ε-NTU las
  salidas (con el ε-NTU de la Fase 8.2).
- **Δp**: Δp = f·(L/D)·ρ·u²/2 de cada lado; en el anular laminar, f·Re del anular
  concéntrico (Shah y London, 1978), entre 64 (d_o/D → 0) y 96 (placas paralelas).

Las propiedades van a la temperatura media de cada corriente y μ_p a la de la pared: con
CoolProp se itera hasta que las salidas no cambian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from core.heat_transfer.convection import (
    RE_LAMINAR_TUBE,
    RE_TURBULENT_TUBE,
    nu_dittus_boelter,
    nu_gnielinski,
    nu_sieder_tate,
)
from core.heat_transfer.design_common import (
    TUBE_SIDE_CORRELATIONS,
    DesignStream,
    FluidProperties,
    Side,
    StreamProps,
    TubeFlow,
    check_single_phase,
    check_stream,
    darcy_f,
    overall_u_per_length,
    regime,
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
    "ANNULUS_CORRELATIONS",
    "ANNULUS_TABLE",
    "DOUBLE_PIPE_EXAMPLES",
    "FLOWS",
    "PROBLEMS",
    "AnnulusFlow",
    "DoublePipeExample",
    "DoublePipeInputs",
    "DoublePipeResult",
    "annulus_flow",
    "annulus_friction_Re",
    "annulus_nu_laminar",
    "double_pipe_notes",
    "double_pipe_to_dict",
    "solve_double_pipe",
    "swap_sides",
]

Flow = Literal["counter", "parallel"]
Problem = Literal["sizing", "rating"]

FLOWS: dict[Flow, str] = {"counter": "Contracorriente", "parallel": "Flujo paralelo"}
PROBLEMS: dict[Problem, str] = {
    "sizing": "Dimensionamiento (el largo para una salida)",
    "rating": "Verificación (las salidas de un equipo dado)",
}
ANNULUS_CORRELATIONS: dict[str, str] = {
    "auto": "Automática (laminar, Kays y Perkins; turbulento, Gnielinski)",
    "dittus_boelter": "Dittus y Boelter con D_h",
    "gnielinski": "Gnielinski con D_h",
    "sieder_tate": "Sieder y Tate con D_h",
}

#: Kays y Perkins (1973), en Incropera, tabla 8.2, y Çengel y Ghajar, tabla 8-4: el Nu
#: laminar desarrollado de la pared interior de un anular (la exterior, aislada), contra
#: d_o/D (D_i/D_o en los libros).
ANNULUS_TABLE: tuple[tuple[float, float], ...] = (
    (0.05, 17.46),
    (0.10, 11.56),
    (0.25, 7.37),
    (0.50, 5.74),
    (1.00, 4.86),
)

_SIDE_ES: dict[Side, str] = {"hot": "caliente", "cold": "frío"}


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _fmt(x: float) -> str:
    """Sin notación científica: 2621, 120, 38,3 o 0,0125."""
    if abs(x) >= 100.0:
        return f"{x:.0f}"
    return _num(x)


def _which(label: str, kind: str) -> str:
    """«la de la convección en el anular», «la del ensuciamiento…», «la de la pared»."""
    rest = label[0].lower() + label[1:]
    return f"la del {rest}" if kind == "ensuciamiento" else f"la de la {rest}"


#: Kern (1950) suele admitir unos 10 psi (69 kPa) por corriente en los líquidos.
DP_LIQUID_KERN_PA = 10.0 * 6894.757293168361


def annulus_nu_laminar(ratio: float) -> float:
    """Nu laminar de la pared interior de un anular, con la exterior aislada (Kays y Perkins,
    1973; Incropera, tabla 8.2), interpolado lineal en d_o/D; bajo 0,05, el valor de 0,05."""
    table = ANNULUS_TABLE
    if ratio <= table[0][0]:
        return table[0][1]
    for (r0, n0), (r1, n1) in zip(table, table[1:], strict=False):
        if ratio <= r1:
            return n0 + (n1 - n0) * (ratio - r0) / (r1 - r0)
    return table[-1][1]


def annulus_friction_Re(ratio: float) -> float:
    """f·Re (Darcy) del flujo laminar desarrollado en un anular concéntrico con κ = d_o/D:
    f·Re = 64·(1 − κ)²/[1 + κ² + (1 − κ²)/ln κ] (Shah y London, 1978): 64 con κ → 0 (un
    tubo) y 96 con κ → 1 (dos placas paralelas)."""
    k = ratio
    if k <= 0.0:
        return 64.0
    if k >= 1.0 - 1e-9:
        return 96.0
    return 64.0 * (1.0 - k) ** 2 / (1.0 + k * k + (1.0 - k * k) / math.log(k))


@dataclass(frozen=True)
class AnnulusFlow(TubeFlow):
    """El flujo por el anular, con d_o/D."""

    ratio: float = 0.0


def annulus_flow(
    props: StreamProps,
    m_dot_kg_s: float,
    d_o: float,
    D: float,
    heating: bool,
    mu_wall_Pa_s: float | None,
    correlation: str = "auto",
) -> AnnulusFlow:
    """h y f del anular entre el tubo (d_o) y el caño (D), con D_h = D − d_o (Incropera,
    §8.7): laminar con la tabla de Kays y Perkins y f·Re del anular; turbulento con las
    correlaciones del tubo (Dittus y Boelter, Gnielinski o Sieder y Tate) y Petukhov."""
    if correlation not in ANNULUS_CORRELATIONS:
        raise ValueError(f"Correlación del anular desconocida: {correlation!r}.")
    D_h = D - d_o
    ratio = d_o / D
    A = math.pi * (D * D - d_o * d_o) / 4.0
    u = m_dot_kg_s / (props.rho_kg_per_m3 * A)
    Re = props.rho_kg_per_m3 * u * D_h / props.mu_Pa_s
    Pr = props.Pr
    mu_ratio = props.mu_Pa_s / mu_wall_Pa_s if mu_wall_Pa_s else 1.0
    warnings: list[str] = []
    if Re < RE_LAMINAR_TUBE:
        key = "kays_perkins"
        Nu = annulus_nu_laminar(ratio)
        f = annulus_friction_Re(ratio) / Re
        if ratio < ANNULUS_TABLE[0][0]:
            warnings.append(
                f"d_o/D = {_num(ratio)} está debajo de la tabla de Kays y Perkins (0,05 a 1): "
                "se usa el valor de 0,05."
            )
        if correlation != "auto":
            warnings.append(
                f"Re = {_num(Re)} en el anular: el flujo es laminar y se usa la tabla de Kays "
                "y Perkins en vez de la correlación turbulenta elegida."
            )
    else:
        key = "gnielinski" if correlation == "auto" else correlation
        if key == "dittus_boelter":
            Nu = nu_dittus_boelter(Re, Pr, heating)
        elif key == "gnielinski":
            Nu = nu_gnielinski(max(Re, 3000.0), Pr)
        else:
            Nu = nu_sieder_tate(Re, Pr, mu_ratio)
        f = darcy_f(Re)
        if key != "gnielinski" and Re < RE_TURBULENT_TUBE:
            warnings.append(
                f"Re = {_num(Re)} en el anular: Dittus y Boelter y Sieder y Tate piden "
                "Re ≥ 10⁴; en la transición, Gnielinski es la mejor estimación."
            )
    h = Nu * props.k_W_per_mK / D_h
    return AnnulusFlow(D_h, A, u, Re, regime(Re), key, Nu, h, f, mu_ratio, tuple(warnings), ratio)


# ---------------------------------------------------------------------
# El intercambiador
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DoublePipeInputs:
    """Un doble tubo: las dos corrientes, la geometría y el problema.

    ``tube_side`` es la corriente que va por el tubo interior. Con ``problem="sizing"``,
    ``target`` y ``target_value`` fijan lo que tiene que hacer (una salida o Q̇); con
    ``"rating"``, el equipo tiene ``hairpins`` horquillas de dos ramas de ``leg_length_m``.
    """

    hot: DesignStream
    cold: DesignStream
    d_i_m: float
    d_o_m: float
    D_m: float
    tube_side: Side = "cold"
    k_wall_W_per_mK: float = 45.0
    flow: Flow = "counter"
    problem: Problem = "sizing"
    target: SizingTarget = "T_hot_out"
    target_value: float = 0.0
    hairpins: int = 1
    leg_length_m: float = 6.0
    tube_correlation: str = "auto"
    annulus_correlation: str = "auto"
    T0_K: float = 298.15


@dataclass(frozen=True)
class DoublePipeResult:
    """El doble tubo resuelto."""

    inputs: DoublePipeInputs
    tube_props: StreamProps
    annulus_props: StreamProps
    tube: TubeFlow
    annulus: AnnulusFlow
    T_wall_tube_K: float
    T_wall_annulus_K: float
    u_result: OverallUResult
    u_clean: OverallUResult
    exchanger: ExchangerResult
    L_m: float
    warnings: tuple[str, ...]
    iterations: int

    @property
    def tube_stream(self) -> Side:
        return self.inputs.tube_side

    @property
    def annulus_stream(self) -> Side:
        return "hot" if self.inputs.tube_side == "cold" else "cold"

    @property
    def U_o_W_per_m2K(self) -> float:
        """U referido al área exterior del tubo, π·d_o por metro."""
        return self.u_result.UA_W_per_K / (math.pi * self.inputs.d_o_m)

    @property
    def U_o_clean_W_per_m2K(self) -> float:
        return self.u_clean.UA_W_per_K / (math.pi * self.inputs.d_o_m)

    @property
    def area_m2(self) -> float:
        return math.pi * self.inputs.d_o_m * self.L_m

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
    def hairpins(self) -> int:
        """Las horquillas: las del equipo (verificación) o las que hacen falta."""
        i = self.inputs
        if i.problem == "rating":
            return i.hairpins
        return max(1, math.ceil(self.L_m / (2.0 * i.leg_length_m) - 1e-9))

    @property
    def L_installed_m(self) -> float:
        return 2.0 * self.hairpins * self.inputs.leg_length_m

    @property
    def overdesign(self) -> float:
        """El largo instalado sobre el necesario, menos 1 (dimensionamiento)."""
        return self.L_installed_m / self.L_m - 1.0

    @property
    def dp_tube_Pa(self) -> float:
        """Por fricción en el largo instalado (Kern, 1950, cap. 6: el equipo que se pone)."""
        t, p = self.tube, self.tube_props
        return t.f * (self.L_installed_m / t.D_h_m) * p.rho_kg_per_m3 * t.velocity_m_s**2 / 2.0

    @property
    def dp_annulus_Pa(self) -> float:
        """Por fricción en el largo instalado, como el del tubo."""
        a, p = self.annulus, self.annulus_props
        return a.f * (self.L_installed_m / a.D_h_m) * p.rho_kg_per_m3 * a.velocity_m_s**2 / 2.0

    @property
    def dp_returns_Pa(self) -> float:
        """Las vueltas de las horquillas en el anular, ~una carga de velocidad por horquilla
        (Kern, 1950, cap. 6): una estimación, aparte del Δp por fricción."""
        a, p = self.annulus, self.annulus_props
        return self.hairpins * p.rho_kg_per_m3 * a.velocity_m_s**2 / 2.0


def _validate(i: DoublePipeInputs) -> None:
    check_stream(i.hot, "caliente")
    check_stream(i.cold, "frío")
    if i.hot.T_in_K <= i.cold.T_in_K:
        raise ValueError(
            f"El fluido caliente entra a {_degC(i.hot.T_in_K)} y el frío a "
            f"{_degC(i.cold.T_in_K)}: el caliente tiene que entrar más caliente que el frío."
        )
    if i.tube_side not in ("hot", "cold"):
        raise ValueError(f"Lado desconocido: {i.tube_side!r}.")
    if i.flow not in FLOWS:
        raise ValueError(f"Arreglo desconocido: {i.flow!r}.")
    if i.problem not in PROBLEMS:
        raise ValueError(f"Problema desconocido: {i.problem!r}.")
    if not i.d_i_m > 0.0:
        raise ValueError("El diámetro interior del tubo tiene que ser positivo.")
    if i.d_o_m < i.d_i_m:
        raise ValueError(
            "El diámetro exterior del tubo no puede ser menor que el interior (con los dos "
            "iguales la pared es delgada y no cuenta)."
        )
    if i.D_m <= i.d_o_m:
        raise ValueError(
            f"El caño exterior (D = {_num(i.D_m * 1e3)} mm) tiene que ser más grande que el "
            f"tubo (d_o = {_num(i.d_o_m * 1e3)} mm): si no, no queda anular."
        )
    if i.d_o_m > i.d_i_m and not i.k_wall_W_per_mK > 0.0:
        raise ValueError("La conductividad de la pared tiene que ser positiva.")
    if i.tube_correlation not in TUBE_SIDE_CORRELATIONS:
        raise ValueError(f"Correlación de tubo desconocida: {i.tube_correlation!r}.")
    if not i.leg_length_m > 0.0:
        raise ValueError("El largo de cada rama de la horquilla tiene que ser positivo.")
    if i.problem == "rating" and i.hairpins < 1:
        raise ValueError("El equipo tiene que tener al menos una horquilla.")


def _sides(i: DoublePipeInputs) -> tuple[DesignStream, DesignStream]:
    """(la corriente del tubo, la del anular)."""
    return (i.cold, i.hot) if i.tube_side == "cold" else (i.hot, i.cold)


def solve_double_pipe(inputs: DoublePipeInputs) -> DoublePipeResult:
    """El doble tubo: los h de cada lado, el U y el largo o las salidas (Incropera, §11.3;
    Çengel y Ghajar, §11-2 y §11-5).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno (geometría, salidas imposibles, cambio de fase).
    """
    i = inputs
    _validate(i)
    tube_s, ann_s = _sides(i)
    tube_label, ann_label = (
        _SIDE_ES[i.tube_side],
        _SIDE_ES["hot" if i.tube_side == "cold" else "cold"],
    )
    heating_tube = i.tube_side == "cold"
    arrangement = Arrangement(i.flow)
    T_h_out = (
        i.target_value if (i.problem == "sizing" and i.target == "T_hot_out") else i.hot.T_in_K
    )
    T_c_out = (
        i.target_value if (i.problem == "sizing" and i.target == "T_cold_out") else i.cold.T_in_K
    )
    L = 2.0 * i.hairpins * i.leg_length_m if i.problem == "rating" else 1.0
    T_w_tube = T_w_ann = 0.5 * (i.hot.T_in_K + i.cold.T_in_K)
    result: ExchangerResult | None = None
    iterations = 0
    for iterations in range(1, 201):  # noqa: B007 — se informa cuántas hicieron falta
        T_h_mean = 0.5 * (i.hot.T_in_K + T_h_out)
        T_c_mean = 0.5 * (i.cold.T_in_K + T_c_out)
        props_h = stream_props(i.hot, T_h_mean, "caliente")
        props_c = stream_props(i.cold, T_c_mean, "frío")
        props_t, props_a = (props_c, props_h) if heating_tube else (props_h, props_c)
        tube = tube_flow(
            props_t,
            tube_s.m_dot_kg_s,
            i.d_i_m,
            L,
            heating_tube,
            wall_viscosity(tube_s, T_w_tube),
            i.tube_correlation,
        )
        ann = annulus_flow(
            props_a,
            ann_s.m_dot_kg_s,
            i.d_o_m,
            i.D_m,
            not heating_tube,
            wall_viscosity(ann_s, T_w_ann),
            i.annulus_correlation,
        )
        u = overall_u_per_length(
            tube.h_W_per_m2K,
            ann.h_W_per_m2K,
            i.d_i_m,
            i.d_o_m,
            i.k_wall_W_per_mK,
            tube_s.R_f_m2K_per_W,
            ann_s.R_f_m2K_per_W,
            "adentro del tubo",
            "en el anular",
        )
        U_o = u.UA_W_per_K / (math.pi * i.d_o_m)
        hot = Stream(stream_name(i.hot), i.hot.T_in_K, i.hot.m_dot_kg_s, props_h.cp_J_per_kgK)
        cold = Stream(stream_name(i.cold), i.cold.T_in_K, i.cold.m_dot_kg_s, props_c.cp_J_per_kgK)
        if i.problem == "sizing":
            result = solve_sizing(
                SizingInputs(arrangement, hot, cold, U_o, i.target, i.target_value, T0_K=i.T0_K)
            )
            L_new = result.A_m2 / (math.pi * i.d_o_m)
        else:
            result = solve_rating(
                RatingInputs(arrangement, hot, cold, U_o, math.pi * i.d_o_m * L, T0_K=i.T0_K)
            )
            L_new = L
        # la pared: T_tubo + (T_anular − T_tubo)·R_lado/R_total, de cada lado
        T_t = T_c_mean if heating_tube else T_h_mean
        T_a = T_h_mean if heating_tube else T_c_mean
        R_total = u.R_total_K_per_W
        new_w_tube = T_t + (T_a - T_t) * (1.0 / (tube.h_W_per_m2K * u.A_i_m2)) / R_total
        new_w_ann = T_a - (T_a - T_t) * (1.0 / (ann.h_W_per_m2K * u.A_o_m2)) / R_total
        new_h, new_c = result.hot.T_out_K, result.cold.T_out_K
        done = (
            abs(new_h - T_h_out) < 1e-9
            and abs(new_c - T_c_out) < 1e-9
            and abs(new_w_tube - T_w_tube) < 1e-6
            and abs(new_w_ann - T_w_ann) < 1e-6
            and abs(L_new - L) <= 1e-9 * L_new
        )
        T_h_out, T_c_out, T_w_tube, T_w_ann, L = new_h, new_c, new_w_tube, new_w_ann, L_new
        if done:
            break
    else:  # pragma: no cover — converge en pocas vueltas
        raise ValueError("Las temperaturas de salida no convergieron.")
    assert result is not None
    check_single_phase(i.hot, T_h_out, "caliente")
    check_single_phase(i.cold, T_c_out, "frío")
    u_clean = overall_u_per_length(
        tube.h_W_per_m2K,
        ann.h_W_per_m2K,
        i.d_i_m,
        i.d_o_m,
        i.k_wall_W_per_mK,
        0.0,
        0.0,
        "adentro del tubo",
        "en el anular",
    )
    warnings = [f"Tubo ({tube_label}): {w}" for w in tube.warnings]
    warnings += [f"Anular ({ann_label}): {w}" for w in ann.warnings]
    warnings += list(result.warnings)
    return DoublePipeResult(
        inputs=i,
        tube_props=props_t,
        annulus_props=props_a,
        tube=tube,
        annulus=ann,
        T_wall_tube_K=T_w_tube,
        T_wall_annulus_K=T_w_ann,
        u_result=u,
        u_clean=u_clean,
        exchanger=result,
        L_m=L,
        warnings=tuple(warnings),
        iterations=iterations,
    )


def swap_sides(inputs: DoublePipeInputs) -> DoublePipeResult | None:
    """El mismo problema con las corrientes al revés (la del tubo por el anular), o None si
    así no se puede resolver."""
    other: Side = "hot" if inputs.tube_side == "cold" else "cold"
    tube_corr = inputs.annulus_correlation if inputs.annulus_correlation != "auto" else "auto"
    ann_corr = (
        inputs.tube_correlation if inputs.tube_correlation in ANNULUS_CORRELATIONS else "auto"
    )
    try:
        return solve_double_pipe(
            replace(
                inputs, tube_side=other, tube_correlation=tube_corr, annulus_correlation=ann_corr
            )
        )
    except ValueError:
        return None


def double_pipe_notes(r: DoublePipeResult) -> list[str]:
    """Interpretación física del resultado (markdown)."""
    i = r.inputs
    notes: list[str] = []
    rows = r.u_result.resistances
    top = max(rows, key=lambda x: x.R_K_per_W)
    share = 100.0 * r.u_result.share(top)
    notes.append(
        f"La resistencia mayor es {_which(top.label, top.kind)} ({_num(share, 2)} % del "
        "total): para mejorar el U hay que atacar ese lado."
    )
    if r.annulus.correlation == "kays_perkins" or r.tube.regime == "laminar":
        notes.append(
            "Con flujo laminar h es chico: el Nu de un anular o de un tubo laminar no crece con "
            "la velocidad. Achicar el anular o pasar el fluido viscoso al otro lado ayuda."
        )
    if r.u_result.R_total_K_per_W > r.u_clean.R_total_K_per_W * (1 + 1e-9):
        extra = 100.0 * (r.u_result.R_total_K_per_W / r.u_clean.R_total_K_per_W - 1.0)
        notes.append(
            f"El ensuciamiento sube la resistencia un {_fmt(extra)} %: limpio, U sería "
            f"{_fmt(r.U_o_clean_W_per_m2K)} W/(m²·K) en vez de {_fmt(r.U_o_W_per_m2K)}."
        )
    for where, dp in (("el tubo", r.dp_tube_Pa), ("el anular", r.dp_annulus_Pa)):
        if dp > DP_LIQUID_KERN_PA:
            notes.append(
                f"La caída de presión en {where} es de {_fmt(dp / 1e3)} kPa: más que los "
                "10 psi (69 kPa) que Kern suele admitir por corriente en los líquidos. "
                "Conviene más caños en paralelo o un caño más grande."
            )
    if i.problem == "sizing":
        notes.append(
            f"Hacen falta {_num(r.L_m)} m de tubo: {r.hairpins} horquillas de dos ramas de "
            f"{_num(i.leg_length_m)} m dan {_num(r.L_installed_m)} m "
            f"({_num(100.0 * r.overdesign, 2)} % de sobrediseño)."
        )
    return notes


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DoublePipeExample:
    description: str
    inputs: DoublePipeInputs


_C = 273.15

#: Aceite de motor a 80 °C (Incropera, tabla A.5, a 353 K; ρ interpolada).
_OIL_INCROPERA = FluidProperties(852.1, 2131.0, 3.25e-2, 0.138, name="aceite de motor")
#: Aceite de motor a 80 °C (Çengel y Ghajar, tabla A-13: ν = 3,794·10⁻⁵ m²/s).
_OIL_CENGEL = FluidProperties(852.0, 2131.0, 852.0 * 3.794e-5, 0.138, name="aceite de motor")

DOUBLE_PIPE_EXAMPLES: dict[str, DoublePipeExample] = {
    "Enfriador de aceite de una turbina de gas (Incropera, ejemplo 11.1)": DoublePipeExample(
        "Aceite de motor (0,1 kg/s) se enfría de 100 a 60 °C con agua (0,2 kg/s, entra a "
        "30 °C) que va por un tubo de 25 mm de pared delgada, en contracorriente, dentro de un "
        "caño de 45 mm. El libro usa Dittus y Boelter en el tubo y la tabla del anular: "
        "h_i = 2250 W/(m²·K), h_o = 38,8 W/(m²·K), U = 38,1 W/(m²·K) y L = 65,9 m. Las "
        "propiedades del aceite son las de la tabla A.5 a 80 °C.",
        DoublePipeInputs(
            hot=DesignStream(100.0 + _C, 0.1, properties=_OIL_INCROPERA),
            cold=DesignStream(30.0 + _C, 0.2, fluid="Water", p_Pa=2.0e5),
            d_i_m=0.025,
            d_o_m=0.025,
            D_m=0.045,
            tube_side="cold",
            flow="counter",
            problem="sizing",
            target="T_hot_out",
            target_value=60.0 + _C,
            leg_length_m=6.0,
            tube_correlation="dittus_boelter",
        ),
    ),
    "U de un doble tubo con agua y aceite (Çengel y Ghajar, ejemplo 11-1)": DoublePipeExample(
        "Agua (0,5 kg/s) por un tubo de cobre de 2 cm de pared delgada y aceite caliente "
        "(0,8 kg/s) por el anular hasta un caño de 3 cm, en contracorriente. El libro da las "
        "temperaturas medias, 45 °C el agua y 80 °C el aceite: acá entra el agua a 40 °C y "
        "sale a 50 °C, y el aceite entra a 86,1 °C, para que las medias coincidan. "
        "El libro da U ≈ 74,5 W/(m²·K).",
        DoublePipeInputs(
            hot=DesignStream(86.1 + _C, 0.8, properties=_OIL_CENGEL),
            cold=DesignStream(40.0 + _C, 0.5, fluid="Water", p_Pa=2.0e5),
            d_i_m=0.02,
            d_o_m=0.02,
            D_m=0.03,
            tube_side="cold",
            flow="counter",
            problem="sizing",
            target="T_cold_out",
            target_value=50.0 + _C,
            leg_length_m=3.0,
            tube_correlation="dittus_boelter",
        ),
    ),
    "Calentador de agua de proceso (caños de acero, con ensuciamiento)": DoublePipeExample(
        "Agua caliente (1,2 kg/s a 90 °C) por el anular calienta agua de proceso (1 kg/s de "
        '20 a 50 °C) que va por un caño de acero de 1" (26,6 mm por dentro y 33,4 mm por '
        'fuera) dentro de uno de 2" (52,5 mm), con 0,0002 m²·K/W de ensuciamiento de cada '
        "lado. Horquillas de dos ramas de 6 m.",
        DoublePipeInputs(
            hot=DesignStream(90.0 + _C, 1.2, fluid="Water", p_Pa=3.0e5, R_f_m2K_per_W=0.0002),
            cold=DesignStream(20.0 + _C, 1.0, fluid="Water", p_Pa=3.0e5, R_f_m2K_per_W=0.0002),
            d_i_m=0.0266,
            d_o_m=0.0334,
            D_m=0.0525,
            tube_side="cold",
            k_wall_W_per_mK=45.0,
            flow="counter",
            problem="sizing",
            target="T_cold_out",
            target_value=50.0 + _C,
            leg_length_m=6.0,
        ),
    ),
}


def _q(value: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value, kind, system), "unidad": unit_label(kind, system)}


def _side_dict(flow: TubeFlow, props: StreamProps, dp: float, system: UnitSystem) -> dict[str, Any]:
    return {
        "T_media": _q(props.T_K, "temperature", system),
        "rho": _q(props.rho_kg_per_m3, "density", system),
        "cp": _q(props.cp_J_per_kgK, "specific_heat", system),
        "mu": _q(props.mu_Pa_s, "dynamic_viscosity", system),
        "k": _q(props.k_W_per_mK, "thermal_conductivity", system),
        "Pr": props.Pr,
        "D_h": _q(flow.D_h_m, "small_length", system),
        "velocidad": _q(flow.velocity_m_s, "speed", system),
        "Re": flow.Re,
        "regimen": flow.regime,
        "correlacion": flow.correlation,
        "Nu": flow.Nu,
        "h": _q(flow.h_W_per_m2K, "heat_transfer_coefficient", system),
        "f": flow.f,
        "Delta_p": _q(dp, "pressure_loss", system),
    }


def double_pipe_to_dict(r: DoublePipeResult, system: UnitSystem) -> dict[str, Any]:
    """El doble tubo resuelto, serializable a JSON y CSV."""
    i = r.inputs
    tube_s, ann_s = _sides(i)
    return {
        "problema": PROBLEMS[i.problem],
        "arreglo": FLOWS[i.flow],
        "por_el_tubo": f"{stream_name(tube_s)} ({_SIDE_ES[r.tube_stream]})",
        "por_el_anular": f"{stream_name(ann_s)} ({_SIDE_ES[r.annulus_stream]})",
        "d_i": _q(i.d_i_m, "small_length", system),
        "d_o": _q(i.d_o_m, "small_length", system),
        "D": _q(i.D_m, "small_length", system),
        "k_pared": _q(i.k_wall_W_per_mK, "thermal_conductivity", system),
        "tubo": _side_dict(r.tube, r.tube_props, r.dp_tube_Pa, system),
        "anular": _side_dict(r.annulus, r.annulus_props, r.dp_annulus_Pa, system),
        "U_o": _q(r.U_o_W_per_m2K, "heat_transfer_coefficient", system),
        "U_o_limpio": _q(r.U_o_clean_W_per_m2K, "heat_transfer_coefficient", system),
        "Q": _q(r.Q_W, "heat_rate", system),
        "T_caliente_salida": _q(r.T_hot_out_K, "temperature", system),
        "T_frio_salida": _q(r.T_cold_out_K, "temperature", system),
        "Delta_T_ml": _q(r.exchanger.dT_lm_K, "temperature_difference", system),
        "efectividad": r.exchanger.effectiveness,
        "NTU": r.exchanger.NTU,
        "area": _q(r.area_m2, "area", system),
        "largo": _q(r.L_m, "length", system),
        "horquillas": r.hairpins,
        "largo_instalado": _q(r.L_installed_m, "length", system),
        "Delta_p_vueltas_anular": _q(r.dp_returns_Pa, "pressure_loss", system),
        "avisos": list(r.warnings),
        "notas": double_pipe_notes(r),
    }

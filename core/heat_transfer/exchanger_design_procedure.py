"""Procedimiento didáctico del diseño de intercambiadores (LaTeX), en el sistema activo.

- :func:`tube_bank_steps`: las propiedades, V_máx, Re, el Nu de Zukauskas con la corrección
  por filas, h, la salida, el calor y la caída de presión con f y χ del gráfico
  (Çengel y Ghajar, §7-4; Incropera, §7.6);
- :func:`double_pipe_steps`: las propiedades, el lado del tubo y el del anular, el U por
  metro con el ensuciamiento, ε-NTU y la LMTD (los pasos de la Fase 8.2), el largo y las
  horquillas y las caídas de presión (Incropera, §8.7 y §11.3);
- :func:`shell_tube_steps`: la geometría, el lado del casco con Kern, el de los tubos,
  U_C y U_D, ε-NTU, el criterio de R_d de Kern y las caídas de presión (Kern, 1950;
  Kakaç y Liu, 2002). Con Bell–Delaware (Fase 12.2; Taborek, 1983), el casco va en cuatro
  pasos (la geometría, las fugas y el bypass, el banco ideal y los factores) y su Δp, por
  zonas.

Factores de unidades a la vista: ṁ·c_p lleva 1000 en el Técnico (c_p en kJ/(kg·K)) y
3600 en el Inglés (para pasar Btu/s a Btu/h, la hora de los h); ρ·u²/2 lleva g_c = 32,174
en el Inglés (para dar lbf/ft²), y la caída de presión se pasa después a kPa o psi.
"""

from __future__ import annotations

import math
from dataclasses import replace

from core.heat_transfer.bell_delaware import IDEAL_BANK, BellDelawareSide
from core.heat_transfer.convection_procedure import big, correlation_latex
from core.heat_transfer.design_common import DesignStream, StreamProps, TubeFlow, stream_name
from core.heat_transfer.double_pipe import ANNULUS_TABLE, DoublePipeResult, annulus_friction_Re
from core.heat_transfer.exchangers import ExchangerResult
from core.heat_transfer.exchangers_procedure import _capacity_units_note, effectiveness_steps
from core.heat_transfer.procedure_common import (
    Pack,
    chain,
    dimensionless,
    flush,
    frac,
    n,
    numbered,
    prod,
    q,
    sub,
    times,
)
from core.heat_transfer.shell_tube import ShellTubeResult
from core.heat_transfer.tube_bank import TubeBankResult
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = ["double_pipe_steps", "shell_tube_steps", "tube_bank_steps"]

_T = "temperature"
_DT = "temperature_difference"
_RHO = "density"
_MU = "dynamic_viscosity"
_CP = "specific_heat"
_K = "thermal_conductivity"
_L = "length"
_A = "area"
_V = "speed"
_W = "heat_rate"
_H = "heat_transfer_coefficient"
_M = "mass_flow"
_G = "mass_flux"
_RF = "fouling_resistance"
_PD = "pressure_drop"
_PL = "pressure_loss"
_R = "thermal_resistance"

#: El factor de ṁ·c_p (vacío en el SI): kW/K → W/K en el Técnico, Btu/s → Btu/h en el Inglés.
_FACTOR: dict[str, str] = {"SI": "", "Técnico": "1000", "Inglés": "3600"}
_GC = "32.174"
_VMAX = r"V_{\text{máx}}"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _d(x: float, sig: int = 4) -> str:
    return dimensionless(x, sig)


def _pw(base: str, exp: str) -> str:
    """``base^{exp}`` con paréntesis si la base lleva ×10ⁿ, es negativa o es una fracción."""
    if r"\times" in base or base.startswith(("-", r"\frac")) or "/" in base:
        return rf"\left({base}\right)^{{{exp}}}"
    return rf"{base}^{{{exp}}}"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _txt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 4) -> str:
    """Una cantidad para el texto, sin la «e» de Python: «25,6 °C», «18 659 Pa», «4,2·10⁻⁵ psi»."""
    value = convert_from_si(value_si, kind, system)
    if 1e3 <= abs(value) < 1e9:
        number = f"{value:,.0f}".replace(",", " ")
    elif value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e9):
        mantissa, exponent = f"{value:.{min(sig, 3) - 1}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".").replace(".", ",")
        number = f"{mantissa}·10{str(int(exponent)).translate(_SUPERSCRIPT)}"
    else:
        number = f"{value:.{sig}g}".replace(".", ",")
    return f"{number} {unit_label(kind, system)}"


def _comma(x: float, sig: int = 4) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _minus(x: float, sig: int = 4) -> str:
    """Como :func:`_comma`, con el signo menos tipográfico (−0,395)."""
    return _comma(x, sig).replace("-", "−")


def _count(x: float) -> str:
    """Un Re de tabla para el texto: «1000», «200 000» (espacio desde 10⁴, como la RAE)."""
    value = round(x)
    return str(value) if value < 10_000 else f"{value:,}".replace(",", " ")


def _props(rows: list[tuple[str, str]]) -> str:
    """Un bloque ``aligned`` con una propiedad por renglón."""
    return r"\begin{aligned}" + r" \\ ".join(rf"{a} &= {b}" for a, b in rows) + r"\end{aligned}"


def _mcp(system: UnitSystem, m_dot: float, cp: float) -> list[str]:
    """Los factores de ṁ·c_p, con el de unidades adelante."""
    f = _FACTOR[system]
    factors = [n(m_dot, _M, system), n(cp, _CP, system)]
    return [f, *factors] if f else factors


def _mcp_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " Con c_p en kJ/(kg·K), el 1000 pasa ṁ·c_p a W/K."
    if system == "Inglés":
        return " Con ṁ en lb/s, el 3600 pasa ṁ·c_p a Btu/(h·°F), la hora de h."
    return ""


def _dyn_sym(system: UnitSystem, rho: str = r"\rho", u: str = "u") -> str:
    return (
        rf"\frac{{{rho}\,{u}^2}}{{2\,g_c}}"
        if system == "Inglés"
        else rf"\frac{{{rho}\,{u}^2}}{{2}}"
    )


def _dyn_num(system: UnitSystem, rho: float, u: float) -> str:
    num = times(n(rho, _RHO, system), _pw(n(u, _V, system), "2"))
    return frac(num, rf"2 \cdot {_GC}" if system == "Inglés" else "2")


def _dp_text(system: UnitSystem) -> str:
    if system == "Inglés":
        return (
            " En el Inglés, g_c = 32,174 lbm·ft/(lbf·s²) deja ρ·u²/2 en lbf/ft², y 144 in² por "
            "ft² lo pasan a psi."
        )
    if system == "Técnico":
        return " La cuenta da Pa; al final, en kPa."
    return ""


def _to_loss(dp: float, system: UnitSystem) -> list[str]:
    """Las unidades finales: Pa en el SI; además kPa o psi en el Técnico y el Inglés."""
    out = [q(dp, _PD, system)]
    if system != "SI":
        out.append(q(dp, _PL, system))
    return out


def _stream_props_block(props: StreamProps, system: UnitSystem, given: bool) -> list[str]:
    rows = [
        (r"\rho", q(props.rho_kg_per_m3, _RHO, system)),
        (r"c_p", q(props.cp_J_per_kgK, _CP, system)),
        (r"\mu", q(props.mu_Pa_s, _MU, system)),
        (r"k", q(props.k_W_per_mK, _K, system)),
        (r"\mathrm{Pr}", _d(props.Pr)),
    ]
    head = [] if given else [(r"T_m", q(props.T_K, _T, system))]
    return [_props(head + rows[:2]), _props(rows[2:])]


def _effectiveness(
    ex: ExchangerResult, hot: str, cold: str, system: UnitSystem
) -> list[ProcedureStep]:
    """ε-NTU y la LMTD de la Fase 8.2, con el c_p de cada corriente del paso «Propiedades»."""
    who = f"caliente ({hot})" if ex.cmin_side == "hot" else f"frío ({cold})"
    text = (
        "El c_p de cada corriente es el del paso «Propiedades», a su temperatura media."
        + _capacity_units_note(system)
        + f" El de menor capacidad (C_mín) es el fluido {who}."
    )
    return [
        replace(s, text=text) if s.title == "Capacidades caloríficas" else s
        for s in effectiveness_steps(ex, system)
    ]


def _correlation_dict(
    flow: TubeFlow, props: StreamProps, D: float, L: float, heating: bool
) -> dict[str, float]:
    Re = max(flow.Re, 3000.0) if flow.correlation == "gnielinski" else flow.Re
    return {
        "Re": Re,
        "Pr": props.Pr,
        "heating": 1.0 if heating else 0.0,
        "mu_ratio": flow.mu_ratio,
        "D_over_L": D / L,
        "constant_T": 1.0,
    }


_CORR_NAMES = {
    "tube_laminar": "laminar desarrollada (Nu = 3,66)",
    "hausen": "Hausen (laminar con entrada térmica)",
    "dittus_boelter": "Dittus y Boelter",
    "gnielinski": "Gnielinski",
    "sieder_tate": "Sieder y Tate (con μ/μ_p)",
    "kays_perkins": "la tabla de Kays y Perkins (laminar)",
}


def _friction_lines(sym: str, flow: TubeFlow, laminar_fRe: float | None = None) -> list[str]:
    """f de Darcy: 64/Re (o f·Re del anular) en laminar, Petukhov en turbulento."""
    Re = flow.Re
    if laminar_fRe is not None or Re < 2300.0:
        fRe = laminar_fRe if laminar_fRe is not None else 64.0
        return [chain(sym, frac(_d(fRe), "Re"), frac(_d(fRe), big(Re)), _d(flow.f))]
    Re_f = max(Re, 3000.0)
    return [
        chain(
            sym,
            r"(0.790\,\ln Re - 1.64)^{-2}",
            rf"(0.790\,\ln {big(Re_f)} - 1.64)^{{-2}}",
            _d(flow.f),
        )
    ]


# ---------------------------------------------------------------------
# Banco de tubos (§7-4)
# ---------------------------------------------------------------------


def _bank_properties(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    given = i.properties is not None
    rows_a = [
        (r"\rho", q(r.rho_kg_per_m3, _RHO, system)),
        (r"\mu", q(r.mu_Pa_s, _MU, system)),
        (r"k", q(r.k_W_per_mK, _K, system)),
    ]
    rows_b = [
        (r"c_p", q(r.cp_J_per_kgK, _CP, system)),
        (r"\mathrm{Pr}", _d(r.Pr)),
        (r"\mathrm{Pr}_s", _d(r.Pr_s)),
        (r"\rho_{\text{ent}}", q(r.rho_in_kg_per_m3, _RHO, system)),
    ]
    head = [] if given else [(r"T_m", q(r.T_mean_K, _T, system))]
    latex = (_props(head + rows_a), _props(rows_b))
    if given:
        text = "Las propiedades son las del enunciado (no se iteran)."
    else:
        text = (
            "El fluido a la temperatura media T_m = (T_ent + T_sal)/2, que se itera porque la "
            "salida depende de h; Pr_s a la temperatura de los tubos y ρ_ent a la entrada, "
            "para el caudal (CoolProp)."
        )
    return ProcedureStep("Propiedades", text, latex)


def _bank_velocity(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    ST, SL, D, V = (
        n(i.S_T_m, _L, system),
        n(i.S_L_m, _L, system),
        n(i.D_m, _L, system),
        n(i.V_m_s, _V, system),
    )
    latex: list[str] = []
    if i.arrangement == "aligned":
        text = "Alineado: el paso más angosto es el transversal, S_T − D."
    else:
        assert r.S_D_m is not None
        latex.append(
            chain(
                "S_D",
                r"\sqrt{S_L^2 + (S_T/2)^2}",
                rf"\sqrt{{{_pw(SL, '2')} + ({ST}/2)^2}}",
                q(r.S_D_m, _L, system),
            )
        )
        latex.append(
            chain(
                r"\frac{S_T + D}{2}",
                frac(f"{ST} + {D}", "2"),
                q((i.S_T_m + i.D_m) / 2.0, _L, system),
            )
        )
        text = "Tresbolillo: el fluido pasa por la diagonal si S_D < (S_T + D)/2 " + (
            "(acá sí)." if r.diagonal else "(acá no: pasa por el plano transversal)."
        )
    if r.diagonal:
        latex.append(
            chain(
                r"V_{\text{máx}}",
                r"\frac{S_T}{2\,(S_D - D)}\,V",
                prod(frac(ST, rf"2 \cdot ({n(r.S_D_m or 0.0, _L, system)} - {D})"), V),
                q(r.V_max_m_s, _V, system),
            )
        )
    else:
        latex.append(
            chain(
                r"V_{\text{máx}}",
                r"\frac{S_T}{S_T - D}\,V",
                prod(frac(ST, f"{ST} - {D}"), V),
                q(r.V_max_m_s, _V, system),
            )
        )
    latex.append(
        chain(
            "Re",
            r"\frac{\rho\,V_{\text{máx}}\,D}{\mu}",
            frac(
                times(n(r.rho_kg_per_m3, _RHO, system), n(r.V_max_m_s, _V, system), D),
                n(r.mu_Pa_s, _MU, system),
            ),
            big(r.Re),
        )
    )
    return ProcedureStep("Velocidad máxima y Reynolds", text, tuple(latex))


def _bank_nusselt(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i, fit = r.inputs, r.fit
    cengel = i.source == "cengel"
    F_sym = "F" if cengel else "C_2"
    rows_lim = 16 if cengel else 20
    m, n_exp = _d(fit.m, 3), _d(fit.n, 3)
    latex: list[str] = []
    if fit.p:
        ratio = i.S_T_m / i.S_L_m
        latex.append(
            chain(
                "C",
                rf"{_d(fit.C, 3)}\,(S_T/S_L)^{{{_d(fit.p, 3)}}}",
                times(_d(fit.C, 3), _pw(_d(ratio), _d(fit.p, 3))),
                _d(r.C),
            )
        )
    latex.append(
        chain(
            r"\mathrm{Nu}_D",
            rf"C\,Re^{{{m}}}\,Pr^{{{n_exp}}}\left(\frac{{Pr}}{{Pr_s}}\right)^{{1/4}}",
            prod(
                _d(r.C),
                _pw(big(r.Re), m),
                _pw(_d(r.Pr), n_exp),
                _pw(frac(_d(r.Pr), _d(r.Pr_s)), "1/4"),
            ),
            _d(r.Nu_inf),
        )
    )
    latex.append(rf"{F_sym} = {_d(r.F, 3)} \quad (N_L = {i.N_L})")
    latex.append(
        chain(
            r"\mathrm{Nu}_{D,N_L}",
            rf"{F_sym}\,\mathrm{{Nu}}_D",
            times(_d(r.F, 3), _d(r.Nu_inf)),
            _d(r.Nu),
        )
    )
    latex.append(
        chain(
            "h",
            r"\frac{\mathrm{Nu}_{D,N_L}\,k}{D}",
            frac(times(_d(r.Nu), n(r.k_W_per_mK, _K, system)), n(i.D_m, _L, system)),
            q(r.h_W_per_m2K, _H, system),
        )
    )
    table = "tabla 7-2" if cengel else "tabla 7.5"
    rows_table = "tabla 7-3" if cengel else "tabla 7.6"
    if fit.single_cylinder:
        which = (
            "Entre Re = 10² y 10³ Incropera aproxima cada tubo como un cilindro aislado "
            "(Zukauskas, tabla 7.4)."
        )
    else:
        which = f"El tramo de Re de {_count(fit.Re_low)} a {_count(fit.Re_high)} de la {table}."
    text = (
        f"Zukauskas en la versión de {'Çengel y Ghajar' if cengel else 'Incropera'} ({table}), "
        f"para {rows_lim} filas o más. {which} Con N_L = {i.N_L} filas, la corrección "
        f"{F_sym} de la {rows_table} (interpolada entre las filas de la tabla)."
    )
    return ProcedureStep("Nusselt (Zukauskas)", text, tuple(latex))


def _bank_outlet(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    D, L = n(i.D_m, _L, system), n(i.L_m, _L, system)
    NTU = r.h_W_per_m2K * r.area_m2 / (r.m_dot_kg_s * r.cp_J_per_kgK)
    dT_in = i.T_s_K - i.T_in_K
    latex = [
        chain(
            r"\dot m",
            r"\rho_{\text{ent}}\,V\,N_T\,S_T\,L",
            prod(
                n(r.rho_in_kg_per_m3, _RHO, system),
                n(i.V_m_s, _V, system),
                str(i.N_T),
                n(i.S_T_m, _L, system),
                L,
            ),
            q(r.m_dot_kg_s, _M, system),
        ),
        chain(
            "A_s",
            r"N_L\,N_T\,\pi\,D\,L",
            prod(str(i.N_L), str(i.N_T), r"\pi", D, L),
            q(r.area_m2, _A, system),
        ),
        chain(
            "NTU",
            frac(r"h\,A_s", r"\dot m\,c_p"),
            frac(
                times(n(r.h_W_per_m2K, _H, system), n(r.area_m2, _A, system)),
                times(*_mcp(system, r.m_dot_kg_s, r.cp_J_per_kgK)),
            ),
            _d(NTU),
        ),
        chain(
            r"T_{\text{sal}}",
            r"T_s - (T_s - T_{\text{ent}})\,e^{-NTU}",
            rf"{n(i.T_s_K, _T, system)} - {n(dT_in, _DT, system)} \cdot e^{{-{_d(NTU)}}}",
            q(r.T_out_K, _T, system),
        ),
    ]
    text = (
        "El caudal entra por el frente del banco, N_T·S_T·L, a la velocidad V; con los tubos "
        "a T_s, la diferencia T_s − T cae exponencialmente a lo largo del banco "
        "(Çengel y Ghajar, §7-4)." + _mcp_text(system)
    )
    return ProcedureStep("Temperatura de salida", text, tuple(latex))


def _bank_heat(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    d_in, d_out = i.T_s_K - i.T_in_K, i.T_s_K - r.T_out_K
    latex = [
        chain(
            r"\Delta T_{\text{ent}}",
            r"T_s - T_{\text{ent}}",
            sub(n(i.T_s_K, _T, system), n(i.T_in_K, _T, system)),
            q(d_in, _DT, system),
        ),
        chain(
            r"\Delta T_{\text{sal}}",
            r"T_s - T_{\text{sal}}",
            sub(n(i.T_s_K, _T, system), n(r.T_out_K, _T, system)),
            q(d_out, _DT, system),
        ),
        chain(
            r"\Delta T_{ml}",
            frac(
                r"\Delta T_{\text{ent}} - \Delta T_{\text{sal}}",
                r"\ln(\Delta T_{\text{ent}}/\Delta T_{\text{sal}})",
            ),
            frac(
                sub(n(d_in, _DT, system), n(d_out, _DT, system)),
                rf"\ln({n(d_in, _DT, system)}/{n(d_out, _DT, system)})",
            ),
            q(r.dT_lm_K, _DT, system),
        ),
        chain(
            r"\dot Q",
            r"h\,A_s\,\Delta T_{ml}",
            prod(
                n(r.h_W_per_m2K, _H, system),
                n(r.area_m2, _A, system),
                n(r.dT_lm_K, _DT, system),
            ),
            q(r.Q_W, _W, system),
        ),
    ]
    text = "La diferencia media logarítmica entre los tubos y el fluido, y el calor."
    return ProcedureStep("Calor", text, tuple(latex))


def _bank_pressure(r: TubeBankResult, system: UnitSystem) -> ProcedureStep:
    i, fr = r.inputs, r.friction
    aligned = i.arrangement == "aligned"
    pitch_sym = "P_L" if aligned else "P_T"
    ratio_sym = r"\frac{P_T - 1}{P_L - 1}" if aligned else r"\frac{P_T}{P_L}"
    ratio_num = (
        frac(f"{_d(r.P_T)} - 1", f"{_d(r.P_L)} - 1") if aligned else frac(_d(r.P_T), _d(r.P_L))
    )
    latex = [
        chain("P_T", r"S_T/D", _d(r.P_T)),
        chain("P_L", r"S_L/D", _d(r.P_L)),
        chain(r"\text{eje de }\chi", ratio_sym, ratio_num, _d(fr.ratio)),
        rf"f = {_d(fr.f, 3)} \qquad \chi = {_d(fr.chi, 3)}",
        chain(
            r"\Delta p",
            r"N_L\,\chi\,f\," + _dyn_sym(system, u=_VMAX),
            prod(
                str(i.N_L),
                _d(fr.chi, 3),
                _d(fr.f, 3),
                _dyn_num(system, r.rho_kg_per_m3, r.V_max_m_s),
            ),
            q(r.dp_Pa, _PD, system),
        ),
    ]
    c0, c1 = fr.f_curves
    if c0 != c1:
        f_part = (
            f"f con {pitch_sym} = {_comma(fr.pitch)} entre las curvas de {_comma(c0)} y "
            f"{_comma(c1)}"
        )
    elif math.isclose(fr.pitch, c0, rel_tol=1e-9):
        f_part = f"f en la curva de {pitch_sym} = {_comma(c0)}"
    else:
        f_part = f"f con {pitch_sym} = {_comma(fr.pitch)}, fuera de las curvas: la de {_comma(c0)}"
    r0, r1 = fr.chi_curves
    if r0 != r1:
        chi_part = f"χ entre las curvas de Re = {_count(r0)} y {_count(r1)}"
    else:
        nearest = "" if math.isclose(r.Re, r0, rel_tol=1e-6) else " (la más cercana)"
        chi_part = f"χ en la curva de Re = {_count(r0)}{nearest}"
    reference = (
        "El alineado de referencia es el cuadrado"
        if aligned
        else "El tresbolillo de referencia es el equilátero"
    )
    text = (
        "f y χ de los gráficos de Zukauskas (Incropera, figs. 7.13 y 7.14; Çengel y Ghajar, "
        f"fig. 7-27), en la digitalización de la biblioteca ht: {f_part}, y {chi_part}. "
        f"{reference} (χ = 1)."
        + (" En el Inglés, g_c deja ρ·V²/2 en lbf/ft²." if system == "Inglés" else "")
    )
    return ProcedureStep("Caída de presión", text, tuple(latex))


def tube_bank_steps(r: TubeBankResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del banco de tubos (Çengel y Ghajar, §7-4; Incropera, §7.6)."""
    return numbered(
        [
            _bank_properties(r, system),
            _bank_velocity(r, system),
            _bank_nusselt(r, system),
            _bank_outlet(r, system),
            _bank_heat(r, system),
            _bank_pressure(r, system),
        ]
    )


# ---------------------------------------------------------------------
# Lo común del doble tubo y el casco y tubos
# ---------------------------------------------------------------------


def _streams_step(
    hot: StreamProps,
    cold: StreamProps,
    hot_name: str,
    cold_name: str,
    hot_given: bool,
    cold_given: bool,
    system: UnitSystem,
) -> ProcedureStep:
    latex = [
        rf"\textbf{{Caliente ({hot_name})}}",
        *_stream_props_block(hot, system, hot_given),
        rf"\textbf{{Frío ({cold_name})}}",
        *_stream_props_block(cold, system, cold_given),
    ]
    parts = []
    for name, given in ((hot_name, hot_given), (cold_name, cold_given)):
        parts.append(f"el {name}, del enunciado" if given else f"el {name}, de CoolProp")
    text = (
        "Cada corriente a su temperatura media, (T_ent + T_sal)/2, que se itera porque las "
        f"salidas dependen del U: {parts[0]}; {parts[1]}."
    )
    return ProcedureStep("Propiedades", text, tuple(latex))


def _tube_side_lines(
    flow: TubeFlow,
    props: StreamProps,
    m_dot_per_tube: float,
    d_i: float,
    L: float,
    heating: bool,
    system: UnitSystem,
) -> list[str]:
    d = n(d_i, _L, system)
    latex = [
        chain(
            "u",
            r"\frac{4\,\dot m}{\rho\,\pi\,d_i^2}",
            frac(
                times("4", n(m_dot_per_tube, _M, system)),
                times(n(props.rho_kg_per_m3, _RHO, system), r"\pi", _pw(d, "2")),
            ),
            q(flow.velocity_m_s, _V, system),
        ),
        chain(
            "Re",
            r"\frac{\rho\,u\,d_i}{\mu}",
            frac(
                times(n(props.rho_kg_per_m3, _RHO, system), n(flow.velocity_m_s, _V, system), d),
                n(props.mu_Pa_s, _MU, system),
            ),
            big(flow.Re),
        ),
    ]
    latex += correlation_latex(
        flow.correlation, _correlation_dict(flow, props, d_i, L, heating), flow.Nu
    )
    latex.append(
        chain(
            "h_i",
            r"\frac{\mathrm{Nu}\,k}{d_i}",
            frac(times(_d(flow.Nu), n(props.k_W_per_mK, _K, system)), d),
            q(flow.h_W_per_m2K, _H, system),
        )
    )
    return latex


def _rlen(R_per_m: float, system: UnitSystem) -> str:
    """R' en K·m/W (o h·ft·°F/Btu): una resistencia por metro (o pie) de tubo."""
    # R' [K·m/W] = R [K/W] de un metro; en el Inglés, h·°F/Btu por pie de tubo
    value = convert_from_si(R_per_m, _R, system) * convert_from_si(1.0, _L, system)
    unit = (
        r"\mathrm{h\cdot ft\cdot {}^{\circ}F/Btu}" if system == "Inglés" else r"\mathrm{m\cdot K/W}"
    )
    return rf"{dimensionless(value, 4)}\ {unit}"


def _u_step(
    u,
    d_i: float,
    d_o: float,
    k_wall: float,
    R_fi: float,
    R_fo: float,
    h_i: float,
    h_o: float,
    U_o: float,
    system: UnitSystem,
    inner: str,
    outer: str,
    title: str = "U global",
) -> ProcedureStep:
    di, do = n(d_i, _L, system), n(d_o, _L, system)
    per = "pie" if system == "Inglés" else "metro"
    lines: list[str] = [
        chain(
            r"R'_{\text{conv},i}",
            frac("1", r"h_i\,\pi\,d_i"),
            frac("1", times(n(h_i, _H, system), r"\pi", di)),
            _rlen(1.0 / (h_i * math.pi * d_i), system),
        )
    ]
    if R_fi > 0.0:
        lines.append(
            chain(
                r"R'_{f,i}",
                frac(r"R''_{f,i}", r"\pi\,d_i"),
                frac(n(R_fi, _RF, system), times(r"\pi", di)),
                _rlen(R_fi / (math.pi * d_i), system),
            )
        )
    if d_o > d_i:
        lines.append(
            chain(
                r"R'_{\text{pared}}",
                frac(r"\ln(d_o/d_i)", r"2\pi\,k"),
                frac(rf"\ln({do}/{di})", times(r"2\pi", n(k_wall, _K, system))),
                _rlen(math.log(d_o / d_i) / (2.0 * math.pi * k_wall), system),
            )
        )
    if R_fo > 0.0:
        lines.append(
            chain(
                r"R'_{f,o}",
                frac(r"R''_{f,o}", r"\pi\,d_o"),
                frac(n(R_fo, _RF, system), times(r"\pi", do)),
                _rlen(R_fo / (math.pi * d_o), system),
            )
        )
    lines.append(
        chain(
            r"R'_{\text{conv},o}",
            frac("1", r"h_o\,\pi\,d_o"),
            frac("1", times(n(h_o, _H, system), r"\pi", do)),
            _rlen(1.0 / (h_o * math.pi * d_o), system),
        )
    )
    R_total = u.R_total_K_per_W
    terms = [_rlen(x.R_K_per_W, system).split(r"\ ")[0] for x in u.resistances]
    lines.append(
        chain(
            r"R'",
            Pack(terms[0], tuple(("+", t) for t in terms[1:])),
            _rlen(R_total, system),
        )
    )
    lines.append(
        chain(
            "U_o",
            frac("1", r"R'\,\pi\,d_o"),
            frac("1", times(_rlen(R_total, system).split(r"\ ")[0], r"\pi", do)),
            q(U_o, _H, system),
        )
    )
    parts = [f"la convección {inner}"]
    if R_fi > 0.0:
        parts.append("el ensuciamiento de adentro")
    if d_o > d_i:
        parts.append("la pared")
    if R_fo > 0.0:
        parts.append("el ensuciamiento de afuera")
    parts.append(f"la convección {outer}")
    wall = " La pared es delgada (d_o = d_i) y no cuenta." if d_o <= d_i else ""
    text = (
        f"Las resistencias de un {per} de tubo en serie: {', '.join(parts[:-1])} y "
        f"{parts[-1]} (Çengel y Ghajar, §11-2). U se refiere al área exterior, π·d_o por "
        f"{per}." + wall
    )
    return ProcedureStep(f"{title} (por {per} de tubo)", text, tuple(lines))


# ---------------------------------------------------------------------
# Doble tubo
# ---------------------------------------------------------------------


def _annulus_lines(r: DoublePipeResult, system: UnitSystem) -> list[str]:
    i, a, p = r.inputs, r.annulus, r.annulus_props
    s = r.annulus_stream
    stream = i.hot if s == "hot" else i.cold
    D, do = n(i.D_m, _L, system), n(i.d_o_m, _L, system)
    latex = [
        chain("D_h", "D - d_o", f"{D} - {do}", q(a.D_h_m, _L, system)),
        chain(
            "A_a",
            r"\frac{\pi\,(D^2 - d_o^2)}{4}",
            frac(rf"\pi \cdot ({_pw(D, '2')} - {_pw(do, '2')})", "4"),
            q(a.area_m2, _A, system),
        ),
        chain(
            "u",
            r"\frac{\dot m}{\rho\,A_a}",
            frac(
                n(stream.m_dot_kg_s, _M, system),
                times(n(p.rho_kg_per_m3, _RHO, system), n(a.area_m2, _A, system)),
            ),
            q(a.velocity_m_s, _V, system),
        ),
        chain(
            "Re",
            r"\frac{\rho\,u\,D_h}{\mu}",
            frac(
                times(
                    n(p.rho_kg_per_m3, _RHO, system),
                    n(a.velocity_m_s, _V, system),
                    n(a.D_h_m, _L, system),
                ),
                n(p.mu_Pa_s, _MU, system),
            ),
            big(a.Re),
        ),
    ]
    if a.correlation == "kays_perkins":
        ratio = a.ratio
        table = ANNULUS_TABLE
        lo = max((row for row in table if row[0] <= ratio), default=table[0])
        hi = min((row for row in table if row[0] >= ratio), default=table[-1])
        latex.append(chain(r"\frac{d_o}{D}", frac(do, D), _d(ratio)))
        if lo == hi or ratio <= table[0][0]:
            latex.append(rf"\mathrm{{Nu}}_i = {_d(a.Nu)}")
        else:
            t = (ratio - lo[0]) / (hi[0] - lo[0])
            latex.append(
                chain(
                    "t",
                    frac(f"{_d(ratio)} - {_d(lo[0], 3)}", f"{_d(hi[0], 3)} - {_d(lo[0], 3)}"),
                    _d(t),
                )
            )
            latex.append(
                chain(
                    r"\mathrm{Nu}_i",
                    rf"{_d(lo[1])} + ({_d(hi[1])} - {_d(lo[1])}) \cdot {_d(t)}",
                    _d(a.Nu),
                )
            )
    else:
        latex += correlation_latex(
            a.correlation,
            _correlation_dict(a, p, a.D_h_m, r.L_m, s == "cold"),
            a.Nu,
        )
    latex.append(
        chain(
            "h_o",
            r"\frac{\mathrm{Nu}\,k}{D_h}",
            frac(times(_d(a.Nu), n(p.k_W_per_mK, _K, system)), n(a.D_h_m, _L, system)),
            q(a.h_W_per_m2K, _H, system),
        )
    )
    return latex


def _dp_double_pipe(r: DoublePipeResult, system: UnitSystem) -> ProcedureStep:
    i, t, a = r.inputs, r.tube, r.annulus
    L = n(r.L_installed_m, _L, system)
    lines: list[str] = []
    lines += _friction_lines("f_t", t)
    lines.append(
        chain(
            r"\Delta p_t",
            rf"f_t\,\frac{{L}}{{d_i}}\,{_dyn_sym(system)}",
            prod(
                _d(t.f),
                frac(L, n(i.d_i_m, _L, system)),
                _dyn_num(system, r.tube_props.rho_kg_per_m3, t.velocity_m_s),
            ),
            *_to_loss(r.dp_tube_Pa, system),
        )
    )
    laminar = a.correlation == "kays_perkins"
    lines += _friction_lines("f_a", a, annulus_friction_Re(a.ratio) if laminar else None)
    lines.append(
        chain(
            r"\Delta p_a",
            rf"f_a\,\frac{{L}}{{D_h}}\,{_dyn_sym(system)}",
            prod(
                _d(a.f),
                frac(L, n(a.D_h_m, _L, system)),
                _dyn_num(system, r.annulus_props.rho_kg_per_m3, a.velocity_m_s),
            ),
            *_to_loss(r.dp_annulus_Pa, system),
        )
    )
    lam = (
        " En el anular laminar, f·Re = 64·(1 − κ)²/[1 + κ² + (1 − κ²)/ln κ] con κ = d_o/D "
        "(Shah y London, 1978)."
        if laminar
        else ""
    )
    installed = (
        f"Con el largo instalado, L = {_txt(r.L_installed_m, _L, system)} (las horquillas "
        "enteras, el equipo que se pone), y fricción de Darcy: "
        if i.problem == "sizing"
        else "Por fricción, con Darcy: "
    )
    returns = (
        "La vuelta de la horquilla suma en el anular ~una carga de velocidad (Kern): "
        if r.hairpins == 1
        else f"Las {r.hairpins} vueltas de las horquillas suman en el anular ~una carga de "
        "velocidad cada una (Kern): "
    )
    text = (
        installed
        + "64/Re en laminar y Petukhov en turbulento."
        + lam
        + f" {returns}{_txt(r.dp_returns_Pa, _PL, system)}."
        + _dp_text(system)
    )
    return ProcedureStep("Caídas de presión", text, tuple(lines))


def double_pipe_steps(r: DoublePipeResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del doble tubo (Incropera, §8.7 y §11.3; Çengel y Ghajar, §11-2)."""
    i = r.inputs
    hot_props = r.annulus_props if r.annulus_stream == "hot" else r.tube_props
    cold_props = r.tube_props if r.annulus_stream == "hot" else r.annulus_props
    steps = [
        _streams_step(
            hot_props,
            cold_props,
            stream_name(i.hot),
            stream_name(i.cold),
            i.hot.properties is not None,
            i.cold.properties is not None,
            system,
        )
    ]
    tube_stream = i.cold if r.tube_stream == "cold" else i.hot
    heating = r.tube_stream == "cold"
    steps.append(
        ProcedureStep(
            f"Lado del tubo ({stream_name(tube_stream)})",
            f"Por el tubo de d_i = {_txt(i.d_i_m, 'small_length', system)}, con "
            f"{_CORR_NAMES.get(r.tube.correlation, r.tube.correlation)} (Fase 8.1).",
            tuple(
                _tube_side_lines(
                    r.tube, r.tube_props, tube_stream.m_dot_kg_s, i.d_i_m, r.L_m, heating, system
                )
            ),
        )
    )
    ann_stream = i.hot if r.annulus_stream == "hot" else i.cold
    ann_text = "El anular con el diámetro hidráulico D_h = D − d_o (Incropera, §8.7). " + (
        "Laminar: el Nu de la pared interior con la exterior aislada (Kays y Perkins; "
        "Incropera, tabla 8.2), interpolado en d_o/D (t es la fracción del tramo de la tabla)."
        if r.annulus.correlation == "kays_perkins"
        else f"Turbulento: {_CORR_NAMES.get(r.annulus.correlation, r.annulus.correlation)} con D_h."
    )
    steps.append(
        ProcedureStep(
            f"Lado del anular ({stream_name(ann_stream)})",
            ann_text,
            tuple(_annulus_lines(r, system)),
        )
    )
    tube_s, ann_s = (i.cold, i.hot) if r.tube_stream == "cold" else (i.hot, i.cold)
    steps.append(
        _u_step(
            r.u_result,
            i.d_i_m,
            i.d_o_m,
            i.k_wall_W_per_mK,
            tube_s.R_f_m2K_per_W,
            ann_s.R_f_m2K_per_W,
            r.tube.h_W_per_m2K,
            r.annulus.h_W_per_m2K,
            r.U_o_W_per_m2K,
            system,
            "adentro del tubo",
            "en el anular",
        )
    )
    do = n(i.d_o_m, _L, system)
    if i.problem == "rating":
        steps.append(
            ProcedureStep(
                "Largo y área",
                f"{i.hairpins} horquillas de dos ramas de {_txt(i.leg_length_m, _L, system)}.",
                (
                    chain(
                        "L",
                        r"2\,N\,L_{\text{rama}}",
                        prod("2", str(i.hairpins), n(i.leg_length_m, _L, system)),
                        q(r.L_m, _L, system),
                    ),
                    chain(
                        "A",
                        r"\pi\,d_o\,L",
                        prod(r"\pi", do, n(r.L_m, _L, system)),
                        q(r.area_m2, _A, system),
                    ),
                ),
            )
        )
    steps += _effectiveness(r.exchanger, stream_name(i.hot), stream_name(i.cold), system)
    if i.problem == "sizing":
        leg = n(i.leg_length_m, _L, system)
        steps.append(
            ProcedureStep(
                "Largo y horquillas",
                f"El largo de tubo para esa área y las horquillas de dos ramas de "
                f"{_txt(i.leg_length_m, _L, system)} que hacen falta (redondeadas para arriba).",
                (
                    chain(
                        "L",
                        frac("A", r"\pi\,d_o"),
                        frac(n(r.area_m2, _A, system), times(r"\pi", do)),
                        q(r.L_m, _L, system),
                    ),
                    chain(
                        "N",
                        r"\left\lceil\frac{L}{2\,L_{\text{rama}}}\right\rceil",
                        rf"\left\lceil{frac(n(r.L_m, _L, system), times('2', leg))}\right\rceil",
                        str(r.hairpins),
                    ),
                    chain(
                        r"L_{\text{inst}}",
                        r"2\,N\,L_{\text{rama}}",
                        prod("2", str(r.hairpins), leg),
                        q(r.L_installed_m, _L, system),
                    ),
                ),
            )
        )
    steps.append(_dp_double_pipe(r, system))
    return numbered(steps)


# ---------------------------------------------------------------------
# Casco y tubos (Kern)
# ---------------------------------------------------------------------


def _geometry_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    latex: list[str] = []
    PT, Ds = n(i.P_T_m, _L, system), n(i.D_s_m, _L, system)
    if i.tube_count is None:
        from core.heat_transfer.shell_tube import _CL, _ctp

        ctp, cl = _ctp(i.tube_passes), _CL[i.layout]
        latex.append(
            chain(
                "N_t",
                r"0.785\,\frac{CTP}{CL}\,\frac{D_s^2}{P_T^2}",
                prod("0.785", frac(_d(ctp, 3), _d(cl, 3)), frac(_pw(Ds, "2"), _pw(PT, "2"))),
                rf"{0.785 * ctp / cl * i.D_s_m**2 / i.P_T_m**2:.1f} \to {r.N_t}",
            )
        )
    else:
        latex.append(rf"N_t = {r.N_t}")
    latex.append(
        chain(
            "A_o",
            r"N_t\,\pi\,d_o\,L",
            prod(str(r.N_t), r"\pi", n(i.d_o_m, _L, system), n(i.L_m, _L, system)),
            q(r.area_m2, _A, system),
        )
    )
    text = (
        f"{r.N_t} tubos de d_o = {_txt(i.d_o_m, 'small_length', system)} y d_i = "
        f"{_txt(i.d_i_m, 'small_length', system)}, {i.tube_passes} "
        f"{'paso' if i.tube_passes == 1 else 'pasos'} por los tubos."
        + (
            " Sin el dato, N_t se estima con D_s (Kakaç y Liu): CTP = 0,93, 0,90 o 0,85 con "
            "1, 2 o más pasos y CL = 0,87 triangular o 1 cuadrado, y se redondea hacia abajo "
            "a un múltiplo de N_p (cada paso, los mismos tubos)."
            if i.tube_count is None
            else ""
        )
    )
    return ProcedureStep("Geometría", text, tuple(latex))


def _shell_side_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, s, p = r.inputs, r.shell, r.shell_props
    shell_s = i.hot if i.shell_side == "hot" else i.cold
    Ds, PT, B, do = (
        n(i.D_s_m, _L, system),
        n(i.P_T_m, _L, system),
        n(i.B_m, _L, system),
        n(i.d_o_m, _L, system),
    )
    square = i.layout in ("square", "rotated_square")
    clearance = i.P_T_m - i.d_o_m
    if square:
        free = i.P_T_m**2 - math.pi * i.d_o_m**2 / 4.0
        perimeter = math.pi * i.d_o_m
        free_formula = r"P_T^2 - \frac{\pi\,d_o^2}{4}"
        free_num = Pack(_pw(PT, "2"), (("-", frac(times(r"\pi", _pw(do, "2")), "4")),))
        perimeter_formula, perimeter_num = r"\pi\,d_o", times(r"\pi", do)
    else:
        free = math.sqrt(3.0) / 4.0 * i.P_T_m**2 - math.pi * i.d_o_m**2 / 8.0
        perimeter = math.pi * i.d_o_m / 2.0
        free_formula = r"\frac{\sqrt{3}}{4}\,P_T^2 - \frac{\pi\,d_o^2}{8}"
        free_num = Pack(
            times("0.43301", _pw(PT, "2")), (("-", frac(times(r"\pi", _pw(do, "2")), "8")),)
        )
        perimeter_formula, perimeter_num = r"\frac{\pi\,d_o}{2}", frac(times(r"\pi", do), "2")
    latex = [
        chain("C'", "P_T - d_o", sub(PT, do), q(clearance, _L, system)),
        chain(
            "a_s",
            r"\frac{D_s\,C'\,B}{P_T}",
            frac(times(Ds, n(clearance, _L, system), B), PT),
            q(s.a_s_m2, _A, system),
        ),
        chain(
            "G_s",
            r"\frac{\dot m}{a_s}",
            frac(n(shell_s.m_dot_kg_s, _M, system), n(s.a_s_m2, _A, system)),
            q(s.G_s, _G, system),
        ),
        chain(r"A_{\text{libre}}", free_formula, free_num, q(free, _A, system)),
        chain(
            r"\mathcal{P}_{\text{moj}}", perimeter_formula, perimeter_num, q(perimeter, _L, system)
        ),
        chain(
            "D_e",
            frac(r"4\,A_{\text{libre}}", r"\mathcal{P}_{\text{moj}}"),
            frac(times("4", n(free, _A, system)), n(perimeter, _L, system)),
            q(s.D_e_m, _L, system),
        ),
        chain(
            "Re_s",
            r"\frac{G_s\,D_e}{\mu}",
            frac(times(n(s.G_s, _G, system), n(s.D_e_m, _L, system)), n(p.mu_Pa_s, _MU, system)),
            big(s.Re),
        ),
    ]
    mu_w = r.shell_props.mu_Pa_s / s.phi ** (1 / 0.14) if s.phi != 1.0 else None
    if mu_w is not None:
        latex.append(
            chain(
                r"\phi_s",
                r"\left(\frac{\mu}{\mu_p}\right)^{0.14}",
                _pw(frac(n(p.mu_Pa_s, _MU, system), n(mu_w, _MU, system)), "0.14"),
                _d(s.phi),
            )
        )
    else:
        latex.append(r"\phi_s = 1")
    latex.append(
        chain(
            r"\mathrm{Nu}_s",
            r"0.36\,Re_s^{0.55}\,Pr^{1/3}\,\phi_s",
            prod("0.36", _pw(big(s.Re), "0.55"), _pw(_d(p.Pr), "1/3"), _d(s.phi)),
            _d(s.Nu),
        )
    )
    latex.append(
        chain(
            "h_o",
            r"\frac{\mathrm{Nu}_s\,k}{D_e}",
            frac(times(_d(s.Nu), n(p.k_W_per_mK, _K, system)), n(s.D_e_m, _L, system)),
            q(s.h_W_per_m2K, _H, system),
        )
    )
    shape = {
        "square": "cuadrado",
        "rotated_square": "cuadrado (Kern no distingue el cuadrado rotado)",
        "triangular": "triangular",
    }[i.layout]
    if mu_w is None:
        wall = "sin μ_p, φ_s = 1"
    elif shell_s.properties is not None:
        wall = "μ_p del enunciado"
    else:
        wall = "μ_p de CoolProp a la temperatura de la pared, que se itera"
    text = (
        "Kern (1950): el área de flujo cruzado entre dos bafles, con la luz C' entre dos "
        f"tubos, la velocidad másica y el diámetro equivalente del arreglo {shape}: cuatro "
        "veces el área libre de la celda sobre el perímetro mojado"
        + ("." if square else " (la celda es medio triángulo equilátero, con medio tubo).")
        + f" φ_s corrige por la viscosidad junto a la pared ({wall})."
    )
    return ProcedureStep(f"Lado del casco: Kern ({stream_name(shell_s)})", text, tuple(latex))


def _tube_side_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    tube_s = i.cold if i.shell_side == "hot" else i.hot
    per_pass = r.N_t / i.tube_passes
    lines = [
        chain(
            r"\dot m_{\text{tubo}}",
            r"\frac{\dot m}{N_t/N_p}",
            frac(n(tube_s.m_dot_kg_s, _M, system), frac(str(r.N_t), str(i.tube_passes))),
            q(tube_s.m_dot_kg_s / per_pass, _M, system),
        )
    ]
    lines += _tube_side_lines(
        r.tube,
        r.tube_props,
        tube_s.m_dot_kg_s / per_pass,
        i.d_i_m,
        i.L_m,
        i.shell_side == "hot",
        system,
    )
    average = "" if r.N_t % i.tube_passes == 0 else " en promedio"
    text = (
        f"Cada paso lleva N_t/N_p = {_comma(per_pass)} tubos en paralelo{average}: el caudal de "
        f"un tubo, y h_i con {_CORR_NAMES.get(r.tube.correlation, r.tube.correlation)} "
        "(Fase 8.1)."
    )
    return ProcedureStep(f"Lado de los tubos ({stream_name(tube_s)})", text, tuple(lines))


def _kern_check_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    ex = r.exchanger
    U_req = r.U_required_W_per_m2K or 0.0
    Rd = r.R_d_available or 0.0
    i = r.inputs
    shell_s, tube_s = (i.hot, i.cold) if i.shell_side == "hot" else (i.cold, i.hot)
    lines = [
        chain(
            r"U_{\text{req}}",
            frac(r"\dot Q", r"A_o\,F\,\Delta T_{ml}"),
            frac(
                n(ex.Q_W, _W, system),
                times(n(r.area_m2, _A, system), _d(ex.F, 4), n(ex.dT_lm_K, _DT, system)),
            ),
            q(U_req, _H, system),
        ),
        chain(
            "R_d",
            frac(r"U_C - U_{\text{req}}", r"U_C\,U_{\text{req}}"),
            frac(
                sub(n(r.U_C_W_per_m2K, _H, system), n(U_req, _H, system)),
                times(n(r.U_C_W_per_m2K, _H, system), n(U_req, _H, system)),
            ),
            q(Rd, _RF, system),
        ),
        chain(
            r"R_{d,\text{pedido}}",
            r"R''_{f,o} + R''_{f,i}\,\frac{d_o}{d_i}",
            Pack(
                n(shell_s.R_f_m2K_per_W, _RF, system),
                (
                    (
                        "+",
                        times(
                            n(tube_s.R_f_m2K_per_W, _RF, system),
                            frac(n(i.d_o_m, _L, system), n(i.d_i_m, _L, system)),
                        ),
                    ),
                ),
            ),
            q(r.R_d_required, _RF, system),
        ),
        chain(
            r"\frac{A_o}{A_{\text{req}}} - 1",
            frac(n(r.area_m2, _A, system), n(r.area_required_m2 or 0.0, _A, system)) + " - 1",
            _d(r.overdesign or 0.0, 3),
        ),
    ]
    verdict = (
        "el equipo sirve (tolera más ensuciamiento que el pedido)"
        if Rd >= r.R_d_required
        else "limpio alcanza, pero no tolera el ensuciamiento pedido"
        if Rd >= 0.0
        else "ni limpio alcanza: falta área"
    )
    text = (
        "Kern: el U que hace falta para el servicio con el área que hay, y el ensuciamiento "
        f"que el equipo tolera, contra el pedido (referido al área exterior): {verdict}. "
        "A_req es el área que hace falta con U_D (los pasos de arriba)."
    )
    return ProcedureStep("El criterio de Kern (R_d)", text, tuple(lines))


def _dp_tube_lines(r: ShellTubeResult, system: UnitSystem) -> list[str]:
    """Δp_t = K_t·q_t, con la fricción de los N_p pasos y cuatro cargas por paso."""
    i, t = r.inputs, r.tube
    lines = _friction_lines("f_t", t)
    rho_t = r.tube_props.rho_kg_per_m3
    dyn = rho_t * t.velocity_m_s**2 / 2.0
    lines.append(
        chain(
            r"q_t", _dyn_sym(system), _dyn_num(system, rho_t, t.velocity_m_s), q(dyn, _PD, system)
        )
    )
    n_p = i.tube_passes
    bracket = t.f * i.L_m * n_p / i.d_i_m + 4.0 * n_p
    friction = frac(times(n(i.L_m, _L, system), str(n_p)), n(i.d_i_m, _L, system))
    lines.append(
        chain(
            "K_t",
            r"f_t\,\frac{L\,N_p}{d_i} + 4\,N_p",
            Pack(times(_d(t.f), friction), (("+", f"4 \\cdot {n_p}"),)),
            _d(bracket),
        )
    )
    lines.append(
        chain(
            r"\Delta p_t",
            r"K_t\,q_t",
            times(_d(bracket), n(dyn, _PD, system)),
            *_to_loss(r.dp_tube_Pa, system),
        )
    )
    return lines


def _dp_shell_tube(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, s = r.inputs, r.shell
    p = r.shell_props
    q_s = s.G_s**2 / (2.0 * p.rho_kg_per_m3 * s.phi)
    num = _pw(n(s.G_s, _G, system), "2")
    den = times("2", n(p.rho_kg_per_m3, _RHO, system), _d(s.phi))
    if system == "Inglés":
        den = times("2", _GC, n(p.rho_kg_per_m3, _RHO, system), _d(s.phi))
    lines = [
        chain(
            "f_s", r"e^{0.576 - 0.19\,\ln Re_s}", rf"e^{{0.576 - 0.19\,\ln {big(s.Re)}}}", _d(s.f)
        ),
        chain(
            "q_s",
            r"\frac{G_s^2}{2\,g_c\,\rho\,\phi_s}"
            if system == "Inglés"
            else r"\frac{G_s^2}{2\,\rho\,\phi_s}",
            frac(num, den),
            q(q_s, _PD, system),
        ),
        chain(
            r"\Delta p_s",
            r"f_s\,(N_b + 1)\,\frac{D_s}{D_e}\,q_s",
            prod(
                _d(s.f),
                f"({i.baffles} + 1)",
                frac(n(i.D_s_m, _L, system), n(s.D_e_m, _L, system)),
                n(q_s, _PD, system),
            ),
            *_to_loss(s.dp_Pa, system),
        ),
    ]
    lines += _dp_tube_lines(r, system)
    text = (
        "Casco (Kern): f con el ajuste de Kakaç y Liu del gráfico de Kern (400 < Re ≤ 10⁶) y "
        "una caída por cada cruce del haz, N_b + 1. Tubos: la fricción de los N_p pasos y "
        "cuatro cargas de velocidad por paso en las vueltas (Kern; Kakaç y Liu)." + _dp_text(system)
    )
    return ProcedureStep("Caídas de presión", text, tuple(lines))


# ---------------------------------------------------------------------
# Casco y tubos: el lado del casco con Bell–Delaware (Fase 12.2)
# ---------------------------------------------------------------------

#: El factor del paso en la dirección del flujo (L_pp) y del normal efectivo (L_tp,ef) de
#: cada ángulo (Taborek, 1983).
_ANGLE_PITCH: dict[int, tuple[str, str]] = {
    30: ("0.866", "1"),
    45: ("0.707", "0.707"),
    90: ("1", "1"),
}
_ANGLE_NAME: dict[int, str] = {
    30: "triangular (30°)",
    45: "cuadrado rotado (45°)",
    90: "cuadrado (90°)",
}
#: Los tramos de Re de la tabla de Taborek, por el Re mínimo de cada uno.
_RE_ROW: dict[float, str] = {
    1.0e4: "de 10⁴ a 10⁵",
    1.0e3: "de 10³ a 10⁴",
    1.0e2: "de 10² a 10³",
    1.0e1: "de 10 a 10²",
    0.0: "de 1 a 10",
}
_ENT = r"\text{ent}"
_SAL = r"\text{sal}"


def _bd(r: ShellTubeResult) -> BellDelawareSide:
    assert r.bell_delaware is not None
    return r.bell_delaware


def _rad(x: float) -> str:
    return rf"{_d(x)}\ \text{{rad}}"


def _paren(tex: str) -> str:
    return f"({tex})"


def _shell_stream(r: ShellTubeResult) -> DesignStream:
    i = r.inputs
    return i.hot if i.shell_side == "hot" else i.cold


def _bd_geometry_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, bd = r.inputs, _bd(r)
    g, a = bd.geometry, bd.areas

    def L(x: float) -> str:
        return n(x, _L, system)

    Bc = _d(g.baffle_cut, 3)
    cut = rf"(1 - 2 \cdot {Bc})"
    k_pp, k_eff = _ANGLE_PITCH[a.angle]
    pp = "P_T" if k_pp == "1" else "L_{pp}"
    eff = "P_T" if k_eff == "1" else r"L_{tp,\text{ef}}"
    lines = [
        chain("D_{otl}", "D_s - L_{bb}", sub(L(i.D_s_m), L(g.L_bb_m)), q(a.D_otl_m, _L, system)),
        chain("D_{ctl}", "D_{otl} - d_o", sub(L(a.D_otl_m), L(i.d_o_m)), q(a.D_ctl_m, _L, system)),
    ]
    if k_pp != "1":
        lines.append(
            chain(
                r"L_{pp} = L_{tp,\text{ef}}" if k_eff == k_pp else "L_{pp}",
                rf"{k_pp}\,P_T",
                times(k_pp, L(i.P_T_m)),
                q(a.L_pp_m, _L, system),
            )
        )
    if k_eff not in ("1", k_pp):
        lines.append(
            chain(
                r"L_{tp,\text{ef}}",
                rf"{k_eff}\,P_T",
                times(k_eff, L(i.P_T_m)),
                q(a.L_tp_eff_m, _L, system),
            )
        )
    ell = a.S_m_m2 / i.B_m
    lines += [
        chain(
            r"\ell_m",
            Pack("L_{bb}", (("+", rf"\frac{{D_{{ctl}}}}{{{eff}}}\,(P_T - d_o)"),)),
            Pack(
                L(g.L_bb_m),
                (
                    ("+", frac(L(a.D_ctl_m), L(a.L_tp_eff_m))),
                    (r"\cdot", _paren(sub(L(i.P_T_m), L(i.d_o_m)))),
                ),
            ),
            q(ell, _L, system),
        ),
        chain("S_m", r"B\,\ell_m", times(L(i.B_m), L(ell)), q(a.S_m_m2, _A, system)),
        chain(r"\theta_{ds}", r"2\,\arccos(1 - 2\,B_c)", rf"2\,\arccos{cut}", _rad(a.theta_ds)),
    ]
    if a.theta_ctl > 0.0:
        lines.append(
            chain(
                r"\theta_{ctl}",
                r"2\,\arccos\frac{D_s\,(1 - 2\,B_c)}{D_{ctl}}",
                r"2\,\arccos" + frac(times(L(i.D_s_m), cut), L(a.D_ctl_m)),
                _rad(a.theta_ctl),
            )
        )
        lines.append(
            chain(
                "F_w",
                frac(r"\theta_{ctl} - \sin\theta_{ctl}", r"2\pi"),
                frac(rf"{_d(a.theta_ctl)} - \sin {_d(a.theta_ctl)}", r"2\pi"),
                _d(a.F_w),
            )
        )
    else:
        lines.append(r"\theta_{ctl} = 0 \quad \Rightarrow \quad F_w = 0")
    lines += [
        chain("F_c", r"1 - 2\,F_w", sub("1", times("2", _d(a.F_w))), _d(a.F_c)),
        chain(
            "S_{wg}",
            r"\frac{\pi\,D_s^2}{4}\,\frac{\theta_{ds} - \sin\theta_{ds}}{2\pi}",
            prod(
                frac(times(r"\pi", _pw(L(i.D_s_m), "2")), "4"),
                frac(rf"{_d(a.theta_ds)} - \sin {_d(a.theta_ds)}", r"2\pi"),
            ),
            q(a.S_wg_m2, _A, system),
        ),
        chain(
            "S_{wt}",
            r"N_t\,F_w\,\frac{\pi\,d_o^2}{4}",
            prod(str(r.N_t), _d(a.F_w), frac(times(r"\pi", _pw(L(i.d_o_m), "2")), "4")),
            q(a.S_wt_m2, _A, system),
        ),
        chain(
            "S_w",
            "S_{wg} - S_{wt}",
            sub(n(a.S_wg_m2, _A, system), n(a.S_wt_m2, _A, system)),
            q(a.S_w_m2, _A, system),
        ),
        chain(
            "N_{tcc}",
            rf"\frac{{D_s}}{{{pp}}}\,(1 - 2\,B_c)",
            times(frac(L(i.D_s_m), L(a.L_pp_m)), cut),
            _d(a.N_tcc),
        ),
    ]
    ell_w = g.baffle_cut * i.D_s_m - (i.D_s_m - a.D_ctl_m) / 2.0
    lines.append(
        chain(
            r"\ell_w",
            Pack(r"B_c\,D_s", (("-", r"\frac{D_s - D_{ctl}}{2}"),)),
            Pack(times(Bc, L(i.D_s_m)), (("-", frac(sub(L(i.D_s_m), L(a.D_ctl_m)), "2")),)),
            q(ell_w, _L, system),
        )
    )
    if ell_w > 0.0:
        lines.append(
            chain(
                "N_{tcw}",
                rf"\frac{{0.8\,\ell_w}}{{{pp}}}",
                frac(times("0.8", L(ell_w)), L(a.L_pp_m)),
                _d(a.N_tcw),
            )
        )
    else:
        lines.append(r"N_{tcw} = 0 \quad (\ell_w \le 0)")
    if bd.factors.laminar:
        perimeter = math.pi * i.d_o_m * r.N_t * a.F_w + a.theta_ds * i.D_s_m
        lines += [
            chain(
                r"\mathcal{P}_w",
                Pack(r"\pi\,d_o\,N_t\,F_w", (("+", r"\theta_{ds}\,D_s"),)),
                Pack(
                    times(r"\pi", L(i.d_o_m), str(r.N_t), _d(a.F_w)),
                    (("+", times(_d(a.theta_ds), L(i.D_s_m))),),
                ),
                q(perimeter, _L, system),
            ),
            chain(
                "D_w",
                frac(r"4\,S_w", r"\mathcal{P}_w"),
                frac(times("4", n(a.S_w_m2, _A, system)), L(perimeter)),
                q(a.D_w_m, _L, system),
            ),
        ]
    if a.angle == 90:
        pitch = "los pasos en la dirección del flujo (L_pp) y normal a él (L_tp,ef) son P_T"
    elif a.angle == 30:
        pitch = "el paso en la dirección del flujo es L_pp = 0,866·P_T y el normal, P_T"
    else:
        pitch = "los pasos en la dirección del flujo y normal a él son 0,707·P_T"
    text = (
        "Bell–Delaware (Taborek, 1983): el haz ocupa hasta D_otl (el casco menos la holgura "
        "L_bb) y los centros de los tubos de la periferia llegan hasta D_ctl. S_m es el área "
        "de flujo cruzado en la línea de centros: la luz entre los tubos más el canal entre el "
        f"haz y el casco. En el arreglo {_ANGLE_NAME[a.angle]}, {pitch}. Con el corte B_c, θ_ds "
        "y θ_ctl son los ángulos que abarca la cuerda del bafle sobre D_s y sobre D_ctl (en "
        "radianes): F_w es la fracción de los tubos que queda en una ventana y F_c, la de los "
        "que cruzan entre las puntas de los bafles. S_w es el área de flujo de una ventana "
        "(sin los tubos); N_tcc y N_tcw, las hileras que cruza el fluido entre las puntas de "
        "los bafles y en cada ventana (ℓ_w es la altura de la ventana dentro del haz)."
        + (
            " Con flujo laminar hace falta además D_w, el diámetro hidráulico de la ventana "
            "(su perímetro mojado: los tubos y el arco del casco)."
            if bd.factors.laminar
            else ""
        )
    )
    return ProcedureStep(
        f"Lado del casco ({stream_name(_shell_stream(r))}): la geometría de Bell–Delaware",
        text,
        tuple(lines),
    )


def _bd_leakage_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, bd = r.inputs, _bd(r)
    g, a = bd.geometry, bd.areas

    def L(x: float) -> str:
        return n(x, _L, system)

    hole = i.d_o_m + a.L_tb_m
    annulus = math.pi / 4.0 * (hole**2 - i.d_o_m**2)
    leak = a.S_sb_m2 + a.S_tb_m2
    lines = [
        chain(
            "S_{sb}",
            r"\frac{\pi\,D_s\,L_{sb}}{2}\left(1 - \frac{\theta_{ds}}{2\pi}\right)",
            prod(
                frac(times(r"\pi", L(i.D_s_m), L(a.L_sb_m)), "2"),
                rf"\left(1 - \frac{{{_d(a.theta_ds)}}}{{2\pi}}\right)",
            ),
            q(a.S_sb_m2, _A, system),
        ),
        chain(
            r"d_{\text{ag}}", "d_o + L_{tb}", f"{L(i.d_o_m)} + {L(a.L_tb_m)}", q(hole, _L, system)
        ),
        chain(
            "s_{tb}",
            r"\frac{\pi}{4}\,(d_{\text{ag}}^2 - d_o^2)",
            times(frac(r"\pi", "4"), _paren(sub(_pw(L(hole), "2"), _pw(L(i.d_o_m), "2")))),
            q(annulus, _A, system),
        ),
        chain(
            "S_{tb}",
            r"s_{tb}\,N_t\,(1 - F_w)",
            prod(n(annulus, _A, system), str(r.N_t), _paren(sub("1", _d(a.F_w)))),
            q(a.S_tb_m2, _A, system),
        ),
    ]
    if g.L_pl_m > 0.0:
        lines.append(
            chain(
                "S_b",
                r"B\,(L_{bb} + L_{pl})",
                times(L(i.B_m), _paren(f"{L(g.L_bb_m)} + {L(g.L_pl_m)}")),
                q(a.S_b_m2, _A, system),
            )
        )
    else:
        lines.append(
            chain("S_b", r"B\,L_{bb}", times(L(i.B_m), L(g.L_bb_m)), q(a.S_b_m2, _A, system))
        )
    lines += [
        chain(
            "S_l",
            "S_{sb} + S_{tb}",
            f"{n(a.S_sb_m2, _A, system)} + {n(a.S_tb_m2, _A, system)}",
            q(leak, _A, system),
        ),
        chain(
            "r_s",
            frac("S_{sb}", "S_l"),
            frac(n(a.S_sb_m2, _A, system), n(leak, _A, system)),
            _d(a.r_s),
        ),
        chain(
            "r_{lm}",
            frac("S_l", "S_m"),
            frac(n(leak, _A, system), n(a.S_m_m2, _A, system)),
            _d(a.r_lm),
        ),
        chain(
            "F_{sbp}",
            frac("S_b", "S_m"),
            frac(n(a.S_b_m2, _A, system), n(a.S_m_m2, _A, system)),
            _d(a.F_sbp),
        ),
        chain(
            "r_{ss}",
            frac("N_{ss}", "N_{tcc}"),
            frac(str(g.sealing_strips), _d(a.N_tcc)),
            _d(a.r_ss),
        ),
    ]
    tema_sb = " (TEMA, RCB-4.3)" if g.L_sb_m is None else ""
    tema_tb = " (TEMA, RCB-4.2)" if g.L_tb_m is None else ""
    lane = (
        f" más el carril de los pasos, L_pl = {_txt(g.L_pl_m, 'small_length', system)}"
        if g.L_pl_m > 0.0
        else ""
    )
    strips = (
        "Sin tiras de sellado (N_ss = 0), nada lo corta."
        if g.sealing_strips == 0
        else f"N_ss = {g.sealing_strips} pares de tiras de sellado lo cortan."
    )
    text = (
        "Las holguras diametrales: casco–bafle L_sb = "
        f"{_txt(a.L_sb_m, 'small_length', system)}{tema_sb} y tubo–bafle L_tb = "
        f"{_txt(a.L_tb_m, 'small_length', system)}{tema_tb}. Por ellas el fluido pasa de un "
        "compartimiento al siguiente sin cruzar el haz: S_sb es el anillo entre el casco y el "
        "bafle (sin el arco de la ventana) y S_tb, el de los agujeros de los tubos que no están "
        "en la ventana (s_tb cada uno, con el agujero de d_ag); S_l = S_sb + S_tb. S_b es el "
        f"canal del bypass entre el haz y el casco{lane}. {strips} r_s, r_lm, F_sbp y r_ss son "
        "los cocientes que usan los factores."
    )
    return ProcedureStep("Lado del casco: las fugas y el bypass", text, tuple(lines))


def _bd_ideal_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, bd = r.inputs, _bd(r)
    a, ib, p = bd.areas, bd.ideal, r.shell_props
    shell_s = _shell_stream(r)
    lines = [
        chain(
            "G_s",
            frac(r"\dot m", "S_m"),
            frac(n(shell_s.m_dot_kg_s, _M, system), n(a.S_m_m2, _A, system)),
            q(bd.G_s, _G, system),
        ),
        chain(
            "Re_s",
            frac(r"d_o\,G_s", r"\mu"),
            frac(times(n(i.d_o_m, _L, system), n(bd.G_s, _G, system)), n(p.mu_Pa_s, _MU, system)),
            big(bd.Re),
        ),
        chain(
            "a",
            frac("a_3", r"1 + 0.14\,Re_s^{a_4}"),
            frac(_d(ib.a3, 4), rf"1 + 0.14 \cdot {_pw(big(bd.Re), _d(ib.a4, 4))}"),
            _d(ib.a),
        ),
        chain(
            "j_i",
            r"a_1\left(\frac{1.33}{P_T/d_o}\right)^{a}\,Re_s^{a_2}",
            prod(
                _d(ib.a1, 4),
                _pw(frac("1.33", _d(ib.pitch_ratio)), _d(ib.a)),
                _pw(big(bd.Re), _d(ib.a2, 4)),
            ),
            _d(ib.j),
        ),
    ]
    mu_w = p.mu_Pa_s / bd.phi ** (1 / 0.14) if bd.phi != 1.0 else None
    if mu_w is not None:
        lines.append(
            chain(
                r"\phi_s",
                r"\left(\frac{\mu}{\mu_p}\right)^{0.14}",
                _pw(frac(n(p.mu_Pa_s, _MU, system), n(mu_w, _MU, system)), "0.14"),
                _d(bd.phi),
            )
        )
    else:
        lines.append(r"\phi_s = 1")
    f = _FACTOR[system]
    factors = [
        _d(ib.j),
        n(p.cp_J_per_kgK, _CP, system),
        n(bd.G_s, _G, system),
        _pw(_d(p.Pr), "-2/3"),
        _d(bd.phi),
    ]
    lines.append(
        chain(
            r"h_{\text{ideal}}",
            r"j_i\,c_p\,G_s\,\mathrm{Pr}^{-2/3}\,\phi_s",
            prod(*([f, *factors] if f else factors)),
            q(bd.h_ideal_W_per_m2K, _H, system),
        )
    )
    row = next(rw[0] for rw in IDEAL_BANK[a.angle].rows if bd.Re >= rw[0])
    stretch = _RE_ROW[row]
    if not 1.0 <= bd.Re <= 1.0e5:
        stretch += ", extrapolado"
    if mu_w is None:
        wall = "sin μ_p, φ_s = 1"
    elif shell_s.properties is not None:
        wall = "μ_p del enunciado"
    else:
        wall = "μ_p de CoolProp a la temperatura de la pared, que se itera"
    if system == "Técnico":
        units = " Con c_p en kJ/(kg·K), el 1000 deja h en W/(m²·K)."
    elif system == "Inglés":
        units = " Con G_s por segundo, el 3600 deja h por hora: Btu/(h·ft²·°F)."
    else:
        units = ""
    text = (
        "El banco ideal: todo el fluido cruza el haz con la velocidad másica de S_m, y Re_s va "
        "con d_o. La tabla de Taborek (1983) da j_i para el arreglo de "
        f"{a.angle}° con Re {stretch}: a₁ = {_comma(ib.a1)}, a₂ = {_minus(ib.a2)}, "
        f"a₃ = {_comma(ib.a3)} y a₄ = {_comma(ib.a4)}. h_ideal es el h de Colburn (j = St·Pr^⅔) "
        f"con φ_s, la corrección por la viscosidad junto a la pared ({wall})." + units
    )
    return ProcedureStep("Lado del casco: el banco ideal", text, tuple(lines))


def _bd_factors_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, bd = r.inputs, _bd(r)
    g, a, fc = bd.geometry, bd.areas, bd.factors
    c_l = 0.44 * (1.0 - a.r_s)
    lines = [
        chain("J_c", r"0.55 + 0.72\,F_c", rf"0.55 + 0.72 \cdot {_d(a.F_c)}", _d(fc.J_c)),
        chain("c_l", r"0.44\,(1 - r_s)", times("0.44", _paren(sub("1", _d(a.r_s)))), _d(c_l)),
        chain(
            "J_l",
            r"c_l + (1 - c_l)\,e^{-2.2\,r_{lm}}",
            Pack(
                _d(c_l),
                (("+", times(_paren(sub("1", _d(c_l))), rf"e^{{-2.2 \cdot {_d(a.r_lm)}}}")),),
            ),
            _d(fc.J_l),
        ),
    ]
    bypass = r"e^{-C_{bh}\,F_{sbp}\,\left(1 - (2\,r_{ss})^{1/3}\right)}"
    if a.r_ss >= 0.5:
        lines.append(r"J_b = 1 \quad (r_{ss} \ge 1/2)")
    elif g.sealing_strips == 0:
        lines.append(
            chain("J_b", bypass, rf"e^{{-{_d(fc.C_bh, 3)} \cdot {_d(a.F_sbp)}}}", _d(fc.J_b))
        )
    else:
        lines.append(
            chain(
                "J_b",
                bypass,
                rf"e^{{-{_d(fc.C_bh, 3)} \cdot {_d(a.F_sbp)} \cdot "
                rf"\left(1 - (2 \cdot {_d(a.r_ss)})^{{1/3}}\right)}}",
                _d(fc.J_b),
            )
        )
    equal = a.B_in_m == i.B_m and a.B_out_m == i.B_m
    if equal:
        lines.append(rf"J_s = 1 \quad (B_{{{_ENT}}} = B_{{{_SAL}}} = B)")
    else:
        Li, Lo = a.B_in_m / i.B_m, a.B_out_m / i.B_m
        e = _d(1.0 - fc.n, 3)
        lines += [
            chain(
                rf"L^*_{{{_ENT}}}",
                frac(rf"B_{{{_ENT}}}", "B"),
                frac(n(a.B_in_m, _L, system), n(i.B_m, _L, system)),
                _d(Li),
            ),
            chain(
                rf"L^*_{{{_SAL}}}",
                frac(rf"B_{{{_SAL}}}", "B"),
                frac(n(a.B_out_m, _L, system), n(i.B_m, _L, system)),
                _d(Lo),
            ),
            flush(
                "J_s",
                frac(
                    rf"(N_b - 1) + L^{{*\,1-n}}_{{{_ENT}}} + L^{{*\,1-n}}_{{{_SAL}}}",
                    rf"(N_b - 1) + L^*_{{{_ENT}}} + L^*_{{{_SAL}}}",
                ),
                frac(
                    f"{i.baffles - 1} + {_pw(_d(Li), e)} + {_pw(_d(Lo), e)}",
                    f"{i.baffles - 1} + {_d(Li)} + {_d(Lo)}",
                ),
                _d(fc.J_s),
            ),
        ]
    if fc.laminar:
        lines += [
            chain(
                "N_c",
                r"(N_{tcc} + N_{tcw})\,(N_b + 1)",
                times(_paren(f"{_d(a.N_tcc)} + {_d(a.N_tcw)}"), _paren(f"{i.baffles} + 1")),
                _d(fc.N_c),
            ),
            chain(
                r"J_r^*",
                r"\left(\frac{10}{N_c}\right)^{0.18}",
                _pw(frac("10", _d(fc.N_c)), "0.18"),
                _d(fc.J_r_star),
            ),
        ]
        raw = fc.J_r_star
        if bd.Re > 20.0:
            raw += (20.0 - bd.Re) / 80.0 * (fc.J_r_star - 1.0)
        clipped = raw < 0.4
        lhs = r"J_r^{\,\prime}" if clipped else "J_r"
        if bd.Re > 20.0:
            lines.append(
                chain(
                    lhs,
                    Pack(r"J_r^*", (("+", r"\frac{20 - Re_s}{80}\,(J_r^* - 1)"),)),
                    Pack(
                        _d(fc.J_r_star),
                        (
                            (
                                "+",
                                times(
                                    frac(sub("20", _d(bd.Re)), "80"),
                                    _paren(f"{_d(fc.J_r_star)} - 1"),
                                ),
                            ),
                        ),
                    ),
                    _d(raw),
                )
            )
        else:
            lines.append(rf"{lhs} = J_r^* = {_d(fc.J_r_star)}")
        if clipped:
            lines.append(r"J_r = \max(0.4,\ J_r^{\,\prime}) = 0.4")
    else:
        lines.append(r"J_r = 1 \quad (Re_s \ge 100)")
    lines.append(
        chain(
            "h_o",
            r"h_{\text{ideal}}\,J_c\,J_l\,J_b\,J_s\,J_r",
            prod(
                n(bd.h_ideal_W_per_m2K, _H, system),
                _d(fc.J_c),
                _d(fc.J_l),
                _d(fc.J_b),
                _d(fc.J_s),
                _d(fc.J_r),
            ),
            q(bd.h_W_per_m2K, _H, system),
        )
    )
    regime = "laminar (Re_s < 100)" if fc.laminar else "turbulento (Re_s ≥ 100)"
    if a.r_ss >= 0.5:
        bypass_text = "con r_ss ≥ ½ las tiras lo anulan"
    elif g.sealing_strips == 0:
        bypass_text = f"C_bh = {_comma(fc.C_bh)}; sin tiras, el paréntesis vale 1"
    else:
        bypass_text = f"C_bh = {_comma(fc.C_bh)}"
    text = (
        f"Los factores de Taborek (1983), con el flujo {regime}: J_c, el corte (los tubos de "
        "las ventanas no ven flujo cruzado; con ventanas chicas el fluido se acelera y J_c pasa "
        "de 1); J_l, las fugas por las holguras (c_l es lo que queda con fugas enormes); J_b, "
        f"el bypass entre el haz y el casco ({bypass_text}); J_s, las separaciones de la "
        "entrada y de la salida"
        + ("" if equal else f" (n = {'⅓' if fc.laminar else _comma(fc.n, 3)})")
        + "; J_r, el gradiente adverso de temperatura del flujo laminar"
        + (
            f" (N_c = {_comma(fc.N_c)} hileras en todo el casco; nunca menos de 0,4)"
            if fc.laminar
            else ""
        )
        + f". En total, h_o = {_comma(fc.J_total, 3)}·h_ideal."
    )
    return ProcedureStep("Lado del casco: los factores de corrección", text, tuple(lines))


def _bd_pressure_step(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    i, bd = r.inputs, _bd(r)
    g, a, fc, ib, p = bd.geometry, bd.areas, bd.factors, bd.ideal, r.shell_props
    shell_s = _shell_stream(r)
    gc = system == "Inglés"
    rho = n(p.rho_kg_per_m3, _RHO, system)
    G_w = shell_s.m_dot_kg_s / math.sqrt(a.S_m_m2 * a.S_w_m2)
    Gw = n(G_w, _G, system)
    lines = [
        chain(
            "b",
            frac("b_3", r"1 + 0.14\,Re_s^{b_4}"),
            frac(_d(ib.b3, 4), rf"1 + 0.14 \cdot {_pw(big(bd.Re), _d(ib.b4, 4))}"),
            _d(ib.b),
        ),
        chain(
            "f_i",
            r"b_1\left(\frac{1.33}{P_T/d_o}\right)^{b}\,Re_s^{b_2}",
            prod(
                _d(ib.b1, 4),
                _pw(frac("1.33", _d(ib.pitch_ratio)), _d(ib.b)),
                _pw(big(bd.Re), _d(ib.b2, 4)),
            ),
            _d(ib.f),
        ),
        chain(
            r"\Delta p_{bi}",
            frac(r"2\,f_i\,N_{tcc}\,G_s^2", r"g_c\,\rho\,\phi_s" if gc else r"\rho\,\phi_s"),
            frac(
                times("2", _d(ib.f), _d(a.N_tcc), _pw(n(bd.G_s, _G, system), "2")),
                times(_GC, rho, _d(bd.phi)) if gc else times(rho, _d(bd.phi)),
            ),
            q(bd.dp_bi_Pa, _PD, system),
        ),
        chain(
            "G_w",
            frac(r"\dot m", r"\sqrt{S_m\,S_w}"),
            frac(
                n(shell_s.m_dot_kg_s, _M, system),
                rf"\sqrt{{{times(n(a.S_m_m2, _A, system), n(a.S_w_m2, _A, system))}}}",
            ),
            q(G_w, _G, system),
        ),
    ]
    rho_den = times(_GC, rho) if gc else rho
    rho_sym = r"g_c\,\rho" if gc else r"\rho"
    if fc.laminar:
        beta = a.N_tcw / (i.P_T_m - i.d_o_m) + i.B_m / a.D_w_m**2
        visc = 26.0 * p.mu_Pa_s * G_w * beta / p.rho_kg_per_m3
        accel = G_w**2 / p.rho_kg_per_m3
        lines += [
            chain(
                r"\beta_w",
                Pack(frac("N_{tcw}", "P_T - d_o"), (("+", frac("B", "D_w^2")),)),
                Pack(
                    frac(_d(a.N_tcw), sub(n(i.P_T_m, _L, system), n(i.d_o_m, _L, system))),
                    (("+", frac(n(i.B_m, _L, system), _pw(n(a.D_w_m, _L, system), "2"))),),
                ),
                q(beta, "inverse_length", system),
            ),
            chain(
                r"\Delta p_{wi}",
                Pack(
                    frac(r"26\,\mu\,G_w\,\beta_w", rho_sym),
                    (("+", frac("G_w^2", rho_sym)),),
                ),
                Pack(
                    frac(
                        times(
                            "26", n(p.mu_Pa_s, _MU, system), Gw, n(beta, "inverse_length", system)
                        ),
                        rho_den,
                    ),
                    (("+", frac(_pw(Gw, "2"), rho_den)),),
                ),
                Pack(n(visc, _PD, system), (("+", n(accel, _PD, system)),)),
                q(bd.dp_wi_Pa, _PD, system),
            ),
        ]
        window_text = (
            " En flujo laminar (Re_s < 100), la ventana suma la fricción viscosa (β_w, con las "
            "hileras de la ventana y su diámetro hidráulico) y la carga de velocidad."
        )
    else:
        lines.append(
            chain(
                r"\Delta p_{wi}",
                rf"(2 + 0.6\,N_{{tcw}})\,\frac{{G_w^2}}{{2\,{rho_sym}}}",
                prod(
                    _paren(rf"2 + 0.6 \cdot {_d(a.N_tcw)}"),
                    frac(_pw(Gw, "2"), times("2", rho_den)),
                ),
                q(bd.dp_wi_Pa, _PD, system),
            )
        )
        window_text = (
            " En cada ventana, dos cargas de velocidad con G_w (la media geométrica de S_m y S_w) "
            "y 0,6 por cada hilera de la ventana."
        )
    lines += [
        chain("p", r"-0.15\,(1 + r_s) + 0.8", rf"-0.15 \cdot (1 + {_d(a.r_s)}) + 0.8", _d(fc.p)),
        chain(
            "R_l",
            r"e^{-1.33\,(1 + r_s)\,r_{lm}^{\,p}}",
            rf"e^{{-1.33 \cdot (1 + {_d(a.r_s)}) \cdot {_pw(_d(a.r_lm), _d(fc.p))}}}",
            _d(fc.R_l),
        ),
    ]
    bypass = r"e^{-C_{bp}\,F_{sbp}\,\left(1 - (2\,r_{ss})^{1/3}\right)}"
    if a.r_ss >= 0.5:
        lines.append(r"R_b = 1 \quad (r_{ss} \ge 1/2)")
    elif g.sealing_strips == 0:
        lines.append(
            chain("R_b", bypass, rf"e^{{-{_d(fc.C_bp, 3)} \cdot {_d(a.F_sbp)}}}", _d(fc.R_b))
        )
    else:
        lines.append(
            chain(
                "R_b",
                bypass,
                rf"e^{{-{_d(fc.C_bp, 3)} \cdot {_d(a.F_sbp)} \cdot "
                rf"\left(1 - (2 \cdot {_d(a.r_ss)})^{{1/3}}\right)}}",
                _d(fc.R_b),
            )
        )
    equal = a.B_in_m == i.B_m and a.B_out_m == i.B_m
    if equal:
        lines.append(rf"R_{{s,{_ENT}}} = R_{{s,{_SAL}}} = 1")
    else:
        e = _d(2.0 - fc.n_prime, 3)
        for sym, B_end, value in (
            (_ENT, a.B_in_m, fc.R_s_in),
            (_SAL, a.B_out_m, fc.R_s_out),
        ):
            lines.append(
                chain(
                    rf"R_{{s,{sym}}}",
                    _pw(frac("B", rf"B_{{{sym}}}"), "2 - n'"),
                    _pw(frac(n(i.B_m, _L, system), n(B_end, _L, system)), e),
                    _d(value),
                )
            )
    dp_bi = n(bd.dp_bi_Pa, _PD, system)
    windows = rf"\left(1 + \frac{{{_d(a.N_tcw)}}}{{{_d(a.N_tcc)}}}\right)"
    lines += [
        chain(
            r"\Delta p_c",
            r"(N_b - 1)\,\Delta p_{bi}\,R_b\,R_l",
            prod(_paren(f"{i.baffles} - 1"), dp_bi, _d(fc.R_b), _d(fc.R_l)),
            q(bd.dp_cross_Pa, _PD, system),
        ),
        chain(
            r"\Delta p_w",
            r"N_b\,\Delta p_{wi}\,R_l",
            prod(str(i.baffles), n(bd.dp_wi_Pa, _PD, system), _d(fc.R_l)),
            q(bd.dp_window_Pa, _PD, system),
        ),
    ]
    if equal:
        lines.append(
            chain(
                r"\Delta p_e",
                r"2\,\Delta p_{bi}\left(1 + \frac{N_{tcw}}{N_{tcc}}\right)R_b",
                prod("2", dp_bi, windows, _d(fc.R_b)),
                q(bd.dp_ends_Pa, _PD, system),
            )
        )
    else:
        lines.append(
            chain(
                r"\Delta p_e",
                Pack(
                    r"\Delta p_{bi}\left(1 + \frac{N_{tcw}}{N_{tcc}}\right)R_b",
                    ((r"\cdot", rf"(R_{{s,{_ENT}}} + R_{{s,{_SAL}}})"),),
                ),
                prod(dp_bi, windows, _d(fc.R_b), _paren(f"{_d(fc.R_s_in)} + {_d(fc.R_s_out)}")),
                q(bd.dp_ends_Pa, _PD, system),
            )
        )
    lines.append(
        chain(
            r"\Delta p_s",
            r"\Delta p_c + \Delta p_w + \Delta p_e",
            Pack(
                n(bd.dp_cross_Pa, _PD, system),
                (("+", n(bd.dp_window_Pa, _PD, system)), ("+", n(bd.dp_ends_Pa, _PD, system))),
            ),
            *_to_loss(bd.dp_Pa, system),
        )
    )
    text = (
        "El Δp del casco por zonas (Taborek, 1983): Δp_bi es un compartimiento central ideal, "
        "con f_i de la misma tabla (los b), y Δp_wi una ventana ideal." + window_text + " R_l "
        "corrige por las fugas el flujo cruzado y las ventanas; R_b, por el bypass, el flujo "
        "cruzado y los extremos. Son N_b − 1 compartimientos centrales, N_b ventanas y dos "
        "extremos, que cruzan además las hileras de una ventana"
        + ("" if equal else f" y llevan R_s por su separación (n′ = {_comma(fc.n_prime, 3)})")
        + "."
        + _dp_text(system)
    )
    return ProcedureStep("Caída de presión en el casco (Bell–Delaware)", text, tuple(lines))


def _dp_tubes_only(r: ShellTubeResult, system: UnitSystem) -> ProcedureStep:
    text = (
        "La fricción de los N_p pasos y cuatro cargas de velocidad por paso en las vueltas "
        "(Kern; Kakaç y Liu)." + _dp_text(system)
    )
    return ProcedureStep("Caída de presión en los tubos", text, tuple(_dp_tube_lines(r, system)))


def shell_tube_steps(r: ShellTubeResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del casco y tubos con Kern (Kern, 1950; Kakaç y Liu, 2002) o con
    Bell–Delaware en el casco (Taborek, 1983)."""
    i = r.inputs
    bell = r.bell_delaware is not None
    hot_props = r.shell_props if i.shell_side == "hot" else r.tube_props
    cold_props = r.tube_props if i.shell_side == "hot" else r.shell_props
    steps = [
        _geometry_step(r, system),
        _streams_step(
            hot_props,
            cold_props,
            stream_name(i.hot),
            stream_name(i.cold),
            i.hot.properties is not None,
            i.cold.properties is not None,
            system,
        ),
        *(
            (
                _bd_geometry_step(r, system),
                _bd_leakage_step(r, system),
                _bd_ideal_step(r, system),
                _bd_factors_step(r, system),
            )
            if bell
            else (_shell_side_step(r, system),)
        ),
        _tube_side_step(r, system),
    ]
    shell_s, tube_s = (i.hot, i.cold) if i.shell_side == "hot" else (i.cold, i.hot)
    steps.append(
        _u_step(
            r.u_dirty,
            i.d_i_m,
            i.d_o_m,
            i.k_wall_W_per_mK,
            tube_s.R_f_m2K_per_W,
            shell_s.R_f_m2K_per_W,
            r.tube.h_W_per_m2K,
            r.h_shell_W_per_m2K,
            r.U_D_W_per_m2K,
            system,
            "en los tubos",
            "en el casco",
            title="U de diseño, con el ensuciamiento",
        )
    )
    if r.u_dirty.R_total_K_per_W > r.u_clean.R_total_K_per_W:
        steps.append(
            ProcedureStep(
                "U limpio",
                "Sin el ensuciamiento: U_C, el que usa Kern para el criterio de R_d."
                if i.problem == "check"
                else "Sin el ensuciamiento: U_C, el del equipo recién limpio.",
                (
                    chain(
                        "U_C",
                        frac("1", r"R'_{\text{limpio}}\,\pi\,d_o"),
                        frac(
                            "1",
                            times(
                                _rlen(r.u_clean.R_total_K_per_W, system).split(r"\ ")[0],
                                r"\pi",
                                n(i.d_o_m, _L, system),
                            ),
                        ),
                        q(r.U_C_W_per_m2K, _H, system),
                    ),
                ),
            )
        )
    steps += _effectiveness(r.exchanger, stream_name(i.hot), stream_name(i.cold), system)
    if i.problem == "check":
        steps.append(_kern_check_step(r, system))
    if bell:
        steps += [_bd_pressure_step(r, system), _dp_tubes_only(r, system)]
    else:
        steps.append(_dp_shell_tube(r, system))
    return numbered(steps)

"""Procedimiento de la psicrometría «como en el pizarrón» — Fase 4.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo, con las fórmulas del vademecum §14 (Çengel & Boles §14-1 a
§14-7):

- **estado del aire húmedo** (:func:`moist_air_steps`): presión (por altura),
  presión de saturación (tablas A-4 o A-8), los datos del par, ω, φ, μ, h, v,
  ρ, R, el punto de rocío, el bulbo húmedo (saturación adiabática), la
  entropía y la exergía;
- **tren de procesos** (:func:`hvac_steps`): caudal de aire seco, cada proceso
  con sus balances de masa y energía (§14.12) y su exergía destruida;
- **torre de enfriamiento** (:func:`cooling_tower_steps`): los balances del
  §14.12.7 y la exergía.

Las ecuaciones con números se escriben con :func:`core.latex.latex_chain` (una
igualdad por renglón) y se cortan antes de un signo cuando llevan ×10ⁿ (SI) o
números negativos, para que entren en un celular. No importa Streamlit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.hvac import (
    AdiabaticHumidification,
    AdiabaticMixing,
    CoolingDehumidification,
    CoolingTowerResult,
    HeatingHumidification,
    HvacResult,
    HvacStream,
    ProcessResult,
    SensibleProcess,
)
from core.latex import (
    latex_chain,
    latex_is_wide,
    latex_number,
    latex_paren,
    latex_quantity,
    latex_unit,
    latex_value,
)
from core.psychrometrics import (
    C_P_AIR,
    C_P_VAPOR,
    EPSILON,
    H_VAPOR_0,
    MOLAR_RATIO,
    P_REF_AIR_PA,
    P_REF_VAPOR_PA,
    P_SEA_LEVEL_PA,
    R_AIR,
    R_VAPOR,
    S_VAPOR_0,
    T_REF_K,
    DeadState,
    MoistAirState,
    over_ice,
    room_masses,
    saturation_humidity_ratio,
    saturation_pressure,
    water_enthalpy,
    water_exergy,
)
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = ["cooling_tower_steps", "hvac_steps", "moist_air_steps"]

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"
_CP: QuantityKind = "specific_heat"
_P: QuantityKind = "pressure"
_DT: QuantityKind = "temperature_difference"
_EPS = latex_number(EPSILON, 4)
_MR = latex_number(MOLAR_RATIO, 4)


# ---------------------------------------------------------------------
# Números, unidades y símbolos
# ---------------------------------------------------------------------


def _q(value_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    return latex_quantity(value_si, kind, system)


def _n(value_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    return latex_value(value_si, kind, system)


def _dh(a_si: float, b_si: float, system: UnitSystem) -> str:
    """``a - b`` de dos entalpías en ``system`` (``b`` entre paréntesis si es negativa)."""
    return rf"{_n(a_si, _EH, system)} - {latex_paren(_n(b_si, _EH, system))}"


def _is_wide(*parts: str) -> bool:
    """Si algún número lleva ×10ⁿ o es negativo (también entre paréntesis)."""
    return any(r"\times" in p or re.search(r"(^|[(\s])-\s?\d", p) for p in parts)


def _lookup(lhs: str, middle: str, value: str) -> str:
    r"""``lhs = middle = value`` en un renglón; con números anchos, en dos (``aligned``)."""
    if _is_wide(middle, value):
        return latex_chain(lhs, middle, value)
    return f"{lhs} = {middle} = {value}"


def _h_lines(T_K: float, omega: float, h: float, system: UnitSystem, s: _Sym) -> list[str]:
    """h_a = c_p,a·t, h_v = r₀ + c_p,v·t y h = h_a + ω·h_v (vademecum §14.5)."""
    t = latex_paren(_t(T_K, system))
    h_a = C_P_AIR * (T_K - T_REF_K)
    h_v = H_VAPOR_0 + C_P_VAPOR * (T_K - T_REF_K)
    return [
        latex_chain(s.ha, rf"c_{{p,a}}\,{s.t}", rf"{_cpa(system)} \cdot {t}", _q(h_a, _EH, system)),
        latex_chain(
            s.hv,
            rf"r_0 + c_{{p,v}}\,{s.t}",
            rf"{_r0(system)} + {_cpv(system)} \cdot {t}",
            _q(h_v, _EH, system),
        ),
        latex_chain(
            s.h,
            rf"{s.ha} + {s.w}\,{s.hv}",
            _wrap(
                _n(h_a, _EH, system),
                "+",
                rf"{_w(omega)} \cdot {latex_paren(_n(h_v, _EH, system))}",
            ),
            _q(h, _EH, system),
        ),
    ]


def _w(omega: float) -> str:
    """Humedad absoluta (adimensional: kg de vapor por kg de aire seco), 4 cifras."""
    return latex_number(omega, 4)


def _pct(fraction: float) -> str:
    return rf"{latex_number(100.0 * fraction, 4)}\ \%"


def _abs(T_K: float, system: UnitSystem) -> str:
    """Temperatura absoluta (número): K en SI y Técnico, °R en Inglés."""
    return latex_number(T_K * 1.8 if system == "Inglés" else T_K, 5)


def _T(T_K: float, system: UnitSystem) -> str:
    return _q(T_K, "temperature", system)


def _t(T_K: float, system: UnitSystem) -> str:
    """t = T − 273,15 K (número, en las unidades de ΔT del sistema)."""
    return _n(T_K - T_REF_K, _DT, system)


def _t_q(T_K: float, system: UnitSystem) -> str:
    return rf"{_t(T_K, system)}\ {latex_unit(unit_label(_DT, system))}"


def _cpa(system: UnitSystem) -> str:
    return _n(C_P_AIR, _CP, system)


def _cpv(system: UnitSystem) -> str:
    return _n(C_P_VAPOR, _CP, system)


def _r0(system: UnitSystem) -> str:
    return _n(H_VAPOR_0, _EH, system)


def _R_pv(R_si: float, system: UnitSystem) -> tuple[str, str]:
    """R en unidades de p·v/T del sistema: (número, unidad en LaTeX).

    Así v = R·T/p sale sin factores de conversión, como hace Cengel con kPa:
    0,287 kJ/(kg·K) = 0,00287 bar·m³/(kg·K) = 0,3704 psia·ft³/(lb·°R).
    """
    p_per_pa = convert_from_si(1.0, _P, system)
    v_per_si = convert_from_si(1.0, "specific_volume", system)
    per_T = 1.0 / 1.8 if system == "Inglés" else 1.0
    value = R_si * p_per_pa * v_per_si * per_T
    units = {
        "SI": r"\mathrm{Pa\cdot m^{3}/(kg\cdot K)}",
        "Técnico": r"\mathrm{bar\cdot m^{3}/(kg\cdot K)}",
        "Inglés": r"\mathrm{psia\cdot ft^{3}/(lb\cdot R)}",
    }
    return latex_number(value, 4), units[system]


def _wrap(head: str, op: str, tail: str) -> str:
    r"""``head op tail``; con números anchos (×10ⁿ o negativos), ``op tail`` en otro renglón."""
    if latex_is_wide(head, tail):
        return rf"{head} \\ &\quad {op} {tail}"
    return f"{head} {op} {tail}"


def _table(T_K: float) -> str:
    return "tabla A-8, sobre hielo" if over_ice(T_K) else "tabla A-4"


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


@dataclass(frozen=True)
class _Sym:
    """Símbolos LaTeX de un estado (sin número o con subíndice)."""

    T: str
    t: str
    phi: str
    w: str
    h: str
    ha: str
    hv: str
    v: str
    pv: str
    pvs: str
    pa: str
    psi: str
    ma: str
    V: str


def _sym(n: int | None = None) -> _Sym:
    if n is None:
        return _Sym(
            "T",
            "t",
            r"\varphi",
            r"\omega",
            "h",
            "h_a",
            "h_v",
            "v",
            "p_v",
            "p_{vs}",
            "p_a",
            r"\psi",
            r"\dot m_a",
            r"\dot V",
        )
    k = str(n) if n < 10 else f"{{{n}}}"
    return _Sym(
        f"T_{k}",
        f"t_{k}",
        rf"\varphi_{k}",
        rf"\omega_{k}",
        f"h_{k}",
        f"h_{{a,{n}}}",
        f"h_{{v,{n}}}",
        f"v_{k}",
        f"p_{{v,{n}}}",
        f"p_{{vs,{n}}}",
        f"p_{{a,{n}}}",
        rf"\psi_{k}",
        rf"\dot m_{{a,{n}}}",
        rf"\dot V_{k}",
    )


# ---------------------------------------------------------------------
# Renglones reutilizables
# ---------------------------------------------------------------------


_STAR = _Sym(
    "T^{*}",
    "t^{*}",
    r"\varphi^{*}",
    r"\omega_s^{*}",
    "h^{*}",
    "h_a^{*}",
    "h_g^{*}",
    "v^{*}",
    "p_{vs}^{*}",
    "p_{vs}^{*}",
    "p_a^{*}",
    r"\psi^{*}",
    r"\dot m_a",
    r"\dot V",
)


def _lookup_pvs(T_K: float, system: UnitSystem, s: _Sym) -> str:
    """``p_vs = p_sat(T) = …`` (tabla A-4 o A-8)."""
    return _lookup(
        s.pvs,
        rf"p_{{\mathrm{{sat}}}}({_T(T_K, system)})",
        _q(saturation_pressure(T_K), _P, system),
    )


def _omega_chain(p_v: float, p: float, omega: float, system: UnitSystem, s: _Sym) -> str:
    """ω = 0,622·p_v/(p − p_v)."""
    return latex_chain(
        s.w,
        rf"{_EPS}\,\frac{{{s.pv}}}{{p - {s.pv}}}",
        rf"{_EPS}\,\frac{{{_n(p_v, _P, system)}}}{{{_n(p, _P, system)} - {_n(p_v, _P, system)}}}",
        _w(omega),
    )


def _pv_from_omega_chain(omega: float, p: float, system: UnitSystem, s: _Sym) -> str:
    """p_v = ω·p/(0,622 + ω)."""
    p_v = omega * p / (EPSILON + omega)
    return latex_chain(
        s.pv,
        rf"\frac{{{s.w}\,p}}{{{_EPS} + {s.w}}}",
        rf"\frac{{{_w(omega)} \cdot {_n(p, _P, system)}}}{{{_EPS} + {_w(omega)}}}",
        _q(p_v, _P, system),
    )


def _pv_from_phi_chain(state: MoistAirState, system: UnitSystem, s: _Sym) -> str:
    """p_v = φ·p_vs."""
    return latex_chain(
        s.pv,
        rf"{s.phi}\,{s.pvs}",
        rf"{latex_number(state.phi, 4)} \cdot {_n(state.p_vs_Pa, _P, system)}",
        _q(state.p_v_Pa, _P, system),
    )


def _phi_chain(state: MoistAirState, system: UnitSystem, s: _Sym) -> str:
    """φ = p_v/p_vs."""
    return latex_chain(
        s.phi,
        rf"\frac{{{s.pv}}}{{{s.pvs}}}",
        rf"\frac{{{_n(state.p_v_Pa, _P, system)}}}{{{_n(state.p_vs_Pa, _P, system)}}}",
        _pct(state.phi),
    )


def _v_chain(state: MoistAirState, system: UnitSystem, s: _Sym) -> str:
    """v = R_a·T/p_a, con R_a en unidades de p·v/T."""
    R_val, _ = _R_pv(R_AIR, system)
    return latex_chain(
        s.v,
        rf"\frac{{R_a\,{s.T}}}{{{s.pa}}}",
        rf"\frac{{{R_val} \cdot {_abs(state.T_K, system)}}}{{{_n(state.p_a_Pa, _P, system)}}}",
        _q(state.v_m3_per_kg, "specific_volume", system),
    )


def _constants_text(system: UnitSystem) -> str:
    return (
        f"Constantes del vademecum §14: c_p,a = {convert_from_si(C_P_AIR, _CP, system):.4g}, "
        f"c_p,v = {convert_from_si(C_P_VAPOR, _CP, system):.4g} {unit_label(_CP, system)} y "
        f"r₀ = {convert_from_si(H_VAPOR_0, _EH, system):.5g} {unit_label(_EH, system)}"
    )


def _wet_bulb_water(T_wb: float, system: UnitSystem) -> tuple[float, str, str]:
    """(h_w*, símbolo, renglón) del agua del bulbo húmedo: líquida o hielo."""
    if over_ice(T_wb):
        h_w = water_enthalpy(T_wb, "ice")
        sym = r"h_w^{*}"
        a, b = _n(-333.4e3, _EH, system), _n(2100.0, _CP, system)
        line = latex_chain(
            sym,
            rf"{a} + {b}\,t^{{*}}",
            rf"{a} \\ &\quad + {b} \cdot {latex_paren(_t(T_wb, system))}",
            _q(h_w, _EH, system),
        )
        return h_w, sym, line
    h_w = water_enthalpy(T_wb, "liquid")
    sym = r"h_w^{*}"
    return h_w, sym, _lookup(sym, rf"h_f({_T(T_wb, system)})", _q(h_w, _EH, system))


def _wb_text(T_wb: float) -> str:
    if over_ice(T_wb):
        return (
            "El bulbo húmedo está bajo 0 °C: el agua del bulbo se congela y su entalpía es la "
            "del hielo, h = −333,4 + 2,1·t kJ/kg (ASHRAE)."
        )
    return "h_w* es la del agua líquida a T_bh: h_f de la tabla A-4."


def _saturation_star_lines(T_wb: float, p: float, system: UnitSystem) -> list[str]:
    """p_vs*, ω_s* y h_w* a la temperatura de bulbo húmedo."""
    omega_s = saturation_humidity_ratio(T_wb, p)
    assert omega_s is not None
    p_vs = saturation_pressure(T_wb)
    _, _, h_line = _wet_bulb_water(T_wb, system)
    return [
        _lookup("p_{vs}^{*}", rf"p_{{\mathrm{{sat}}}}({_T(T_wb, system)})", _q(p_vs, _P, system)),
        _omega_chain(p_vs, p, omega_s, system, _STAR),
        h_line,
    ]


def _omega_from_wet_bulb_lines(
    T_K: float, T_wb: float, omega: float, p: float, system: UnitSystem
) -> list[str]:
    """ω con T y T_bh (Cengel ec. 14-14): ω = [c_p,a·(t* − t) + ω_s*·h_fg*]/(h_g − h_f*)."""
    omega_s = saturation_humidity_ratio(T_wb, p)
    assert omega_s is not None
    h_w, hw, _ = _wet_bulb_water(T_wb, system)
    fg = r"h_{ig}^{*}" if over_ice(T_wb) else r"h_{fg}^{*}"
    t, ts = latex_paren(_t(T_K, system)), latex_paren(_t(T_wb, system))
    h_w_n = latex_paren(_n(h_w, _EH, system))
    h_fg = H_VAPOR_0 + C_P_VAPOR * (T_wb - T_REF_K) - h_w
    h_g_minus = H_VAPOR_0 + C_P_VAPOR * (T_K - T_REF_K) - h_w
    num = (
        rf"{_cpa(system)}\,({_t(T_wb, system)} - {t}) + {_w(omega_s)} \cdot "
        rf"{_n(h_fg, _EH, system)}"
    )
    den = _n(h_g_minus, _EH, system)
    if _is_wide(num, den):
        numbers = (
            rf"\bigl[{_cpa(system)}\,({_t(T_wb, system)} - {t}) \\ &\quad + {_w(omega_s)} \cdot "
            rf"{_n(h_fg, _EH, system)}\bigr] \\ &\quad /\,{den}"
        )
    else:
        numbers = rf"\frac{{{num}}}{{{den}}}"
    return [
        *_saturation_star_lines(T_wb, p, system),
        latex_chain(
            fg,
            rf"r_0 + c_{{p,v}}\,t^{{*}} - {hw}",
            rf"{_r0(system)} + {_cpv(system)} \cdot {ts} \\ &\quad - {h_w_n}",
            _q(h_fg, _EH, system),
        ),
        latex_chain(
            r"h_g - h_w^{*}",
            rf"r_0 + c_{{p,v}}\,t - {hw}",
            rf"{_r0(system)} + {_cpv(system)} \cdot {t} \\ &\quad - {h_w_n}",
            _q(h_g_minus, _EH, system),
        ),
        latex_chain(
            r"\omega",
            rf"\frac{{c_{{p,a}}\,(t^{{*}} - t) + \omega_s^{{*}}\,{fg}}}{{h_g - h_w^{{*}}}}",
            numbers,
            _w(omega),
        ),
    ]


# ---------------------------------------------------------------------
# Estado del aire húmedo
# ---------------------------------------------------------------------

_NAMES_TEX = {
    "T": "T",
    "phi": r"\varphi",
    "T_wb": r"T_{bh}",
    "T_dp": r"T_{pr}",
    "omega": r"\omega",
    "h": "h",
}


def _given_line(name: str, value: float, system: UnitSystem) -> str:
    sym = _NAMES_TEX[name]
    if name in ("T", "T_wb", "T_dp"):
        return rf"{sym} = {_T(value, system)}"
    if name == "phi":
        return rf"{sym} = {_pct(value)}"
    if name == "omega":
        return rf"{sym} = {_w(value)}"
    return rf"{sym} = {_q(value, _EH, system)}"


def _pressure_step(
    state: MoistAirState, system: UnitSystem, altitude_m: float | None
) -> ProcedureStep:
    if altitude_m is None:
        return ProcedureStep(
            title="Presión total",
            text=(
                "La presión total es la del aire seco más la del vapor: p = p_a + p_v "
                "(vademecum §14.1, ley de Dalton)."
            ),
            latex=(rf"p = p_a + p_v = {_q(state.p_Pa, _P, system)}",),
        )
    z = latex_number(altitude_m, 5)
    base = 1.0 - 2.25577e-5 * altitude_m
    return ProcedureStep(
        title="Presión total a la altura del lugar",
        text=(
            "La presión atmosférica normal baja con la altura Z, en metros (atmósfera estándar; "
            "ASHRAE HoF 2017, cap. 1, ec. 3), desde p₀ al nivel del mar. La presión total es la "
            "del aire seco más la del vapor: p = p_a + p_v (vademecum §14.1)."
        ),
        latex=(
            rf"p_0 = {_q(P_SEA_LEVEL_PA, _P, system)},\quad Z = {z}\ \mathrm{{m}}",
            rf"1 - 2.25577\times 10^{{-5}} \cdot {z} = {latex_number(base, 5)}",
            latex_chain(
                "p",
                r"p_0\,(1 - 2.25577\times 10^{-5}\,Z)^{5.2559}",
                rf"{_n(P_SEA_LEVEL_PA, _P, system)} \cdot {latex_number(base, 5)}^{{5.2559}}",
                _q(state.p_Pa, _P, system),
            ),
        ),
    )


def _inputs_step(
    state: MoistAirState, system: UnitSystem, pair: tuple[str, str], given: tuple[float, float]
) -> ProcedureStep:
    """Cómo salen T, ω y φ de los dos datos (vademecum §14.2 a §14.5)."""
    values = dict(zip(pair, given, strict=True))
    names = set(pair)
    p, T, omega = state.p_Pa, state.T_K, state.omega
    s = _sym()
    lines = [_given_line(name, values[name], system) for name in pair]
    pv_dp = rf"{s.pv} = p_{{\mathrm{{sat}}}}(T_{{pr}}) = {_q(state.p_v_Pa, _P, system)}"
    if names == {"T", "phi"}:
        lines += [
            _lookup_pvs(T, system, s),
            _pv_from_phi_chain(state, system, s),
            _omega_chain(state.p_v_Pa, p, omega, system, s),
        ]
        text = (
            f"Con T sale la presión de saturación ({_table(T)}); la humedad relativa da la "
            "presión parcial del vapor, φ = p_v/p_vs (§14.3), y con ella la humedad absoluta "
            "(§14.2)."
        )
    elif names == {"T", "T_wb"}:
        lines += _omega_from_wet_bulb_lines(T, values["T_wb"], omega, p, system)
        lines += [
            _pv_from_omega_chain(omega, p, system, s),
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = (
            "El psicrómetro mide la temperatura de bulbo húmedo, que es la de saturación "
            "adiabática (§14.12.5): el aire se satura con agua a T_bh sin intercambiar calor. Del "
            "balance h + (ω_s* − ω)·h_f* = h* se despeja ω (Cengel ec. 14-14); el * indica "
            f"T_bh. {_wb_text(values['T_wb'])}"
        )
    elif names == {"T", "T_dp"}:
        lines += [
            pv_dp,
            _omega_chain(state.p_v_Pa, p, omega, system, s),
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = (
            "En el punto de rocío el vapor se satura sin cambiar su presión parcial: p_v es la "
            f"de saturación a T_pr ({_table(values['T_dp'])})."
        )
    elif names == {"T", "omega"}:
        lines += [
            _pv_from_omega_chain(omega, p, system, s),
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = "De ω = 0,622·p_v/(p − p_v) se despeja p_v (§14.2); φ = p_v/p_vs (§14.3)."
    elif names == {"T", "h"}:
        t = latex_paren(_t(T, system))
        lines += [
            latex_chain(
                r"\omega",
                r"\frac{h - c_{p,a}\,t}{r_0 + c_{p,v}\,t}",
                rf"\frac{{{_n(state.h_J_per_kg, _EH, system)} - {_cpa(system)} \cdot {t}}}"
                rf"{{{_r0(system)} + {_cpv(system)} \cdot {t}}}",
                _w(omega),
            ),
            _pv_from_omega_chain(omega, p, system, s),
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = "De h = c_p,a·t + ω·(r₀ + c_p,v·t) se despeja ω (§14.5)."
    elif names == {"h", "phi"}:
        lines += [
            rf"T = {_T(T, system)}",
            _lookup_pvs(T, system, s),
            _pv_from_phi_chain(state, system, s),
            _omega_chain(state.p_v_Pa, p, omega, system, s),
            *_h_lines(T, omega, state.h_J_per_kg, system, s),
        ]
        text = (
            "T no está entre los datos y la entalpía depende de T también a través de p_vs(T): "
            "se busca numéricamente la T con la que h(T, ω(T, φ)) da la entalpía dada (en la "
            "carta, el cruce de la recta de h con la curva de φ). Abajo, la verificación con la "
            "T encontrada."
        )
    elif names == {"T_wb", "phi"}:
        lines.append(rf"T = {_T(T, system)}")
        lines += _omega_from_wet_bulb_lines(T, values["T_wb"], omega, p, system)
        lines += [
            _pv_from_omega_chain(omega, p, system, s),
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = (
            "T no está entre los datos: se busca numéricamente la T con la que la ω de saturación "
            "adiabática (Cengel ec. 14-14) da la φ pedida (en la carta, el cruce de la línea de "
            f"T_bh con la curva de φ). Abajo, la verificación. {_wb_text(values['T_wb'])}"
        )
    elif names in ({"omega", "phi"}, {"T_dp", "phi"}):
        if "omega" in names:
            lines.append(_pv_from_omega_chain(omega, p, system, s))
        else:
            lines += [pv_dp, _omega_chain(state.p_v_Pa, p, omega, system, s)]
        lines += [
            latex_chain(
                s.pvs,
                rf"\frac{{{s.pv}}}{{{s.phi}}}",
                rf"\frac{{{_n(state.p_v_Pa, _P, system)}}}{{{latex_number(state.phi, 4)}}}",
                _q(state.p_vs_Pa, _P, system),
            ),
            rf"T = T_{{\mathrm{{sat}}}}({s.pvs}) = {_T(T, system)}",
        ]
        text = (
            "Con la humedad sale p_v; φ = p_v/p_vs da la presión de saturación, y T es la "
            f"temperatura a la que el agua satura a esa presión ({_table(T)})."
        )
    elif names in ({"h", "omega"}, {"h", "T_dp"}):
        if "T_dp" in names:
            lines += [pv_dp, _omega_chain(state.p_v_Pa, p, omega, system, s)]
        else:
            lines.append(_pv_from_omega_chain(omega, p, system, s))
        lines += [
            _t_from_h_chain(state.h_J_per_kg, omega, T, system, s),
            rf"T = {_T(T, system)}",
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = (
            "De h = c_p,a·t + ω·(r₀ + c_p,v·t) se despeja t = T − 273,15 K (§14.5): con ω "
            "conocida es lineal."
        )
    else:  # T_bh con ω o con T_pr
        T_wb = values["T_wb"]
        _, hw, _ = _wet_bulb_water(T_wb, system)
        if "T_dp" in names:
            lines += [pv_dp, _omega_chain(state.p_v_Pa, p, omega, system, s)]
        else:
            lines.append(_pv_from_omega_chain(omega, p, system, s))
        lines += _saturation_star_lines(T_wb, p, system)
        lines += [
            latex_chain(
                "t",
                rf"\bigl[c_{{p,a}}\,t^{{*}} + \omega_s^{{*}}\,(r_0 + c_{{p,v}}\,t^{{*}} - {hw}) \\ "
                rf"&\quad - \omega\,(r_0 - {hw})\bigr]\,/\,(c_{{p,a}} + \omega\,c_{{p,v}})",
                _t_q(T, system),
            ),
            rf"T = {_T(T, system)}",
            _lookup_pvs(T, system, s),
            _phi_chain(state, system, s),
        ]
        text = (
            "Con la humedad conocida, el balance de saturación adiabática "
            "h + (ω_s* − ω)·h_w* = h* (§14.12.5) es lineal en t: se despeja la temperatura de "
            f"bulbo seco. {_wb_text(T_wb)}"
        )
    return ProcedureStep(
        title="Humedad absoluta y relativa con los datos", text=text, latex=tuple(lines)
    )


def _saturation_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    p = state.p_Pa
    lines = [
        latex_chain(
            "p_a",
            "p - p_v",
            rf"{_n(p, _P, system)} - {_n(state.p_v_Pa, _P, system)}",
            _q(state.p_a_Pa, _P, system),
        )
    ]
    omega_s = state.omega_s
    if omega_s is not None:
        sat = _Sym(
            "T",
            "t",
            r"\varphi",
            r"\omega_s",
            "h",
            "h_a",
            "h_v",
            "v",
            "p_{vs}",
            "p_{vs}",
            "p_a",
            r"\psi",
            r"\dot m_a",
            r"\dot V",
        )
        lines += [
            _omega_chain(state.p_vs_Pa, p, omega_s, system, sat),
            latex_chain(
                r"\mu",
                r"\frac{\omega}{\omega_s}",
                rf"\frac{{{_w(state.omega)}}}{{{_w(omega_s)}}}",
                _pct(state.mu or 0.0),
            ),
        ]
        text = (
            "ω_s es la humedad máxima a esa T (aire saturado) y μ = ω/ω_s, el grado de saturación "
            "(§14.4): casi igual a φ."
        )
    else:
        text = (
            "A esta T el agua hierve a la presión total: el aire no se puede saturar (ω_s no "
            "existe) y φ queda por debajo de p/p_vs."
        )
    return ProcedureStep(
        title="Presión del aire seco y grado de saturación",
        text="La presión del aire seco es lo que falta para llegar a la total (§14.1). " + text,
        latex=tuple(lines),
    )


def _enthalpy_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    return ProcedureStep(
        title="Entalpía",
        text=(
            "Por kg de aire seco: la del aire seco más la del vapor, "
            "h = c_p,a·t + ω·(r₀ + c_p,v·t), "
            "con t = T − 273,15 K: h = 0 para el aire seco a 0 °C y para el agua líquida en el "
            f"punto triple (§14.5). {_constants_text(system)}."
        ),
        latex=tuple(_h_lines(state.T_K, state.omega, state.h_J_per_kg, system, _sym())),
    )


def _volume_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    R_val, R_unit = _R_pv(R_AIR, system)
    return ProcedureStep(
        title="Volumen específico, densidad, constante y calor específico",
        text=(
            "El volumen por kg de aire seco sale de la ecuación de estado del aire seco a su "
            "presión parcial, v = R_a·T/p_a (§14.7), con R_a en unidades de presión por volumen; "
            "la densidad cuenta también el vapor (§14.8), R_ah es la constante del aire húmedo "
            "(§14.9) y c_p,ah = c_p,a + ω·c_p,v, por kg de aire seco (§14.6)."
        ),
        latex=(
            rf"R_a = {R_val}\ {R_unit}",
            _v_chain(state, system, _sym()),
            latex_chain(
                r"\rho",
                r"\frac{1 + \omega}{v}",
                rf"\frac{{1 + {_w(state.omega)}}}"
                rf"{{{_n(state.v_m3_per_kg, 'specific_volume', system)}}}",
                _q(state.rho_kg_per_m3, "density", system),
            ),
            latex_chain(
                r"R_{ah}",
                rf"R_a\,\frac{{1 + {_MR}\,\omega}}{{1 + \omega}}",
                _q(state.R_J_per_kg_K, _CP, system),
            ),
            latex_chain(
                r"c_{p,ah}",
                r"c_{p,a} + \omega\,c_{p,v}",
                _q(state.cp_J_per_kg_K, _CP, system),
            ),
        ),
    )


def _dew_point_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    if state.T_dp_K is None:
        return ProcedureStep(
            title="Temperatura de punto de rocío",
            text="Es aire seco (ω = 0): no tiene punto de rocío.",
        )
    frost = over_ice(state.T_dp_K)
    tail = (
        "Está bajo 0 °C: el vapor pasa directo a hielo (escarcha; tabla A-8)."
        if frost
        else "Por debajo de ella, el vapor condensa (tabla A-4)."
    )
    return ProcedureStep(
        title="Temperatura de punto de rocío" + (" (de escarcha)" if frost else ""),
        text=(
            "Si el aire se enfría a presión constante, el vapor se satura a la temperatura de "
            f"saturación de su presión parcial (Cengel §14-3). {tail}"
        ),
        latex=(
            latex_chain(
                r"T_{pr}",
                r"T_{\mathrm{sat}}(p_v)",
                rf"T_{{\mathrm{{sat}}}}({_q(state.p_v_Pa, _P, system)})",
                _T(state.T_dp_K, system),
            ),
        ),
    )


def _wet_bulb_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    T_wb = state.T_wb_K
    omega_s = saturation_humidity_ratio(T_wb, state.p_Pa)
    assert omega_s is not None
    h_w, hw, _ = _wet_bulb_water(T_wb, system)
    h_star = state.h_J_per_kg + (omega_s - state.omega) * h_w
    return ProcedureStep(
        title="Temperatura de bulbo húmedo (saturación adiabática)",
        text=(
            "Es la temperatura a la que el aire queda saturado si se le agrega agua a esa misma "
            "temperatura sin intercambiar calor (§14.12.5): h* = h + (ω_s* − ω)·h_w*, con * a "
            "T_bh. La ecuación es implícita (T_bh aparece en ω_s*, h_w* y h*): se busca "
            "numéricamente. Abajo, la verificación: h* del balance y h* del aire saturado a T_bh "
            f"dan lo mismo. {_wb_text(T_wb)}"
        ),
        latex=(
            rf"T_{{bh}} = {_T(T_wb, system)}",
            *_saturation_star_lines(T_wb, state.p_Pa, system),
            latex_chain(
                r"h^{*}",
                rf"h + (\omega_s^{{*}} - \omega)\,{hw}",
                rf"{_n(state.h_J_per_kg, _EH, system)} \\ &\quad + ({_w(omega_s)} - "
                rf"{_w(state.omega)}) \\ &\qquad \cdot {latex_paren(_n(h_w, _EH, system))}",
                _q(h_star, _EH, system),
            ),
            *_h_lines(T_wb, omega_s, h_star, system, _STAR),
        ),
    )


def _entropy_step(state: MoistAirState, system: UnitSystem) -> ProcedureStep:
    T = _abs(state.T_K, system)
    T_ref = _abs(T_REF_K, system)
    Ra = _n(R_AIR, _CP, system)
    s_a = state.s_air
    lines = [
        latex_chain(
            r"s_a",
            r"c_{p,a}\ln\frac{T}{T_{\mathrm{ref}}} - R_a\ln\frac{p_a}{p_{\mathrm{ref},a}}",
            rf"{_cpa(system)}\ln\frac{{{T}}}{{{T_ref}}} \\ &\quad - {Ra}\ln"
            rf"\frac{{{_n(state.p_a_Pa, _P, system)}}}{{{_n(P_REF_AIR_PA, _P, system)}}}",
            _q(s_a, _ES, system),
        )
    ]
    s_v = state.s_vapor
    if s_v is not None:
        lines += [
            latex_chain(
                r"s_v",
                r"s_0 + c_{p,v}\ln\frac{T}{T_{\mathrm{ref}}} \\ &\quad"
                r" - R_v\ln\frac{p_v}{p_{\mathrm{ref},v}}",
                rf"{_n(S_VAPOR_0, _ES, system)} + {_cpv(system)}\ln\frac{{{T}}}{{{T_ref}}}"
                rf" \\ &\quad - {_n(R_VAPOR, _CP, system)}\ln"
                rf"\frac{{{_n(state.p_v_Pa, _P, system)}}}"
                rf"{{{_n(P_REF_VAPOR_PA, _P, system)}}}",
                _q(s_v, _ES, system),
            ),
            latex_chain(
                "s",
                r"s_a + \omega\,s_v",
                _wrap(
                    _n(s_a, _ES, system),
                    "+",
                    rf"{_w(state.omega)} \cdot {latex_paren(_n(s_v, _ES, system))}",
                ),
                _q(state.s_J_per_kg_K, _ES, system),
            ),
        ]
    return ProcedureStep(
        title="Entropía",
        text=(
            "Por kg de aire seco, s = s_a + ω·s_v, cada gas a su presión parcial (§14.10), con "
            "T_ref = 273,15 K. Referencias de ASHRAE: s = 0 para el aire seco a 0 °C y "
            "101,325 kPa y para el agua líquida en el punto triple; por eso aparecen s₀, la "
            "entropía del vapor saturado en el punto triple, y p_ref,v = 611,657 Pa. R_v = "
            "1,608·R_a = 0,4615 kJ/(kg·K) (Cengel A-1)."
        ),
        latex=tuple(lines),
    )


def _exergy_step(state: MoistAirState, dead: DeadState, system: UnitSystem) -> ProcedureStep:
    thermo, mixing = state.psi_parts(dead)
    T, T0 = _abs(state.T_K, system), _abs(dead.T0_K, system)
    cp_mix = _n(state.cp_J_per_kg_K, _CP, system)
    Ra = _n(R_AIR, _CP, system)
    nu = 1.0 + MOLAR_RATIO * state.omega
    nu0 = 1.0 + MOLAR_RATIO * dead.omega0
    pressure_term = ""
    if not math.isclose(state.p_Pa, dead.p0_Pa, rel_tol=1e-9):
        pressure_term = (
            rf" \\ &\quad + {latex_number(nu, 5)} \cdot {Ra} \cdot {T0} \\ &\qquad \cdot \ln"
            rf"\frac{{{_n(state.p_Pa, _P, system)}}}{{{_n(dead.p0_Pa, _P, system)}}}"
        )
    chem = (
        rf" \\ &\quad + {latex_number(nu - 1.0, 4)}\ln\frac{{{_w(state.omega)}}}"
        rf"{{{_w(dead.omega0)}}}\Bigr]"
        if state.omega > 0.0
        else r"\Bigr]"
    )
    return ProcedureStep(
        title="Exergía de flujo",
        text=(
            "Es el trabajo máximo por kg de aire seco al llevar el aire al equilibrio con el "
            "ambiente (vademecum §14.11; Wepfer, Gaggioli y Obert, 1979): ψ_tm lo lleva a T₀ y p₀ "
            "con la misma humedad (parte térmica y mecánica), y ψ_qu (química, de mezcla) "
            "iguala su humedad con la del ambiente. ν = 1 + 1,608·ω son los moles de aire húmedo "
            "por mol de aire seco, y c_p,ah = c_p,a + ω·c_p,v. Un aire más seco que el ambiente "
            "también tiene exergía: con él se puede enfriar por evaporación."
        ),
        latex=(
            rf"T_0 = {_T(dead.T0_K, system)},\quad \varphi_0 = {_pct(dead.phi0)}",
            rf"\omega_0 = {_w(dead.omega0)}",
            rf"p_0 = {_q(dead.p0_Pa, _P, system)}",
            latex_chain(r"\nu", rf"1 + {_MR}\,\omega", latex_number(nu, 5)),
            latex_chain(r"\nu_0", rf"1 + {_MR}\,\omega_0", latex_number(nu0, 5)),
            latex_chain(
                r"\psi_{tm}",
                r"c_{p,ah}\Bigl[(T - T_0) - T_0\ln\frac{T}{T_0}\Bigr] \\ &\quad"
                r" + \nu\,R_a\,T_0\ln\frac{p}{p_0}",
                rf"{cp_mix}\,\Bigl[({T} - {T0}) \\ &\quad - {T0}\ln\frac{{{T}}}{{{T0}}}\Bigr]"
                + pressure_term,
                _q(thermo, _EH, system),
            ),
            latex_chain(
                r"\psi_{qu}",
                r"R_a\,T_0\Bigl[\nu\ln\frac{\nu_0}{\nu} \\ &\quad + (\nu - 1)\ln"
                r"\frac{\omega}{\omega_0}\Bigr]",
                rf"{Ra} \cdot {T0} \\ &\quad \cdot \Bigl[{latex_number(nu, 5)}\ln"
                rf"\frac{{{latex_number(nu0, 5)}}}{{{latex_number(nu, 5)}}}" + chem,
                _q(mixing, _EH, system),
            ),
            latex_chain(
                r"\psi",
                r"\psi_{tm} + \psi_{qu}",
                _wrap(_n(thermo, _EH, system), "+", latex_paren(_n(mixing, _EH, system))),
                _q(thermo + mixing, _EH, system),
            ),
        ),
    )


def _room_step(state: MoistAirState, volume_m3: float, system: UnitSystem) -> ProcedureStep:
    m_a, m_v = room_masses(state, volume_m3)
    R_val, _ = _R_pv(R_AIR, system)
    if system == "Inglés":
        V, V_unit = volume_m3 / 0.3048**3, r"\mathrm{ft^{3}}"
        m_a, m_v, m_unit = m_a / 0.45359237, m_v / 0.45359237, r"\mathrm{lb}"
    else:
        V, V_unit, m_unit = volume_m3, r"\mathrm{m^{3}}", r"\mathrm{kg}"
    return ProcedureStep(
        title="Masas de aire seco y de vapor en el recinto",
        text=(
            "El aire seco ocupa todo el volumen a su presión parcial: m_a = p_a·V/(R_a·T); el "
            "vapor, también: su masa es ω veces la del aire seco."
        ),
        latex=(
            rf"V = {latex_number(V, 5)}\ {V_unit}",
            latex_chain(
                "m_a",
                r"\frac{p_a\,V}{R_a\,T}",
                rf"\frac{{{_n(state.p_a_Pa, _P, system)} \cdot {latex_number(V, 5)}}}"
                rf"{{{R_val} \cdot {_abs(state.T_K, system)}}}",
                rf"{latex_number(m_a, 5)}\ {m_unit}",
            ),
            latex_chain(
                "m_v",
                r"\omega\,m_a",
                rf"{_w(state.omega)} \cdot {latex_number(m_a, 5)}",
                rf"{latex_number(m_v, 5)}\ {m_unit}",
            ),
        ),
    )


def moist_air_steps(
    state: MoistAirState,
    system: UnitSystem,
    *,
    pair: tuple[str, str],
    given: tuple[float, float],
    dead: DeadState | None = None,
    altitude_m: float | None = None,
    room_volume_m3: float | None = None,
) -> list[ProcedureStep]:
    """Pasos del estado del aire húmedo, en el orden del vademecum §14."""
    names = set(pair)
    steps = [
        _pressure_step(state, system, altitude_m),
        _inputs_step(state, system, pair, given),
        _saturation_step(state, system),
    ]
    if "h" not in names:
        steps.append(_enthalpy_step(state, system))
    steps.append(_volume_step(state, system))
    if "T_dp" not in names:
        steps.append(_dew_point_step(state, system))
    if "T_wb" not in names:
        steps.append(_wet_bulb_step(state, system))
    steps.append(_entropy_step(state, system))
    if dead is not None:
        steps.append(_exergy_step(state, dead, system))
    if room_volume_m3 is not None:
        steps.append(_room_step(state, room_volume_m3, system))
    return steps


# ---------------------------------------------------------------------
# Tren de procesos
# ---------------------------------------------------------------------


def _stream_state_lines(stream: HvacStream, system: UnitSystem) -> list[str]:
    """Estado de una corriente dada por T y φ: p_vs, p_v, ω y h."""
    st = stream.state
    s = _sym(stream.number)
    return [
        rf"{s.T} = {_T(st.T_K, system)},\quad {s.phi} = {_pct(st.phi)}",
        _lookup_pvs(st.T_K, system, s),
        _pv_from_phi_chain(st, system, s),
        _omega_chain(st.p_v_Pa, st.p_Pa, st.omega, system, s),
        *_h_lines(st.T_K, st.omega, st.h_J_per_kg, system, s),
    ]


def _flow_lines(stream: HvacStream, by_volume: bool, system: UnitSystem) -> list[str]:
    """v y el caudal de aire seco de una corriente."""
    st = stream.state
    s = _sym(stream.number)
    lines = [
        latex_chain(
            s.pa,
            f"p - {s.pv}",
            rf"{_n(st.p_Pa, _P, system)} - {_n(st.p_v_Pa, _P, system)}",
            _q(st.p_a_Pa, _P, system),
        ),
        _v_chain(st, system, s),
    ]
    if by_volume:
        lines.append(
            latex_chain(
                s.ma,
                rf"\frac{{{s.V}}}{{{s.v}}}",
                rf"\frac{{{_n(stream.V_m3_s, 'volume_flow', system)}}}"
                rf"{{{_n(st.v_m3_per_kg, 'specific_volume', system)}}}",
                _q(stream.m_dry_air_kg_s, "mass_flow", system),
            )
        )
    else:
        lines.append(rf"{s.ma} = {_q(stream.m_dry_air_kg_s, 'mass_flow', system)}")
    return lines


def _water_line(proc: ProcessResult, system: UnitSystem) -> str:
    h_w = proc.h_water_J_per_kg or 0.0
    T_w = proc.T_water_K or proc.outlet.state.T_K
    phase = "g" if proc.water_phase == "vapor" else "f"
    return _lookup("h_w", rf"h_{phase}({_T(T_w, system)})", _q(h_w, _EH, system))


def _exergy_lines(proc: ProcessResult, dead: DeadState, system: UnitSystem) -> list[str]:
    """X_dest: Σṁψ que entra − Σṁψ que sale, más el agua y el calor."""
    si, so = _sym(proc.inlet.number), _sym(proc.outlet.number)
    lines = [rf"{si.psi} = {_q(proc.inlet.state.psi(dead), _EH, system)}"]
    if proc.other is not None:
        sk = _sym(proc.other.number)
        lines.append(rf"{sk.psi} = {_q(proc.other.state.psi(dead), _EH, system)}")
    lines.append(rf"{so.psi} = {_q(proc.outlet.state.psi(dead), _EH, system)}")
    psi_in = _n(proc.inlet.state.psi(dead), _EH, system)
    psi_out = latex_paren(_n(proc.outlet.state.psi(dead), _EH, system))
    if proc.other is not None:
        sk = _sym(proc.other.number)
        terms = [rf"{si.ma}\,{si.psi} + {sk.ma}\,{sk.psi} \\ &\quad - {so.ma}\,{so.psi}"]
        numbers = [
            rf"{_n(proc.inlet.m_dry_air_kg_s, 'mass_flow', system)} \cdot {latex_paren(psi_in)}"
            rf" \\ &\quad + {_n(proc.other.m_dry_air_kg_s, 'mass_flow', system)} \cdot "
            rf"{latex_paren(_n(proc.other.state.psi(dead), _EH, system))}"
            rf" \\ &\quad - {_n(proc.outlet.m_dry_air_kg_s, 'mass_flow', system)} \cdot {psi_out}"
        ]
    else:
        terms = [rf"\dot m_a\,({si.psi} - {so.psi})"]
        m = _n(proc.m_dry_air_kg_s, "mass_flow", system)
        numbers = [_wrap(rf"{m}\,({psi_in}", "-", rf"{psi_out})")]
    if proc.m_water_kg_s != 0.0 and proc.T_water_K is not None and proc.water_phase is not None:
        psi_w = water_exergy(proc.T_water_K, proc.water_phase, dead)
        lines.append(
            latex_chain(
                r"\psi_w",
                r"(h_w - h_{g0}) - T_0\,(s_w - s_{g0}) \\ &\quad + R_v\,T_0\ln(1/\varphi_0)",
                _q(psi_w, _EH, system),
            )
        )
        sign = "+" if proc.m_water_kg_s > 0.0 else "-"
        terms.append(rf"{sign} \dot m_w\,\psi_w")
        numbers.append(
            rf"{sign} {_n(abs(proc.m_water_kg_s), 'mass_flow', system)} \cdot "
            rf"{latex_paren(_n(psi_w, _EH, system))}"
        )
    if proc.Q_W != 0.0 and proc.T_source_K is not None:
        lines.append(rf"T_b = {_T(proc.T_source_K, system)}")
        Q = latex_paren(_n(proc.Q_W, "power", system))
        factor = rf"(1 - {_abs(dead.T0_K, system)}/{_abs(proc.T_source_K, system)})"
        lines.append(
            latex_chain(
                r"\dot X_Q",
                r"\dot Q\,(1 - T_0/T_b)",
                rf"{Q} \\ &\quad \cdot {factor}" if _is_wide(Q) else rf"{Q}\,{factor}",
                _q(proc.X_heat_W, "power", system),
            )
        )
        terms.append(r"+ \dot X_Q")
        numbers.append(rf"+ {latex_paren(_n(proc.X_heat_W, 'power', system))}")
    lines.append(
        latex_chain(
            r"\dot X_{\mathrm{dest}}",
            r" \\ &\quad ".join(terms),
            r" \\ &\quad ".join(numbers),
            _q(proc.X_destroyed_W, "power", system),
        )
    )
    return lines


def _sensible_lines(proc: ProcessResult, system: UnitSystem) -> tuple[list[str], str]:
    si, so = _sym(proc.inlet.number), _sym(proc.outlet.number)
    s1, s2 = proc.inlet.state, proc.outlet.state
    m = _n(proc.m_dry_air_kg_s, "mass_flow", system)
    lines = [
        rf"{so.w} = {si.w} = {_w(s1.omega)}",
        rf"{so.T} = {_T(s2.T_K, system)}",
        *_h_lines(s2.T_K, s2.omega, s2.h_J_per_kg, system, so),
        latex_chain(
            r"\dot Q",
            rf"\dot m_a\,({so.h} - {si.h})",
            rf"{m}\,({_dh(s2.h_J_per_kg, s1.h_J_per_kg, system)})",
            _q(proc.Q_W, "power", system),
        ),
        _lookup_pvs(s2.T_K, system, so),
        latex_chain(
            so.phi,
            rf"\frac{{{si.pv}}}{{{so.pvs}}}",
            rf"\frac{{{_n(s2.p_v_Pa, _P, system)}}}{{{_n(s2.p_vs_Pa, _P, system)}}}",
            _pct(s2.phi),
        ),
    ]
    verb = "calienta" if proc.Q_W > 0 else "enfría"
    text = (
        f"El aire se {verb} sin cambiar su humedad (vademecum §14.12.1): ω y p_v no cambian, pero "
        "φ sí, porque cambia p_vs(T). Masa de aire seco: ṁ_a constante. Energía: "
        "Q̇ = ṁ_a·(h₂ − h₁)."
    )
    return lines, text


def _humidify_lines(proc: ProcessResult, system: UnitSystem) -> tuple[list[str], str]:
    spec = proc.spec
    assert isinstance(spec, HeatingHumidification)
    si, so = _sym(proc.inlet.number), _sym(proc.outlet.number)
    s1, s2 = proc.inlet.state, proc.outlet.state
    m = _n(proc.m_dry_air_kg_s, "mass_flow", system)
    h_w = proc.h_water_J_per_kg or 0.0
    lines = [
        *_stream_state_lines(proc.outlet, system),
        latex_chain(
            r"\dot m_w",
            rf"\dot m_a\,({so.w} - {si.w})",
            rf"{m}\,({_w(s2.omega)} - {_w(s1.omega)})",
            _q(proc.m_water_kg_s, "mass_flow", system),
        ),
        _water_line(proc, system),
        latex_chain(
            r"\dot Q",
            rf"\dot m_a\,({so.h} - {si.h}) - \dot m_w\,h_w",
            rf"{m}\,({_dh(s2.h_J_per_kg, s1.h_J_per_kg, system)})"
            rf" \\ &\quad - {_n(proc.m_water_kg_s, 'mass_flow', system)} \cdot "
            rf"{_n(h_w, _EH, system)}",
            _q(proc.Q_W, "power", system),
        ),
    ]
    water = "vapor saturado" if spec.water == "vapor" else "agua líquida"
    text = (
        f"El aire recibe calor y {water} a {_degC(spec.T_water_K)} (vademecum §14.12.2). Masa de "
        "agua: ṁ_w = ṁ_a·(ω₂ − ω₁). Energía: Q̇ + ṁ_w·h_w = ṁ_a·(h₂ − h₁), con h_w de tablas "
        "(Cengel A-4)."
    )
    return lines, text


def _coil_lines(proc: ProcessResult, system: UnitSystem) -> tuple[list[str], str]:
    si, so = _sym(proc.inlet.number), _sym(proc.outlet.number)
    s1, s2 = proc.inlet.state, proc.outlet.state
    m = _n(proc.m_dry_air_kg_s, "mass_flow", system)
    h_w = proc.h_water_J_per_kg or 0.0
    T_w = proc.T_water_K or s2.T_K
    m_w = -proc.m_water_kg_s
    lines = [
        *_stream_state_lines(proc.outlet, system),
        latex_chain(
            r"\dot m_w",
            rf"\dot m_a\,({si.w} - {so.w})",
            rf"{m}\,({_w(s1.omega)} - {_w(s2.omega)})",
            _q(m_w, "mass_flow", system),
        ),
        _water_line(proc, system),
        latex_chain(
            r"\dot Q",
            rf"\dot m_a\,({so.h} - {si.h}) + \dot m_w\,h_w",
            rf"{m}\,({_dh(s2.h_J_per_kg, s1.h_J_per_kg, system)})"
            rf" \\ &\quad + {_n(m_w, 'mass_flow', system)} \cdot {_n(h_w, _EH, system)}",
            _q(proc.Q_W, "power", system),
        ),
    ]
    text = (
        "El aire se enfría por debajo de su punto de rocío y parte del vapor condensa sobre el "
        f"serpentín; el condensado sale a {_degC(T_w)}, la temperatura de la superficie fría "
        "(vademecum §14.12.3). Masa de agua: ṁ_w = ṁ_a·(ω₁ − ω₂). Energía: "
        "Q̇ = ṁ_a·[(h₂ − h₁) + (ω₁ − ω₂)·h_w] = ṁ_a·(h₂ − h₁) + ṁ_w·h_w, negativo: el calor "
        "sale del aire."
    )
    return lines, text


def _adiabatic_lines(proc: ProcessResult, system: UnitSystem) -> tuple[list[str], str]:
    spec = proc.spec
    assert isinstance(spec, AdiabaticHumidification)
    si, so = _sym(proc.inlet.number), _sym(proc.outlet.number)
    s1, s2 = proc.inlet.state, proc.outlet.state
    m = _n(proc.m_dry_air_kg_s, "mass_flow", system)
    h_w = proc.h_water_J_per_kg or 0.0
    T_w = proc.T_water_K or s1.T_wb_K
    lines = [
        _water_line(proc, system),
        rf"{so.T} = {_T(s2.T_K, system)},\quad {so.phi} = {_pct(s2.phi)}",
        _lookup_pvs(s2.T_K, system, so),
        _pv_from_phi_chain(s2, system, so),
        _omega_chain(s2.p_v_Pa, s2.p_Pa, s2.omega, system, so),
        *_h_lines(s2.T_K, s2.omega, s2.h_J_per_kg, system, so),
        latex_chain(
            so.h,
            rf"{si.h} + ({so.w} - {si.w})\,h_w",
            rf"{_n(s1.h_J_per_kg, _EH, system)} \\ &\quad + ({_w(s2.omega)} - {_w(s1.omega)})"
            rf" \\ &\qquad \cdot {_n(h_w, _EH, system)}",
            _q(s1.h_J_per_kg + (s2.omega - s1.omega) * h_w, _EH, system),
        ),
        latex_chain(
            r"\dot m_w",
            rf"\dot m_a\,({so.w} - {si.w})",
            rf"{m}\,({_w(s2.omega)} - {_w(s1.omega)})",
            _q(proc.m_water_kg_s, "mass_flow", system),
        ),
    ]
    if spec.water == "liquid":
        water = f"agua líquida a {_degC(T_w)}"
        if spec.T_water_K is None:
            water += " (la que recircula llega a la temperatura de bulbo húmedo del aire)"
    else:
        water = f"vapor saturado a {_degC(T_w)}"
    text = (
        f"Sin calor (vademecum §14.12.4), con {water}: h₂ = h₁ + (ω₂ − ω₁)·h_w. La salida tiene "
        "la φ pedida y T₂ es la que cumple el balance (se busca numéricamente, porque ω₂ "
        "depende de T₂); abajo, la verificación: h₂ con la T encontrada y h₂ del balance dan lo "
        "mismo. Como (ω₂ − ω₁)·h_w es chico frente a h₁, el proceso casi sigue una recta de h "
        "constante en la carta."
    )
    return lines, text


def _t_from_h_chain(h: float, omega: float, T_K: float, system: UnitSystem, s: _Sym) -> str:
    """t = (h − ω·r₀)/(c_p,a + ω·c_p,v); con números anchos, el cociente en tres renglones."""
    h_n = latex_paren(_n(h, _EH, system))
    num = rf"{_n(h, _EH, system)} - {_w(omega)} \cdot {_r0(system)}"
    den = rf"{_cpa(system)} + {_w(omega)} \cdot {_cpv(system)}"
    if _is_wide(num):
        numbers = (
            rf"\bigl[{h_n} \\ &\quad - {_w(omega)} \cdot {_r0(system)}\bigr] \\ &\quad /\,({den})"
        )
    else:
        numbers = rf"\frac{{{num}}}{{{den}}}"
    return latex_chain(
        s.t,
        rf"\frac{{{s.h} - {s.w}\,r_0}}{{c_{{p,a}} + {s.w}\,c_{{p,v}}}}",
        numbers,
        _t_q(T_K, system),
    )


def _mixing_lines(proc: ProcessResult, system: UnitSystem) -> tuple[list[str], str]:
    spec = proc.spec
    assert isinstance(spec, AdiabaticMixing) and proc.other is not None
    si, sk, so = _sym(proc.inlet.number), _sym(proc.other.number), _sym(proc.outlet.number)
    s1, s2, s3 = proc.inlet.state, proc.other.state, proc.outlet.state
    m1 = _n(proc.inlet.m_dry_air_kg_s, "mass_flow", system)
    m2 = _n(proc.other.m_dry_air_kg_s, "mass_flow", system)
    m3 = _n(proc.outlet.m_dry_air_kg_s, "mass_flow", system)
    lines = [
        *_stream_state_lines(proc.other, system),
        *_flow_lines(proc.other, spec.flow.kind == "volume", system),
        latex_chain(
            so.ma,
            rf"{si.ma} + {sk.ma}",
            rf"{m1} + {m2}",
            _q(proc.outlet.m_dry_air_kg_s, "mass_flow", system),
        ),
        latex_chain(
            so.w,
            rf"\frac{{{si.ma}\,{si.w} + {sk.ma}\,{sk.w}}}{{{so.ma}}}",
            rf"\bigl({m1} \cdot {_w(s1.omega)} \\ &\quad + {m2} \cdot {_w(s2.omega)}\bigr)"
            rf" \\ &\quad /\,{m3}",
            _w(s3.omega),
        ),
        latex_chain(
            so.h,
            rf"\frac{{{si.ma}\,{si.h} + {sk.ma}\,{sk.h}}}{{{so.ma}}}",
            rf"\bigl({m1} \cdot {latex_paren(_n(s1.h_J_per_kg, _EH, system))}"
            rf" \\ &\quad + {m2} \cdot {latex_paren(_n(s2.h_J_per_kg, _EH, system))}\bigr)"
            rf" \\ &\quad /\,{m3}",
            _q(s3.h_J_per_kg, _EH, system),
        ),
        _t_from_h_chain(s3.h_J_per_kg, s3.omega, s3.T_K, system, so),
        rf"{so.T} = {_T(s3.T_K, system)}",
        _pv_from_omega_chain(s3.omega, s3.p_Pa, system, so),
        _lookup_pvs(s3.T_K, system, so),
        _phi_chain(s3, system, so),
        latex_chain(
            so.pa,
            f"p - {so.pv}",
            rf"{_n(s3.p_Pa, _P, system)} - {_n(s3.p_v_Pa, _P, system)}",
            _q(s3.p_a_Pa, _P, system),
        ),
        _v_chain(s3, system, so),
        latex_chain(so.V, rf"{so.ma}\,{so.v}", _q(proc.outlet.V_m3_s, "volume_flow", system)),
    ]
    text = (
        "Dos corrientes se mezclan sin intercambiar calor (vademecum §14.12.6). Se conservan el "
        "aire seco, el agua y la energía: ω y h de la mezcla son promedios pesados con los "
        "caudales de aire seco, y en la carta la mezcla cae sobre la recta que une las dos "
        "entradas (ṁ₁/ṁ₂ = (ω₂ − ω₃)/(ω₃ − ω₁) = (h₂ − h₃)/(h₃ − h₁)). La temperatura sale de "
        "h y ω (§14.5)."
    )
    return lines, text


def _process_step(proc: ProcessResult, dead: DeadState, system: UnitSystem) -> ProcedureStep:
    spec = proc.spec
    if isinstance(spec, SensibleProcess):
        lines, text = _sensible_lines(proc, system)
    elif isinstance(spec, HeatingHumidification):
        lines, text = _humidify_lines(proc, system)
    elif isinstance(spec, CoolingDehumidification):
        lines, text = _coil_lines(proc, system)
    elif isinstance(spec, AdiabaticHumidification):
        lines, text = _adiabatic_lines(proc, system)
    else:
        lines, text = _mixing_lines(proc, system)
    lines += _exergy_lines(proc, dead, system)
    if proc.kind == "mixing":
        exergy = (
            "Exergía destruida: mezclar dos corrientes de distinta temperatura y humedad es "
            "irreversible (vademecum §14.11)."
        )
    else:
        exergy = (
            "Exergía destruida: lo que entra (ψ del aire, la del agua y la del calor a la "
            "temperatura T_b de la fuente o del serpentín) menos lo que sale (vademecum §11 y "
            "§14.11)."
        )
    inlets = f"{proc.inlet.number}"
    if proc.other is not None:
        inlets += f" + {proc.other.number}"
    return ProcedureStep(
        title=f"Proceso {proc.index}: {proc.name.lower()} ({inlets} → {proc.outlet.number})",
        text=f"{text} {exergy}",
        latex=tuple(lines),
    )


def hvac_steps(result: HvacResult, system: UnitSystem) -> list[ProcedureStep]:
    """Pasos del tren de procesos (vademecum §14.12; Cengel §14-7)."""
    inlet = result.inlet
    dead = result.dead
    by_volume = result.inputs.inlet.flow.kind == "volume"
    steps = [
        ProcedureStep(
            title="Estado 1 y caudal de aire seco",
            text=(
                "El aire que entra queda fijado por T y φ (vademecum §14.2 a §14.5): φ da p_v y de "
                "ahí salen ω y h. El caudal que se conserva en todos los procesos es el de aire "
                "seco: ṁ_a = V̇/v, con v por kg de aire seco. "
                f"Presión total: {result.inputs.p_Pa / 1e5:.5g} bar. {_constants_text(system)}."
            ),
            latex=(
                rf"p = {_q(result.inputs.p_Pa, _P, system)}",
                *_stream_state_lines(inlet, system),
                *_flow_lines(inlet, by_volume, system),
            ),
        ),
        ProcedureStep(
            title="Ambiente para la exergía",
            text=(
                "La exergía se mide contra el aire del ambiente (estado muerto): ψ = ψ_tm + ψ_qu "
                "(vademecum §14.11). El agua líquida también tiene exergía: en un ambiente no "
                "saturado se puede evaporar (Wepfer, Gaggioli y Obert, 1979)."
            ),
            latex=(
                rf"T_0 = {_T(dead.T0_K, system)},\quad \varphi_0 = {_pct(dead.phi0)}",
                rf"\omega_0 = {_w(dead.omega0)}",
            ),
        ),
    ]
    steps += [_process_step(proc, dead, system) for proc in result.processes]
    steps.append(
        ProcedureStep(
            title="Totales",
            text="La suma de lo que entrega y quita cada proceso.",
            latex=(
                rf"\dot Q_{{\mathrm{{entregado}}}} = {_q(result.heating_W, 'power', system)}",
                rf"\dot Q_{{\mathrm{{quitado}}}} = {_q(result.cooling_W, 'power', system)}",
                rf"\dot m_{{w,\mathrm{{agregada}}}} = "
                rf"{_q(result.water_added_kg_s, 'mass_flow', system)}",
                rf"\dot m_{{w,\mathrm{{condensada}}}} = "
                rf"{_q(result.water_removed_kg_s, 'mass_flow', system)}",
                rf"\dot X_{{\mathrm{{dest}}}} = {_q(result.X_destroyed_W, 'power', system)}",
            ),
        )
    )
    return steps


# ---------------------------------------------------------------------
# Torre de enfriamiento
# ---------------------------------------------------------------------


def cooling_tower_steps(result: CoolingTowerResult, system: UnitSystem) -> list[ProcedureStep]:
    """Pasos de la torre de enfriamiento (vademecum §14.12.7; Cengel ejemplo 14-9)."""
    ti = result.inputs
    dead = result.dead
    a1, a2 = result.air_in, result.air_out
    m_a = result.m_dry_air_kg_s
    air_in = HvacStream(1, "aire que entra", a1, m_a)
    air_out = HvacStream(2, "aire que sale", a2, m_a)
    h3, h4 = result.h_water_in, result.h_water_out
    ma = _n(m_a, "mass_flow", system)
    mw3 = _n(ti.m_water_in_kg_s, "mass_flow", system)
    psi1, psi2 = a1.psi(dead), a2.psi(dead)
    psi3 = water_exergy(ti.T_water_in_K, "liquid", dead)
    psi4 = water_exergy(ti.T_water_out_K, "liquid", dead)
    dh_w = h3 - h4
    dh_a = (a2.h_J_per_kg - a1.h_J_per_kg) - (a2.omega - a1.omega) * h4
    s1 = _sym(1)
    return [
        ProcedureStep(
            title="Aire que entra (1) y que sale (2)",
            text=(
                "Los dos estados del aire quedan fijados por T y φ (vademecum §14.2 a §14.5). "
                f"{_constants_text(system)}."
            ),
            latex=(
                rf"p = {_q(ti.p_Pa, _P, system)}",
                *_stream_state_lines(air_in, system),
                latex_chain(
                    s1.pa,
                    f"p - {s1.pv}",
                    rf"{_n(a1.p_Pa, _P, system)} - {_n(a1.p_v_Pa, _P, system)}",
                    _q(a1.p_a_Pa, _P, system),
                ),
                _v_chain(a1, system, s1),
                *_stream_state_lines(air_out, system),
            ),
        ),
        ProcedureStep(
            title="Agua caliente (3) y enfriada (4)",
            text="Agua líquida: h = h_f(T), de la tabla A-4.",
            latex=(
                _lookup("h_3", rf"h_f({_T(ti.T_water_in_K, system)})", _q(h3, _EH, system)),
                _lookup("h_4", rf"h_f({_T(ti.T_water_out_K, system)})", _q(h4, _EH, system)),
            ),
        ),
        ProcedureStep(
            title="Balances: caudal de aire y agua de reposición",
            text=(
                "Masa de aire seco: ṁ_a constante. Masa de agua: ṁ₃ − ṁ₄ = ṁ_a·(ω₂ − ω₁), lo que "
                "se evapora. Energía, sin calor hacia el ambiente: ṁ₃·h₃ + ṁ_a·h₁ = ṁ₄·h₄ + "
                "ṁ_a·h₂. Reemplazando ṁ₄ se despeja ṁ_a (vademecum §14.12.7): el agua entrega "
                "ṁ₃·(h₃ − h₄) y cada kg de aire seco se lleva (h₂ − h₁) − (ω₂ − ω₁)·h₄."
            ),
            latex=(
                latex_chain(r"\Delta h_w", "h_3 - h_4", _dh(h3, h4, system), _q(dh_w, _EH, system)),
                latex_chain(
                    r"\Delta h_a",
                    r"(h_2 - h_1) - (\omega_2 - \omega_1)\,h_4",
                    rf"({_dh(a2.h_J_per_kg, a1.h_J_per_kg, system)}) \\ &\quad - "
                    rf"({_w(a2.omega)} - {_w(a1.omega)}) \\ &\qquad \cdot "
                    rf"{_n(h4, _EH, system)}",
                    _q(dh_a, _EH, system),
                ),
                latex_chain(
                    r"\dot m_a",
                    r"\frac{\dot m_3\,\Delta h_w}{\Delta h_a}",
                    rf"\frac{{{mw3} \cdot {_n(dh_w, _EH, system)}}}{{{_n(dh_a, _EH, system)}}}",
                    _q(m_a, "mass_flow", system),
                ),
                latex_chain(
                    r"\dot V_1",
                    r"\dot m_a\,v_1",
                    rf"{ma} \cdot {_n(a1.v_m3_per_kg, 'specific_volume', system)}",
                    _q(result.V_air_in_m3_s, "volume_flow", system),
                ),
                latex_chain(
                    r"\dot m_{\mathrm{rep}}",
                    r"\dot m_a\,(\omega_2 - \omega_1)",
                    rf"{ma}\,({_w(a2.omega)} - {_w(a1.omega)})",
                    _q(result.makeup_kg_s, "mass_flow", system),
                ),
                latex_chain(
                    r"\dot m_4",
                    r"\dot m_3 - \dot m_{\mathrm{rep}}",
                    _q(result.m_water_out_kg_s, "mass_flow", system),
                ),
            ),
        ),
        ProcedureStep(
            title="Rango, aproximación y efectividad",
            text=(
                "El rango es cuánto se enfría el agua; la aproximación, cuánto le falta para "
                "llegar a la temperatura de bulbo húmedo del aire que entra, que es el límite: "
                "el agua no puede enfriarse por debajo de ella. La efectividad compara el rango "
                "con ese máximo. L/G es la relación entre los caudales de agua y de aire seco."
            ),
            latex=(
                rf"T_{{bh,1}} = {_T(a1.T_wb_K, system)}",
                latex_chain(r"\text{rango}", "T_3 - T_4", _q(result.range_K, _DT, system)),
                latex_chain(
                    r"\text{aproximación}", r"T_4 - T_{bh,1}", _q(result.approach_K, _DT, system)
                ),
                latex_chain(
                    r"\varepsilon",
                    r"\frac{T_3 - T_4}{T_3 - T_{bh,1}}",
                    latex_number(result.effectiveness, 4),
                ),
                latex_chain(
                    r"L/G",
                    r"\frac{\dot m_3}{\dot m_a}",
                    latex_number(result.water_to_air, 4),
                ),
            ),
        ),
        ProcedureStep(
            title="Exergía destruida",
            text=(
                "Contra el ambiente (vademecum §14.11 para el aire; Wepfer, Gaggioli y Obert, "
                "1979, para el agua): lo que entra menos lo que sale. Casi toda la exergía que "
                "pierde el agua caliente se destruye al evaporarse y mezclarse con el aire."
            ),
            latex=(
                rf"T_0 = {_T(dead.T0_K, system)},\quad \varphi_0 = {_pct(dead.phi0)}",
                rf"\psi_1 = {_q(psi1, _EH, system)}",
                rf"\psi_2 = {_q(psi2, _EH, system)}",
                rf"\psi_3 = {_q(psi3, _EH, system)}",
                rf"\psi_4 = {_q(psi4, _EH, system)}",
                latex_chain(
                    r"\dot X_{\mathrm{dest}}",
                    r"\dot m_a\,(\psi_1 - \psi_2) \\ &\quad + \dot m_3\,\psi_3 - \dot m_4\,\psi_4",
                    _wrap(
                        rf"{ma}\,({_n(psi1, _EH, system)}",
                        "-",
                        rf"{latex_paren(_n(psi2, _EH, system))})",
                    )
                    + rf" \\ &\quad + {mw3} \cdot {_n(psi3, _EH, system)}"
                    rf" \\ &\quad - {_n(result.m_water_out_kg_s, 'mass_flow', system)} \cdot "
                    rf"{_n(psi4, _EH, system)}",
                    _q(result.X_destroyed_W, "power", system),
                ),
            ),
        ),
    ]

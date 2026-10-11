"""Procedimiento didáctico de los balances (LaTeX), en el sistema de unidades activo.

Las fórmulas son las del vademecum: §3.1 y §3.3–§3.6 (primer principio y trabajo), §10.3
(balances de entropía), §10.4 (rendimientos isoentrópicos) y §13 (incompresibles), con la
velocidad ω del vademecum. Dos factores de unidades quedan a la vista, como en Çengel y
Boles:

- p·v: 1 bar·m³/kg = 100 kJ/kg (Técnico) y 1 psia·ft³/lb = 0,18505 Btu/lb (Inglés);
- ω²/2 y g·z: 1 kJ/kg = 1000 m²/s² y 1 Btu/lb = 25 037 ft²/s².

El resto de las sustituciones cierra sin factores (``core.units_system``). Para que cada
renglón entre en el ancho de un celular, una resta con números largos (×10ⁿ o negativos)
se calcula antes de multiplicarla, y los balances con varios términos van de a uno.
"""

from __future__ import annotations

import math
import re

from core.balances.closed import ClosedResult, EquilibriumResult
from core.balances.common import G
from core.balances.steady_flow import DEVICES, DeviceResult, ExchangerResult, MixingResult
from core.balances.substance import PAIR_NAMES, PairCode, Substance, ThermoState
from core.balances.transient import INTEGRATION_POINTS, ChargingResult, DischargingResult
from core.heat_transfer.procedure_common import frac, n, numbered, q, times
from core.latex import latex_chain, latex_is_wide, latex_number, latex_paren
from core.state_report import ProcedureStep, pv_energy_factor
from core.units_system import QuantityKind, UnitSystem, convert_from_si

__all__ = [
    "charging_steps",
    "closed_steps",
    "device_steps",
    "discharging_steps",
    "equilibrium_steps",
    "exchanger_steps",
    "ke_factor",
    "mixing_steps",
]

_P = "pressure"
_T = "temperature"
_TA = "absolute_temperature"
_DT = "temperature_difference"
_V = "specific_volume"
_E = "specific_enthalpy"
_S = "specific_entropy"
_C = "specific_heat"
_EN = "energy"
_EN_S = "entropy"
_M = "mass"
_VOL = "volume"
_MF = "mass_flow"
_VF = "volume_flow"
_PW = "power"
_SF = "entropy_flow"
_SP = "speed"
_AR = "area"
_L = "length"
_G = "acceleration"

_LETTER = {"T": "T", "P": "p", "V": "v", "H": "h", "S": "s", "U": "u", "X": "x"}
_KIND: dict[str, QuantityKind] = {"T": _T, "p": _P, "v": _V, "h": _E, "s": _S, "u": _E}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _r(x: float, sig: int = 5) -> str:
    return latex_number(x, sig)


def _pv(system: UnitSystem) -> str:
    """El factor de p·v en LaTeX (vacío en el SI)."""
    f = pv_energy_factor(system)
    return "" if math.isclose(f, 1.0) else latex_number(f, 5)


def _pv_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " Ojo con las unidades de p·v: 1 bar·m³/kg = 100 kJ/kg."
    if system == "Inglés":
        return " Ojo con las unidades de p·v: 1 psia·ft³/lb = 0,18505 Btu/lb."
    return ""


def ke_factor(system: UnitSystem) -> float:
    """Cuántos (unidad de velocidad)² hay en una unidad de energía específica: 1, 1000 o
    25 037 (1 kJ/kg = 1000 m²/s², 1 Btu/lb = 25 037 ft²/s²)."""
    speed = 1.0 / convert_from_si(1.0, _SP, system)  # una unidad de velocidad en m/s
    return 1.0 / convert_from_si(speed**2, _E, system)


def _ke_tex(system: UnitSystem) -> str:
    f = ke_factor(system)
    if math.isclose(f, 1.0):
        return ""
    return r"1000" if math.isclose(f, 1000.0) else r"25\,037"


def _ke_den(system: UnitSystem) -> str:
    """El denominador de ω²/2: ``2``, ``2 \\cdot 1000`` o ``2 \\cdot 25\\,037``."""
    k = _ke_tex(system)
    return times("2", k) if k else "2"


def _ke_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " Las energías cinética y potencial van divididas por 1000 (1 kJ/kg = 1000 m²/s²)."
    if system == "Inglés":
        return (
            " Las energías cinética y potencial van divididas por 25 037 "
            "(1 Btu/lb = 25 037 ft²/s²)."
        )
    return ""


def _with(f: str, tex: str) -> str:
    return times(f, tex) if f else tex


def _stack(lhs: str, *steps: str) -> str:
    """``lhs`` en el primer renglón y cada ``= paso`` abajo, contra el margen izquierdo."""
    rows = [f"&{lhs}", *(rf"&= {s}" for s in steps)]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _vis(tex: str) -> int:
    """Cuántos caracteres se ven, más o menos, en un pedazo de LaTeX (para decidir si un lado
    izquierdo entra al lado de cada renglón o si un producto entra en uno)."""
    s = re.sub(r"\\(left|right|mathrm|text|dot|,|;|quad)", "", tex)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: max(m[1], m[2], key=len), s)
    s = s.replace(r"\cdot", "··").replace(r"\times", "×")
    s = re.sub(r"\\[a-zA-Z]+", "x", s)
    return len(re.sub(r"[{}_^ &]", "", s))


def _chain(lhs: str, *steps: str) -> str:
    """Una cadena alineada en el signo; apilada si algún paso lleva ×10ⁿ o el lado izquierdo
    es largo (en un ``aligned`` su ancho se suma al de cada renglón)."""
    if any(r"\times" in s for s in steps) or _vis(lhs) > 6:
        return _stack(lhs, *steps)
    return latex_chain(lhs, *steps)


def _diff(a: str, b: str) -> str:
    return rf"{a} - {latex_paren(b)}"


def _paren_diff(a: str, b: str) -> str:
    return rf"\left({_diff(a, b)}\right)"


def _sum(*terms: str) -> str:
    """``a + b + c`` con los negativos entre paréntesis (salvo el primero)."""
    return " + ".join(latex_paren(t) if i else t for i, t in enumerate(terms))


def _times_diff(
    lhs: str,
    sym: str,
    factors: list[str],
    a: float,
    b: float,
    kind: QuantityKind,
    d_lhs: str,
    result: str,
    system: UnitSystem,
    d_kind: QuantityKind | None = None,
) -> list[str]:
    """``lhs = sym = f₁·f₂·(a − b) = result``; con números largos (×10ⁿ o negativos), antes
    ``d_lhs = a − b = d`` en su renglón y después ``lhs = f₁·f₂·d``. ``d_kind`` es la
    magnitud de la resta si no es la de a y b (una diferencia de temperaturas)."""
    at, bt = n(a, kind, system), n(b, kind, system)
    dk = d_kind or kind
    steps = [sym] if sym else []
    if latex_is_wide(at, bt) or len(factors) >= 2 or _vis(times(*factors, at, bt)) > 24:
        return [
            _chain(d_lhs, _diff(at, bt), q(a - b, dk, system)),
            _chain(lhs, *steps, times(*factors, n(a - b, dk, system)), result),
        ]
    return [_chain(lhs, *steps, times(*factors, _paren_diff(at, bt)), result)]


def _model_text(sub: Substance) -> str:
    if sub.kind == "fluid":
        return (
            f"{sub.label}: las propiedades salen de las tablas (la ecuación de estado de "
            "CoolProp, con su referencia para u, h y s)."
        )
    if sub.kind == "ideal_gas":
        how = (
            "c_p constante a 25 °C"
            if sub.model == "constant"
            else "c_p variable (polinomios NASA, vademecum §4.8)"
        )
        return (
            f"{sub.gas.name} como gas ideal con {how}: p·v = R·T, u y h dependen solo de T "
            "(h = u + R·T) y s = s°(T) − R·ln(p/p°), con u = 0 y s = 0 a 25 °C y 1 bar."
        )
    m = sub.material
    return (
        f"{m.name} como incompresible (vademecum §13): v = 1/ρ constante, "
        "u = c·(T − 25 °C), h = u + p·v y s = c·ln(T/298,15 K)."
    )


def _args(pair: PairCode, label: str) -> str:
    return ", ".join(f"{_LETTER[c]}_{{{label}}}" for c in pair)


def _given(pair: PairCode | None) -> set[str]:
    return {_LETTER[c] for c in pair} if pair else set()


def _value(st: ThermoState, key: str) -> float:
    return {"T": st.T, "p": st.p, "v": st.v, "h": st.h, "s": st.s, "u": st.u}[key]


def _prop_lines(
    sub: Substance,
    st: ThermoState,
    label: str,
    pair: PairCode | None,
    keys: tuple[str, ...],
    system: UnitSystem,
) -> list[str]:
    """Las propiedades ``keys`` que no son dato, con de dónde salen."""
    given = _given(pair)
    lines: list[str] = []
    for key in keys:
        if key in given:
            continue
        value = _value(st, key)
        lhs = f"{key}_{{{label}}}"
        if sub.kind == "incompressible" and key in ("u", "s", "v", "h"):
            lines += _incompressible_lines(sub, st, key, lhs, label, system)
            continue
        if sub.kind == "ideal_gas" and key == "v" and given >= {"T", "p"}:
            den = _with(_pv(system), n(st.p, _P, system))
            lines.append(
                _chain(
                    lhs,
                    frac(rf"R\,T_{{{label}}}", f"p_{{{label}}}"),
                    frac(times(n(sub.gas.R, _S, system), n(st.T, _TA, system)), den),
                    q(value, _V, system),
                )
            )
            continue
        if sub.kind == "ideal_gas" and key in ("u", "h"):
            fn = rf"{key}(T_{{{label}}})"
        else:
            fn = rf"{key}({_args(pair, label)})" if pair else key
        lines.append(_chain(lhs, fn, q(value, _KIND[key], system)))
    if sub.kind == "fluid" and st.x is not None and "x" not in given:
        lines.append(latex_chain(f"x_{{{label}}}", _r(st.x, 4)))
    return lines


def _incompressible_lines(
    sub: Substance, st: ThermoState, key: str, lhs: str, label: str, system: UnitSystem
) -> list[str]:
    c = sub.material.c
    if key == "v":
        return [latex_chain(lhs, frac("1", r"\rho"), q(st.v, _V, system))]
    if key == "u":
        return [
            _chain(
                lhs,
                rf"c\,(T_{{{label}}} - T_{{\mathrm{{ref}}}})",
                times(n(c, _C, system), n(st.T - 298.15, _DT, system)),
                q(st.u, _E, system),
            )
        ]
    if key == "s":
        ratio = rf"\ln\frac{{{n(st.T, _TA, system)}}}{{{n(298.15, _TA, system)}}}"
        return [
            _chain(
                lhs,
                rf"c\,\ln\frac{{T_{{{label}}}}}{{T_{{\mathrm{{ref}}}}}}",
                times(n(c, _C, system), ratio),
                q(st.s, _S, system),
            )
        ]
    pv = st.p * st.v
    term = _with(_pv(system), times(n(st.p, _P, system), n(st.v, _V, system)))
    pv_line = _chain(rf"p_{{{label}}}\,v_{{{label}}}", term, q(pv, _E, system))
    h_line = _chain(
        lhs,
        rf"u_{{{label}}} + p_{{{label}}}\,v_{{{label}}}",
        _sum(n(st.u, _E, system), n(pv, _E, system)),
        q(st.h, _E, system),
    )
    return [pv_line, h_line]


def _verdict_text(S_gen: float, violation: str | None, what: str = "S_gen") -> str:
    if violation:
        return violation
    if S_gen == 0.0:
        return f"{what} = 0: el proceso es reversible."
    return f"{what} > 0: el proceso es irreversible (vademecum §10.2)."


def _sgen_lines(
    dS: float,
    Q: float,
    T_b: float,
    S_gen: float,
    T0: float,
    system: UnitSystem,
    rate: bool = False,
) -> list[str]:
    """Q/T_b, S_gen = ΔS − Q/T_b y X_dest = T₀·S_gen (totales o por unidad de tiempo)."""
    s_kind: QuantityKind = _SF if rate else _EN_S
    e_kind: QuantityKind = _PW if rate else _EN
    if rate:
        Qs, Ss, Xs, dSs = (
            r"\dot{Q}",
            r"\dot{S}_{\mathrm{gen}}",
            r"\dot{X}_{\mathrm{dest}}",
            r"\Delta\dot{S}",
        )
    else:
        Qs, Ss, Xs, dSs = "Q", r"S_{\mathrm{gen}}", r"X_{\mathrm{dest}}", r"\Delta S"
    lines = []
    if Q != 0.0:
        lines.append(
            _chain(
                frac(Qs, "T_b"),
                frac(n(Q, e_kind, system), n(T_b, _TA, system)),
                q(Q / T_b, s_kind, system),
            )
        )
        lines.append(
            _chain(
                Ss,
                _diff(dSs, frac(Qs, "T_b")),
                _diff(n(dS, s_kind, system), n(Q / T_b, s_kind, system)),
                q(S_gen, s_kind, system),
            )
        )
    else:
        lines.append(latex_chain(Ss, dSs, q(S_gen, s_kind, system)))
    lines.append(
        _chain(
            Xs,
            rf"T_0\,{Ss}",
            times(n(T0, _TA, system), n(S_gen, s_kind, system)),
            q(T0 * S_gen, e_kind, system),
        )
    )
    return lines


def _mass_from_volume(
    lhs: str, V: float, v: float, m: float, v_sym: str, system: UnitSystem
) -> str:
    return latex_chain(
        lhs, frac("V", v_sym), frac(n(V, _VOL, system), n(v, _V, system)), q(m, _M, system)
    )


# ---------------------------------------------------------------------
# Sistema cerrado
# ---------------------------------------------------------------------

_CLOSED_KEYS: dict[str, tuple[str, ...]] = {
    "v_const": ("T", "p", "v", "u", "s"),
    "p_const": ("T", "p", "v", "u", "h", "s"),
    "T_const": ("T", "p", "v", "u", "s"),
    "polytropic": ("T", "p", "v", "u", "s"),
    "isentropic": ("T", "p", "v", "u", "s"),
    "p_ext": ("T", "p", "v", "u", "h", "s"),
    "general": ("T", "p", "v", "u", "s"),
}


def closed_steps(r: ClosedResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de un sistema cerrado: estados, trabajo, primer y segundo principio."""
    inp = r.inputs
    sub, proc = inp.substance, inp.process
    s1, s2, m = r.state1, r.state2, r.mass
    keys = _CLOSED_KEYS[proc] if sub.kind != "incompressible" else ("T", "p", "u", "s")
    steps = [
        ProcedureStep(
            "Estado 1",
            f"{_model_text(sub)} Datos: {PAIR_NAMES[inp.pair1]}.",
            tuple(_prop_lines(sub, s1, "1", inp.pair1, keys, system)),
        )
    ]
    if inp.volume_m3 is not None:
        steps.append(
            ProcedureStep(
                "Masa",
                "Con el volumen inicial: m = V₁/v₁.",
                (
                    latex_chain(
                        "m",
                        frac("V_1", "v_1"),
                        frac(n(inp.volume_m3, _VOL, system), n(s1.v, _V, system)),
                        q(m, _M, system),
                    ),
                ),
            )
        )
    steps.append(_closed_state2(r, keys, system))
    steps.extend(_closed_work_and_energy(r, system))
    m_t = n(m, _M, system)
    lines = _times_diff(
        r"\Delta S",
        r"m\,(s_2 - s_1)",
        [m_t],
        s2.s,
        s1.s,
        _S,
        "s_2 - s_1",
        q(r.dS, _EN_S, system),
        system,
    )
    lines += _sgen_lines(r.dS, r.Q, r.T_b, r.S_gen, inp.T0_K, system)
    tb = (
        "la del sistema (el calor reversible entra a la temperatura del sistema)"
        if proc == "T_const" and inp.T_b_K is None
        else "la de la fuente o el medio en la frontera (el T_k del vademecum §10.3)"
    )
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "Vademecum §10.3.1: S₂ − S₁ = Q/T_b + S_gen, con T_b "
            f"{tb}; la exergía destruida es T₀·S_gen (§11). {_verdict_text(r.S_gen, r.violation)}",
            tuple(lines),
        )
    )
    if r.reversible_W is not None and r.reversible_state is not None:
        rs = r.reversible_state
        lines = [latex_chain("T_{2s}", r"T(p_{\mathrm{ext}}, s_1)", q(rs.T, _T, system))]
        lines += _times_diff(
            r"W_{\mathrm{rev}}",
            r"m\,(u_1 - u_{2s})",
            [m_t],
            s1.u,
            rs.u,
            _E,
            "u_1 - u_{2s}",
            q(r.reversible_W, _EN, system),
            system,
        )
        steps.append(
            ProcedureStep(
                "Comparación con la expansión reversible",
                "Adiabática y reversible hasta la misma presión: s₂ₛ = s₁ y W = m·(u₁ − u₂ₛ). "
                "Hace más trabajo y termina más fría: la diferencia es la irreversibilidad.",
                tuple(lines),
            )
        )
    return numbered(steps)


def _heat_per_kg(r: ClosedResult, system: UnitSystem) -> tuple[str, str]:
    """El renglón de (Q + W_ent)/m y su símbolo."""
    W_in = r.W_in
    sym = r"\frac{Q + W_{\mathrm{ent}}}{m}" if W_in else r"\frac{Q}{m}"
    num = _sum(n(r.Q, _EN, system), n(W_in, _EN, system)) if W_in else n(r.Q, _EN, system)
    value = (r.Q + W_in) / r.mass
    line = _chain(sym, frac(num, n(r.mass, _M, system)), q(value, _E, system))
    return line, sym


def _closed_state2(r: ClosedResult, keys: tuple[str, ...], system: UnitSystem) -> ProcedureStep:
    inp = r.inputs
    sub, proc, end = inp.substance, inp.process, inp.end
    s1, s2, m = r.state1, r.state2, r.mass
    lines: list[str] = []
    pair: PairCode | None
    heat = (r.Q + r.W_in) / m

    def plus_heat(sym: str, base: float, result: float) -> None:
        line, hsym = _heat_per_kg(r, system)
        lines.append(line)
        lines.append(
            _chain(
                f"{sym}_2",
                f"{sym}_1 + {hsym}",
                _sum(n(base, _E, system), n(heat, _E, system)),
                q(result, _E, system),
            )
        )

    def volume_from_V() -> None:
        lines.append(
            latex_chain(
                "v_2",
                frac("V_2", "m"),
                frac(n(r.V2, _VOL, system), n(m, _M, system)),
                q(s2.v, _V, system),
            )
        )

    if sub.kind == "incompressible":
        text = "Incompresible: v₂ = v₁ y no hay trabajo de frontera."
        if end == "T":
            pair = "TP"
        else:
            text += " Con el calor: u₂ = u₁ + (Q + W_ent)/m."
            plus_heat("u", s1.u, s2.u)
            pair = "PU"
        lines += _prop_lines(sub, s2, "2", pair, keys, system)
        return ProcedureStep("Estado 2", text, tuple(lines))
    if proc == "v_const":
        text = "Tanque rígido: v₂ = v₁."
        lines.append(latex_chain("v_2", "v_1", q(s2.v, _V, system)))
        if end == "Q":
            text += " Con el calor dado, el primer principio da u₂ (W_b = 0)."
            plus_heat("u", s1.u, s2.u)
            pair = "VU"
        else:
            pair = {"T": "TV", "p": "PV", "x": "VX"}[end]  # type: ignore[assignment]
    elif proc == "p_const":
        text = "Pistón libre: p₂ = p₁."
        lines.append(latex_chain("p_2", "p_1", q(s2.p, _P, system)))
        if end == "Q":
            text += (
                " Con el calor dado, como W_b = m·p·Δv, el primer principio queda "
                "Q + W_ent = m·(h₂ − h₁) (Çengel §4-2)."
            )
            plus_heat("h", s1.h, s2.h)
            pair = "PH"
        elif end in ("v", "V"):
            if end == "V":
                volume_from_V()
            pair = "PV"
        else:
            pair = {"T": "TP", "x": "PX"}[end]  # type: ignore[assignment]
    elif proc == "T_const":
        text = "Temperatura constante: T₂ = T₁."
        lines.append(latex_chain("T_2", "T_1", q(s2.T, _T, system)))
        if end == "Q":
            text += " Cuasiestático: el calor reversible es Q = m·T·(s₂ − s₁)."
            ds = r.Q / (m * s1.T)
            lines.append(
                _chain(
                    r"\frac{Q}{m\,T}",
                    frac(n(r.Q, _EN, system), times(n(m, _M, system), n(s1.T, _TA, system))),
                    q(ds, _S, system),
                )
            )
            lines.append(
                _chain(
                    "s_2",
                    r"s_1 + \frac{Q}{m\,T}",
                    _sum(n(s1.s, _S, system), n(ds, _S, system)),
                    q(s2.s, _S, system),
                )
            )
            pair = "TS"
        elif end == "V":
            volume_from_V()
            pair = "TV"
        else:
            pair = {"p": "TP", "v": "TV", "x": "TX"}[end]  # type: ignore[assignment]
    elif proc == "polytropic":
        nn = inp.n
        text = f"Politrópica (vademecum §6.1): p₁·v₁ⁿ = p₂·v₂ⁿ con n = {_num_text(nn)}."
        if end == "p":
            ratio = frac(n(s1.p, _P, system), n(s2.p, _P, system))
            lines.append(
                _chain(
                    "v_2",
                    r"v_1 \left(\frac{p_1}{p_2}\right)^{1/n}",
                    rf"{n(s1.v, _V, system)} \left({ratio}\right)^{{1/{_r(nn, 4)}}}",
                    q(s2.v, _V, system),
                )
            )
        elif end in ("v", "V"):
            if end == "V":
                volume_from_V()
            ratio = frac(n(s1.v, _V, system), n(s2.v, _V, system))
            lines.append(
                _chain(
                    "p_2",
                    r"p_1 \left(\frac{v_1}{v_2}\right)^{n}",
                    rf"{n(s1.p, _P, system)} \left({ratio}\right)^{{{_r(nn, 4)}}}",
                    q(s2.p, _P, system),
                )
            )
        else:
            text += " Con T₂ dada, se busca p₂ tal que T(p₂, v₂) = T₂ sobre la politrópica."
            lines.append(latex_chain("p_2", q(s2.p, _P, system)))
            lines.append(latex_chain("v_2", q(s2.v, _V, system)))
        pair = "PV"
    elif proc == "isentropic":
        text = "Adiabático y reversible: s₂ = s₁ (vademecum §10.1)."
        lines.append(latex_chain("s_2", "s_1", q(s2.s, _S, system)))
        if end == "V":
            volume_from_V()
        pair = {"p": "PS", "v": "VS", "V": "VS", "T": "TS"}[end]  # type: ignore[assignment]
    elif proc == "p_ext":
        text = "Contra una presión exterior constante: el sistema termina a p₂ = p_ext."
        lines.append(latex_chain("p_2", r"p_{\mathrm{ext}}", q(s2.p, _P, system)))
        if end == "Q":
            assert inp.p_ext_Pa is not None
            text += (
                " Con W_b = m·p_ext·(v₂ − v₁), el primer principio da h₂ = u₂ + p_ext·v₂ = "
                f"u₁ + p_ext·v₁ + (Q + W_ent)/m (vademecum §3.6).{_pv_text(system)}"
            )
            pv1 = inp.p_ext_Pa * s1.v
            lines.append(
                _chain(
                    r"p_{\mathrm{ext}}\,v_1",
                    _with(_pv(system), times(n(inp.p_ext_Pa, _P, system), n(s1.v, _V, system))),
                    q(pv1, _E, system),
                )
            )
            line, hsym = _heat_per_kg(r, system)
            lines.append(line)
            base = s1.u + pv1
            lines.append(
                _chain(
                    r"u_1 + p_{\mathrm{ext}}\,v_1",
                    _sum(n(s1.u, _E, system), n(pv1, _E, system)),
                    q(base, _E, system),
                )
            )
            lines.append(
                _chain(
                    "h_2",
                    rf"(u_1 + p_{{\mathrm{{ext}}}}\,v_1) + {hsym}",
                    _sum(n(base, _E, system), n(heat, _E, system)),
                    q(s2.h, _E, system),
                )
            )
            pair = "PH"
        else:
            pair = "TP"
    else:  # general
        text = f"Estado final dado ({PAIR_NAMES[inp.end_pair or 'TP']})."  # type: ignore[index]
        pair = inp.end_pair
    lines += _prop_lines(sub, s2, "2", pair, keys, system)
    return ProcedureStep("Estado 2", text, tuple(lines))


def _num_text(x: float) -> str:
    return f"{x:.4g}".replace(".", ",")


def _dU_lines(r: ClosedResult, system: UnitSystem) -> list[str]:
    return _times_diff(
        r"\Delta U",
        r"m\,(u_2 - u_1)",
        [n(r.mass, _M, system)],
        r.state2.u,
        r.state1.u,
        _E,
        "u_2 - u_1",
        q(r.dU, _EN, system),
        system,
    )


def _pv_line(lhs: str, p: float, v: float, system: UnitSystem) -> str:
    return _chain(
        lhs,
        _with(_pv(system), times(n(p, _P, system), n(v, _V, system))),
        q(p * v, _E, system),
    )


def _pdv_lines(
    m: float, p: float, v1: float, v2: float, p_sym: str, W: float, system: UnitSystem
) -> list[str]:
    """v₂ − v₁, p·(v₂ − v₁) (con el factor de unidades) y W_b = m·p·(v₂ − v₁)."""
    dv, pdv = v2 - v1, p * (v2 - v1)
    return [
        _chain("v_2 - v_1", _diff(n(v2, _V, system), n(v1, _V, system)), q(dv, _V, system)),
        _chain(
            rf"{p_sym}\,(v_2 - v_1)",
            _with(_pv(system), times(n(p, _P, system), n(dv, _V, system))),
            q(pdv, _E, system),
        ),
        _chain(
            "W_b",
            rf"m\,{p_sym}\,(v_2 - v_1)",
            times(n(m, _M, system), n(pdv, _E, system)),
            q(W, _EN, system),
        ),
    ]


def _closed_work_and_energy(r: ClosedResult, system: UnitSystem) -> list[ProcedureStep]:
    inp = r.inputs
    proc, sub = inp.process, inp.substance
    s1, s2, m = r.state1, r.state2, r.mass
    m_t = n(m, _M, system)
    lines: list[str] = []
    if sub.kind == "incompressible" or proc == "v_const":
        W_text = "El volumen no cambia: W_b = 0."
    elif proc == "p_const":
        W_text = f"Presión constante (vademecum §3.5): W_b = m·p·(v₂ − v₁).{_pv_text(system)}"
        lines += _pdv_lines(m, s1.p, s1.v, s2.v, "p", r.W_b, system)
    elif proc == "T_const":
        W_text = (
            "A T constante y cuasiestático, el calor es reversible: Q = m·T·(s₂ − s₁), y el "
            "primer principio da W_b = Q − m·(u₂ − u₁) (vale también dentro de la campana, "
            "sin integrar p·dv)."
        )
        lines += _times_diff(
            "Q",
            r"m\,T\,(s_2 - s_1)",
            [m_t, n(s1.T, _TA, system)],
            s2.s,
            s1.s,
            _S,
            "s_2 - s_1",
            q(r.Q, _EN, system),
            system,
        )
        lines += _dU_lines(r, system)
        lines.append(
            _chain(
                "W_b",
                r"Q - \Delta U",
                _diff(n(r.Q, _EN, system), n(r.dU, _EN, system)),
                q(r.W_b, _EN, system),
            )
        )
        return [ProcedureStep("Calor y trabajo (primer principio)", W_text, tuple(lines))]
    elif proc == "polytropic":
        nn = inp.n
        lines.append(_pv_line("p_1 v_1", s1.p, s1.v, system))
        if math.isclose(nn, 1.0):
            W_text = f"Politrópica con n = 1: W_b = m·p₁·v₁·ln(v₂/v₁).{_pv_text(system)}"
            log = rf"\ln\frac{{{n(s2.v, _V, system)}}}{{{n(s1.v, _V, system)}}}"
            lines.append(
                _chain(
                    "W_b",
                    r"m\,p_1 v_1 \ln\frac{v_2}{v_1}",
                    times(m_t, n(s1.p * s1.v, _E, system), log),
                    q(r.W_b, _EN, system),
                )
            )
        else:
            W_text = (
                "Politrópica (vademecum §6.2): W_b = m·(p₂·v₂ − p₁·v₁)/(1 − n), para cualquier "
                f"sustancia.{_pv_text(system)}"
            )
            lines.append(_pv_line("p_2 v_2", s2.p, s2.v, system))
            a, b = n(s2.p * s2.v, _E, system), n(s1.p * s1.v, _E, system)
            if latex_is_wide(a, b):
                d = s2.p * s2.v - s1.p * s1.v
                lines.append(_chain("p_2 v_2 - p_1 v_1", _diff(a, b), q(d, _E, system)))
                num = times(m_t, n(d, _E, system))
            else:
                num = times(m_t, rf"\left({_diff(a, b)}\right)")
            lines.append(
                _chain(
                    "W_b",
                    frac(r"m\,(p_2 v_2 - p_1 v_1)", "1 - n"),
                    frac(num, f"1 - {_r(nn, 4)}"),
                    q(r.W_b, _EN, system),
                )
            )
    elif proc == "isentropic":
        W_text = "Adiabático (Q = 0): el primer principio da W_b = −m·(u₂ − u₁)."
        lines += _dU_lines(r, system)
        lines.append(latex_chain("W_b", r"-\Delta U", q(r.W_b, _EN, system)))
        return [ProcedureStep("Trabajo (primer principio)", W_text, tuple(lines))]
    elif proc == "p_ext":
        assert inp.p_ext_Pa is not None
        W_text = (
            "No cuasiestático (vademecum §3.6): el sistema no tiene una presión definida y el "
            f"trabajo se calcula con la exterior, W_b = m·p_ext·(v₂ − v₁).{_pv_text(system)}"
        )
        lines += _pdv_lines(m, inp.p_ext_Pa, s1.v, s2.v, r"p_{\mathrm{ext}}", r.W_b, system)
    else:
        W_text = ""
    steps = []
    if proc != "general":
        steps.append(ProcedureStep("Trabajo de frontera", W_text, tuple(lines)))
    energy = _dU_lines(r, system)
    W_in = r.W_in
    if proc == "general" and inp.end == "W":
        energy.append(
            _chain(
                "Q",
                r"\Delta U + W_b",
                _sum(n(r.dU, _EN, system), n(r.W_b, _EN, system)),
                q(r.Q, _EN, system),
            )
        )
        text = (
            "Sin camino no hay ∫p·dv: el trabajo es dato y el calor sale del primer principio "
            "(vademecum §3.1: Q − W = ΔU)."
        )
    elif proc == "general":
        energy.append(
            _chain(
                "W_b",
                r"Q - \Delta U",
                _diff(n(r.Q, _EN, system), n(r.dU, _EN, system)),
                q(r.W_b, _EN, system),
            )
        )
        text = (
            "Sin camino no hay ∫p·dv: el calor es dato y el trabajo sale del primer principio "
            "(vademecum §3.1: Q − W = ΔU)."
        )
    else:
        if W_in:
            energy.append(
                _chain(
                    "W",
                    r"W_b - W_{\mathrm{ent}}",
                    _diff(n(r.W_b, _EN, system), n(W_in, _EN, system)),
                    q(r.W, _EN, system),
                )
            )
            energy.append(
                _chain(
                    "Q",
                    r"\Delta U + W",
                    _sum(n(r.dU, _EN, system), n(r.W, _EN, system)),
                    q(r.Q, _EN, system),
                )
            )
        else:
            energy.append(
                _chain(
                    "Q",
                    r"\Delta U + W_b",
                    _sum(n(r.dU, _EN, system), n(r.W_b, _EN, system)),
                    q(r.Q, _EN, system),
                )
            )
        text = (
            "Vademecum §3.1: Q − W = ΔU, con W = W_b − W_ent (el trabajo de frontera que hace "
            "el sistema menos el que recibe por una resistencia o una paleta)."
            if W_in
            else "Vademecum §3.1: Q − W = ΔU (sin energía cinética ni potencial), con W = W_b."
        )
    steps.append(ProcedureStep("Primer principio", text, tuple(energy)))
    return steps


# ---------------------------------------------------------------------
# Equilibrio térmico
# ---------------------------------------------------------------------


def equilibrium_steps(r: EquilibriumResult, system: UnitSystem) -> list[ProcedureStep]:
    """T final, el calor y la entropía de cada cuerpo y la S_gen (Çengel §4-5 y §7-13)."""
    inp = r.inputs
    names = ["a", "b"]
    mc = [c.body.mass_kg * c.body.substance.material.c for c in r.changes]
    cap_lines = [
        latex_chain(
            rf"(m c)_{names[k]}",
            rf"m_{names[k]}\,c_{names[k]}",
            times(n(c.body.mass_kg, _M, system), n(c.body.substance.material.c, _C, system)),
            q(mc[k], _EN_S, system),
        )
        for k, c in enumerate(r.changes)
    ]
    if inp.body_b is not None:
        a, b = r.changes
        # (m c)·T con T en la unidad del sistema (°C, °F o K): el promedio pesado es lineal
        Ta = convert_from_si(a.body.T_K, _T, system)
        Tb = convert_from_si(b.body.T_K, _T, system)
        prods = [mc[0] * Ta, mc[1] * Tb]
        prod_lines = [
            _chain(
                rf"(m c)_{lbl} T_{lbl}",
                times(n(mc[k], _EN_S, system), n(T, _T, system)),
                _r(convert_from_si(mc[k], _EN_S, system) * Tu),
            )
            for k, (lbl, T, Tu) in enumerate((("a", a.body.T_K, Ta), ("b", b.body.T_K, Tb)))
        ]
        num = _sum(*(_r(convert_from_si(mc[k], _EN_S, system) * (Ta, Tb)[k]) for k in range(2)))
        den = _sum(n(mc[0], _EN_S, system), n(mc[1], _EN_S, system))
        del prods
        lines = [
            *cap_lines,
            *prod_lines,
            _stack(
                r"T_f = \frac{(m c)_a T_a + (m c)_b T_b}{(m c)_a + (m c)_b}",
                frac(num, den),
                q(r.T_final, _T, system),
            ),
        ]
        text = (
            "Recipiente aislado y rígido: Q_a + Q_b = 0 con Q = m·c·(T_f − T) (vademecum "
            "§13), así que T_f es el promedio de las temperaturas pesado con m·c."
        )
    else:
        lines = [*cap_lines, latex_chain("T_f", r"T_{\mathrm{res}}", q(r.T_final, _T, system))]
        text = "El reservorio no cambia de temperatura: el cuerpo termina a la suya."
    steps = [ProcedureStep("Temperatura final", text, tuple(lines))]
    heat: list[str] = []
    entropy: list[str] = []
    for k, c in enumerate(r.changes):
        lbl = names[k]
        heat += _times_diff(
            f"Q_{lbl}",
            rf"(m c)_{lbl}\,(T_f - T_{lbl})",
            [n(mc[k], _EN_S, system)],
            r.T_final,
            c.body.T_K,
            _T,
            rf"T_f - T_{lbl}",
            q(c.Q, _EN, system),
            system,
            _DT,
        )
        ratio = rf"\ln\frac{{{n(r.T_final, _TA, system)}}}{{{n(c.body.T_K, _TA, system)}}}"
        entropy.append(
            _chain(
                rf"\Delta S_{lbl}",
                rf"(m c)_{lbl} \ln\frac{{T_f}}{{T_{lbl}}}",
                times(n(mc[k], _EN_S, system), ratio),
                q(c.dS, _EN_S, system),
            )
        )
    steps.append(ProcedureStep("Calor", "Q > 0 si el cuerpo recibe calor.", tuple(heat)))
    if r.reservoir_dS is not None:
        entropy.append(
            _chain(
                r"\Delta S_{\mathrm{res}}",
                r"-\frac{Q_a}{T_{\mathrm{res}}}",
                frac(n(-r.changes[0].Q, _EN, system), n(r.T_final, _TA, system)),
                q(r.reservoir_dS, _EN_S, system),
            )
        )
    steps.append(
        ProcedureStep(
            "Entropía de cada parte",
            "Vademecum §13: ΔS = m·c·ln(T₂/T₁), con T absoluta."
            + (
                " El reservorio recibe el calor a su temperatura: ΔS = −Q_a/T_res."
                if r.reservoir_dS is not None
                else ""
            ),
            tuple(entropy),
        )
    )
    terms = [n(c.dS, _EN_S, system) for c in r.changes]
    sym = [rf"\Delta S_{names[k]}" for k in range(len(r.changes))]
    if r.reservoir_dS is not None:
        terms.append(n(r.reservoir_dS, _EN_S, system))
        sym.append(r"\Delta S_{\mathrm{res}}")
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "El conjunto es aislado (vademecum §10.2): S_gen = ΣΔS. "
            + _verdict_text(r.S_gen, None),
            (
                _chain(
                    r"S_{\mathrm{gen}}", " + ".join(sym), _sum(*terms), q(r.S_gen, _EN_S, system)
                ),
                _chain(
                    r"X_{\mathrm{dest}}",
                    r"T_0\,S_{\mathrm{gen}}",
                    times(n(inp.T0_K, _TA, system), n(r.S_gen, _EN_S, system)),
                    q(r.X_dest, _EN, system),
                ),
            ),
        )
    )
    return numbered(steps)


# ---------------------------------------------------------------------
# Flujo estacionario
# ---------------------------------------------------------------------


def device_steps(r: DeviceResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de un dispositivo: estados, caudal, primer principio, potencias y entropía."""
    inp = r.inputs
    sub = inp.substance
    s1, s2 = r.state1, r.state2
    keys = ("T", "p", "v", "h", "s")
    steps = [
        ProcedureStep(
            "Entrada",
            f"{_model_text(sub)} Datos: {PAIR_NAMES[inp.pair1]}.",
            tuple(_prop_lines(sub, s1, "1", inp.pair1, keys, system)),
        )
    ]
    flow = _flow_lines(r, system)
    if flow:
        steps.append(ProcedureStep("Caudal", flow[0], tuple(flow[1:])))
    if inp.rates:
        steps.append(_per_kg_step(r, system))
    steps.append(_device_outlet(r, keys, system))
    steps.append(_device_energy(r, system))
    rates = _device_rates(r, system)
    if rates is not None:
        steps.append(rates)
    if r.eta_s is not None and inp.out != "eta" and r.state2s is not None:
        steps.append(_eta_step(r, system))
    ds = s2.s - s1.s
    lines = [
        _chain("s_2 - s_1", _diff(n(s2.s, _S, system), n(s1.s, _S, system)), q(ds, _S, system))
    ]
    if r.q != 0.0:
        lines.append(
            _chain(
                r"\frac{q}{T_b}",
                frac(n(r.q, _E, system), n(r.T_b, _TA, system)),
                q(r.q / r.T_b, _S, system),
            )
        )
        lines.append(
            _chain(
                r"s_{\mathrm{gen}}",
                r"(s_2 - s_1) - \frac{q}{T_b}",
                _diff(n(ds, _S, system), n(r.q / r.T_b, _S, system)),
                q(r.s_gen, _S, system),
            )
        )
    else:
        lines.append(latex_chain(r"s_{\mathrm{gen}}", r"s_2 - s_1", q(r.s_gen, _S, system)))
    lines.append(
        _chain(
            r"\dot{S}_{\mathrm{gen}}",
            r"\dot{m}\,s_{\mathrm{gen}}",
            times(n(r.m_dot, _MF, system), n(r.s_gen, _S, system)),
            q(r.S_gen_dot, _SF, system),
        )
    )
    lines.append(
        _chain(
            r"\dot{X}_{\mathrm{dest}}",
            r"T_0\,\dot{S}_{\mathrm{gen}}",
            times(n(inp.T0_K, _TA, system), n(r.S_gen_dot, _SF, system)),
            q(r.X_dest_dot, _PW, system),
        )
    )
    tb = " (T_b: la del medio en la frontera)" if r.q != 0.0 else ""
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "Vademecum §10.3.2, por kg: s_gen = s₂ − s₁ − q/T_b"
            f"{tb}. {_verdict_text(r.s_gen, r.violation, 's_gen')}",
            tuple(lines),
        )
    )
    return numbered(steps)


def _flow_lines(r: DeviceResult, system: UnitSystem) -> list[str]:
    inp = r.inputs
    s1 = r.state1
    if inp.flow == "Vdot":
        return [
            "Con el caudal volumétrico de entrada: ṁ = V̇₁/v₁.",
            latex_chain(
                r"\dot{m}",
                frac(r"\dot{V}_1", "v_1"),
                frac(n(inp.flow_value, _VF, system), n(s1.v, _V, system)),
                q(r.m_dot, _MF, system),
            ),
        ]
    if inp.flow == "inlet":
        num = times(n(inp.flow_value, _AR, system), n(r.V1, _SP, system))
        return [
            "Con el área y la velocidad de entrada: ṁ = A₁·ω₁/v₁ (vademecum §3.3).",
            latex_chain(
                r"\dot{m}",
                frac(r"A_1\,\omega_1", "v_1"),
                frac(num, n(s1.v, _V, system)),
                q(r.m_dot, _MF, system),
            ),
        ]
    if inp.A1_m2 is not None:
        num = times(n(r.m_dot, _MF, system), n(s1.v, _V, system))
        return [
            "Con el caudal y el área de entrada: ω₁ = ṁ·v₁/A₁.",
            latex_chain(
                r"\omega_1",
                frac(r"\dot{m}\,v_1", "A_1"),
                frac(num, n(inp.A1_m2, _AR, system)),
                q(r.V1, _SP, system),
            ),
        ]
    return []


def _per_kg_step(r: DeviceResult, system: UnitSystem) -> ProcedureStep:
    """Q̇ y Ẇ dados como potencias, por kg: q = Q̇/ṁ y w = Ẇ/ṁ."""
    lines = []
    for sym, big, value in (("q", r"\dot{Q}", r.q), ("w", r"\dot{W}", r.w)):
        if r.unknown == sym or value == 0.0:
            continue
        lines.append(
            _chain(
                sym,
                frac(big, r"\dot{m}"),
                frac(n(value * r.m_dot, _PW, system), n(r.m_dot, _MF, system)),
                q(value, _E, system),
            )
        )
    return ProcedureStep(
        "Calor y trabajo por kg",
        "El calor y el trabajo se dieron como potencias (Ẇ < 0 si entra, como en una "
        "resistencia): por kg, q = Q̇/ṁ y w = Ẇ/ṁ.",
        tuple(lines),
    )


def _device_outlet(r: DeviceResult, keys: tuple[str, ...], system: UnitSystem) -> ProcedureStep:
    inp = r.inputs
    sub, dev, out = inp.substance, inp.device, inp.out
    s1, s2 = r.state1, r.state2
    lines: list[str] = []
    pair: PairCode
    dpe = G * (inp.z2 - inp.z1)
    if dev == "valve":
        text = (
            "Válvula de estrangulamiento (Çengel §5-4): adiabática, sin trabajo y con Δe_c ≈ 0, "
            "así que h₂ = h₁" + (" − Δe_p." if dpe else ".")
        )
        if dpe:
            lines.append(
                _chain(
                    "h_2",
                    r"h_1 - \Delta e_p",
                    _diff(n(s1.h, _E, system), n(dpe, _E, system)),
                    q(s2.h, _E, system),
                )
            )
        else:
            lines.append(latex_chain("h_2", "h_1", q(s2.h, _E, system)))
        pair = "PH"
    elif out == "eta":
        s2s = r.state2s
        assert s2s is not None
        lines.append(latex_chain("h_{2s}", "h(p_2, s_1)", q(s2s.h, _E, system)))
        eta = _r(inp.out_value, 4)
        if dev == "turbine":
            text = "Con el rendimiento isoentrópico de la turbina (vademecum §10.4.1)."
            dhs = s1.h - s2s.h
            lines.append(
                _chain(
                    "h_1 - h_{2s}",
                    _diff(n(s1.h, _E, system), n(s2s.h, _E, system)),
                    q(dhs, _E, system),
                )
            )
            lines.append(
                _chain(
                    "h_2",
                    r"h_1 - \eta_s\,(h_1 - h_{2s})",
                    _diff(n(s1.h, _E, system), times(eta, n(dhs, _E, system))),
                    q(s2.h, _E, system),
                )
            )
        elif dev in ("compressor", "pump"):
            name = "del compresor (§10.4.2)" if dev == "compressor" else "de la bomba (§10.4.3)"
            text = f"Con el rendimiento isoentrópico {name}."
            dhs = s2s.h - s1.h
            lines.append(
                _chain(
                    "h_{2s} - h_1",
                    _diff(n(s2s.h, _E, system), n(s1.h, _E, system)),
                    q(dhs, _E, system),
                )
            )
            lines.append(
                _chain(
                    "h_2",
                    r"h_1 + \frac{h_{2s} - h_1}{\eta_s}",
                    f"{n(s1.h, _E, system)} + {frac(n(dhs, _E, system), eta)}",
                    q(s2.h, _E, system),
                )
            )
        else:  # tobera
            text = (
                "Tobera (vademecum §10.4.4): η_N = ω₂²/ω₂ₛ², con la energía cinética que daría "
                f"la expansión isoentrópica.{_ke_text(system)}"
            )
            den = _ke_den(system)
            ke1 = r.V1**2 / 2.0
            dhs = s1.h - s2s.h
            ke2s = ke1 + dhs - dpe
            ke2 = inp.out_value * ke2s
            lines.append(
                _chain(
                    "h_1 - h_{2s}",
                    _diff(n(s1.h, _E, system), n(s2s.h, _E, system)),
                    q(dhs, _E, system),
                )
            )
            sym = rf"\frac{{\omega_1^2}}{{{den}}} + (h_1 - h_{{2s}})" + (
                r" - \Delta e_p" if dpe else ""
            )
            nums = _sum(n(ke1, _E, system), n(dhs, _E, system))
            if dpe:
                nums = _diff(nums, n(dpe, _E, system))
            lines.append(
                _stack(rf"\frac{{\omega_{{2s}}^2}}{{{den}}}", sym, nums, q(ke2s, _E, system))
            )
            lines.append(
                _chain(
                    rf"\frac{{\omega_2^2}}{{{den}}}",
                    rf"\eta_s\,\frac{{\omega_{{2s}}^2}}{{{den}}}",
                    times(eta, n(ke2s, _E, system)),
                    q(ke2, _E, system),
                )
            )
            k = _ke_tex(system)
            lines.append(
                _chain(
                    r"\omega_2",
                    rf"\sqrt{{{times('2', *([k] if k else []), n(ke2, _E, system))}}}",
                    q(r.V2, _SP, system),
                )
            )
            dke = ke2 - ke1
            lines.append(
                _chain(
                    r"\Delta e_c",
                    rf"\frac{{\omega_2^2 - \omega_1^2}}{{{den}}}",
                    _diff(n(ke2, _E, system), n(ke1, _E, system)),
                    q(dke, _E, system),
                )
            )
            if dpe:
                lines.append(
                    _chain(
                        r"\Delta e_c + \Delta e_p",
                        _sum(n(dke, _E, system), n(dpe, _E, system)),
                        q(dke + dpe, _E, system),
                    )
                )
                lines.append(
                    _chain(
                        "h_2",
                        r"h_1 - (\Delta e_c + \Delta e_p)",
                        _diff(n(s1.h, _E, system), n(dke + dpe, _E, system)),
                        q(s2.h, _E, system),
                    )
                )
            else:
                lines.append(
                    _chain(
                        "h_2",
                        r"h_1 - \Delta e_c",
                        _diff(n(s1.h, _E, system), n(dke, _E, system)),
                        q(s2.h, _E, system),
                    )
                )
        pair = "PH"
    elif r.unknown == "h":
        text = "El primer principio (el paso siguiente) da h₂; con p₂ queda el estado."
        pair = "PH"
    else:
        text = f"Con p₂ y {'T₂' if out == 'T' else 'x₂'}."
        pair = "TP" if out == "T" else "PX"
    if dev != "valve" and r.unknown == "h":
        return ProcedureStep("Salida", text, tuple(lines))
    lines += _prop_lines(sub, s2, "2", pair, keys, system)
    return ProcedureStep("Salida", text, tuple(lines))


def _device_energy(r: DeviceResult, system: UnitSystem) -> ProcedureStep:
    inp = r.inputs
    dev, sub = inp.device, inp.substance
    s1, s2 = r.state1, r.state2
    den = _ke_den(system)
    k = _ke_tex(system)
    lines: list[str] = []
    has_ke = r.V1 != 0.0 or r.V2 != 0.0
    has_pe = inp.z1 != inp.z2
    if has_ke:
        lines.append(
            _chain(
                r"\Delta e_c",
                frac(r"\omega_2^2 - \omega_1^2", den),
                frac(_diff(rf"{n(r.V2, _SP, system)}^2", rf"{n(r.V1, _SP, system)}^2"), den),
                q(r.dke, _E, system),
            )
        )
    if has_pe:
        dz = times(n(G, _G, system), _paren_diff(n(inp.z2, _L, system), n(inp.z1, _L, system)))
        lines.append(
            _chain(
                r"\Delta e_p",
                frac(r"g\,(z_2 - z_1)", k) if k else r"g\,(z_2 - z_1)",
                frac(dz, k) if k else dz,
                q(r.dpe, _E, system),
            )
        )
    terms_sym = [r"\Delta h"] + [r"\Delta e_c"] * has_ke + [r"\Delta e_p"] * has_pe
    terms_num = [n(r.dh, _E, system)]
    terms_num += [n(r.dke, _E, system)] * has_ke + [n(r.dpe, _E, system)] * has_pe
    text = "Vademecum §3.3, por kg: q − w = Δh + Δe_c + Δe_p."
    if has_ke or has_pe:
        text += _ke_text(system)
    unknown = r.unknown
    if unknown in ("w", "q", "V"):
        lines.insert(
            0,
            _chain(
                r"\Delta h", _diff(n(s2.h, _E, system), n(s1.h, _E, system)), q(r.dh, _E, system)
            ),
        )
    total = r.dh + r.dke + r.dpe
    if len(terms_sym) > 1 and unknown in ("w", "q"):
        if has_ke and has_pe:
            kp = r.dke + r.dpe
            lines.append(
                _chain(
                    r"\Delta e_c + \Delta e_p",
                    _sum(n(r.dke, _E, system), n(r.dpe, _E, system)),
                    q(kp, _E, system),
                )
            )
            lines.append(
                _chain(
                    " + ".join(terms_sym),
                    _sum(n(r.dh, _E, system), n(kp, _E, system)),
                    q(total, _E, system),
                )
            )
        else:
            lines.append(_chain(" + ".join(terms_sym), _sum(*terms_num), q(total, _E, system)))
    group = f"({' + '.join(terms_sym)})" if len(terms_sym) > 1 else terms_sym[0]
    if unknown == "w":
        text += f" {DEVICES[dev]}: el balance da el trabajo (w > 0 si lo entrega)."
        lines.append(
            _chain(
                "w",
                f"q - {group}",
                _diff(n(r.q, _E, system), n(total, _E, system)),
                q(r.w, _E, system),
            )
        )
    elif unknown == "q":
        text += " El balance da el calor (q > 0 si lo recibe el fluido)."
        lines.append(
            _chain(
                "q",
                f"w + {group}",
                _sum(n(r.w, _E, system), n(total, _E, system)),
                q(r.q, _E, system),
            )
        )
    elif unknown == "V":
        text += " Sin trabajo, el balance da la velocidad de salida."
        ke1, ke2 = r.V1**2 / 2.0, r.V2**2 / 2.0
        text += " Con e_c = ω²/2 la energía cinética por kg:"
        lines.append(
            _chain(
                "e_{c1}",
                frac(r"\omega_1^2", den),
                frac(rf"{n(r.V1, _SP, system)}^2", den),
                q(ke1, _E, system),
            )
        )
        gain_sym = ("q - " if r.q else "-") + r"\Delta h" + (r" - \Delta e_p" if has_pe else "")
        gain = r.q - r.dh - r.dpe
        nums = _diff(n(r.q, _E, system), n(r.dh, _E, system)) if r.q else n(-r.dh, _E, system)
        if has_pe:
            nums = _diff(nums, n(r.dpe, _E, system))
        if r.q or has_pe:
            lines.append(_chain(gain_sym, nums, q(gain, _E, system)))
        sym = rf"e_{{c1}} + ({gain_sym})" if r.q or has_pe else r"e_{c1} - \Delta h"
        lines.append(
            _chain("e_{c2}", sym, _sum(n(ke1, _E, system), n(gain, _E, system)), q(ke2, _E, system))
        )
        lines.append(
            _chain(
                r"\omega_2",
                rf"\sqrt{{{times('2', *([k] if k else []), n(ke2, _E, system))}}}",
                q(r.V2, _SP, system),
            )
        )
    elif dev == "valve":
        text = (
            "Vademecum §3.3 con q = w = 0 y Δe_c = 0: h₂ = h₁ − Δe_p (= h₁ sin cambio de "
            "altura). Ya está en el paso anterior."
        )
        lines.insert(
            0,
            _chain(
                r"\Delta h", _diff(n(s2.h, _E, system), n(s1.h, _E, system)), q(r.dh, _E, system)
            ),
        )
    else:  # h₂ del balance
        text += " Con el calor, el trabajo y las velocidades dados, el balance da h₂."
        qw = r.q - r.w
        lines.append(
            _chain("q - w", _diff(n(r.q, _E, system), n(r.w, _E, system)), q(qw, _E, system))
        )
        if len(terms_sym) > 1:
            avail = qw - (r.dke if has_ke else 0.0) - (r.dpe if has_pe else 0.0)
            extra_sym = "".join(f" - {t}" for t in terms_sym[1:])
            nums = n(qw, _E, system)
            for t in terms_num[1:]:
                nums = _diff(nums, t)
            lines.append(_chain(rf"q - w{extra_sym}", nums, q(avail, _E, system)))
            lines.append(
                _stack(
                    "h_2",
                    rf"h_1 + (q - w{extra_sym})",
                    _sum(n(s1.h, _E, system), n(avail, _E, system)),
                    q(s2.h, _E, system),
                )
            )
        else:
            lines.append(
                _chain(
                    "h_2",
                    "h_1 + (q - w)",
                    _sum(n(s1.h, _E, system), n(qw, _E, system)),
                    q(s2.h, _E, system),
                )
            )
        lines += _prop_lines(sub, s2, "2", "PH", ("T", "v", "s"), system)
    return ProcedureStep("Primer principio", text, tuple(lines))


def _device_rates(r: DeviceResult, system: UnitSystem) -> ProcedureStep | None:
    inp = r.inputs
    lines: list[str] = []
    text = ""
    if inp.flow == "power":
        term, sym = (r.w, "w") if r.unknown == "w" else (r.q, "q")
        big = r"\dot{W}" if sym == "w" else r"\dot{Q}"
        text = f"Con la potencia como dato: ṁ = |{'Ẇ' if sym == 'w' else 'Q̇'}|/|{sym}|."
        lines.append(
            latex_chain(
                r"\dot{m}",
                frac(rf"|{big}|", rf"|{sym}|"),
                frac(n(inp.flow_value, _PW, system), n(abs(term), _E, system)),
                q(r.m_dot, _MF, system),
            )
        )
    if r.w != 0.0:
        lines.append(
            _chain(
                r"\dot{W}",
                r"\dot{m}\,w",
                times(n(r.m_dot, _MF, system), n(r.w, _E, system)),
                q(r.W_dot, _PW, system),
            )
        )
    if r.q != 0.0:
        lines.append(
            _chain(
                r"\dot{Q}",
                r"\dot{m}\,q",
                times(n(r.m_dot, _MF, system), n(r.q, _E, system)),
                q(r.Q_dot, _PW, system),
            )
        )
    if r.V2 > 0.0 and inp.device in ("nozzle", "diffuser"):
        assert r.A2 is not None
        num = times(n(r.m_dot, _MF, system), n(r.state2.v, _V, system))
        lines.append(
            latex_chain(
                "A_2",
                frac(r"\dot{m}\,v_2", r"\omega_2"),
                frac(num, n(r.V2, _SP, system)),
                q(r.A2, _AR, system),
            )
        )
    if not lines:
        return None
    if not text:
        text = "Con el caudal: Ẇ = ṁ·w y Q̇ = ṁ·q."
    return ProcedureStep("Potencias", text, tuple(lines))


def _eta_step(r: DeviceResult, system: UnitSystem) -> ProcedureStep:
    s1, s2, s2s = r.state1, r.state2, r.state2s
    assert s2s is not None and r.eta_s is not None
    dev = r.inputs.device
    lines = [latex_chain("h_{2s}", "h(p_2, s_1)", q(s2s.h, _E, system))]
    if dev == "turbine":
        lines.append(
            _chain(
                r"\eta_s",
                frac("h_1 - h_2", "h_1 - h_{2s}"),
                frac(n(s1.h - s2.h, _E, system), n(s1.h - s2s.h, _E, system)),
                _r(r.eta_s, 4),
            )
        )
        text = "Adiabática: η_s = (h₁ − h₂)/(h₁ − h₂ₛ) (vademecum §10.4.1)."
    elif dev in ("compressor", "pump"):
        lines.append(
            _chain(
                r"\eta_s",
                frac("h_{2s} - h_1", "h_2 - h_1"),
                frac(n(s2s.h - s1.h, _E, system), n(s2.h - s1.h, _E, system)),
                _r(r.eta_s, 4),
            )
        )
        text = "Adiabático: η_s = (h₂ₛ − h₁)/(h₂ − h₁) (vademecum §10.4.2 y §10.4.3)."
    else:
        lines.append(latex_chain(r"\eta_s", frac(r"\omega_2^2", r"\omega_{2s}^2"), _r(r.eta_s, 4)))
        text = "Adiabática: η_N = ω₂²/ω₂ₛ² (vademecum §10.4.4)."
    return ProcedureStep("Rendimiento isoentrópico", text, tuple(lines))


def mixing_steps(r: MixingResult, system: UnitSystem) -> list[ProcedureStep]:
    """Estados, ṁ₁h₁ + ṁ₂h₂ + Q̇ = ṁ₃h₃ y la entropía generada."""
    inp = r.inputs
    sub = inp.substance
    (a, b), c = r.inlets, r.outlet
    m1, m2 = r.m_dots
    keys = ("T", "p", "h", "s")
    lines = [latex_chain("p_3", q(c.p, _P, system))]
    lines += _prop_lines(sub, a, "1", inp.inlet_1.pair, keys, system)
    lines += _prop_lines(sub, b, "2", inp.inlet_2.pair, keys, system)
    if inp.out != "h":
        lines += _prop_lines(sub, c, "3", "TP" if inp.out == "T" else "PX", keys, system)
    steps = [
        ProcedureStep(
            "Estados", f"{_model_text(sub)} La salida está a la presión p₃.", tuple(lines)
        )
    ]
    Q = inp.Q_dot_W
    energy: list[str] = []
    if inp.out == "h":
        qsym = r" + \dot{Q}" if Q else ""
        for k, (m, st) in enumerate(((m1, a), (m2, b)), start=1):
            energy.append(
                _chain(
                    rf"\dot{{m}}_{k} h_{k}",
                    times(n(m, _MF, system), n(st.h, _E, system)),
                    q(m * st.h, _PW, system),
                )
            )
        num_terms = [n(m1 * a.h, _PW, system), n(m2 * b.h, _PW, system)]
        if Q:
            num_terms.append(n(Q, _PW, system))
        energy.append(
            _stack(
                rf"h_3 = \frac{{\dot{{m}}_1 h_1 + \dot{{m}}_2 h_2{qsym}}}"
                r"{\dot{m}_1 + \dot{m}_2}",
                frac(_sum(*num_terms), _sum(n(m1, _MF, system), n(m2, _MF, system))),
                q(c.h, _E, system),
            )
        )
        energy += _prop_lines(sub, c, "3", "PH", ("T", "s"), system)
        text = "Vademecum §3.3: ṁ₁h₁ + ṁ₂h₂ + Q̇ = ṁ₃h₃ con ṁ₃ = ṁ₁ + ṁ₂: da h₃."
    else:
        u = 0 if inp.inlet_1.m_dot is None else 1
        kk = 1 - u
        h = (a.h, b.h)
        mk = (m1, m2)[kk]
        lu, lk = str(u + 1), str(kk + 1)
        energy.append(
            _chain(
                f"h_3 - h_{lk}",
                _diff(n(c.h, _E, system), n(h[kk], _E, system)),
                q(c.h - h[kk], _E, system),
            )
        )
        energy.append(
            _chain(
                f"h_{lu} - h_3",
                _diff(n(h[u], _E, system), n(c.h, _E, system)),
                q(h[u] - c.h, _E, system),
            )
        )
        num = times(n(mk, _MF, system), n(c.h - h[kk], _E, system))
        if Q:
            num = _diff(num, n(Q, _PW, system))
        energy.append(
            _stack(
                rf"\dot{{m}}_{lu} = \frac{{\dot{{m}}_{lk}\,(h_3 - h_{lk})"
                + (r" - \dot{Q}" if Q else "")
                + rf"}}{{h_{lu} - h_3}}",
                frac(num, n(h[u] - c.h, _E, system)),
                q((m1, m2)[u], _MF, system),
            )
        )
        text = (
            "Vademecum §3.3: ṁ₁h₁ + ṁ₂h₂ + Q̇ = (ṁ₁ + ṁ₂)·h₃, con la salida dada: da el caudal "
            "que falta."
        )
    steps.append(ProcedureStep("Primer principio", text, tuple(energy)))
    flows = (m1 * (c.s - a.s), m2 * (c.s - b.s))
    lines = []
    for k, (m, st) in enumerate(((m1, a), (m2, b)), start=1):
        lines += _times_diff(
            rf"\dot{{m}}_{k} (s_3 - s_{k})",
            "",
            [n(m, _MF, system)],
            c.s,
            st.s,
            _S,
            f"s_3 - s_{k}",
            q(flows[k - 1], _SF, system),
            system,
        )
    dS = sum(flows)
    lines.append(
        _chain(
            r"\Delta\dot{S}",
            r"\dot{m}_1 (s_3 - s_1) + \dot{m}_2 (s_3 - s_2)",
            _sum(n(flows[0], _SF, system), n(flows[1], _SF, system)),
            q(dS, _SF, system),
        )
    )
    lines += _sgen_lines(dS, Q, r.T_b, r.S_gen_dot, inp.T0_K, system, rate=True)
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "Vademecum §10.3.2: Ṡ_gen = Σ_sal ṁ·s − Σ_ent ṁ·s − Q̇/T_b; con ṁ₃ = ṁ₁ + ṁ₂, "
            "Σ_sal ṁ·s − Σ_ent ṁ·s = ṁ₁·(s₃ − s₁) + ṁ₂·(s₃ − s₂). "
            + _verdict_text(r.S_gen_dot, r.violation, "Ṡ_gen"),
            tuple(lines),
        )
    )
    return numbered(steps)


def exchanger_steps(r: ExchangerResult, system: UnitSystem) -> list[ProcedureStep]:
    """Estados, Σṁ·Δh = Q̇, el calor que pasa y la entropía generada."""
    inp = r.inputs
    streams = (inp.stream_a, inp.stream_b)
    tags = ("A", "B")
    lines: list[str] = []
    for i, s in enumerate(streams):
        keys = ("T", "p", "h", "s")
        lines += _prop_lines(s.substance, r.inlets[i], f"{tags[i]}1", s.pair, keys, system)
        if s.out != "h":
            pair: PairCode = "TP" if s.out == "T" else "PX"
            lines += _prop_lines(s.substance, r.outlets[i], f"{tags[i]}2", pair, keys, system)
    names = " y ".join(
        f"{tags[i]} = {s.label or 'corriente ' + tags[i]} ({s.substance.label})"
        for i, s in enumerate(streams)
    )
    steps = [ProcedureStep("Estados", f"Corrientes: {names}.", tuple(lines))]
    Q = inp.Q_dot_W
    unknown_flow = [s.m_dot is None for s in streams]
    energy: list[str] = []
    if any(unknown_flow):
        u = unknown_flow.index(True)
    else:
        u = 0 if streams[0].out == "h" else 1
    kk = 1 - u
    tu, tk = tags[u], tags[kk]
    dhk = r.outlets[kk].h - r.inlets[kk].h
    energy.append(
        _chain(
            rf"h_{{{tk}2}} - h_{{{tk}1}}",
            _diff(n(r.outlets[kk].h, _E, system), n(r.inlets[kk].h, _E, system)),
            q(dhk, _E, system),
        )
    )
    rest_sym = (r"\dot{Q} - " if Q else "-") + rf"\dot{{m}}_{tk}\,(h_{{{tk}2}} - h_{{{tk}1}})"
    rest_num = (f"{n(Q, _PW, system)} - " if Q else "-") + times(
        n(r.m_dots[kk], _MF, system), latex_paren(n(dhk, _E, system))
    )
    Qu = Q - r.m_dots[kk] * dhk  # el calor que recibe la corriente incógnita
    energy.append(_stack(rf"\dot{{Q}}_{tu} = {rest_sym}", rest_num, q(Qu, _PW, system)))
    rest_sym, rest_num = rf"\dot{{Q}}_{tu}", n(Qu, _PW, system)
    if any(unknown_flow):
        dhu = r.outlets[u].h - r.inlets[u].h
        energy.append(
            _chain(
                rf"h_{{{tu}2}} - h_{{{tu}1}}",
                _diff(n(r.outlets[u].h, _E, system), n(r.inlets[u].h, _E, system)),
                q(dhu, _E, system),
            )
        )
        energy.append(
            _stack(
                rf"\dot{{m}}_{tu} = \frac{{{rest_sym}}}{{h_{{{tu}2}} - h_{{{tu}1}}}}",
                frac(rest_num, n(dhu, _E, system)),
                q(r.m_dots[u], _MF, system),
            )
        )
    else:
        energy.append(
            _stack(
                rf"h_{{{tu}2}} = h_{{{tu}1}} + \frac{{{rest_sym}}}{{\dot{{m}}_{tu}}}",
                f"{n(r.inlets[u].h, _E, system)} + " + frac(rest_num, n(r.m_dots[u], _MF, system)),
                q(r.outlets[u].h, _E, system),
            )
        )
        energy += _prop_lines(
            streams[u].substance, r.outlets[u], f"{tu}2", "PH", ("T", "s"), system
        )
    energy.append(latex_chain(r"\dot{Q}_{\mathrm{transf}}", q(r.Q_transfer, _PW, system)))
    text = (
        "Vademecum §3.3 para todo el intercambiador: ṁ_A·(h_A2 − h_A1) + ṁ_B·(h_B2 − h_B1) = Q̇"
        + (" (el calor de afuera: negativo si se pierde)." if Q else " = 0 (aislado por fuera).")
    )
    steps.append(ProcedureStep("Primer principio", text, tuple(energy)))
    sl: list[str] = []
    for i in range(2):
        t = tags[i]
        sl += _times_diff(
            rf"\Delta\dot{{S}}_{t}",
            rf"\dot{{m}}_{t}\,(s_{{{t}2}} - s_{{{t}1}})",
            [n(r.m_dots[i], _MF, system)],
            r.outlets[i].s,
            r.inlets[i].s,
            _S,
            rf"s_{{{t}2}} - s_{{{t}1}}",
            q(r.entropy_change(i), _SF, system),
            system,
        )
    dS = r.entropy_change(0) + r.entropy_change(1)
    sl.append(
        _chain(
            r"\Delta\dot{S}",
            r"\Delta\dot{S}_A + \Delta\dot{S}_B",
            _sum(n(r.entropy_change(0), _SF, system), n(r.entropy_change(1), _SF, system)),
            q(dS, _SF, system),
        )
    )
    sl += _sgen_lines(dS, Q, r.T_b, r.S_gen_dot, inp.T0_K, system, rate=True)
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "Vademecum §10.3.2: Ṡ_gen = Σṁ·(s_sal − s_ent) − Q̇/T_b. "
            + _verdict_text(r.S_gen_dot, r.violation, "Ṡ_gen"),
            tuple(sl),
        )
    )
    return numbered(steps)


# ---------------------------------------------------------------------
# Régimen transitorio
# ---------------------------------------------------------------------


def charging_steps(r: ChargingResult, system: UnitSystem) -> list[ProcedureStep]:
    """Masas, el estado final, Q = m₁(u₂ − u₁) + m_e(u₂ − h_ℓ) y la entropía generada."""
    inp = r.inputs
    sub = inp.substance
    s1, s2, line = r.state1, r.state2, r.line
    V = inp.volume_m3
    lines = _prop_lines(sub, line, r"\ell", inp.line_pair, ("T", "p", "h", "s"), system)
    if s1 is not None:
        lines += _prop_lines(sub, s1, "1", inp.initial_pair, ("T", "p", "v", "u", "s"), system)
        lines.append(_mass_from_volume("m_1", V, s1.v, r.m1, "v_1", system))
        text = f"{_model_text(sub)} La línea (ℓ) y el tanque al principio."
    else:
        text = f"{_model_text(sub)} La línea (ℓ); el tanque está vacío (m₁ = 0)."
    steps = [ProcedureStep("Línea y estado inicial", text, tuple(lines))]
    lines = []
    if inp.end == "T":
        lines += _prop_lines(sub, s2, "2", "TP", ("v", "u", "s"), system)
        lines.append(_mass_from_volume("m_2", V, s2.v, r.m2, "v_2", system))
        text = "Con p₂ y T₂ el estado final queda fijo, y la masa sale del volumen."
    elif s1 is None and r.Q == 0.0:
        text = (
            "Tanque vacío y adiabático: el primer principio da m₂·u₂ = m₂·h_ℓ, o sea u₂ = h_ℓ. "
            "Con p₂ y u₂ queda el estado final."
        )
        lines.append(latex_chain("u_2", r"h_{\ell}", q(s2.u, _E, system)))
        lines += _prop_lines(sub, s2, "2", "PU", ("T", "v", "s"), system)
        lines.append(_mass_from_volume("m_2", V, s2.v, r.m2, "v_2", system))
    else:
        text = (
            "Del primer principio, m₂·u₂ = m₁·u₁ + (m₂ − m₁)·h_ℓ + Q, o sea "
            "u₂ = h_ℓ + [m₁·(u₁ − h_ℓ) + Q]/m₂. La masa final m₂ es la que hace que el estado "
            "con v₂ = V/m₂ y ese u₂ tenga la presión p₂: se busca por tanteo (acá, con brentq). "
            "El resultado:"
        )
        lines.append(latex_chain("m_2", q(r.m2, _M, system)))
        lines.append(
            latex_chain(
                "v_2",
                frac("V", "m_2"),
                frac(n(V, _VOL, system), n(r.m2, _M, system)),
                q(s2.v, _V, system),
            )
        )
        num_sym, num = "", ""
        if s1 is not None:
            lines.append(
                _chain(
                    r"u_1 - h_{\ell}",
                    _diff(n(s1.u, _E, system), n(line.h, _E, system)),
                    q(s1.u - line.h, _E, system),
                )
            )
            num_sym = r"m_1 (u_1 - h_{\ell})"
            num = times(n(r.m1, _M, system), n(s1.u - line.h, _E, system))
        if r.Q:
            num_sym = f"{num_sym} + Q" if num_sym else "Q"
            num = _sum(num, n(r.Q, _EN, system)) if num else n(r.Q, _EN, system)
        excess = r.m1 * (s1.u - line.h) + r.Q if s1 is not None else r.Q
        if s1 is not None and r.Q:
            lines.append(_chain(num_sym, num, q(excess, _EN, system)))
            num = n(excess, _EN, system)
        lines.append(
            _stack(
                rf"u_2 = h_{{\ell}} + \frac{{{num_sym}}}{{m_2}}",
                f"{n(line.h, _E, system)} + {frac(num, n(r.m2, _M, system))}",
                q(s2.u, _E, system),
            )
        )
        lines += _prop_lines(sub, s2, "2", "VU", ("T", "p", "s"), system)
    steps.append(ProcedureStep("Estado final", text, tuple(lines)))
    lines = [
        latex_chain(
            "m_e",
            "m_2 - m_1",
            _diff(n(r.m2, _M, system), n(r.m1, _M, system)),
            q(r.m_in, _M, system),
        )
    ]
    t1 = r.m1 * (s2.u - s1.u) if s1 is not None else 0.0
    t2 = r.m_in * (s2.u - line.h)
    if s1 is not None:
        lines += _times_diff(
            r"m_1 (u_2 - u_1)",
            "",
            [n(r.m1, _M, system)],
            s2.u,
            s1.u,
            _E,
            "u_2 - u_1",
            q(t1, _EN, system),
            system,
        )
    lines += _times_diff(
        r"m_e (u_2 - h_{\ell})",
        "",
        [n(r.m_in, _M, system)],
        s2.u,
        line.h,
        _E,
        r"u_2 - h_{\ell}",
        q(t2, _EN, system),
        system,
    )
    if s1 is not None:
        lines.append(
            _chain(
                "Q",
                r"m_1 (u_2 - u_1) + m_e (u_2 - h_{\ell})",
                _sum(n(t1, _EN, system), n(t2, _EN, system)),
                q(r.Q, _EN, system),
            )
        )
    else:
        lines.append(latex_chain("Q", r"m_e (u_2 - h_{\ell})", q(r.Q, _EN, system)))
    steps.append(
        ProcedureStep(
            "Primer principio",
            "Vademecum §3.4 con un tanque rígido (W = 0) y sin energía cinética: "
            "Q = m₂u₂ − m₁u₁ − m_e·h_ℓ, que es lo mismo que Q = m₁·(u₂ − u₁) + m_e·(u₂ − h_ℓ) "
            "(lo que ya estaba y lo que entra; así no depende de la referencia de u y h).",
            tuple(lines),
        )
    )
    a1 = r.m1 * (s2.s - s1.s) if s1 is not None else 0.0
    a2 = r.m_in * (s2.s - line.s)
    lines = []
    if s1 is not None:
        lines += _times_diff(
            r"m_1 (s_2 - s_1)",
            "",
            [n(r.m1, _M, system)],
            s2.s,
            s1.s,
            _S,
            "s_2 - s_1",
            q(a1, _EN_S, system),
            system,
        )
    lines += _times_diff(
        r"m_e (s_2 - s_{\ell})",
        "",
        [n(r.m_in, _M, system)],
        s2.s,
        line.s,
        _S,
        r"s_2 - s_{\ell}",
        q(a2, _EN_S, system),
        system,
    )
    dS = a1 + a2
    if s1 is not None:
        lines.append(
            _chain(
                r"\Delta S",
                r"m_1 (s_2 - s_1) + m_e (s_2 - s_{\ell})",
                _sum(n(a1, _EN_S, system), n(a2, _EN_S, system)),
                q(dS, _EN_S, system),
            )
        )
    else:
        lines.append(latex_chain(r"\Delta S", r"m_e (s_2 - s_{\ell})", q(dS, _EN_S, system)))
    lines += _sgen_lines(dS, r.Q, r.T_b, r.S_gen, inp.T0_K, system)
    steps.append(
        ProcedureStep(
            "Segundo principio",
            "Vademecum §10.3.3: (S₂ − S₁)_VC = Q/T_b + m_e·s_ℓ + S_gen; reagrupado, "
            "ΔS = m₂s₂ − m₁s₁ − m_e·s_ℓ = m₁·(s₂ − s₁) + m_e·(s₂ − s_ℓ). "
            + _verdict_text(r.S_gen, r.violation),
            tuple(lines),
        )
    )
    return numbered(steps)


def discharging_steps(r: DischargingResult, system: UnitSystem) -> list[ProcedureStep]:
    """Masas, ∫h·dm (Simpson) contra el flujo uniforme, el calor y la entropía generada."""
    inp = r.inputs
    sub = inp.substance
    s1, s2 = r.state1, r.state2
    V = inp.volume_m3
    lines = _prop_lines(sub, s1, "1", inp.pair1, ("T", "p", "v", "u", "h", "s"), system)
    lines.append(_mass_from_volume("m_1", V, s1.v, r.m1, "v_1", system))
    steps = [ProcedureStep("Estado inicial", _model_text(sub), tuple(lines))]
    if inp.mode == "adiabatic":
        text = (
            "Adiabático y sin irreversibilidades adentro: lo que queda en el tanque se expande "
            "en forma isoentrópica (con s constante, d(m·u) = h·dm). s₂ = s₁."
        )
        lines = [latex_chain("s_2", "s_1", q(s2.s, _S, system))]
        lines += _prop_lines(sub, s2, "2", "PS", ("T", "v", "u", "h"), system)
    else:
        text = "Isotérmico: el calor mantiene T₂ = T₁."
        lines = [latex_chain("T_2", "T_1", q(s2.T, _T, system))]
        lines += _prop_lines(sub, s2, "2", "TP", ("v", "u", "h", "s"), system)
    lines.append(_mass_from_volume("m_2", V, s2.v, r.m2, "v_2", system))
    steps.append(ProcedureStep("Estado final", text, tuple(lines)))
    m_s = r.m_out
    h_mean = (s1.h + s2.h) / 2.0
    lines = [
        latex_chain(
            "m_s", "m_1 - m_2", _diff(n(r.m1, _M, system), n(r.m2, _M, system)), q(m_s, _M, system)
        ),
        latex_chain(r"\int h\,dm", q(r.H_out, _EN, system)),
        _chain(
            r"\frac{h_1 + h_2}{2}",
            frac(_sum(n(s1.h, _E, system), n(s2.h, _E, system)), "2"),
            q(h_mean, _E, system),
        ),
        _chain(
            r"m_s\,\frac{h_1 + h_2}{2}",
            times(n(m_s, _M, system), n(h_mean, _E, system)),
            q(r.H_out_uniform, _EN, system),
        ),
    ]
    steps.append(
        ProcedureStep(
            "Entalpía que sale",
            "Lo que sale lleva la entalpía del tanque en cada momento, y esa entalpía cambia: "
            "Σ m·h es la integral ∫h·dm a lo largo del camino (Simpson con "
            f"{INTEGRATION_POINTS} puntos). La aproximación de flujo uniforme de Çengel §5-5 "
            "usa el promedio de los extremos.",
            tuple(lines),
        )
    )
    U1, U2 = r.m1 * s1.u, r.m2 * s2.u
    dU = U2 - U1
    lines = [
        _chain(
            "U_1", r"m_1 u_1", times(n(r.m1, _M, system), n(s1.u, _E, system)), q(U1, _EN, system)
        ),
        _chain(
            "U_2", r"m_2 u_2", times(n(r.m2, _M, system), n(s2.u, _E, system)), q(U2, _EN, system)
        ),
        _chain(
            r"\Delta U",
            "U_2 - U_1",
            _diff(n(U2, _EN, system), n(U1, _EN, system)),
            q(dU, _EN, system),
        ),
    ]
    if inp.mode == "adiabatic":
        text = (
            "Vademecum §3.4 con Q = 0 y W = 0: −ΔU = m₁u₁ − m₂u₂ = ∫h·dm. Se cumple (la "
            "diferencia es el error de la integración numérica); con el promedio de los "
            "extremos no cierra."
        )
        lines.append(
            latex_chain(r"-\Delta U - \int h\,dm", q(r.integration_residual, _EN, system, 3))
        )
        lines.append(
            _chain(
                r"Q_{\mathrm{unif}}",
                r"\Delta U + m_s\,\frac{h_1 + h_2}{2}",
                _sum(n(dU, _EN, system), n(r.H_out_uniform, _EN, system)),
                q(r.Q_uniform, _EN, system),
            )
        )
    else:
        text = "Vademecum §3.4 con W = 0: Q = m₂u₂ − m₁u₁ + ∫h·dm."
        lines.append(
            _chain(
                "Q",
                r"\Delta U + \int h\,dm",
                _sum(n(dU, _EN, system), n(r.H_out, _EN, system)),
                q(r.Q, _EN, system),
            )
        )
    steps.append(ProcedureStep("Primer principio", text, tuple(lines)))
    S1, S2 = r.m1 * s1.s, r.m2 * s2.s
    dS = S2 - S1 + r.S_out
    valve = inp.p_out_Pa is not None
    s_int = r"\int s_{\mathrm{sal}}\,dm" if valve else r"\int s\,dm"
    lines = [
        _chain(
            "S_1", r"m_1 s_1", times(n(r.m1, _M, system), n(s1.s, _S, system)), q(S1, _EN_S, system)
        ),
        _chain(
            "S_2", r"m_2 s_2", times(n(r.m2, _M, system), n(s2.s, _S, system)), q(S2, _EN_S, system)
        ),
        latex_chain(s_int, q(r.S_out, _EN_S, system)),
        _stack(
            rf"\Delta S = S_2 - S_1 + {s_int}",
            _sum(_diff(n(S2, _EN_S, system), n(S1, _EN_S, system)), n(r.S_out, _EN_S, system)),
            q(dS, _EN_S, system),
        ),
        *_sgen_lines(dS, r.Q, r.T_b, r.S_gen, inp.T0_K, system),
    ]
    text = "Vademecum §10.3.3: (S₂ − S₁)_VC = Q/T_b − Σ m·s + S_gen, con Σ m·s = ∫s·dm."
    if valve:
        text += (
            " Con la válvula en el volumen de control, lo que sale está a p_sal con la entalpía "
            "del tanque: s_sal = s(p_sal, h)."
        )
    steps.append(
        ProcedureStep(
            "Segundo principio", f"{text} {_verdict_text(r.S_gen, r.violation)}", tuple(lines)
        )
    )
    return numbered(steps)

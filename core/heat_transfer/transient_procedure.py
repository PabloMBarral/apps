"""Procedimiento didáctico de la conducción transitoria (LaTeX), en el sistema de unidades activo.

Çengel y Ghajar (2015), §4-1 a §4-4; Incropera et al. (2007), cap. 5:

- la capacidad concentrada: L_c, Bi, b = h·A/(ρ·V·c), la constante de tiempo, T(t) o el
  tiempo, y el calor;
- la pared, el cilindro largo y la esfera con la **solución de un término** (λ₁ y A₁ de la
  ecuación característica, como las tablas del libro) y después la **serie completa**, que
  es el resultado;
- el sólido semi-infinito con la superficie a T_s, un flujo q̇_s o convección, el flujo en
  la superficie, la energía y la penetración; y el contacto de dos sólidos;
- el producto de soluciones, con la regla de Langston para Q/Q_máx.

Factores de unidades a la vista (como el ṁ·c_p de los intercambiadores): α = k/(ρ·c) y
b = h/(ρ·c·L_c) llevan 1000 en el Técnico (c en kJ/(kg·K)) y 3600 en el Inglés (k y h
por hora; el tiempo va en segundos en los tres sistemas). La energía por unidad de área
que sale de un flujo por un tiempo (q̇·t) lleva los mismos números: 1000 (J → kJ) y 3600
(Btu/h por segundo → Btu). Bi, τ, h·√(α·t)/k y ρ·c·V·ΔT cierran solos.

Ancho: los renglones con números se parten (``\\ &\quad``) cuando :func:`_vis` estima que
no entran en un celular; con raíces chicas o tiempos largos aparecen números ×10ⁿ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import erfc, erfcinv, erfcx, j0, j1

from core.heat_transfer.procedure_common import dimensionless, frac, n, numbered, q, sub, times
from core.heat_transfer.transient import (
    BI_LUMPED,
    TAU_ONE_TERM,
    ContactResult,
    Factor,
    FactorResult,
    LumpedResult,
    OneDimResult,
    ProductResult,
    SemiInfiniteResult,
    Solid,
    _n_terms,
    coefficient,
    eigenvalues,
    heat_ratio,
    shape_function,
)
from core.latex import latex_chain, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import UnitSystem, convert_from_si

__all__ = [
    "contact_steps",
    "lumped_steps",
    "one_dim_steps",
    "product_steps",
    "semi_infinite_steps",
]

_T = "temperature"
_DT = "temperature_difference"
_L = "length"
_K = "thermal_conductivity"
_RHO = "density"
_CP = "specific_heat"
_H = "heat_transfer_coefficient"
_ALPHA = "diffusivity"
_TIME = "time"
_FLUX = "heat_flux"
_QA = "energy_per_area"
_QL = "energy_per_length"
_E = "energy"

#: El factor de unidades de α = k/(ρ·c), b = h/(ρ·c·L_c) y q̇·t (vacío en el SI).
_FACTOR: dict[str, str] = {"SI": "", "Técnico": "1000", "Inglés": "3600"}
#: δ = 2·η·√(α·t) con erfc(η) = 0,01.
_PENETRATION = 2.0 * float(erfcinv(0.01))
#: Más allá de este exponente e^x desborda o da números ilegibles: el producto va junto.
_EXP_MAX = 300.0
#: Un renglón de ``latex_chain`` en «caracteres» de :func:`_vis` (324 px son unos 33 dígitos
#: de 9,7 px, menos el signo = y un margen).
_ROW = 29.0


# ---------------------------------------------------------------------
# Ancho de los renglones
# ---------------------------------------------------------------------

_NARROW_CMDS = {",": 0.2, " ": 0.3, ";": 0.3, "quad": 1.0, "qquad": 2.0, "left": 0.0, "right": 0.0}
_FUNCS = {"sin", "cos", "tan", "cot", "ln", "exp"}


def _group(tex: str, i: int) -> tuple[str, int]:
    """El contenido del grupo ``{…}`` que empieza en ``i`` y la posición siguiente."""
    if i >= len(tex) or tex[i] != "{":
        return (tex[i] if i < len(tex) else ""), i + 1
    depth = 0
    for j in range(i, len(tex)):
        if tex[j] == "{":
            depth += 1
        elif tex[j] == "}":
            depth -= 1
            if depth == 0:
                return tex[i + 1 : j], j + 1
    return tex[i + 1 :], len(tex)


def _vis(tex: str) -> float:
    """El ancho aproximado de un renglón de LaTeX, en «caracteres» (un dígito = 1).

    Alcanza para decidir si un renglón se parte: una fracción cuenta lo que su parte más
    ancha, un exponente o un subíndice el 70 %, y los signos +, − e = con sus espacios, 2.
    """
    total = 0.0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            j = i + 1
            if j < len(tex) and not tex[j].isalpha():
                total += _NARROW_CMDS.get(tex[j], 0.5)
                i = j + 1
                continue
            while j < len(tex) and tex[j].isalpha():
                j += 1
            cmd = tex[i + 1 : j]
            if cmd in ("frac", "tfrac"):
                a, j = _group(tex, j)
                b, j = _group(tex, j)
                total += max(_vis(a), _vis(b)) * (0.75 if cmd == "tfrac" else 1.0) + 1.0
            elif cmd in ("text", "mathrm", "operatorname"):
                a, j = _group(tex, j)
                total += float(len(a))
            elif cmd == "sqrt":
                a, j = _group(tex, j)
                total += _vis(a) + 1.0
            elif cmd == "times":
                total += 1.6
            elif cmd == "cdot":
                total += 1.4
            elif cmd in _FUNCS:
                total += len(cmd) + 0.5
            elif cmd in _NARROW_CMDS:
                total += _NARROW_CMDS[cmd]
            else:
                total += 1.0
            i = j
            continue
        if ch in "^_":
            a, i = _group(tex, i + 1)
            total += 0.7 * _vis(a)
            continue
        if ch == "=":
            total += 2.7
        elif ch in "+-":
            total += 2.4
        elif ch in "()":
            total += 0.8
        elif ch in ".,":
            total += 0.55
        elif ch in "{}& ":
            pass
        else:
            total += 1.0
        i += 1
    return total


@dataclass(frozen=True)
class _Pack:
    """Un renglón de números que :func:`_chain` parte si no entra: ``first`` y después cada
    ``(operador, término)``."""

    first: str
    rest: tuple[tuple[str, str], ...] = ()


def _pack(p: _Pack, budget: float) -> str:
    r"""Junta los términos y pasa a otro renglón (``\\ &\quad``) cuando no entran."""
    out, width = p.first, _vis(p.first)
    for op, term in p.rest:
        piece = f" {op} {term}"
        w = _vis(piece)
        if width + w > budget:
            out += rf" \\ &\quad {op} {term}"
            width = 2.0 + w
        else:
            out += piece
            width += w
    return out


def _budget(lhs: str) -> float:
    """Lo que queda para los números en un renglón con ese lado izquierdo."""
    return _ROW - _vis(lhs)


def _chain(lhs: str, *steps: str | _Pack) -> str:
    """:func:`~core.latex.latex_chain` con los renglones de números partidos si no entran (en
    un ``aligned`` el lado izquierdo se suma al ancho de cada renglón)."""
    budget = _budget(lhs)
    return latex_chain(lhs, *(st if isinstance(st, str) else _pack(st, budget) for st in steps))


def _flush(lhs: str, *steps: str | _Pack) -> str:
    """Como :func:`_chain`, con los renglones contra el margen: para un lado izquierdo ancho
    (una fracción), que en un ``aligned`` se sumaría a cada renglón."""
    out = [st if isinstance(st, str) else _pack(st, _ROW) for st in steps]
    rows = [rf"&{lhs} = {out[0]}", *(rf"&= {st}" for st in out[1:])]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _prod(*factors: str) -> _Pack:
    r"""``a \cdot b \cdot c`` (los negativos entre paréntesis), partido si no entra."""
    return _Pack(factors[0], tuple((r"\cdot", latex_paren(f)) for f in factors[1:]))


def _affine(a: str, diff: str, factor: str) -> _Pack:
    r"""``a + (diff) \cdot factor``, como T = T∞ + (T_i − T∞)·θ, partido si no entra."""
    return _Pack(f"{a} + ({diff})", ((r"\cdot", latex_paren(factor)),))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _d(x: float, sig: int = 4) -> str:
    """Un número sin unidad."""
    return dimensionless(x, sig)


def _pw(base: str, exp: str) -> str:
    """``base^{exp}`` con paréntesis si la base lleva ×10ⁿ o es negativa."""
    if r"\times" in base or base.startswith("-"):
        return rf"({base})^{{{exp}}}"
    return rf"{base}^{{{exp}}}"


def _exp_term(A: float, x: float, sig: int = 3) -> str:
    """A·e^(−x) en LaTeX con ``sig`` cifras, aunque e^(−x) sea menor que el menor float."""
    value = A * math.exp(-x) if x < 700.0 else 0.0
    if value != 0.0 or A == 0.0:
        return latex_number(value, sig)
    log10 = math.log10(abs(A)) - x / math.log(10.0)
    e = math.floor(log10)
    m = 10.0 ** (log10 - e)
    if round(m, 2) >= 10.0:
        m, e = m / 10.0, e + 1
    return rf"{'-' if A < 0 else ''}{m:.3g}\times 10^{{{e}}}"


def _with_factor(system: UnitSystem, *factors: str) -> str:
    r"""``1000 \cdot a \cdot b`` (el factor del sistema adelante; nada en el SI)."""
    f = _FACTOR[system]
    return times(f, *factors) if f else times(*factors)


def _over_factor(num: str, system: UnitSystem) -> str:
    """``num/1000`` o ``num/3600`` (``num`` en el SI)."""
    f = _FACTOR[system]
    return frac(num, f) if f else num


def _alpha_latex(solid: Solid, system: UnitSystem, sub_: str = "") -> str:
    r"""α = k/(ρ·c) con el factor del sistema (``sub_`` es el subíndice: ``_A``)."""
    return _chain(
        rf"\alpha{sub_}",
        frac(f"k{sub_}", rf"\rho{sub_}\,c{sub_}"),
        frac(
            n(solid.k, _K, system),
            _with_factor(system, n(solid.rho, _RHO, system), n(solid.c, _CP, system)),
        ),
        q(solid.alpha, _ALPHA, system),
    )


def _alpha_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " Con c en kJ/(kg·K), ρ·c sale en kJ/(m³·K): el 1000 la pasa a J, y α queda en m²/s."
    if system == "Inglés":
        return (
            " Con k en Btu/(h·ft·°F), k/(ρ·c) sale en ft²/h: el 3600 la pasa a ft²/s, porque el "
            "tiempo va en segundos."
        )
    return ""


def _energy_factor_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " q̇·t sale en J/m²: dividido por 1000, en kJ/m²."
    if system == "Inglés":
        return " Con q̇ en Btu/(h·ft²) y t en segundos, q̇·t se divide por 3600 para dar Btu/ft²."
    return ""


def _T_line(
    lhs: str, theta_sym: str, theta_val: float, T_i: float, T_inf: float, system: UnitSystem
) -> str:
    """T = T∞ + (T_i − T∞)·θ."""
    Ti, Tinf = n(T_i, _T, system), n(T_inf, _T, system)
    return _chain(
        lhs,
        rf"T_\infty + (T_i - T_\infty)\,{theta_sym}",
        _affine(Tinf, sub(Ti, Tinf), _d(theta_val)),
        q(T_inf + (T_i - T_inf) * theta_val, _T, system),
    )


def _theta_target(
    lhs: str, sym_T: str, T: float, T_i: float, T_inf: float, system: UnitSystem
) -> str:
    """θ = (T − T∞)/(T_i − T∞) con los números."""
    Tn, Ti, Tinf = n(T, _T, system), n(T_i, _T, system), n(T_inf, _T, system)
    return _chain(
        lhs,
        frac(rf"{sym_T} - T_\infty", r"T_i - T_\infty"),
        frac(sub(Tn, Tinf), sub(Ti, Tinf)),
        _d((T - T_inf) / (T_i - T_inf)),
    )


def _delta_T(T_i: float, T_inf: float, system: UnitSystem) -> tuple[str, str, float]:
    """La diferencia de Q_máx, ordenada para que sea positiva: (símbolo, LaTeX, valor)."""
    Ti, Tinf = n(T_i, _T, system), n(T_inf, _T, system)
    if T_i >= T_inf:
        sym, value = r"T_i - T_\infty", T_i - T_inf
        tex = _chain(sym, sub(Ti, Tinf), q(value, _DT, system))
    else:
        sym, value = r"T_\infty - T_i", T_inf - T_i
        tex = _chain(sym, sub(Tinf, Ti), q(value, _DT, system))
    return sym, tex, value


def _direction(T_i: float, T_inf: float) -> tuple[str, str]:
    """(«entrega», «al fluido») si se enfría; («recibe», «del fluido») si se calienta."""
    return ("entrega", "al fluido") if T_i > T_inf else ("recibe", "del fluido")


# ---------------------------------------------------------------------
# Capacidad concentrada (§4-1)
# ---------------------------------------------------------------------


def _lumped_geometry_step(r: LumpedResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    Lc = q(r.Lc, _L, system)
    V, A = q(r.V, "volume", system), q(r.A, "area", system)
    V_n, A_n = n(r.V, "volume", system), n(r.A, "area", system)
    latex: list[str] = []
    if i.body == "custom":
        latex.append(_chain("L_c", frac("V", "A"), frac(V_n, A_n), Lc))
        text = "Con el volumen V y el área de intercambio A dados, L_c = V/A."
    elif i.body == "sphere":
        D = n(i.dims[0], _L, system)
        latex += [
            _chain("V", frac(r"\pi D^3", "6"), frac(rf"\pi \cdot {_pw(D, '3')}", "6"), V),
            _chain("A", r"\pi D^2", rf"\pi \cdot {_pw(D, '2')}", A),
            _chain("L_c", frac("V", "A"), frac("D", "6"), frac(D, "6"), Lc),
        ]
        text = "Una **esfera** de diámetro D: V = π·D³/6 y A = π·D², así que L_c = D/6."
    elif i.body == "cylinder":
        D, H = n(i.dims[0], _L, system), n(i.dims[1], _L, system)
        caps = frac(rf"\pi \cdot {_pw(D, '2')}", "2")
        quarter = frac(rf"\pi \cdot {_pw(D, '2')}", "4")
        latex += [
            _chain(
                "V",
                r"\frac{\pi D^2}{4}\,L",
                rf"{quarter} \cdot {H}",
                V,
            ),
            _chain(
                "A",
                r"\pi D L + \frac{\pi D^2}{2}",
                _Pack(rf"\pi \cdot {times(D, H)}", (("+", caps),)),
                A,
            ),
            _chain("L_c", frac("V", "A"), frac(V_n, A_n), Lc),
        ]
        text = (
            "Un **cilindro** de diámetro D y largo L que intercambia también por las tapas: "
            "V = π·D²·L/4 y A = π·D·L + 2·π·D²/4."
        )
    elif i.body == "long_cylinder":
        D = n(i.dims[0], _L, system)
        latex += [
            _chain(
                r"\frac{V}{L}",
                frac(r"\pi D^2", "4"),
                frac(rf"\pi \cdot {_pw(D, '2')}", "4"),
                q(r.V, "area", system),
            ),
            _chain("L_c", frac("V", "A"), frac("D", "4"), frac(D, "4"), Lc),
        ]
        text = (
            "Un **cilindro largo** (sin las tapas), por unidad de largo: V = π·D²·L/4 y "
            "A = π·D·L, así que L_c = D/4."
        )
    elif i.body == "plate":
        e = n(i.dims[0], _L, system)
        latex.append(_chain("L_c", frac("V", "A"), frac("e", "2"), frac(e, "2"), Lc))
        text = (
            "Una **placa** de espesor e que intercambia por las dos caras: V = e·A_cara y "
            "A = 2·A_cara, así que L_c = e/2 (el semiespesor)."
        )
    elif i.body == "cube":
        a = n(i.dims[0], _L, system)
        latex += [
            _chain("V", "a^3", _pw(a, "3"), V),
            _chain("A", r"6\,a^2", rf"6 \cdot {_pw(a, '2')}", A),
            _chain("L_c", frac("V", "A"), frac("a", "6"), frac(a, "6"), Lc),
        ]
        text = "Un **cubo** de arista a: V = a³ y A = 6·a², así que L_c = a/6."
    else:  # prisma
        a, b, c = (n(x, _L, system) for x in i.dims)
        latex += [
            _chain("V", r"a\,b\,c", _prod(a, b, c), V),
            _chain(
                "A",
                r"2\,(a\,b + b\,c + c\,a)",
                _Pack(rf"2 \cdot ({times(a, b)}", (("+", times(b, c)), ("+", f"{times(c, a)})"))),
                A,
            ),
            _chain("L_c", frac("V", "A"), frac(V_n, A_n), Lc),
        ]
        text = "Un **prisma** de aristas a, b y c: V = a·b·c y A = 2·(a·b + b·c + c·a)."
    text += (
        " La **longitud característica** L_c = V/A mide cuánto volumen se enfría (o se calienta) "
        "por cada unidad de área que intercambia (Çengel y Ghajar, §4-1)."
    )
    return ProcedureStep("Longitud característica", text, tuple(latex))


def _lumped_biot_step(r: LumpedResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    latex = _chain(
        "Bi",
        frac(r"h\,L_c", "k"),
        frac(times(n(i.h, _H, system), n(r.Lc, _L, system)), n(i.solid.k, _K, system)),
        _d(r.Bi),
    )
    text = (
        "**Número de Biot**: compara la resistencia de conducción adentro (L_c/k) con la de "
        "convección afuera (1/h)."
    )
    if r.Bi <= BI_LUMPED:
        text += (
            " Con Bi ≤ 0,1 la temperatura del cuerpo es prácticamente uniforme y vale la "
            "**capacidad concentrada** (Çengel y Ghajar, §4-1)."
        )
    else:
        text += (
            " Con Bi > 0,1 la temperatura adentro **no** es uniforme: la capacidad concentrada es "
            "solo una estimación (las notas la comparan con la solución exacta)."
        )
    return ProcedureStep("Número de Biot", text, (latex,))


def _lumped_rate_step(r: LumpedResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    s = i.solid
    h = n(i.h, _H, system)
    rho, c, Lc = n(s.rho, _RHO, system), n(s.c, _CP, system), n(r.Lc, _L, system)
    numbers = frac(h, _with_factor(system, rho, c, Lc))
    if _vis(numbers) > _budget("b"):  # h/(f·ρ) por 1/(c·L_c), en dos renglones
        numbers = rf"{frac(h, _with_factor(system, rho))} \\ &\quad \cdot {frac('1', times(c, Lc))}"
    b = latex_number(r.b, 5)
    latex = (
        _chain(
            "b",
            frac(r"h\,A", r"\rho\,V\,c"),
            frac("h", r"\rho\,c\,L_c"),
            numbers,
            rf"{b}\ \mathrm{{1/s}}",
        ),
        _chain(r"\frac{1}{b}", frac("1", b), q(r.time_constant_s, _TIME, system)),
    )
    text = (
        "El balance de energía del cuerpo, −h·A·(T − T∞) = ρ·V·c·dT/dt, da "
        "θ = (T − T∞)/(T_i − T∞) = e^(−b·t). Su inversa 1/b es la **constante de tiempo**: en "
        "ese tiempo el cuerpo recorre el 63,2 % de la diferencia con el fluido (e⁻¹ = 0,368)."
    )
    if system == "Técnico":
        text += " Con c en kJ/(kg·K), el 1000 pasa los kJ a J (W·s) y b queda en 1/s."
    elif system == "Inglés":
        text += " Con h en Btu/(h·ft²·°F), h/(ρ·c·L_c) sale en 1/h: el 3600 la pasa a 1/s."
    return ProcedureStep("Constante de tiempo", text, latex)


def _lumped_time_step(r: LumpedResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    b = latex_number(r.b, 5)
    if i.T_target_K is None:
        latex = (
            _chain(
                r"\theta",
                frac(r"T - T_\infty", r"T_i - T_\infty"),
                r"e^{-b\,t}",
                rf"e^{{-{times(b, n(r.time_s, _TIME, system))}}}",
                _exp_term(1.0, r.b * r.time_s, 4),
            ),
            _T_line("T", r"\theta", r.theta, i.T_i, i.T_inf, system),
        )
        return ProcedureStep("Temperatura", "La temperatura del cuerpo en el instante t.", latex)
    latex = (
        _theta_target(r"\theta", "T", i.T_target_K, i.T_i, i.T_inf, system),
        _chain(
            "t",
            r"\frac{1}{b}\,\ln\frac{1}{\theta}",
            rf"{latex_number(r.time_constant_s, 5)} \cdot \ln{frac('1', _d(r.theta))}",
            q(r.time_s, _TIME, system),
        ),
    )
    text = (
        "El tiempo en que el cuerpo llega a la temperatura buscada: se despeja t de θ = e^(−b·t)."
    )
    return ProcedureStep("Tiempo", text, latex)


def _lumped_heat_step(r: LumpedResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    s = i.solid
    sym, dT_tex, dT = _delta_T(i.T_i, i.T_inf, system)
    rho, c, dTn = n(s.rho, _RHO, system), n(s.c, _CP, system), n(dT, _DT, system)
    Q, Qmax = abs(r.Q_J), abs(r.Q_max_J)
    latex = [dT_tex]
    if i.body == "plate":
        lhs, lhs_q, kind = (
            r"\frac{Q_{\text{máx}}}{A_{\text{cara}}}",
            r"\frac{Q}{A_{\text{cara}}}",
            _QA,
        )
        latex.append(
            _chain(
                lhs,
                rf"\rho\,e\,c\,({sym})",
                _prod(rho, n(i.dims[0], _L, system), c, dTn),
                q(Qmax, kind, system),
            )
        )
        per = " Va por unidad de área de la placa (cada m² o ft², con sus dos caras)."
    elif i.body == "long_cylinder":
        lhs, lhs_q, kind = r"\frac{Q_{\text{máx}}}{L}", r"\frac{Q}{L}", _QL
        latex.append(
            _chain(
                lhs,
                rf"\rho\,\frac{{V}}{{L}}\,c\,({sym})",
                _prod(rho, n(r.V, "area", system), c, dTn),
                q(Qmax, kind, system),
            )
        )
        per = " Va por unidad de largo."
    else:
        lhs, lhs_q, kind = r"Q_{\text{máx}}", "Q", _E
        latex += [
            _chain(
                "m", r"\rho\,V", _prod(rho, n(r.V, "volume", system)), q(r.mass, "mass", system)
            ),
            _chain(
                lhs,
                rf"m\,c\,({sym})",
                _prod(n(r.mass, "mass", system), c, dTn),
                q(Qmax, kind, system),
            ),
        ]
        per = ""
    latex.append(
        _chain(
            lhs_q,
            rf"{lhs}\,(1 - \theta)",
            _prod(n(Qmax, kind, system), rf"(1 - {_d(r.theta)})"),  # type: ignore[arg-type]
            q(Q, kind, system),  # type: ignore[arg-type]
        )
    )
    verb, side = _direction(i.T_i, i.T_inf)
    text = (
        f"Q_máx es lo máximo que el cuerpo {verb} {side} (hasta quedar a T∞); hasta el instante "
        f"t {verb} Q = Q_máx·(1 − θ) (Çengel y Ghajar, §4-1).{per}"
    )
    return ProcedureStep("Calor", text, tuple(latex))


def lumped_steps(r: LumpedResult, system: UnitSystem) -> list[ProcedureStep]:
    """Capacidad concentrada (Çengel y Ghajar §4-1), con los valores en ``system``."""
    return numbered(
        [
            _lumped_geometry_step(r, system),
            _lumped_biot_step(r, system),
            _lumped_rate_step(r, system),
            _lumped_time_step(r, system),
            _lumped_heat_step(r, system),
        ]
    )


# ---------------------------------------------------------------------
# Pared, cilindro y esfera (§4-2)
# ---------------------------------------------------------------------

_SIZE = {"wall": "L", "cylinder": "r_0", "sphere": "r_0"}
_EQUATION = {
    "wall": r"\lambda_1 \tan\lambda_1",
    "cylinder": r"\lambda_1\,\frac{J_1(\lambda_1)}{J_0(\lambda_1)}",
    "sphere": r"1 - \lambda_1 \cot\lambda_1",
}


def _sphere_s1(lam: float) -> float:
    """s₁ = sin λ₁ − λ₁·cos λ₁ (≈ λ³/3 con λ chico, sin perder cifras)."""
    if lam < 1e-3:
        return lam**3 / 3.0 - lam**5 / 30.0
    return math.sin(lam) - lam * math.cos(lam)


def _eigen_latex(geometry: str, Bi: float, lam: float, A1: float, sub_: str = "") -> list[str]:
    """La ecuación característica, λ₁ y A₁ (la tabla 4-2 de Çengel y Ghajar, calculada)."""
    lam_n, A1_n = latex_number(lam, 5), latex_number(A1, 5)
    rows = [
        r"\begin{aligned}"
        + rf"{_EQUATION[geometry]} &= \mathrm{{Bi}}{sub_} = {_d(Bi)} \\ "
        + rf"\lambda_1 &= {lam_n}"
        + r"\end{aligned}"
    ]
    two_lam = latex_number(2.0 * lam, 5)
    if geometry == "wall":
        rows.append(
            _chain(
                "A_1",
                frac(r"4\sin\lambda_1", r"2\lambda_1 + \sin 2\lambda_1"),
                frac(rf"4\sin {lam_n}", rf"2 \cdot {lam_n} + \sin {two_lam}"),
                A1_n,
            )
        )
    elif geometry == "cylinder":
        J0, J1 = latex_number(float(j0(lam)), 5), latex_number(float(j1(lam)), 5)
        rows += [
            rf"\begin{{aligned}}J_0(\lambda_1) &= {J0} \\ J_1(\lambda_1) &= {J1}\end{{aligned}}",
            _chain(
                "A_1",
                r"\frac{2}{\lambda_1}\,\frac{J_1(\lambda_1)}{J_0^2(\lambda_1) + J_1^2(\lambda_1)}",
                _Pack(
                    frac("2", lam_n), ((r"\cdot", frac(J1, f"{_pw(J0, '2')} + {_pw(J1, '2')}")),)
                ),
                A1_n,
            ),
        ]
    else:
        s1 = latex_number(_sphere_s1(lam), 5)
        rows += [
            _chain(
                "s_1",
                r"\sin\lambda_1 - \lambda_1\cos\lambda_1",
                _Pack(rf"\sin {lam_n}", (("-", rf"{lam_n}\cos {lam_n}"),)),
                s1,
            ),
            _chain(
                "A_1",
                frac(r"4\,s_1", r"2\lambda_1 - \sin 2\lambda_1"),
                frac(rf"4 \cdot {latex_paren(s1)}", rf"2 \cdot {lam_n} - \sin {two_lam}"),
                A1_n,
            ),
        ]
    return rows


def _shape_latex(geometry: str, lam: float, xi: float) -> tuple[str, str]:
    """(X simbólica, X con los números) en la posición ξ."""
    lam_n, xi_n = latex_number(lam, 5), _d(xi)
    arg = times(lam_n, xi_n)
    if geometry == "wall":
        return r"\cos(\lambda_1 x/L)", rf"\cos({arg})"
    if geometry == "cylinder":
        return r"J_0(\lambda_1 r/r_0)", rf"J_0({arg})"
    return r"\frac{\sin(\lambda_1 r/r_0)}{\lambda_1 r/r_0}", frac(rf"\sin({arg})", arg)


def _weight(geometry: str, lam: float) -> float:
    """El peso del primer término en Q/Q_máx = 1 − θ₀·w (Çengel y Ghajar, tabla 4-1)."""
    if geometry == "wall":
        return math.sin(lam) / lam
    if geometry == "cylinder":
        return 2.0 * float(j1(lam)) / lam
    return 3.0 * _sphere_s1(lam) / lam**3


def _ratio_latex(geometry: str, lam: float, th0: float, lhs: str, ratio: float) -> str:
    """Q/Q_máx = 1 − θ₀·w(λ₁) (Çengel y Ghajar, tabla 4-1)."""
    lam_n, th = latex_number(lam, 5), latex_paren(_d(th0))
    if geometry == "wall":
        formula = r"1 - \theta_0\,\frac{\sin\lambda_1}{\lambda_1}"
        first, last = f"1 - {th}", frac(rf"\sin {lam_n}", lam_n)
    elif geometry == "cylinder":
        J1 = latex_number(float(j1(lam)), 5)
        formula = r"1 - 2\,\theta_0\,\frac{J_1(\lambda_1)}{\lambda_1}"
        first, last = rf"1 - 2 \cdot {th}", frac(J1, lam_n)
    else:
        s1 = latex_number(_sphere_s1(lam), 5)
        formula = r"1 - 3\,\theta_0\,\frac{s_1}{\lambda_1^3}"
        first, last = rf"1 - 3 \cdot {th}", frac(s1, _pw(lam_n, "3"))
    return _chain(lhs, formula, _Pack(first, ((r"\cdot", last),)), _d(ratio))


def _one_dim_numbers_step(r: OneDimResult, system: UnitSystem, inverse: bool) -> ProcedureStep:
    i = r.inputs
    L = _SIZE[i.geometry]
    latex = [
        _alpha_latex(i.solid, system),
        _chain(
            "Bi",
            frac(rf"h\,{L}", "k"),
            frac(times(n(i.h, _H, system), n(i.size, _L, system)), n(i.solid.k, _K, system)),
            _d(r.Bi),
        ),
    ]
    if not inverse:
        latex.append(
            _chain(
                r"\tau",
                frac(r"\alpha\,t", f"{L}^2"),
                frac(
                    times(n(i.solid.alpha, _ALPHA, system), n(r.time_s, _TIME, system)),
                    _pw(n(i.size, _L, system), "2"),
                ),
                _d(r.tau),
            )
        )
    what = (
        "el **semiespesor L** (la pared, de espesor 2L, se enfría por las dos caras)"
        if i.geometry == "wall"
        else "el **radio r₀**"
    )
    text = f"Bi se calcula con {what}, no con L_c como en la capacidad concentrada. "
    if inverse:
        text += (
            "El número de Fourier τ = α·t/L² sale al final, del tiempo que se busca (Çengel y "
            "Ghajar, §4-2)."
        )
    else:
        text += (
            "τ = α·t/L² es el **número de Fourier**: el tiempo adimensional (Çengel y Ghajar, "
            "§4-2)."
        )
    return ProcedureStep("Difusividad, Biot y Fourier", text + _alpha_text(system), tuple(latex))


def _one_dim_eigen_step(r: OneDimResult) -> ProcedureStep:
    i = r.inputs
    eq = {
        "wall": "λ·tan λ = Bi",
        "cylinder": "λ·J₁(λ)/J₀(λ) = Bi",
        "sphere": "1 − λ·cot λ = Bi",
    }[i.geometry]
    text = (
        f"λ₁ es la primera raíz de la **ecuación característica** {eq}; acá se resuelve, y la "
        "tabla 4-2 de Çengel y Ghajar la tabula junto con A₁ en función de Bi."
    )
    if i.geometry == "sphere":
        text += " En la esfera, s₁ = sin λ₁ − λ₁·cos λ₁ se usa también en el calor."
    return ProcedureStep("λ₁ y A₁", text, tuple(_eigen_latex(i.geometry, r.Bi, r.lambda1, r.A1)))


def _one_dim_one_term(r: OneDimResult) -> tuple[float, float]:
    """(τ, θ₀) de la solución de un término: el τ del dato, o el que da la θ₀ buscada."""
    i = r.inputs
    if i.T_center_target_K is None:
        return r.tau, r.theta0_one
    th0 = (i.T_center_target_K - i.T_inf) / (i.T_i - i.T_inf)
    return math.log(r.A1 / th0) / r.lambda1**2, th0


def _decay(A: float, lam: float, tau: float, *more: str) -> _Pack:
    r"""``A \cdot e^{-λ²·τ}`` con los números (el exponente ya calculado si es largo) y los
    factores de ``more``."""
    A_n, lam_n, tau_n = latex_number(A, 5), latex_number(lam, 5), _d(tau)
    expo = rf"-{times(_pw(lam_n, '2'), tau_n)}"
    if _vis(expo) > 20.0:  # en superíndice: unos 14 caracteres
        expo = f"-{latex_number(lam * lam * tau, 4)}"
    rest = [(r"\cdot", rf"e^{{{expo}}}"), *((r"\cdot", m) for m in more)]
    return _Pack(A_n, tuple(rest))


def _one_dim_center_step(r: OneDimResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    lam_n, A1_n = latex_number(r.lambda1, 5), latex_number(r.A1, 5)
    tau1, th0 = _one_dim_one_term(r)
    if i.T_center_target_K is None:
        latex = (
            _chain(
                r"\theta_0",
                frac(r"T_0 - T_\infty", r"T_i - T_\infty"),
                r"A_1\,e^{-\lambda_1^2\,\tau}",
                _decay(r.A1, r.lambda1, r.tau),
                _exp_term(r.A1, r.lambda1**2 * r.tau, 4),
            ),
            _T_line("T_0", r"\theta_0", th0, i.T_i, i.T_inf, system),
        )
        text = (
            "Con un término de la serie, la temperatura del centro es θ₀ = A₁·e^(−λ₁²·τ) "
            "(Çengel y Ghajar, §4-2; es lo que leen los gráficos de Heisler)."
        )
        if r.tau < TAU_ONE_TERM:
            text += (
                " Ojo: con τ < 0,2 el primer término no alcanza; el último paso suma la serie "
                "completa."
            )
        return ProcedureStep("Temperatura del centro", text, latex)
    L = _SIZE[i.geometry]
    t1 = tau1 * i.size**2 / i.solid.alpha
    latex = (
        _theta_target(r"\theta_0", "T_0", i.T_center_target_K, i.T_i, i.T_inf, system),
        _chain(
            r"\tau",
            r"\frac{1}{\lambda_1^2}\,\ln\frac{A_1}{\theta_0}",
            rf"\frac{{1}}{{{_pw(lam_n, '2')}}} \cdot \ln{frac(A1_n, _d(th0))}",
            _d(tau1),
        ),
        _chain(
            "t",
            frac(rf"\tau\,{L}^2", r"\alpha"),
            frac(
                times(_d(tau1), _pw(n(i.size, _L, system), "2")),
                n(i.solid.alpha, _ALPHA, system),
            ),
            q(t1, _TIME, system),
        ),
    )
    text = (
        "Se despeja τ de θ₀ = A₁·e^(−λ₁²·τ) (la solución de un término, Çengel y Ghajar, §4-2) "
        "y de τ = α·t/L² sale el tiempo."
    )
    if tau1 < TAU_ONE_TERM:
        text += (
            " Ojo: da τ < 0,2, donde el primer término no alcanza; el último paso usa la serie "
            "completa."
        )
    return ProcedureStep("Tiempo", text, latex)


def _one_dim_position_step(r: OneDimResult, system: UnitSystem) -> ProcedureStep | None:
    i = r.inputs
    if i.position == 0.0:
        return None
    _tau1, th0 = _one_dim_one_term(r)
    th = th0 * shape_function(i.geometry, r.lambda1, i.position)
    sym, numbers = _shape_latex(i.geometry, r.lambda1, i.position)
    latex = (
        _chain(r"\theta", rf"\theta_0\,{sym}", _prod(_d(th0), numbers), _d(th)),
        _T_line("T", r"\theta", th, i.T_i, i.T_inf, system),
    )
    if i.position == 1.0:
        where = "la superficie"
    else:
        where = ("x/L" if i.geometry == "wall" else "r/r₀") + f" = {_d(i.position)}"
    text = (
        f"En {where} la temperatura es la del centro por la forma X del primer término: "
        "cos(λ₁·x/L) en la pared, J₀(λ₁·r/r₀) en el cilindro y sin(λ₁·r/r₀)/(λ₁·r/r₀) en la "
        "esfera."
    )
    return ProcedureStep("Temperatura en la posición", text, latex)


def _one_dim_heat_step(r: OneDimResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    s = i.solid
    _tau1, th0 = _one_dim_one_term(r)
    ratio = 1.0 - th0 * _weight(i.geometry, r.lambda1)
    lhs_ratio = r"\frac{Q}{Q_{\text{máx}}}"
    sym, dT_tex, dT = _delta_T(i.T_i, i.T_inf, system)
    rho, c, dTn = n(s.rho, _RHO, system), n(s.c, _CP, system), n(dT, _DT, system)
    size = n(i.size, _L, system)
    Qmax = abs(r.Q_max)
    if i.geometry == "wall":
        lhs, lhs_q, kind = r"\frac{Q_{\text{máx}}}{A}", r"\frac{Q}{A}", _QA
        formula = rf"\rho\,(2L)\,c\,({sym})"
        numbers = _prod(rho, n(2.0 * i.size, _L, system), c, dTn)
    elif i.geometry == "cylinder":
        lhs, lhs_q, kind = r"\frac{Q_{\text{máx}}}{L}", r"\frac{Q}{L}", _QL
        formula = rf"\rho\,\pi r_0^2\,c\,({sym})"
        numbers = _prod(rho, rf"\pi \cdot {_pw(size, '2')}", c, dTn)
    else:
        lhs, lhs_q, kind = r"Q_{\text{máx}}", "Q", _E
        formula = rf"\rho\,\tfrac{{4}}{{3}}\pi r_0^3\,c\,({sym})"
        numbers = _prod(rho, rf"\tfrac{{4}}{{3}}\pi \cdot {_pw(size, '3')}", c, dTn)
    latex = (
        _ratio_latex(i.geometry, r.lambda1, th0, lhs_ratio, ratio),
        dT_tex,
        _chain(lhs, formula, numbers, q(Qmax, kind, system)),  # type: ignore[arg-type]
        _chain(
            lhs_q,
            rf"{lhs_ratio}\,{lhs}",
            _prod(_d(ratio), n(Qmax, kind, system)),  # type: ignore[arg-type]
            q(ratio * Qmax, kind, system),  # type: ignore[arg-type]
        ),
    )
    verb, side = _direction(i.T_i, i.T_inf)
    dT_text = "T_i − T∞" if i.T_i >= i.T_inf else "T∞ − T_i"
    per = {
        "wall": " Va por unidad de área de cada cara, para toda la pared de espesor 2L.",
        "cylinder": " Va por unidad de largo del cilindro.",
        "sphere": "",
    }[i.geometry]
    text = (
        f"Q_máx = m·c·({dT_text}) es lo máximo que el cuerpo {verb} {side}; la fracción que "
        "ya intercambió sale de θ₀ con un término (Çengel y Ghajar, tabla 4-1)." + per
    )
    return ProcedureStep("Calor", text, latex)


def _series_terms(r: OneDimResult, tau: float) -> list[str]:
    """Los términos 2 y 3 de la serie en el centro, para ver cuánto pesan."""
    i = r.inputs
    rows = []
    for k, lam in enumerate(eigenvalues(i.geometry, r.Bi, 3)[1:], start=2):
        A = coefficient(i.geometry, lam)
        rows += [
            rf"\begin{{aligned}}\lambda_{k} &= {latex_number(lam, 5)} \\ "
            rf"A_{k} &= {latex_number(A, 5)}\end{{aligned}}",
            _chain(
                rf"A_{k}\,e^{{-\lambda_{k}^2\,\tau}}",
                _decay(A, lam, tau),
                _exp_term(A, lam * lam * tau),
            ),
        ]
    return rows


def _one_dim_series_step(r: OneDimResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    N = _n_terms(r.tau)
    latex = _series_terms(r, r.tau)
    sum0 = rf"\sum_{{n=1}}^{{{N}}} A_n\,e^{{-\lambda_n^2\,\tau}}"
    inverse = i.T_center_target_K is not None
    if inverse:
        L = _SIZE[i.geometry]
        latex += [
            _chain(r"\theta_0", sum0, _d(r.theta0)),
            rf"\tau = {_d(r.tau)}",
            _chain(
                "t",
                frac(rf"\tau\,{L}^2", r"\alpha"),
                frac(
                    times(_d(r.tau), _pw(n(i.size, _L, system), "2")),
                    n(i.solid.alpha, _ALPHA, system),
                ),
                q(r.time_s, _TIME, system),
            ),
        ]
    else:
        latex += [
            _chain(r"\theta_0", sum0, _d(r.theta0)),
            _T_line("T_0", r"\theta_0", r.theta0, i.T_i, i.T_inf, system),
        ]
    if i.position > 0.0:
        latex.append(
            rf"\begin{{aligned}}\theta &= {_d(r.theta)} \\ T &= {q(r.T_K, _T, system)}"
            r"\end{aligned}"
        )
    latex.append(rf"\frac{{Q}}{{Q_{{\text{{máx}}}}}} = {_d(r.heat_ratio)}")
    text = (
        "La solución exacta es la **serie** θ = Σ A_n·e^(−λ_n²·τ)·X(λ_n·ξ), con una raíz λ_n de "
        "la misma ecuación en cada intervalo y A_n con la misma fórmula que A₁. Los términos "
        "que siguen al primero caen como e^(−λ_n²·τ): con τ ≥ 0,2 pesan poco y alcanza con el "
        f"primero. Acá se suman {N} términos; es el **resultado** que muestra la página."
    )
    if inverse:
        text += " El tiempo sale de buscar el τ en que la serie da la θ₀ pedida."
    return ProcedureStep("Serie completa", text, tuple(latex))


def one_dim_steps(r: OneDimResult, system: UnitSystem) -> list[ProcedureStep]:
    """Pared, cilindro largo o esfera (Çengel y Ghajar §4-2), con los valores en ``system``."""
    inverse = r.inputs.T_center_target_K is not None
    steps: list[ProcedureStep | None] = [
        _one_dim_numbers_step(r, system, inverse),
        _one_dim_eigen_step(r),
        _one_dim_center_step(r, system),
        _one_dim_position_step(r, system),
        _one_dim_heat_step(r, system),
        _one_dim_series_step(r, system),
    ]
    return numbered([s for s in steps if s is not None])


# ---------------------------------------------------------------------
# Sólido semi-infinito y contacto (§4-3)
# ---------------------------------------------------------------------


def _sqrt_alpha_t(alpha: float, t: float, system: UnitSystem) -> str:
    return _chain(
        r"\sqrt{\alpha\,t}",
        rf"\sqrt{{{times(n(alpha, _ALPHA, system), n(t, _TIME, system))}}}",
        q(math.sqrt(alpha * t), _L, system),
    )


def _eta(x: float, sq: float, system: UnitSystem) -> str:
    return _chain(
        r"\eta",
        frac("x", r"2\sqrt{\alpha\,t}"),
        frac(n(x, _L, system), rf"2 \cdot {n(sq, _L, system)}"),
        _d(x / (2.0 * sq)),
    )


def _erfc_pair(eta: float, beta: float, sign: str) -> tuple[str, tuple[str, str]]:
    r"""``sign e^{2ηβ + β²}·erfc(η + β)`` (``sign`` es «-» o «+»): los números y el par
    (operador, valor) del renglón siguiente, con los dos factores si se leen o con el
    producto calculado junto (e^x·erfc(y) = e^(−η²)·erfcx(η + β), sin desbordes)."""
    expo = 2.0 * eta * beta + beta * beta
    numbers = rf"{sign} e^{{{_d(expo)}}}\,\operatorname{{erfc}}({_d(eta + beta)})"
    product = math.exp(-eta * eta) * float(erfcx(eta + beta))
    if expo <= _EXP_MAX:
        e_n, erfc_n = latex_number(math.exp(expo), 4), latex_number(float(erfc(eta + beta)), 4)
        if r"\times" not in e_n + erfc_n:
            return numbers, (sign, times(e_n, erfc_n))
    return numbers, (sign, latex_paren(latex_number(product, 4)))


def _semi_inverse_temperature(r: SemiInfiniteResult, system: UnitSystem) -> list[str]:
    """Con la superficie a T_s, erfc(η) = (T − T_i)/(T_s − T_i) se invierte en cerrado."""
    i = r.inputs
    T, Ti, Ts = (n(x, _T, system) for x in (i.T_target_K or 0.0, i.T_i, i.T_s or 0.0))
    ratio = ((i.T_target_K or 0.0) - i.T_i) / ((i.T_s or 0.0) - i.T_i)
    latex = [
        _chain(
            r"\operatorname{erfc}(\eta)",
            frac("T - T_i", "T_s - T_i"),
            frac(sub(T, Ti), sub(Ts, Ti)),
            _d(ratio),
        ),
        _chain(r"\eta", rf"\operatorname{{erfc}}^{{-1}}({_d(ratio)})", _d(r.eta)),
    ]
    alpha = n(i.solid.alpha, _ALPHA, system)
    if i.depth_m is None:  # la profundidad
        latex += [
            _sqrt_alpha_t(i.solid.alpha, r.time_s, system),
            _chain(
                "x",
                r"2\,\eta\,\sqrt{\alpha\,t}",
                times("2", _d(r.eta), n(r.sqrt_alpha_t, _L, system)),
                q(r.depth_m, _L, system),
            ),
        ]
    else:  # el tiempo
        latex += [
            _chain(
                "t",
                frac("x^2", r"4\,\eta^2\,\alpha"),
                frac(_pw(n(r.depth_m, _L, system), "2"), times("4", _pw(_d(r.eta), "2"), alpha)),
                q(r.time_s, _TIME, system),
            ),
            _sqrt_alpha_t(i.solid.alpha, r.time_s, system),
        ]
    return latex


def _semi_temperature_latex(r: SemiInfiniteResult, system: UnitSystem) -> list[str]:
    """La temperatura en (x, t) con la fórmula de cada borde (§4-3)."""
    i = r.inputs
    Ti = n(i.T_i, _T, system)
    T_out = q(r.T_K, _T, system)
    if i.boundary == "temperature":
        Ts = n(i.T_s or 0.0, _T, system)
        ratio = float(erfc(r.eta))
        return [
            _flush(
                frac("T - T_i", "T_s - T_i"),
                r"\operatorname{erfc}(\eta)",
                rf"\operatorname{{erfc}}({_d(r.eta)})",
                _d(ratio),
            ),
            _chain(
                "T",
                r"T_i + (T_s - T_i)\,\operatorname{erfc}(\eta)",
                _affine(Ti, sub(Ts, Ti), _d(ratio)),
                T_out,
            ),
        ]
    k = n(i.solid.k, _K, system)
    sq = n(r.sqrt_alpha_t, _L, system)
    if i.boundary == "flux":
        qs = n(i.q_s or 0.0, _FLUX, system)
        first = frac(times("2", qs, sq), rf"{k} \cdot \sqrt{{\pi}}")
        if r.depth_m == 0.0:
            return [
                _chain(
                    "T_s",
                    r"T_i + \frac{2\,\dot{q}_s\,\sqrt{\alpha\,t}}{k\,\sqrt{\pi}}",
                    rf"{Ti} + {first}",
                    T_out,
                )
            ]
        eta = _d(r.eta)
        x = n(r.depth_m, _L, system)
        return [
            _chain(
                "T",
                r"T_i + \frac{2\,\dot{q}_s\,\sqrt{\alpha\,t}}{k\,\sqrt{\pi}}\,e^{-\eta^2}"
                r" \\ &\quad - \frac{\dot{q}_s\,x}{k}\,\operatorname{erfc}(\eta)",
                _Pack(
                    Ti,
                    (
                        ("+", rf"{first} \cdot e^{{-{_pw(eta, '2')}}}"),
                        ("-", rf"{frac(times(qs, x), k)} \cdot \operatorname{{erfc}}({eta})"),
                    ),
                ),
                T_out,
            )
        ]
    beta = r.beta or 0.0
    Tinf = n(i.T_inf or 0.0, _T, system)
    ratio = (r.T_K - i.T_i) / ((i.T_inf or 0.0) - i.T_i)
    numbers, pair = _erfc_pair(r.eta, beta, "-")
    a = latex_number(float(erfc(r.eta)), 4)
    return [
        _chain(
            r"\beta",
            frac(r"h\,\sqrt{\alpha\,t}", "k"),
            frac(times(n(i.h or 0.0, _H, system), sq), k),
            _d(beta),
        ),
        _flush(
            frac("T - T_i", r"T_\infty - T_i"),
            r"\operatorname{erfc}(\eta) \\ &\quad - e^{2\eta\beta + \beta^2}\,"
            r"\operatorname{erfc}(\eta + \beta)",
            rf"\operatorname{{erfc}}({_d(r.eta)}) \\ &\quad {numbers}",
            _Pack(a, (pair,)),
            _d(ratio),
        ),
        _chain("T", _affine(Ti, sub(Tinf, Ti), _d(ratio)), T_out),
    ]


def _semi_temperature_step(r: SemiInfiniteResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    inverse = i.T_target_K is not None
    latex: list[str] = [_alpha_latex(i.solid, system)]
    if inverse and i.boundary == "temperature":
        latex += _semi_inverse_temperature(r, system)
        unknown = "la profundidad" if i.depth_m is None else "el tiempo"
        text = (
            "Con la superficie a T_s, (T − T_i)/(T_s − T_i) = erfc(η) con η = x/(2·√(α·t)) "
            f"(Çengel y Ghajar, §4-3). Con la temperatura buscada sale η, y de η, {unknown}."
        )
        title = "Temperatura, η y " + unknown
        return ProcedureStep(title, text + _alpha_text(system), tuple(latex))
    latex += [
        _sqrt_alpha_t(i.solid.alpha, r.time_s, system),
        _eta(r.depth_m, r.sqrt_alpha_t, system),
        *_semi_temperature_latex(r, system),
    ]
    text = {
        "temperature": (
            "Con la superficie a T_s desde t = 0: (T − T_i)/(T_s − T_i) = erfc(η), con "
            "η = x/(2·√(α·t)) (Çengel y Ghajar, §4-3)."
        ),
        "flux": (
            "Con un flujo q̇_s constante que entra por la superficie: "
            "T − T_i = (2·q̇_s·√(α·t/π)/k)·e^(−η²) − (q̇_s·x/k)·erfc(η) (Çengel y Ghajar, §4-3)."
        ),
        "convection": (
            "Con convección (h, T∞): (T − T_i)/(T∞ − T_i) = erfc(η) − e^(h·x/k + h²·α·t/k²)·"
            "erfc(η + h·√(α·t)/k) (Çengel y Ghajar, §4-3). Con β = h·√(α·t)/k el exponente es "
            "2·η·β + β²."
        ),
    }[i.boundary]
    if i.boundary == "flux" and r.depth_m == 0.0:
        text += " En la superficie (x = 0, η = 0) queda solo el primer término."
    if inverse:
        unknown = "la profundidad" if i.depth_m is None else "el tiempo"
        text += (
            f" Con la temperatura buscada, {unknown} se despeja numéricamente; los números de "
            "abajo verifican que la fórmula da esa temperatura."
        )
    return ProcedureStep("Temperatura", text + _alpha_text(system), tuple(latex))


def _semi_heat_step(r: SemiInfiniteResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    k = n(i.solid.k, _K, system)
    t = n(r.time_s, _TIME, system)
    qs_val = r.surface_flux
    qs = n(qs_val, _FLUX, system)
    QA = q(r.energy_per_area, _QA, system)
    latex: list[str] = []
    if i.boundary == "temperature":
        Ts, Ti = n(i.T_s or 0.0, _T, system), n(i.T_i, _T, system)
        latex += [
            _chain(
                r"\dot{q}_s",
                frac(r"k\,(T_s - T_i)", r"\sqrt{\pi}\,\sqrt{\alpha\,t}"),
                frac(
                    rf"{k} \cdot ({sub(Ts, Ti)})",
                    rf"\sqrt{{\pi}} \cdot {n(r.sqrt_alpha_t, _L, system)}",
                ),
                q(qs_val, _FLUX, system),
            ),
            _chain(
                r"\frac{Q}{A}",
                r"\int_0^t \dot{q}_s\,dt = 2\,\dot{q}_s\,t",
                _over_factor(times("2", qs, t), system),
                QA,
            ),
        ]
        text = (
            "El flujo en la superficie es q̇_s = k·(T_s − T_i)/√(π·α·t): baja como 1/√t, así que "
            "su integral es 2·q̇_s·t."
        )
    elif i.boundary == "flux":
        latex.append(
            _chain(r"\frac{Q}{A}", r"\dot{q}_s\,t", _over_factor(times(qs, t), system), QA)
        )
        text = "Con el flujo constante, la energía que entró es q̇_s·t."
    else:
        beta = r.beta or 0.0
        Ti, Tinf = n(i.T_i, _T, system), n(i.T_inf or 0.0, _T, system)
        surf = (r.T_surface_K - i.T_i) / ((i.T_inf or 0.0) - i.T_i)
        latex += [
            _flush(
                frac("T_s - T_i", r"T_\infty - T_i"),
                r"1 - e^{\beta^2}\,\operatorname{erfc}(\beta)",
                rf"1 - e^{{{latex_number(beta * beta, 4)}}}\,\operatorname{{erfc}}({_d(beta)})",
                f"1 - {latex_number(float(erfcx(beta)), 4)}",
                _d(surf),
            ),
            _chain("T_s", _affine(Ti, sub(Tinf, Ti), _d(surf)), q(r.T_surface_K, _T, system)),
            _chain(
                r"\dot{q}_s",
                r"h\,(T_\infty - T_s)",
                times(n(i.h or 0.0, _H, system), f"({sub(Tinf, n(r.T_surface_K, _T, system))})"),
                q(qs_val, _FLUX, system),
            ),
            _chain(r"\frac{Q}{A}", r"\int_0^t \dot{q}_s\,dt", QA),
        ]
        text = (
            "La superficie está a T_s (la fórmula con x = 0) y el fluido le entrega "
            "q̇_s = h·(T∞ − T_s). La energía es la integral de q̇_s en el tiempo, que se hace "
            "numéricamente."
        )
    text += " q̇_s > 0 si el calor entra al sólido."
    if i.boundary != "convection":
        text += _energy_factor_text(system)
    return ProcedureStep("Flujo en la superficie y energía", text, tuple(latex))


def _semi_penetration_step(r: SemiInfiniteResult, system: UnitSystem) -> ProcedureStep:
    c = latex_number(_PENETRATION, 4)
    latex = _chain(
        r"\delta",
        rf"{c}\,\sqrt{{\alpha\,t}}",
        times(c, n(r.sqrt_alpha_t, _L, system)),
        q(r.penetration_m, _L, system),
    )
    text = (
        "La perturbación llega hasta donde erfc(η) = 0,01 (η = 1,82): δ = 3,64·√(α·t). Un cuerpo "
        "más grueso que δ se comporta como semi-infinito."
    )
    return ProcedureStep("Profundidad de penetración", text, (latex,))


def semi_infinite_steps(r: SemiInfiniteResult, system: UnitSystem) -> list[ProcedureStep]:
    """Sólido semi-infinito (Çengel y Ghajar §4-3), con los valores en ``system``."""
    return numbered(
        [
            _semi_temperature_step(r, system),
            _semi_heat_step(r, system),
            _semi_penetration_step(r, system),
        ]
    )


def _effusivity_unit(system: UnitSystem) -> str:
    if system == "Inglés":
        return r"\ \mathrm{Btu/(h^{1/2}\cdot ft^{2}\cdot {}^{\circ}F)}"
    return r"\ \mathrm{W\cdot s^{1/2}/(m^{2}\cdot K)}"


def _effusivity(solid: Solid, system: UnitSystem) -> tuple[str, float]:
    """(k·ρ·c con los números, √(k·ρ·c)) en las unidades que muestra el procedimiento: en
    el Técnico con el 1000 que pasa los kJ a J (W·s^½/(m²·K), como el SI); en el Inglés,
    Btu/(h^½·ft²·°F)."""
    k, rho, c = n(solid.k, _K, system), n(solid.rho, _RHO, system), n(solid.c, _CP, system)
    if system == "Inglés":
        value = math.sqrt(
            convert_from_si(solid.k, _K, system)
            * convert_from_si(solid.rho, _RHO, system)
            * convert_from_si(solid.c, _CP, system)
        )
        return times(k, rho, c), value
    product = times("1000", k, rho, c) if system == "Técnico" else times(k, rho, c)
    return product, solid.effusivity


def contact_steps(r: ContactResult, system: UnitSystem) -> list[ProcedureStep]:
    """Contacto de dos sólidos semi-infinitos (Çengel y Ghajar §4-3)."""
    i = r.inputs
    latex: list[str] = []
    e_vals = []
    for sub_, s in (("A", i.solid_a), ("B", i.solid_b)):
        product, e = _effusivity(s, system)
        e_vals.append(latex_number(e, 5))
        latex.append(
            _chain(
                f"e_{sub_}",
                rf"\sqrt{{k_{sub_}\,\rho_{sub_}\,c_{sub_}}}",
                rf"\sqrt{{{product}}}",
                e_vals[-1] + _effusivity_unit(system),
            )
        )
    TA, TB = n(i.T_a, _T, system), n(i.T_b, _T, system)
    latex += [
        _chain(
            r"\frac{e_A}{e_A + e_B}",
            frac(e_vals[0], f"{e_vals[0]} + {e_vals[1]}"),
            _d(r.share_a),
        ),
        _chain(
            "T_s",
            frac(r"e_A\,T_A + e_B\,T_B", "e_A + e_B"),
            r"T_B + (T_A - T_B)\,\frac{e_A}{e_A + e_B}",
            _affine(TB, sub(TA, TB), _d(r.share_a)),
            q(r.T_contact_K, _T, system),
        ),
    ]
    text = (
        "Dos sólidos semi-infinitos que se tocan: la superficie de contacto queda a una "
        "temperatura constante, el promedio de T_A y T_B pesado con la **efusividad** "
        "e = √(k·ρ·c) de cada uno (Çengel y Ghajar, §4-3). Solo importa el cociente, así que "
        "la unidad de e se cancela."
    )
    if system == "Técnico":
        text += " Con c en kJ/(kg·K), el 1000 la pasa a J para que e quede en W·s^½/(m²·K)."
    steps = [ProcedureStep("Temperatura de contacto", text, tuple(latex))]
    if i.time_s is not None and r.flux_W_m2 is not None:
        a = i.solid_a
        at = times(n(a.alpha, _ALPHA, system), n(i.time_s, _TIME, system))
        flux_latex = (
            _alpha_latex(a, system, "_A"),
            _chain(
                r"\dot{q}",
                frac(r"k_A\,(T_A - T_s)", r"\sqrt{\pi\,\alpha_A\,t}"),
                frac(
                    rf"{n(a.k, _K, system)} \cdot ({sub(TA, n(r.T_contact_K, _T, system))})",
                    rf"\sqrt{{\pi \cdot {at}}}",
                ),
                q(r.flux_W_m2, _FLUX, system),
            ),
        )
        text2 = (
            "El flujo que pasa de A a B es el de un semi-infinito con la superficie a T_s: "
            "q̇ = k_A·(T_A − T_s)/√(π·α_A·t). Baja como 1/√t." + _alpha_text(system)
        )
        steps.append(ProcedureStep("Flujo por el contacto", text2, flux_latex))
    return numbered(steps)


# ---------------------------------------------------------------------
# Producto de soluciones (§4-4)
# ---------------------------------------------------------------------


def _factor_symbols(factors: tuple[Factor, ...]) -> list[str]:
    """El subíndice de cada factor: cil, pared o semi (numerados si se repiten)."""
    names = {"wall": "pared", "cylinder": "cil", "semi": "semi"}
    counts: dict[str, int] = {}
    for f in factors:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for f in factors:
        seen[f.kind] = seen.get(f.kind, 0) + 1
        name = rf"\text{{{names[f.kind]}}}"
        out.append(f"{name},{seen[f.kind]}" if counts[f.kind] > 1 else name)
    return out


def _semi_factor_step(
    fr: FactorResult, symbol: str, r: ProductResult, system: UnitSystem
) -> ProcedureStep:
    i = r.inputs
    f = fr.factor
    s = i.solid
    sq = math.sqrt(s.alpha * i.time_s)
    eta = f.position / (2.0 * sq)
    beta = i.h * sq / s.k
    numbers, pair = _erfc_pair(eta, beta, "+")
    a = latex_number(float(erfc(eta)), 4)
    latex = (
        _sqrt_alpha_t(s.alpha, i.time_s, system),
        _eta(f.position, sq, system),
        _chain(
            r"\beta",
            frac(r"h\,\sqrt{\alpha\,t}", "k"),
            frac(times(n(i.h, _H, system), n(sq, _L, system)), n(s.k, _K, system)),
            _d(beta),
        ),
        _chain(
            rf"\theta_{{{symbol}}}",
            r"1 - \operatorname{erfc}(\eta) \\ &\quad + e^{2\eta\beta + \beta^2}\,"
            r"\operatorname{erfc}(\eta + \beta)",
            rf"1 - \operatorname{{erfc}}({_d(eta)}) \\ &\quad {numbers}",
            _Pack(f"1 - {latex_paren(a)}", (pair,)),
            _d(fr.theta),
        ),
    )
    text = (
        "El factor **semi-infinito** es S = 1 − (T − T_i)/(T∞ − T_i) del sólido semi-infinito "
        "con convección, a la profundidad x medida desde la cara expuesta (Çengel y Ghajar, "
        "§4-4)."
    )
    return ProcedureStep(f"Factor {f.label or 'semi-infinito'}", text, latex)


def _factor_step(
    fr: FactorResult, symbol: str, r: ProductResult, system: UnitSystem
) -> ProcedureStep:
    f = fr.factor
    if f.kind == "semi":
        return _semi_factor_step(fr, symbol, r, system)
    i = r.inputs
    s = i.solid
    g = "wall" if f.kind == "wall" else "cylinder"
    L = "L" if f.kind == "wall" else "r_0"
    size = f.size or 0.0
    Bi, tau = fr.Bi or 0.0, fr.tau or 0.0
    lam = eigenvalues(g, Bi, 1)[0]  # type: ignore[arg-type]
    A1 = coefficient(g, lam)  # type: ignore[arg-type]
    th_one = A1 * math.exp(-lam * lam * tau) * shape_function(g, lam, f.position)  # type: ignore[arg-type]
    th_sym = rf"\theta_{{{symbol}}}"
    latex = [
        _chain(
            rf"\mathrm{{Bi}}_{{{symbol}}}",
            frac(rf"h\,{L}", "k"),
            frac(times(n(i.h, _H, system), n(size, _L, system)), n(s.k, _K, system)),
            _d(Bi),
        ),
        _chain(
            rf"\tau_{{{symbol}}}",
            frac(r"\alpha\,t", f"{L}^2"),
            frac(
                times(n(s.alpha, _ALPHA, system), n(i.time_s, _TIME, system)),
                _pw(n(size, _L, system), "2"),
            ),
            _d(tau),
        ),
        *_eigen_latex(g, Bi, lam, A1, rf"_{{{symbol}}}"),
    ]
    if f.position == 0.0:
        latex.append(
            _chain(th_sym, r"A_1\,e^{-\lambda_1^2\,\tau}", _decay(A1, lam, tau), _d(th_one))
        )
    else:
        sym_X, num_X = _shape_latex(g, lam, f.position)
        latex.append(
            _chain(
                th_sym,
                rf"A_1\,e^{{-\lambda_1^2\,\tau}}\,{sym_X}",
                _decay(A1, lam, tau, num_X),
                _d(th_one),
            )
        )
    what = (
        "una **pared** de espesor 2L"
        if f.kind == "wall"
        else "un **cilindro infinito** de radio r₀"
    )
    text = (
        f"El factor {f.label or g} es {what}, con su propio Bi y su propio τ (Çengel y Ghajar, "
        "§4-2 y §4-4)."
    )
    if abs(th_one - fr.theta) >= 5e-5:
        latex.append(rf"{th_sym} = {_d(fr.theta)}\ \ (\text{{serie}})")
        text += (
            " La solución de un término se aparta de la serie (τ chico): el producto usa la serie."
        )
    return ProcedureStep(f"Factor {f.label or g}", text, tuple(latex))


def _product_step(r: ProductResult, symbols: list[str], system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    syms = r" \, ".join(rf"\theta_{{{s}}}" for s in symbols)
    latex = (
        _chain(r"\theta", syms, _prod(*(_d(fr.theta) for fr in r.factors)), _d(r.theta)),
        _T_line("T", r"\theta", r.theta, i.T_i, i.T_inf, system),
    )
    text = (
        "El sólido es la **intersección** de los de una dimensión, así que su θ es el producto "
        "de los factores (Çengel y Ghajar, §4-4)."
    )
    return ProcedureStep("Producto", text, latex)


def _langston(ratios: list[str]) -> tuple[str, _Pack]:
    """La regla de Langston (1982): fórmula y números (dos o tres factores), cada término
    entero en su renglón si no entran juntos."""
    q1, q2 = ratios[0], ratios[1]
    terms = [("+", rf"{q2} \cdot (1 - {q1})")]
    formula = r"q_1 + q_2\,(1 - q_1)"
    if len(ratios) == 3:
        terms.append(("+", rf"{ratios[2]} \cdot (1 - {q1}) \cdot (1 - {q2})"))
        formula += r" \\ &\quad + q_3\,(1 - q_1)(1 - q_2)"
    return formula, _Pack(q1, tuple(terms))


def _product_heat_step(
    r: ProductResult, symbols: list[str], system: UnitSystem
) -> ProcedureStep | None:
    i = r.inputs
    if r.heat_ratio is None:
        return None
    latex: list[str] = []
    ratios = []
    for k, fr in enumerate(r.factors, start=1):
        g = "wall" if fr.factor.kind == "wall" else "cylinder"
        Bi, tau = fr.Bi or 0.0, fr.tau or 0.0
        lam = eigenvalues(g, Bi, 1)[0]  # type: ignore[arg-type]
        th0 = coefficient(g, lam) * math.exp(-lam * lam * tau)  # type: ignore[arg-type]
        ratio_one = 1.0 - th0 * _weight(g, lam)
        ratio = heat_ratio(g, Bi, tau)  # type: ignore[arg-type]
        ratios.append(ratio)
        latex.append(_ratio_latex(g, lam, th0, f"q_{k}", ratio_one))
        if abs(ratio_one - ratio) >= 5e-5:
            latex.append(rf"q_{k} = {_d(ratio)}\ \ (\text{{serie}})")
    formula, numbers = _langston([_d(x) for x in ratios])
    latex.append(_chain(r"\frac{Q}{Q_{\text{máx}}}", formula, numbers, _d(r.heat_ratio)))
    if r.Q_max is not None and r.Q is not None and r.volume is not None:
        sym, dT_tex, dT = _delta_T(i.T_i, i.T_inf, system)
        s = i.solid
        if i.shape == "short_cylinder":
            cyl = next(f for f in i.factors if f.kind == "cylinder")
            wall = next(f for f in i.factors if f.kind == "wall")
            r0_sq = _pw(n(cyl.size or 0.0, _L, system), "2")
            V_latex = _chain(
                "V",
                r"\pi r_0^2\,(2L)",
                _Pack(
                    rf"\pi \cdot {r0_sq}", ((r"\cdot", n(2.0 * (wall.size or 0.0), _L, system)),)
                ),
                q(r.volume, "volume", system),
            )
        else:
            sides = [n(2.0 * (f.size or 0.0), _L, system) for f in i.factors]
            V_latex = _chain(
                "V", r"(2L_1)(2L_2)(2L_3)", _prod(*sides), q(r.volume, "volume", system)
            )
        Qmax = abs(r.Q_max)
        latex += [
            V_latex,
            dT_tex,
            _chain(
                r"Q_{\text{máx}}",
                rf"\rho\,V\,c\,({sym})",
                _prod(
                    n(s.rho, _RHO, system),
                    n(r.volume, "volume", system),
                    n(s.c, _CP, system),
                    n(dT, _DT, system),
                ),
                q(Qmax, _E, system),
            ),
            _chain(
                "Q",
                r"\frac{Q}{Q_{\text{máx}}}\,Q_{\text{máx}}",
                _prod(_d(r.heat_ratio), n(Qmax, _E, system)),
                q(abs(r.Q), _E, system),
            ),
        ]
    verb, side = _direction(i.T_i, i.T_inf)
    names = ", ".join(
        f"{'q₁ q₂ q₃'.split()[k]}: {fr.factor.label or fr.factor.kind}"
        for k, fr in enumerate(r.factors)
    )
    text = (
        f"q_k es el Q/Q_máx de cada factor ({names}), que sale de su θ₀ (Çengel y Ghajar, "
        "tabla 4-1). Se combinan con la **regla de Langston** (1982): lo que no intercambió un "
        f"factor lo intercambia el siguiente. Q_máx = ρ·V·c·ΔT es lo máximo que el cuerpo {verb} "
        f"{side}."
    )
    return ProcedureStep("Calor", text, tuple(latex))


def product_steps(r: ProductResult, system: UnitSystem) -> list[ProcedureStep]:
    """Producto de soluciones de una dimensión (Çengel y Ghajar §4-4)."""
    i = r.inputs
    symbols = _factor_symbols(i.factors)
    steps: list[ProcedureStep | None] = [
        ProcedureStep(
            "Difusividad",
            "La misma α para todos los factores." + _alpha_text(system),
            (_alpha_latex(i.solid, system),),
        ),
        *(_factor_step(fr, s, r, system) for fr, s in zip(r.factors, symbols, strict=True)),
        _product_step(r, symbols, system),
        _product_heat_step(r, symbols, system),
    ]
    return numbered([s for s in steps if s is not None])

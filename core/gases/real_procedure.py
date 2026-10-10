"""Procedimiento didáctico de los gases reales (LaTeX), en el sistema de unidades activo.

Las fórmulas son las del vademecum: §7 (estados correspondientes), §8 (Van der Waals) y
§15 (funciones características), con Lee y Kesler (1975), Peng y Robinson (1976) y
Çengel §12-2 a §12-5 (diferencias finitas, Clapeyron y Joule–Thomson). Las variables
reducidas no tienen unidades; p·v lleva el factor de unidades del Técnico
(1 bar·m³/kg = 100 kJ/kg) y del Inglés (1 psia·ft³/lb = 0,18505 Btu/lb), y una derivada
con s pasa a las unidades del lado p–v–T con ese mismo factor.
"""

from __future__ import annotations

import math

from core.gases.cubic import CUBIC_MODELS, ROOT_NAMES, CubicResult, CubicSolution, maxwell
from core.gases.real import (
    LK_OMEGA_REF,
    CompressibilityResult,
    ModelResult,
    lee_kesler,
    lk_coefficients,
    lk_z_of_vr,
    pr_kappa,
)
from core.gases.relations import (
    ClapeyronResult,
    Derivative,
    JouleThomsonResult,
    MaxwellRelation,
    PathState,
    RelationsResult,
)
from core.heat_transfer.procedure_common import frac, n, numbered, q, sub, times
from core.ideal_gas import R_U
from core.latex import latex_chain, latex_number, latex_paren, latex_unit
from core.state_report import ProcedureStep, pv_energy_factor
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "clapeyron_steps",
    "compressibility_steps",
    "cubic_steps",
    "joule_thomson_steps",
    "relations_steps",
]

_P: QuantityKind = "pressure"
_T: QuantityKind = "absolute_temperature"
_V: QuantityKind = "specific_volume"
_E: QuantityKind = "specific_enthalpy"
_S: QuantityKind = "specific_entropy"
_RU: QuantityKind = "molar_entropy"
_M: QuantityKind = "molar_mass"


def _r(x: float, sig: int = 5) -> str:
    """Un número sin unidad (variables reducidas, Z, A, B…)."""
    return latex_number(x, sig)


def _pv(system: UnitSystem) -> tuple[float, str]:
    """El factor de p·v (1, 100 o 0,18505) y su LaTeX (vacío en el SI)."""
    f = pv_energy_factor(system)
    return f, ("" if math.isclose(f, 1.0) else latex_number(f, 5))


def _pv_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " Ojo con las unidades de p·v: 1 bar·m³/kg = 100 kJ/kg."
    if system == "Inglés":
        return " Ojo con las unidades de p·v: 1 psia·ft³/lb = 0,18505 Btu/lb."
    return ""


def _with_factor(f_tex: str, *factors: str) -> str:
    """``f · a · b`` si hay factor (Técnico e Inglés)."""
    return times(f_tex, *factors) if f_tex else times(*factors)


def _sq(tex: str) -> str:
    """``x^2``, con paréntesis si x lleva ×10ⁿ o es negativo."""
    if r"\times" in tex or tex.startswith("-"):
        return rf"\left({tex}\right)^2"
    return f"{tex}^2"


def _rows(*pairs: tuple[str, str]) -> str:
    """Un ``aligned`` con un renglón por par ``(lado izquierdo, lado derecho)``."""
    return r"\begin{aligned}" + r" \\ ".join(f"{a} &= {b}" for a, b in pairs) + r"\end{aligned}"


def _num_txt(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _pct_txt(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:+.{decimals}f} %".replace(".", ",").replace("-", "−")


def _cubic_poly(c2: float, c1: float, c0: float) -> str:
    """``Z^3 - 1.0357\\,Z^2 \\ + 0.13941\\,Z - 0.0049713 = 0`` en dos renglones."""

    def term(c: float, z: str) -> str:
        sign = "-" if c < 0 else "+"
        return rf"{sign} {_r(abs(c))}{z}"

    z2, z1 = term(c2, r"\,Z^2"), term(c1, r"\,Z")
    return rf"\begin{{aligned}}&Z^3 {z2} \\ &\quad {z1} {term(c0, '')} = 0\end{{aligned}}"


def _flush_chain(lhs: str, *steps: str, relation: str = "=") -> str:
    """``lhs = paso₁ = paso₂…`` contra el margen izquierdo (un lado izquierdo largo no se
    suma al ancho del renglón más largo, como en un ``aligned`` alineado en el signo)."""
    rows = [rf"&{lhs} {relation} {steps[0]}", *(rf"&= {step}" for step in steps[1:])]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _stack_chain(lhs: str, *steps: str, relation: str = "=") -> str:
    """``lhs`` solo en el primer renglón y cada ``= paso`` abajo, contra el margen."""
    first, *rest = steps
    rows = [f"&{lhs}", rf"&{relation} {first}", *(rf"&= {step}" for step in rest)]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _chain_or_stack(lhs: str, *steps: str, relation: str = "=") -> str:
    """Una cadena alineada; apilada si algún paso lleva ×10ⁿ (no entraría al lado de lhs)."""
    if any(r"\times" in step for step in steps):
        return _stack_chain(lhs, *steps, relation=relation)
    return latex_chain(lhs, *steps, relation=relation)


def _kappa_term(kappa: float) -> str:
    """``1 + 0.37\\left(…`` o ``1 - 0.26\\left(…`` (κ negativo: el helio, el hidrógeno)."""
    sign = "-" if kappa < 0 else "+"
    return rf"1 {sign} {_r(abs(kappa))}"


_ROOT_SUB = {"liquid": r"\text{líq}", "vapor": r"\text{vap}", "unstable": r"\text{medio}"}


def _roots_rows(zs: tuple[float, ...], kinds: tuple[str, ...]) -> str:
    if len(zs) == 1:
        return latex_chain("Z", _r(zs[0]))
    return _rows(
        *((f"Z_{{{_ROOT_SUB.get(k, '')}}}", _r(z)) for z, k in zip(zs, kinds, strict=True))
    )


# ---------------------------------------------------------------------
# El factor de compresibilidad
# ---------------------------------------------------------------------


def _fluid_step(r: CompressibilityResult | CubicResult, system: UnitSystem) -> ProcedureStep:
    fl = r.fluid
    _, f_tex = _pv(system)
    lines = [
        latex_chain(
            "R",
            frac("R_u", "M"),
            frac(q(R_U, _RU, system), q(fl.M, _M, system)),
            q(fl.R, _S, system),
        ),
        _rows(
            (r"T_{cr}", q(fl.T_cr, _T, system)),
            (r"p_{cr}", q(fl.p_cr, _P, system)),
            (r"v_{cr}", q(fl.v_cr, _V, system)),
            (r"\omega", _r(fl.omega, 4)),
        ),
        latex_chain(
            r"Z_{cr}",
            frac(r"p_{cr}\,v_{cr}", r"R\,T_{cr}"),
            frac(
                _with_factor(f_tex, n(fl.p_cr, _P, system), n(fl.v_cr, _V, system)),
                times(n(fl.R, _S, system), n(fl.T_cr, _T, system)),
            ),
            _r(fl.Z_cr, 4),
        ),
    ]
    return ProcedureStep(
        "El fluido y su punto crítico",
        f"El punto crítico y el factor acéntrico ω del {fl.noun} son los de su ecuación de estado "
        "(CoolProp; vademecum §7.5), y R = R_u/M. Z_cr = p_cr·v_cr/(R·T_cr) (§7.4)."
        + _pv_text(system),
        tuple(lines),
    )


def _vr_pseudo(fl, v: float, system: UnitSystem) -> str:  # noqa: ANN001
    _, f_tex = _pv(system)
    return latex_chain(
        r"v'_R",
        frac(r"v\,p_{cr}", r"R\,T_{cr}"),
        frac(
            _with_factor(f_tex, n(v, _V, system), n(fl.p_cr, _P, system)),
            times(n(fl.R, _S, system), n(fl.T_cr, _T, system)),
        ),
        _r(v * fl.p_cr / (fl.R * fl.T_cr)),
    )


def _reduced_step(r: CompressibilityResult, system: UnitSystem) -> ProcedureStep:
    fl = r.fluid
    pair = r.inputs.pair
    lines = []
    if pair in ("pT", "Tv"):
        lines.append(
            latex_chain(
                "T_R",
                frac("T", r"T_{cr}"),
                frac(n(r.T_K, _T, system), n(fl.T_cr, _T, system)),
                _r(r.T_R),
            )
        )
    if pair in ("pT", "pv"):
        lines.append(
            latex_chain(
                "p_R",
                frac("p", r"p_{cr}"),
                frac(n(r.p_Pa, _P, system), n(fl.p_cr, _P, system)),
                _r(r.p_R),
            )
        )
    if pair in ("Tv", "pv"):
        lines.append(_vr_pseudo(fl, r.v_m3_per_kg, system))
    text = (
        "Las variables reducidas (vademecum §7.1) con la temperatura absoluta. Con el volumen "
        "como dato se usa el volumen pseudorreducido de Nelson y Obert (§7.4), v′_R = "
        "v·p_cr/(R·T_cr), que no necesita el v_cr real."
        if pair != "pT"
        else "Las variables reducidas (vademecum §7.1), con la temperatura absoluta."
    )
    return ProcedureStep("Las variables reducidas", text + _pv_text(system), tuple(lines))


def _unknown_line(
    r: CompressibilityResult, m: ModelResult, Z_tex: str, system: UnitSystem, Z: float | None = None
) -> str:
    """La incógnita con Z: v = Z·R·T/p, p = Z·R·T/v o T = p·v/(Z·R); sin Z si es "" (gas
    ideal)."""
    fl = r.fluid
    _, f_tex = _pv(system)
    pair = r.inputs.pair
    assert m.p_Pa is not None and m.T_K is not None and m.v_m3_per_kg is not None
    Zv = m.Z if Z is None else Z
    assert Zv is not None
    z_sym = rf"{Z_tex}\," if Z_tex else ""
    z_num = [_r(Zv)] if Z_tex else []
    R_n, T_n = n(fl.R, _S, system), n(m.T_K, _T, system)
    if pair == "pT":
        return latex_chain(
            "v",
            frac(rf"{z_sym}R\,T", "p"),
            frac(times(*z_num, R_n, T_n), _with_factor(f_tex, n(m.p_Pa, _P, system))),
            q(m.v_m3_per_kg, _V, system),
        )
    if pair == "Tv":
        return latex_chain(
            "p",
            frac(rf"{z_sym}R\,T", "v"),
            frac(times(*z_num, R_n, T_n), _with_factor(f_tex, n(m.v_m3_per_kg, _V, system))),
            q(m.p_Pa, _P, system),
        )
    return latex_chain(
        "T",
        frac(r"p\,v", rf"{z_sym}R"),
        frac(
            _with_factor(f_tex, n(m.p_Pa, _P, system), n(m.v_m3_per_kg, _V, system)),
            times(*z_num, R_n),
        ),
        q(m.T_K, _T, system),
    )


def _lk_equation() -> str:
    return latex_chain(
        "Z",
        r"1 + \frac{B}{v_R} + \frac{C}{v_R^2} + \frac{D}{v_R^5} \\ &\quad + "
        r"\frac{c_4}{T_R^3\,v_R^2}\left(\beta + \frac{\gamma}{v_R^2}\right) e^{-\gamma/v_R^2}",
    )


def _lk_coefficient_lines(Tr: float) -> list[str]:
    B, C, D = lk_coefficients(Tr, "simple")
    return [
        latex_chain(
            "B",
            r"b_1 - \frac{b_2}{T_R} - \frac{b_3}{T_R^2} - \frac{b_4}{T_R^3}",
            _r(B),
        ),
        latex_chain("C", r"c_1 - \frac{c_2}{T_R} + \frac{c_3}{T_R^3}", _r(C)),
        latex_chain("D", r"d_1 + \frac{d_2}{T_R}", _r(D)),
    ]


def _lk0_step(r: CompressibilityResult, system: UnitSystem) -> ProcedureStep:
    title = "La carta generalizada: Z⁰ de Lee y Kesler"
    m = r.model("lk0")
    if m.Z is None:
        return ProcedureStep(title, m.note)
    fl = r.fluid
    assert m.p_Pa is not None and m.T_K is not None and m.v_m3_per_kg is not None
    Tr, pr = m.T_K / fl.T_cr, m.p_Pa / fl.p_cr
    vr = m.v_m3_per_kg * fl.p_cr / (fl.R * fl.T_cr)
    lines = [_lk_equation()]
    pair = r.inputs.pair
    if pair == "pv":
        lines.append(latex_chain("T", q(m.T_K, _T, system)))
        lines.append(latex_chain("T_R", frac("T", r"T_{cr}"), _r(Tr)))
    lines += _lk_coefficient_lines(Tr)
    if pair == "pT":
        lines.append(latex_chain(r"v_R^0", _r(vr)))
        lines.append(
            latex_chain(
                r"Z^0",
                frac(r"p_R\,v_R^0", "T_R"),
                frac(times(_r(pr), _r(vr)), _r(Tr)),
                _r(m.Z),
            )
        )
        text = (
            "La carta de Çengel (figura 3-51, de Nelson y Obert) es la del fluido simple de Lee y "
            "Kesler (1975): una ecuación del tipo de Benedict–Webb–Rubin con 12 constantes, en "
            "variables reducidas. Con T_R y p_R se busca el v_R que cumple p_R·v_R/T_R = Z(T_R, "
            "v_R): la raíz mayor si es vapor, la menor si es líquido."
        )
    elif pair == "Tv":
        lines.append(latex_chain(r"Z^0", r"Z(T_R,\,v'_R)", _r(float(lk_z_of_vr(Tr, vr, "simple")))))
        text = (
            "Con T y v como datos, v_R = v′_R: Z⁰ sale directo de la ecuación, como en la carta "
            "de Çengel, que tiene las líneas de v′_R constante. La ecuación es la del fluido "
            "simple de Lee y Kesler (1975), del tipo de Benedict–Webb–Rubin."
        )
    else:
        lines.append(
            latex_chain(
                "p_R",
                frac(r"Z^0\,T_R", r"v'_R"),
                frac(times(_r(m.Z), _r(Tr)), _r(vr)),
                _r(pr),
            )
        )
        text = (
            "Con p y v como datos se busca la T con la que la ecuación del fluido simple de Lee "
            "y Kesler (1975) da la p_R del dato: p_R = Z⁰(T_R, v′_R)·T_R/v′_R."
        )
    if pair != "pv":
        lines.append(_unknown_line(r, m, "Z^0", system))
    return ProcedureStep(title, text, tuple(lines))


def _lk_step(r: CompressibilityResult, system: UnitSystem) -> ProcedureStep:
    title = "Lee–Kesler con el factor acéntrico"
    m = r.model("lk")
    if m.Z is None:
        return ProcedureStep(title, m.note)
    fl = r.fluid
    assert m.p_Pa is not None and m.T_K is not None
    Tr, pr = m.T_K / fl.T_cr, m.p_Pa / fl.p_cr
    lk = lee_kesler(Tr, pr, fl.omega, r.phase)
    lines = []
    pair = r.inputs.pair
    if pair == "Tv":
        lines.append(latex_chain("p", q(m.p_Pa, _P, system)))
        lines.append(latex_chain("p_R", frac("p", r"p_{cr}"), _r(pr)))
    elif pair == "pv":
        lines.append(latex_chain("T", q(m.T_K, _T, system)))
        lines.append(latex_chain("T_R", frac("T", r"T_{cr}"), _r(Tr)))
    lines += [
        _rows((r"Z^0", _r(lk.Z0)), (r"Z^r", _r(lk.Z_ref))),
        latex_chain(
            r"\frac{\omega}{\omega_r}",
            frac(_r(fl.omega, 4), _r(LK_OMEGA_REF, 4)),
            _r(fl.omega / LK_OMEGA_REF),
        ),
        latex_chain(r"Z^r - Z^0", sub(_r(lk.Z_ref), _r(lk.Z0)), _r(lk.Z_ref - lk.Z0)),
        latex_chain(
            "Z",
            r"Z^0 + \frac{\omega}{\omega_r}\left(Z^r - Z^0\right)",
            rf"{_r(lk.Z0)} \\ &\quad {'-' if fl.omega < 0 else '+'} "
            + times(_r(abs(fl.omega) / LK_OMEGA_REF), _r(lk.Z_ref - lk.Z0)),
            _r(lk.Z),
        ),
        _unknown_line(r, m, "Z", system, lk.Z),
    ]
    text = (
        "Lee y Kesler suman un fluido de referencia (el n-octano, ω_r = 0,3978) con su propia "
        "ecuación, a los mismos T_R y p_R, e interpolan con el factor acéntrico ω: los estados "
        "correspondientes de tres parámetros."
    )
    if pair != "pT":
        text += (
            f" Con {'T y v' if pair == 'Tv' else 'p y v'} hay que iterar: se busca la "
            f"{'presión' if pair == 'Tv' else 'temperatura'} con la que Z·R·T/p da el v del dato."
        )
    return ProcedureStep(title, text, tuple(lines))


def _vdw_step(r: CompressibilityResult, system: UnitSystem) -> ProcedureStep:
    title = "Van der Waals"
    m = r.model("vdw")
    if m.Z is None:
        return ProcedureStep(title, m.note)
    fl = r.fluid
    pair = r.inputs.pair
    lines = []
    if pair == "pT":
        Tr, pr = r.T_R, r.p_R
        A, B = 27.0 / 64.0 * pr / Tr**2, pr / (8.0 * Tr)
        lines += [
            latex_chain(
                "A",
                r"\frac{27}{64}\,\frac{p_R}{T_R^2}",
                rf"\frac{{27}}{{64}}\cdot\frac{{{_r(pr)}}}{{{_sq(_r(Tr))}}}",
                _r(A),
            ),
            latex_chain("B", frac("p_R", r"8\,T_R"), frac(_r(pr), times("8", _r(Tr))), _r(B)),
            r"Z^3 - (1 + B)\,Z^2 + A\,Z - A\,B = 0",
            _cubic_poly(-(1.0 + B), A, -A * B),
            _roots_rows(m.roots, _kinds(len(m.roots))),
            _unknown_line(r, m, "Z", system),
        ]
        text = (
            "La ecuación de Van der Waals (vademecum §8.1) con a y b del punto crítico (§8.2), "
            "escrita en Z: Z³ − (1 + B)·Z² + A·Z − A·B = 0, con A = a·p/(R·T)² = 27·p_R/(64·T_R²) "
            "y B = b·p/(R·T) = p_R/(8·T_R). "
            + (
                "Tiene tres raíces: la mayor es la del vapor, la menor la del líquido y la del "
                f"medio no tiene sentido físico; acá va la del {_phase_word(r.phase)}."
                if len(m.roots) == 3
                else "Tiene una sola raíz real."
            )
        )
    else:
        vr_p = r.v_R_pseudo
        vr = 8.0 / 3.0 * vr_p
        lines.append(_vr_pseudo(fl, r.v_m3_per_kg, system))
        lines.append(
            latex_chain("v_R", r"\frac{8}{3}\,v'_R", times(r"\frac{8}{3}", _r(vr_p)), _r(vr))
        )
        if pair == "Tv":
            Tr = r.T_R
            pr = 8.0 * Tr / (3.0 * vr - 1.0) - 3.0 / vr**2
            lines += [
                latex_chain(
                    "p_R",
                    r"\frac{8\,T_R}{3\,v_R - 1} - \frac{3}{v_R^2}",
                    rf"\frac{{8 \cdot {_r(Tr)}}}{{3 \cdot {_r(vr)} - 1}} - "
                    rf"\frac{{3}}{{{_sq(_r(vr))}}}",
                    _r(pr),
                ),
                latex_chain(
                    "p",
                    r"p_R\,p_{cr}",
                    times(_r(pr), n(fl.p_cr, _P, system)),
                    q(m.p_Pa, _P, system),
                ),  # type: ignore[arg-type]
            ]
        else:
            pr = r.p_R
            Tr = (pr + 3.0 / vr**2) * (3.0 * vr - 1.0) / 8.0
            lines += [
                latex_chain(
                    r"p_R + \frac{3}{v_R^2}",
                    rf"{_r(pr)} + \frac{{3}}{{{_sq(_r(vr))}}}",
                    _r(pr + 3.0 / vr**2),
                ),
                latex_chain(
                    "T_R",
                    r"\frac{(p_R + 3/v_R^2)(3\,v_R - 1)}{8}",
                    rf"\frac{{{_r(pr + 3.0 / vr**2)}\,(3 \cdot {_r(vr)} - 1)}}{{8}}",
                    _r(Tr),
                ),
                latex_chain(
                    "T", r"T_R\,T_{cr}", times(_r(Tr), n(fl.T_cr, _T, system)), q(m.T_K, _T, system)
                ),  # type: ignore[arg-type]
            ]
        text = (
            "Van der Waals en forma reducida (vademecum §8.4): (p_R + 3/v_R²)(3·v_R − 1) = 8·T_R, "
            "con v_R = v/v_cr y el v_cr = 3b del modelo, que no es el real: v_R = (8/3)·v′_R."
        )
    return ProcedureStep(title, text, tuple(lines))


def _kinds(count: int) -> tuple[str, ...]:
    return ("liquid", "unstable", "vapor") if count == 3 else ("vapor",) * count


def _phase_word(phase: str) -> str:
    return {"liquid": "líquido", "vapor": "vapor"}.get(phase, "fluido")


def _pr_step(r: CompressibilityResult, system: UnitSystem) -> ProcedureStep:
    title = "Peng–Robinson"
    m = r.model("pr")
    if m.Z is None:
        return ProcedureStep(title, m.note)
    fl = r.fluid
    assert m.T_K is not None
    pair = r.inputs.pair
    Tr = m.T_K / fl.T_cr
    kappa = pr_kappa(fl.omega)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(Tr))) ** 2
    om = _r(fl.omega, 4)
    lines = []
    if pair == "pv":
        lines.append(latex_chain("T", q(m.T_K, _T, system)))
        lines.append(latex_chain("T_R", frac("T", r"T_{cr}"), _r(Tr)))
    lines += [
        latex_chain(
            r"\kappa",
            r"0.37464 + 1.54226\,\omega \\ &\quad - 0.26992\,\omega^2",
            rf"0.37464 + 1.54226 \cdot {latex_paren(om)} \\ &\quad - 0.26992 \cdot {_sq(om)}",
            _r(kappa),
        ),
        latex_chain(
            r"\alpha",
            r"\left[1 + \kappa\left(1 - \sqrt{T_R}\right)\right]^2",
            rf"\left[{_kappa_term(kappa)}\left(1 - \sqrt{{{_r(Tr)}}}\right)\right]^2",
            _r(alpha),
        ),
    ]
    if pair == "pT":
        pr = r.p_R
        A, B = 0.45724 * alpha * pr / Tr**2, 0.07780 * pr / Tr
        lines += [
            latex_chain(
                "A",
                r"0.45724\,\alpha\,\frac{p_R}{T_R^2}",
                rf"0.45724 \cdot {_r(alpha)} \cdot \frac{{{_r(pr)}}}{{{_sq(_r(Tr))}}}",
                _r(A),
            ),
            latex_chain(
                "B",
                r"0.07780\,\frac{p_R}{T_R}",
                rf"0.07780 \cdot \frac{{{_r(pr)}}}{{{_r(Tr)}}}",
                _r(B),
            ),
            r"\begin{aligned}&Z^3 - (1 - B)\,Z^2 \\ &\quad + (A - 3B^2 - 2B)\,Z \\ &\quad"
            r" - (A\,B - B^2 - B^3) = 0\end{aligned}",
            _cubic_poly(-(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)),
            _roots_rows(m.roots, _kinds(len(m.roots))),
            _unknown_line(r, m, "Z", system),
        ]
        text = (
            "Peng y Robinson (1976): p = R·T/(v − b) − a·α/(v² + 2·b·v − b²), con a y b del "
            "punto crítico y α(T) ajustada con ω para que acierte la presión de vapor. En Z, "
            "A = a·α·p/(R·T)² y B = b·p/(R·T)."
        )
    else:
        vr = r.v_R_pseudo
        assert m.p_Pa is not None
        pr = m.p_Pa / fl.p_cr
        b = 0.07780
        if pair == "Tv":
            lines.append(_vr_pseudo(fl, r.v_m3_per_kg, system))
        lines += [
            latex_chain(
                "p_R",
                r"\frac{T_R}{v'_R - b'} \\ &\quad - "
                r"\frac{\Omega_a\,\alpha}{v'^2_R + 2\,b'\,v'_R - b'^2}",
                rf"\frac{{{_r(Tr)}}}{{{_r(vr)} - {_r(b)}}} \\ &\quad - "
                rf"\frac{{0.45724 \cdot {_r(alpha)}}}{{{_r(vr * vr + 2 * b * vr - b * b)}}}",
                _r(pr),
            ),
        ]
        if pair == "Tv":
            lines.append(
                latex_chain(
                    "p",
                    r"p_R\,p_{cr}",
                    times(_r(pr), n(fl.p_cr, _P, system)),
                    q(m.p_Pa, _P, system),
                )
            )
        text = (
            "Peng y Robinson (1976) en variables reducidas, con v′_R: b′ = 0,07780 y Ω_a = "
            "0,45724 (sus constantes del punto crítico) y α(T) ajustada con ω."
            + (
                " Con p y v hay que iterar: se busca la T con la que la ecuación da la p_R del "
                "dato."
                if pair == "pv"
                else ""
            )
        )
    return ProcedureStep(title, text, tuple(lines))


def compressibility_steps(r: CompressibilityResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del factor de compresibilidad con los seis modelos (vademecum §7 y §8)."""
    fl = r.fluid
    ideal = r.model("ideal")
    steps = [_fluid_step(r, system), _reduced_step(r, system)]
    steps.append(
        ProcedureStep(
            "Como gas ideal (Z = 1)",
            "p·v = R·T (vademecum §4.2): el punto de partida." + _pv_text(system),
            (_unknown_line(r, ideal, "", system),),
        )
    )
    steps += [_lk0_step(r, system), _lk_step(r, system), _vdw_step(r, system), _pr_step(r, system)]
    _, f_tex = _pv(system)
    real_lines = [
        latex_chain(
            "Z",
            frac(r"p\,v", r"R\,T"),
            frac(
                _with_factor(f_tex, n(r.p_Pa, _P, system), n(r.v_m3_per_kg, _V, system)),
                times(n(fl.R, _S, system), n(r.T_K, _T, system)),
            ),
            _r(r.Z),
        )
    ]
    errors = []
    unknown = {"v": "v", "p": "p", "T": "T"}[r.unknown]
    for m in r.models:
        if m.model == "coolprop" or m.error is None:
            continue
        errors.append(f"{_model_name(m.model)} {_pct_txt(m.error)}")
    steps.append(
        ProcedureStep(
            "El fluido real (CoolProp)",
            f"Con la ecuación de estado de referencia del {fl.noun}, Z = p·v/(R·T) (vademecum "
            f"§7.3). El error de cada modelo en {unknown}: " + "; ".join(errors) + ".",
            tuple(real_lines),
        )
    )
    return numbered(steps)


def _model_name(kind: str) -> str:
    return {
        "ideal": "gas ideal",
        "lk0": "carta generalizada",
        "lk": "Lee–Kesler con ω",
        "vdw": "Van der Waals",
        "pr": "Peng–Robinson",
    }[kind]


# ---------------------------------------------------------------------
# Las cúbicas
# ---------------------------------------------------------------------


def _a_quantity(a_si: float, system: UnitSystem) -> str:
    """a en [p]·[v]² (Pa·m⁶/kg², bar·m⁶/kg² o psia·ft⁶/lb²)."""
    value = a_si * convert_from_si(1.0, _P, system) * convert_from_si(1.0, _V, system) ** 2
    p_unit = unit_label(_P, system)
    v_num, v_den = unit_label(_V, system).split("/")
    unit = rf"\mathrm{{{p_unit}\,{v_num[:-1]}^6/{v_den}^2}}"
    return rf"{latex_number(value, 5)}\ {unit}"


def _cubic_solution_lines(
    sol: CubicSolution, r: CubicResult, system: UnitSystem
) -> tuple[list[str], str]:
    Tr, pr = r.T_R, r.p_R
    A, B = sol.A, sol.B
    lines = []
    if sol.model == "vdw":
        lines += [
            latex_chain(
                "A",
                r"\frac{27}{64}\,\frac{p_R}{T_R^2}",
                rf"\frac{{27}}{{64}}\cdot\frac{{{_r(pr)}}}{{{_sq(_r(Tr))}}}",
                _r(A),
            ),
            latex_chain("B", frac("p_R", r"8\,T_R"), frac(_r(pr), times("8", _r(Tr))), _r(B)),
            r"Z^3 - (1 + B)\,Z^2 + A\,Z - A\,B = 0",
        ]
    else:
        lines += [
            latex_chain(
                "A",
                r"0.45724\,\alpha\,\frac{p_R}{T_R^2}",
                rf"0.45724 \cdot {_r(sol.alpha)} \cdot \frac{{{_r(pr)}}}{{{_sq(_r(Tr))}}}",
                _r(A),
            ),
            latex_chain(
                "B",
                r"0.07780\,\frac{p_R}{T_R}",
                rf"0.07780 \cdot \frac{{{_r(pr)}}}{{{_r(Tr)}}}",
                _r(B),
            ),
            r"\begin{aligned}&Z^3 - (1 - B)\,Z^2 \\ &\quad + (A - 3B^2 - 2B)\,Z \\ &\quad"
            r" - (A\,B - B^2 - B^3) = 0\end{aligned}",
        ]
    lines.append(_cubic_poly(*sol.coefficients))
    lines.append(_roots_rows(tuple(x.Z for x in sol.roots), tuple(x.kind for x in sol.roots)))
    phys = [x for x in sol.roots if x.ln_phi is not None]
    for x in phys if len(phys) > 1 else ():
        label = _ROOT_SUB.get(x.kind, "")
        if sol.model == "vdw":
            formula = r"Z - 1 - \ln(Z - B) - \frac{A}{Z}"
        else:
            formula = (
                r"Z - 1 - \ln(Z - B) \\ &\quad - \frac{A}{2\sqrt{2}\,B}"
                r"\ln\frac{Z + 2.414\,B}{Z - 0.414\,B}"
            )
        lines.append(latex_chain(rf"\ln\varphi_{{{label}}}", formula, _r(x.ln_phi or 0.0)))
    stable = sol.stable
    if len(phys) > 1:
        text = (
            f"La raíz estable es la de menor coeficiente de fugacidad φ (menor energía libre de "
            f"Gibbs): la del {ROOT_NAMES[stable.kind]}. La del medio no tiene sentido físico (ahí "
            "∂p/∂v > 0)."
        )
    elif stable.kind == "supercritical":
        text = "Sobre la temperatura crítica la cúbica tiene una sola raíz: un fluido supercrítico."
    else:
        text = f"Una sola raíz real: la del {ROOT_NAMES[stable.kind]}."
    _, f_tex = _pv(system)
    lines.append(
        latex_chain(
            "v",
            frac(r"Z\,R\,T", "p"),
            frac(
                times(_r(stable.Z), n(r.fluid.R, _S, system), n(r.inputs.T_K, _T, system)),
                _with_factor(f_tex, n(r.inputs.p_Pa, _P, system)),
            ),
            q(stable.v_m3_per_kg, _V, system),
        )
    )
    return lines, text


def cubic_steps(r: CubicResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de Van der Waals y Peng–Robinson en (p, T), con Maxwell (vademecum §8)."""
    fl = r.fluid
    steps = [_fluid_step(r, system)]
    steps.append(
        ProcedureStep(
            "Las variables reducidas",
            "Con la temperatura absoluta (vademecum §7.1).",
            (
                latex_chain(
                    "T_R",
                    frac("T", r"T_{cr}"),
                    frac(n(r.inputs.T_K, _T, system), n(fl.T_cr, _T, system)),
                    _r(r.T_R),
                ),
                latex_chain(
                    "p_R",
                    frac("p", r"p_{cr}"),
                    frac(n(r.inputs.p_Pa, _P, system), n(fl.p_cr, _P, system)),
                    _r(r.p_R),
                ),
            ),
        )
    )
    vdw = r.vdw
    steps.append(
        ProcedureStep(
            "Van der Waals: a, b y su punto crítico",
            "a = 27·R²·T_cr²/(64·p_cr) y b = R·T_cr/(8·p_cr) (vademecum §8.2). Con ellas el punto "
            "crítico del modelo cae en el real, pero con v_cr = 3b y Z_cr = 3/8 (§8.3), más que "
            f"el Z_cr = {_num_txt(fl.Z_cr, '.3f')} del {fl.noun}.",
            (
                latex_chain(
                    "a", r"\frac{27\,R^2\,T_{cr}^2}{64\,p_{cr}}", _a_quantity(vdw.a, system)
                ),
                latex_chain("b", r"\frac{R\,T_{cr}}{8\,p_{cr}}", q(vdw.b, _V, system)),
                latex_chain(r"v_{cr}", r"3\,b", q(3.0 * vdw.b, _V, system)),
                latex_chain(r"Z_{cr}", r"\frac{3}{8}", "0.375"),
            ),
        )
    )
    lines, text = _cubic_solution_lines(vdw, r, system)
    steps.append(ProcedureStep("Van der Waals: la cúbica en Z", text, tuple(lines)))
    steps.append(_vdw_maxwell_step(r, system))
    pr = r.pr
    kappa = pr.kappa or 0.0
    om = _r(fl.omega, 4)
    steps.append(
        ProcedureStep(
            "Peng–Robinson: a, b y α(T)",
            "a = 0,45724·R²·T_cr²/p_cr y b = 0,07780·R·T_cr/p_cr (Peng y Robinson, 1976); el "
            "término de atracción se corrige con α(T), que usa el factor acéntrico ω.",
            (
                latex_chain(
                    "a",
                    r"0.45724\,\frac{R^2\,T_{cr}^2}{p_{cr}}",
                    _a_quantity(pr.a_cr, system),
                ),
                latex_chain("b", r"0.07780\,\frac{R\,T_{cr}}{p_{cr}}", q(pr.b, _V, system)),
                latex_chain(
                    r"\kappa",
                    r"0.37464 + 1.54226\,\omega \\ &\quad - 0.26992\,\omega^2",
                    rf"0.37464 + 1.54226 \cdot {latex_paren(om)} \\ &\quad - 0.26992 \cdot "
                    rf"{_sq(om)}",
                    _r(kappa),
                ),
                latex_chain(
                    r"\alpha",
                    r"\left[1 + \kappa\left(1 - \sqrt{T_R}\right)\right]^2",
                    rf"\left[{_kappa_term(kappa)}"
                    rf"\left(1 - \sqrt{{{_r(r.T_R)}}}\right)\right]^2",
                    _r(pr.alpha),
                ),
            ),
        )
    )
    lines, text = _cubic_solution_lines(pr, r, system)
    steps.append(ProcedureStep("Peng–Robinson: la cúbica en Z", text, tuple(lines)))
    steps.append(_pr_maxwell_step(r, system))
    steps.append(_omega_step(r, system))
    steps.append(_real_cubic_step(r, system))
    return numbered(steps)


def _vdw_maxwell_step(r: CubicResult, system: UnitSystem) -> ProcedureStep:
    title = "Van der Waals: la saturación (Maxwell)"
    sat = r.vdw.saturation
    if sat is None:
        return ProcedureStep(title, "Sobre la temperatura crítica no hay saturación: no hay lazo.")
    fl = r.fluid
    Tr = r.T_R
    v3b = 3.0 * r.vdw.b
    ps = sat.p_sat_Pa / fl.p_cr
    vf, vg = sat.v_f / v3b, sat.v_g / v3b
    integral = 8.0 * Tr / 3.0 * math.log((3 * vg - 1) / (3 * vf - 1)) + 3.0 * (1 / vg - 1 / vf)
    lines = [
        latex_chain(r"p_{sat}", q(sat.p_sat_Pa, _P, system)),
        _rows(
            (r"p_{R,sat}", _r(ps)),
            (r"v_{R,f}", _r(vf)),
            (r"v_{R,g}", _r(vg)),
        ),
        _flush_chain(
            r"\int_{v_{R,f}}^{v_{R,g}} p_R\,dv_R",
            r"\frac{8\,T_R}{3}\ln\frac{3\,v_{R,g} - 1}{3\,v_{R,f} - 1} \\ &\quad + "
            r"3\left(\frac{1}{v_{R,g}} - \frac{1}{v_{R,f}}\right)",
            _r(integral),
        ),
        latex_chain(r"v_{R,g} - v_{R,f}", sub(_r(vg), _r(vf)), _r(vg - vf)),
        _flush_chain(
            r"p_{R,sat}\,(v_{R,g} - v_{R,f})",
            times(_r(ps), _r(vg - vf)),
            _r(ps * (vg - vf)),
        ),
    ]
    return ProcedureStep(
        title,
        "La construcción de Maxwell: la presión de saturación es la horizontal que corta dos "
        "áreas iguales en el lazo de la isoterma, ∫p·dv = p_sat·(v_g − v_f). Es lo mismo que "
        "igualar la energía libre de Gibbs (los φ) del líquido y del vapor, que es como se "
        "calcula. En variables reducidas (v_R = v/3b) la integral se hace a mano.",
        tuple(lines),
    )


def _pr_maxwell_step(r: CubicResult, system: UnitSystem) -> ProcedureStep:
    title = "Peng–Robinson: la saturación"
    sat = r.pr.saturation
    if sat is None:
        return ProcedureStep(title, "Sobre la temperatura crítica no hay saturación.")
    from core.gases.cubic import ln_phi
    from core.gases.real import cubic_ab

    A, B = cubic_ab(r.fluid, "pr", r.inputs.T_K, sat.p_sat_Pa)
    lines = [
        latex_chain(r"p_{sat}", q(sat.p_sat_Pa, _P, system)),
        _rows((r"Z_f", _r(sat.Z_f)), (r"Z_g", _r(sat.Z_g))),
        _rows(
            (r"\ln\varphi_f", _r(ln_phi("pr", sat.Z_f, A, B))),
            (r"\ln\varphi_g", _r(ln_phi("pr", sat.Z_g, A, B))),
        ),
        _rows((r"v_f", q(sat.v_f, _V, system)), (r"v_g", q(sat.v_g, _V, system))),
    ]
    return ProcedureStep(
        title,
        "La presión de saturación del modelo es la que iguala los coeficientes de fugacidad "
        "del líquido y del vapor (la energía libre de Gibbs): las dos áreas iguales de Maxwell.",
        tuple(lines),
    )


def _omega_step(r: CubicResult, system: UnitSystem) -> ProcedureStep:
    fl = r.fluid
    lines = []
    for model, omega in (("vdw", r.omega_vdw), ("pr", r.omega_pr)):
        sat = maxwell(fl, model, 0.7 * fl.T_cr)  # type: ignore[arg-type]
        if sat is None:  # pragma: no cover
            continue
        ps = sat.p_sat_Pa / fl.p_cr
        label = r"\text{VdW}" if model == "vdw" else r"\text{PR}"
        lines.append(
            latex_chain(
                rf"\omega_{{{label}}}",
                r"-\log_{10} p_{R,sat}\big|_{T_R = 0.7} - 1",
                rf"-\log_{{10}} {_r(ps)} - 1",
                _r(omega, 4),
            )
        )
    return ProcedureStep(
        "El factor acéntrico de cada cúbica",
        "El ω de Pitzer mide la presión de vapor a T_R = 0,7: ω = −log₁₀(p_sat/p_cr) − 1. Van "
        "der Waals da la misma curva para todos los fluidos, con ω = −0,30; Peng–Robinson "
        f"reproduce el ω del {fl.noun} ({_num_txt(fl.omega, '.3f')}).",
        tuple(lines),
    )


def _real_cubic_step(r: CubicResult, system: UnitSystem) -> ProcedureStep:
    fl = r.fluid
    lines = []
    if r.p_sat_Pa is not None:
        lines.append(latex_chain(r"p_{sat}(T)", q(r.p_sat_Pa, _P, system)))
    lines.append(latex_chain("v", q(r.v_m3_per_kg, _V, system)))
    _, f_tex = _pv(system)
    lines.append(
        latex_chain(
            "Z",
            frac(r"p\,v", r"R\,T"),
            frac(
                _with_factor(f_tex, n(r.inputs.p_Pa, _P, system), n(r.v_m3_per_kg, _V, system)),
                times(n(fl.R, _S, system), n(r.inputs.T_K, _T, system)),
            ),
            _r(r.Z),
        )
    )
    parts = []
    for model in ("vdw", "pr"):
        v = r.model_v(model)  # type: ignore[arg-type]
        name = CUBIC_MODELS[model]  # type: ignore[index]
        if v is None:
            parts.append(f"{name} no tiene raíz en esa fase")
        else:
            parts.append(f"{name} se equivoca {_pct_txt(v / r.v_m3_per_kg - 1.0)} en v")
    phase = {"liquid": "líquido", "vapor": "vapor", "supercritical": "supercrítico"}[r.phase]
    return ProcedureStep(
        "El fluido real (CoolProp)",
        f"El {fl.noun} a esa p y T es {phase}. Con la raíz de esa fase, " + "; ".join(parts) + ".",
        tuple(lines),
    )


# ---------------------------------------------------------------------
# Funciones características y relaciones de Maxwell
# ---------------------------------------------------------------------

_DERIV_TEX: dict[str, str] = {
    "dT_dv_s": r"\left(\frac{\partial T}{\partial v}\right)_s",
    "dp_ds_v": r"\left(\frac{\partial p}{\partial s}\right)_v",
    "dT_dp_s": r"\left(\frac{\partial T}{\partial p}\right)_s",
    "dv_ds_p": r"\left(\frac{\partial v}{\partial s}\right)_p",
    "ds_dv_T": r"\left(\frac{\partial s}{\partial v}\right)_T",
    "dp_dT_v": r"\left(\frac{\partial p}{\partial T}\right)_v",
    "ds_dp_T": r"\left(\frac{\partial s}{\partial p}\right)_T",
    "dv_dT_p": r"\left(\frac{\partial v}{\partial T}\right)_p",
}
_TARGET: dict[str, QuantityKind] = {
    "u": "temperature_per_volume",
    "h": "temperature_per_pressure",
    "f": "pressure_per_temperature",
    "g": "volume_per_temperature",
}
_VALUE_KIND: dict[str, QuantityKind] = {"T": "temperature", "p": _P, "v": _V, "s": _S}
_DIFF_KIND: dict[str, QuantityKind] = {"T": "temperature_difference", "p": _P, "v": _V, "s": _S}
_POTENTIAL_NAMES = {
    "u": "u(s, v)",
    "h": "h(s, p)",
    "f": "f(T, v) = u − T·s",
    "g": "g(T, p) = h − T·s",
}
_PATH_TEXT = {
    "isothermal": "la isoterma a p ± Δp",
    "isentropic": "la isoentrópica a p ± Δp",
    "isobaric": "la isobara a T ± ΔT",
    "isochoric": "la isócora a T ± ΔT",
}


def _diff(a: float, b: float, kind: QuantityKind, system: UnitSystem) -> str:
    """``a − b`` con las cifras que hagan falta para que la diferencia tenga dos."""
    ua, ub = convert_from_si(a, kind, system), convert_from_si(b, kind, system)
    sig = 5
    if ua != ub:
        scale = max(abs(ua), abs(ub))
        sig = min(6, max(5, 2 + math.ceil(math.log10(scale / abs(ua - ub)))))
    return sub(n(a, kind, system, sig), n(b, kind, system, sig))


def _side_lines(
    d: Derivative, r: RelationsResult, target: QuantityKind, system: UnitSystem
) -> list[str]:
    """Una diferencia centrada con sus números y, si tiene s, el pase a las unidades p–v–T."""
    lo, hi = r.paths[d.path]
    num, den = d.numerator, d.denominator
    num_hi, num_lo = getattr(hi, num), getattr(lo, num)
    den_hi, den_lo = getattr(hi, den), getattr(lo, den)
    central = (den == "p" and d.path in ("isothermal", "isentropic")) or (
        den == "T" and d.path in ("isobaric", "isochoric")
    )
    vk = _VALUE_KIND
    if central:
        step = r.dp if den == "p" else r.dT
        den_sym = r"2\,\Delta p" if den == "p" else r"2\,\Delta T"
        den_num = times("2", n(step, _DIFF_KIND[den], system))
    else:
        den_sym = f"{den}_+ - {den}_-"
        den_num = _diff(den_hi, den_lo, vk[den], system)
    num_sym = f"{num}_+ - {num}_-"
    num_num = _diff(num_hi, num_lo, vk[num], system)
    value_si = (num_hi - num_lo) / (den_hi - den_lo)
    lines: list[str] = []
    if len(num_num) + len(den_num) > 36 or (r"\times" in num_num + den_num):
        # números largos (×10ⁿ del SI, muchas cifras): primero las diferencias
        dk = _DIFF_KIND
        lines.append(latex_chain(rf"\Delta {num}", num_num, q(num_hi - num_lo, dk[num], system)))
        if not central:
            lines.append(
                latex_chain(rf"\Delta {den}", den_num, q(den_hi - den_lo, dk[den], system))
            )
            den_sym2, den_num = rf"\Delta {den}", n(den_hi - den_lo, dk[den], system)
        else:
            den_sym2 = den_sym
        num_num = n(num_hi - num_lo, dk[num], system)
        den_sym = den_sym2
        num_sym = rf"\Delta {num}"
    chain = [frac(num_sym, den_sym), frac(num_num, den_num)]
    f, _ = _pv(system)
    if "s" in (num, den) and not math.isclose(f, 1.0):
        raw = (
            value_si
            * convert_from_si(1.0, _DIFF_KIND[num], system)
            / convert_from_si(1.0, _DIFF_KIND[den], system)
        )
        raw_tex = latex_number(raw, 5)
        if r"\times" not in raw_tex:  # con ×10ⁿ no entra: el texto dice el factor
            u_num = latex_unit(unit_label(_DIFF_KIND[num], system))
            u_den = latex_unit(unit_label(_DIFF_KIND[den], system))
            chain.append(rf"{raw_tex}\ \frac{{{u_num}}}{{{u_den}}}")
    chain.append(q(value_si, target, system))
    return [*lines, _chain_or_stack(_DERIV_TEX[d.key], *chain, relation=r"\approx")]


def _relation_step(m: MaxwellRelation, r: RelationsResult, system: UnitSystem) -> ProcedureStep:
    target = _TARGET[m.potential]
    sign = "" if m.sign > 0 else "-"
    lines = [
        rf"{_DERIV_TEX[m.left.key]} = {sign}{_DERIV_TEX[m.right.key]}",
        *_side_lines(m.left, r, target, system),
        *_side_lines(m.right, r, target, system),
        _chain_or_stack(
            _DERIV_TEX[m.left.key],
            rf"{sign}{_DERIV_TEX[m.right.key]}",
            q(m.left.exact, target, system),
        ),
    ]
    gap = (
        "Las dos diferencias finitas difieren "
        + (f"{_pct_txt(m.gap_finite, 2)}" if not r.degenerate else "(α ≈ 0: las dos valen casi 0)")
        + "; las derivadas exactas de CoolProp (el último renglón) coinciden."
    )
    s_conv = ""
    if system == "Técnico":
        s_conv = " Una derivada con s pasa a las unidades p–v–T con 1 kJ = 0,01 bar·m³."
    elif system == "Inglés":
        s_conv = " Una derivada con s pasa a las unidades p–v–T con 1 Btu = 5,4039 psia·ft³."
    return ProcedureStep(
        f"La relación de Maxwell de {_POTENTIAL_NAMES[m.potential]}",
        f"Sale de igualar las derivadas cruzadas de {m.potential} (vademecum §15.3). El lado "
        f"izquierdo por {_PATH_TEXT[m.left.path]} y el derecho por {_PATH_TEXT[m.right.path]}, "
        f"como con las tablas (Çengel §12-2). {gap}{s_conv}",
        tuple(lines),
    )


def relations_steps(r: RelationsResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de las funciones características y las relaciones de Maxwell (vademecum §15)."""
    s = r.state
    _, f_tex = _pv(system)
    E = _E
    potentials = [
        _rows((r"u", q(s.u, E, system)), (r"s", q(s.s, _S, system)), (r"v", q(s.v, _V, system))),
        latex_chain(
            "h",
            r"u + p\,v",
            rf"{n(s.u, E, system)} \\ &\quad + "
            + _with_factor(f_tex, n(s.p, _P, system), n(s.v, _V, system)),
            q(s.h, E, system),
        ),
        latex_chain(
            "f",
            r"u - T\,s",
            sub(n(s.u, E, system), times(n(s.T, _T, system), n(s.s, _S, system))),
            q(s.f, E, system),
        ),
        latex_chain(
            "g",
            r"h - T\,s",
            sub(n(s.h, E, system), times(n(s.T, _T, system), n(s.s, _S, system))),
            q(s.g, E, system),
        ),
    ]
    steps = [
        ProcedureStep(
            "Los potenciales",
            "u, s y v de la ecuación de estado (CoolProp); h, f y g de sus definiciones "
            "(vademecum §15.1). Sus valores dependen del estado de referencia del fluido; las "
            "derivadas, no." + _pv_text(system),
            tuple(potentials),
        )
    ]
    steps.append(
        ProcedureStep(
            "Los pasos de las diferencias finitas",
            f"ΔT = {_txt_q(r.dT, 'temperature_difference', system)} y Δp = "
            f"{_txt_q(r.dp, _P, system)}. Cada derivada se estima con los dos estados de un "
            "camino (− y +): la isoterma y la isoentrópica en p ± Δp, la isobara y la isócora "
            "en T ± ΔT. Las diferencias centradas tienen un error proporcional a Δ².",
        )
    )
    steps += [_relation_step(m, r, system) for m in r.maxwell]
    cpcv = r.cp - r.cv
    steps.append(
        ProcedureStep(
            "α, κ_T y la relación de Mayer generalizada",
            "α = (1/v)·(∂v/∂T)_p y κ_T = −(1/v)·(∂v/∂p)_T (vademecum §15.4); con ellos, "
            "c_p − c_v = T·v·α²/κ_T (§15.5). En un gas ideal α = 1/T, κ_T = 1/p y c_p − c_v = R."
            + _pv_text(system),
            (
                latex_chain(r"\alpha", q(r.alpha, "expansion_coefficient", system)),
                latex_chain(r"\kappa_T", q(r.kappa_T, "isothermal_compressibility", system)),
                latex_chain(
                    r"v\,\alpha",
                    times(n(s.v, _V, system), n(r.alpha, "expansion_coefficient", system)),
                    q(s.v * r.alpha, "volume_per_temperature", system),
                ),
                latex_chain(
                    r"\frac{\alpha}{\kappa_T}",
                    frac(
                        n(r.alpha, "expansion_coefficient", system),
                        n(r.kappa_T, "isothermal_compressibility", system),
                    ),
                    q(r.alpha / r.kappa_T, "pressure_per_temperature", system),
                ),
                _mayer_line(r, f_tex, system),
                latex_chain(
                    r"c_p - c_v",
                    sub(n(r.cp, "specific_heat", system), n(r.cv, "specific_heat", system)),
                    q(cpcv, "specific_heat", system),
                ),
            ),
        )
    )
    steps.append(
        ProcedureStep(
            "El coeficiente de Joule–Thomson",
            _JT_TEXT + _pv_text(system),
            _jt_lines(
                s,
                r.relation("g").right.exact,
                r.cp,
                r.mu_JT_formula,
                r.mu_JT_finite,
                r.isenthalpic,
                r.dp,
                system,
            ),
        )
    )
    return numbered(steps)


_JT_TEXT = (
    "μ_JT = (∂T/∂p)_h = [T·(∂v/∂T)_p − v]/c_p (Çengel §12-5): positivo, el fluido se "
    "enfría al estrangularlo; negativo, se calienta; en un gas ideal, 0. Por diferencias, "
    "con los estados de h constante a p ± Δp."
)


def _jt_lines(
    s: PathState,
    dvdT: float,
    cp: float,
    mu_formula: float,
    mu_finite: float,
    isenthalpic: tuple[PathState, PathState],
    dp: float,
    system: UnitSystem,
) -> tuple[str, ...]:
    """μ_JT con la fórmula y por diferencias a h constante."""
    _, f_tex = _pv(system)
    lo, hi = isenthalpic
    Tdv = s.T * dvdT
    jt_num = n(Tdv - s.v, _V, system)
    return (
        latex_chain(
            r"T\left(\frac{\partial v}{\partial T}\right)_p",
            times(n(s.T, _T, system), n(dvdT, "volume_per_temperature", system)),
            q(Tdv, _V, system),
        ),
        _stack_chain(
            r"T\left(\frac{\partial v}{\partial T}\right)_p - v",
            sub(n(Tdv, _V, system), n(s.v, _V, system)),
            q(Tdv - s.v, _V, system),
        ),
        latex_chain(
            r"\mu_{JT}",
            frac(r"T\left(\frac{\partial v}{\partial T}\right)_p - v", "c_p"),
            frac(
                _with_factor(f_tex, f"({jt_num})") if f_tex else jt_num,
                n(cp, "specific_heat", system),
            ),
            q(mu_formula, "temperature_per_pressure", system),
        ),
        latex_chain(
            r"\mu_{JT}",
            frac(r"T_+ - T_-", r"2\,\Delta p"),
            frac(
                sub(n(hi.T, "temperature", system), n(lo.T, "temperature", system)),
                times("2", n(dp, _P, system)),
            ),
            q(mu_finite, "temperature_per_pressure", system),
            relation=r"\approx",
        ),
    )


def joule_thomson_steps(r: JouleThomsonResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del coeficiente de Joule–Thomson en un estado (Çengel §12-5)."""
    s = r.state
    steps = [
        ProcedureStep(
            "El estado",
            f"De la ecuación de estado del {r.fluid.noun} (CoolProp).",
            (
                _rows(
                    ("T", q(s.T, _T, system)),
                    ("p", q(s.p, _P, system)),
                    ("v", q(s.v, _V, system)),
                    ("c_p", q(r.cp, "specific_heat", system)),
                ),
                latex_chain(
                    r"\left(\frac{\partial v}{\partial T}\right)_p",
                    q(r.dv_dT, "volume_per_temperature", system),
                ),
            ),
        ),
        ProcedureStep(
            "El coeficiente de Joule–Thomson",
            _JT_TEXT + _pv_text(system),
            _jt_lines(s, r.dv_dT, r.cp, r.mu_formula, r.mu_finite, r.isenthalpic, r.dp, system),
        ),
    ]
    lo, hi = r.T_inversion_low, r.T_inversion_high
    text = "La curva de inversión es donde μ_JT = 0, es decir T·(∂v/∂T)_p = v. "
    if lo is not None and hi is not None:
        text += (
            f"A esta presión pasa por {_txt_q(lo, 'temperature', system)} y "
            f"{_txt_q(hi, 'temperature', system)}: entre esas temperaturas el fluido se enfría "
            "al estrangularlo."
        )
    elif lo is not None:
        text += (
            f"A esta presión pasa por {_txt_q(lo, 'temperature', system)}: arriba de esa "
            "temperatura el fluido se enfría al estrangularlo (la rama alta queda por encima de "
            "la temperatura máxima de la ecuación de estado)."
        )
    elif hi is not None:
        text += (
            f"A esta presión pasa por {_txt_q(hi, 'temperature', system)}: abajo de esa "
            "temperatura el fluido se enfría al estrangularlo."
        )
    else:
        how = "se enfría" if r.mu > 0 else "se calienta"
        text += (
            "A esta presión no la corta dentro del rango de la ecuación de estado: el fluido "
            f"{how} al estrangularlo a cualquier temperatura de ese rango."
        )
    steps.append(ProcedureStep("La temperatura de inversión", text))
    return numbered(steps)


def _mayer_line(r: RelationsResult, f_tex: str, system: UnitSystem) -> str:
    """c_p − c_v = T·(v·α)·(α/κ_T), con el factor de p·v; partido si hay ×10ⁿ."""
    s = r.state
    T_n = n(s.T, _T, system)
    va = n(s.v * r.alpha, "volume_per_temperature", system)
    ak = n(r.alpha / r.kappa_T, "pressure_per_temperature", system)
    if r"\times" in va and r"\times" in ak:
        body = (
            rf"{_with_factor(f_tex, T_n)} \\ &\quad \cdot {latex_paren(va)} "
            rf"\\ &\quad \cdot {latex_paren(ak)}"
        )
    elif f_tex or r"\times" in va + ak:
        body = (
            rf"{_with_factor(f_tex, T_n)} \\ &\quad \cdot {latex_paren(va)} \cdot "
            rf"{latex_paren(ak)}"
        )
    else:
        body = _with_factor(f_tex, T_n, va, ak)
    return latex_chain(
        r"c_p - c_v",
        r"\frac{T\,v\,\alpha^2}{\kappa_T}",
        r"T\,(v\,\alpha)\,\frac{\alpha}{\kappa_T}",
        body,
        q(r.mayer, "specific_heat", system),
    )


def _txt_num(x: float) -> str:
    """Un número para el texto, con 4 cifras y sin notación exponencial hasta 10⁶."""
    if x != 0.0 and not (1e-3 <= abs(x) < 1e6):
        exp = math.floor(math.log10(abs(x)))
        sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
        return f"{_num_txt(x / 10**exp, '.3g')}·10{sup}"
    digits = max(0, 3 - math.floor(math.log10(abs(x)))) if x else 0
    text = f"{x:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "−")


def _txt_q(value_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    return f"{_txt_num(convert_from_si(value_si, kind, system))} {unit_label(kind, system)}"


# ---------------------------------------------------------------------
# Clapeyron
# ---------------------------------------------------------------------


def clapeyron_steps(r: ClapeyronResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de la ecuación de Clapeyron y de Clausius–Clapeyron (Çengel §12-3)."""
    fl = r.fluid
    _, f_tex = _pv(system)
    ppt: QuantityKind = "pressure_per_temperature"
    steps = [
        ProcedureStep(
            "La saturación a T",
            f"De la ecuación de estado del {fl.noun} (CoolProp), como de una tabla de saturación.",
            (
                _rows(
                    ("T", q(r.T, _T, system)),
                    (r"p_{sat}", q(r.p_sat, _P, system)),
                    (r"v_f", q(r.v_f, _V, system)),
                    (r"v_g", q(r.v_g, _V, system)),
                    (r"h_{fg}", q(r.h_fg, _E, system)),
                ),
            ),
        ),
        ProcedureStep(
            "La pendiente de la curva de saturación",
            f"Una diferencia centrada con p_sat a T ± ΔT (ΔT = "
            f"{_txt_q(r.inputs.dT_K, 'temperature_difference', system)}), como con las tablas.",
            (
                _rows(
                    (r"p_- = p_{sat}(T - \Delta T)", q(r.p_minus, _P, system)),
                    (r"p_+ = p_{sat}(T + \Delta T)", q(r.p_plus, _P, system)),
                ),
                latex_chain(
                    r"\left(\frac{dp}{dT}\right)_{sat}",
                    frac(r"p_+ - p_-", r"2\,\Delta T"),
                    frac(
                        n(r.p_plus - r.p_minus, _P, system),
                        times("2", n(r.inputs.dT_K, "temperature_difference", system)),
                    ),
                    q(r.dpdT_finite, ppt, system),
                    relation=r"\approx",
                ),
            ),
        ),
        ProcedureStep(
            "La ecuación de Clapeyron",
            "h_fg = T·v_fg·(dp/dT)_sat: sale de la relación de Maxwell (∂s/∂v)_T = (∂p/∂T)_v "
            "aplicada al cambio de fase, a T y p constantes (Çengel §12-3)." + _pv_text(system),
            (
                latex_chain(
                    r"v_{fg}",
                    r"v_g - v_f",
                    sub(n(r.v_g, _V, system), n(r.v_f, _V, system)),
                    q(r.v_fg, _V, system),
                ),
                latex_chain(
                    r"h_{fg}",
                    r"T\,v_{fg}\left(\frac{dp}{dT}\right)_{sat}",
                    (
                        rf"{_with_factor(f_tex, n(r.T, _T, system))} \\ &\quad \cdot "
                        + times(n(r.v_fg, _V, system), n(r.dpdT_finite, ppt, system))
                    )
                    if f_tex
                    else times(
                        n(r.T, _T, system), n(r.v_fg, _V, system), n(r.dpdT_finite, ppt, system)
                    ),
                    q(r.h_fg_clapeyron, _E, system),
                ),
            ),
        ),
        ProcedureStep(
            "Clausius–Clapeyron",
            "Si v_f ≪ v_g y el vapor es un gas ideal (v_g = R·T/p), Clapeyron queda "
            f"h_fg ≈ R·T²/p·(dp/dT)_sat: se equivoca {_pct_txt(r.h_fg_cc / r.h_fg - 1.0)} "
            "contra el h_fg real.",
            (
                latex_chain(
                    r"h_{fg}",
                    r"\frac{R\,T^2}{p_{sat}}\left(\frac{dp}{dT}\right)_{sat}",
                    frac(
                        times(n(fl.R, _S, system), _sq(n(r.T, _T, system))),
                        n(r.p_sat, _P, system),
                    )
                    + rf" \cdot {n(r.dpdT_finite, ppt, system)}",
                    q(r.h_fg_cc, _E, system),
                    relation=r"\approx",
                ),
            ),
        ),
    ]
    if r.T2 is not None and r.p_sat2 is not None and r.p_sat2_cc is not None:
        ratio = r.h_fg / fl.R * (1.0 / r.T - 1.0 / r.T2)
        steps.append(
            ProcedureStep(
                "Extrapolar la presión de saturación",
                "Integrando Clausius–Clapeyron con h_fg constante: ln(p₂/p₁) = (h_fg/R)·(1/T₁ − "
                f"1/T₂). La real a T₂ es {_txt_q(r.p_sat2, _P, system)}: el error es "
                f"{_pct_txt(r.p_sat2_cc / r.p_sat2 - 1.0)}.",
                (
                    latex_chain(
                        r"\frac{h_{fg}}{R}",
                        frac(n(r.h_fg, _E, system), n(fl.R, _S, system)),
                        q(r.h_fg / fl.R, _T, system),
                    ),
                    latex_chain(
                        r"\ln\frac{p_2}{p_1}",
                        rf"{n(r.h_fg / fl.R, _T, system)}\left(\frac{{1}}{{{n(r.T, _T, system)}}}"
                        rf" - \frac{{1}}{{{n(r.T2, _T, system)}}}\right)",
                        _r(ratio),
                    ),
                    latex_chain(
                        r"p_2",
                        times(n(r.p_sat, _P, system), rf"e^{{{_r(ratio)}}}"),
                        q(r.p_sat2_cc, _P, system),
                        relation=r"\approx",
                    ),
                ),
            )
        )
    return numbered(steps)

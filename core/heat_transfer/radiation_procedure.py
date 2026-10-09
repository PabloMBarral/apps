"""Procedimiento didáctico de la radiación (LaTeX), en el sistema de unidades activo.

Cengel y Ghajar (2015), §12-3 y §12-5 (cuerpo negro y superficies reales) y
§13-1 a §13-5 (factor de forma, superficies grises, pantallas, termocuplas).

Las fórmulas de la radiación usan la temperatura absoluta (K, o °R en el Inglés)
y λ en μm; σ, C₁, C₂ y C₃ se escriben en las unidades del sistema.
"""

from __future__ import annotations

import math

from core.heat_transfer.procedure_common import frac, n, numbered, q, sub, sum_rows, times
from core.heat_transfer.radiation import (
    C1,
    C2,
    C3,
    SIGMA,
    VIEW_FACTORS,
    BlackbodyResult,
    EnclosureResult,
    SurfaceBalanceResult,
    ThermocoupleResult,
    TwoSurfaceResult,
    ViewFactorResult,
    blackbody_fraction,
)
from core.latex import latex_chain, latex_is_wide, latex_number, latex_paren, latex_unit
from core.state_report import ProcedureStep
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "blackbody_steps",
    "constants_latex",
    "enclosure_steps",
    "surface_balance_steps",
    "thermocouple_steps",
    "two_surface_steps",
    "view_factor_steps",
]

_TA = "absolute_temperature"
_T = "temperature"
_E = "heat_flux"
_EL = "spectral_emissive_power"
_LAM = "wavelength"
_LT = "wavelength_temperature"
_Q = "heat_rate"
_H = "heat_transfer_coefficient"


def _x(v: float, sig: int = 4) -> str:
    return latex_number(v, sig)


def _sigma(system: UnitSystem) -> float:
    return convert_from_si(SIGMA, _E, system) / convert_from_si(1.0, _TA, system) ** 4


def _sigma_q(system: UnitSystem) -> str:
    """σ con su unidad: W/(m²·K⁴) o Btu/(h·ft²·°R⁴)."""
    if system == "Inglés":
        unit = r"\mathrm{Btu/(h\cdot ft^{2}\cdot {}^{\circ}R^{4})}"
    else:
        unit = r"\mathrm{W/(m^{2}\cdot K^{4})}"
    return rf"{_x(_sigma(system))}\ {unit}"


def _c1(system: UnitSystem) -> float:
    """C₁ en (unidad de flujo)·μm⁴: W·μm⁴/m² o Btu·μm⁴/(h·ft²)."""
    return convert_from_si(C1 * 1e24, _E, system)


def _c2(system: UnitSystem) -> float:
    return convert_from_si(C2, _LT, system)


def _c3(system: UnitSystem) -> float:
    return convert_from_si(C3, _LT, system)


def constants_latex(system: UnitSystem) -> tuple[str, ...]:
    """σ, C₁, C₂ y C₃ en las unidades del sistema (CODATA 2018)."""
    if system == "Inglés":
        c1u = r"\mathrm{Btu\cdot \mu m^{4}/(h\cdot ft^{2})}"
    else:
        c1u = r"\mathrm{W\cdot \mu m^{4}/m^{2}}"
    lt = latex_unit(unit_label(_LT, system))
    return (
        rf"\sigma = {_sigma_q(system)}",
        rf"C_1 = {_x(_c1(system))}\ {c1u}",
        rf"C_2 = {_x(_c2(system), 5)}\ {lt}",
        rf"C_3 = {_x(_c3(system), 5)}\ {lt}",
    )


def _flush_chain(lhs: str, *steps: str) -> str:
    r"""``lhs = paso₁ = paso₂…`` con todos los renglones contra el margen izquierdo.

    En un ``aligned`` el ancho del lado izquierdo se suma al del renglón más ancho:
    con un lado izquierdo largo (E_{bλ,máx}, una ecuación de radiosidades) los
    renglones van sin alinear el signo.
    """
    rows = [rf"&{lhs} = {steps[0]}", *(rf"&= {step}" for step in steps[1:])]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _abs_line(symbol: str, T_K: float, system: UnitSystem) -> str | None:
    """La conversión a temperatura absoluta (en el Técnico y el Inglés)."""
    if system == "SI":
        return None
    offset = "273.15" if system == "Técnico" else "459.67"
    return latex_chain(
        symbol,
        rf"{n(T_K, _T, system)} + {offset}",
        q(T_K, _TA, system),
    )


# ---------------------------------------------------------------------
# Cuerpo negro
# ---------------------------------------------------------------------


def _fraction_text() -> str:
    return (
        "f(λT) es la fracción de la emisión del cuerpo negro por debajo de λ: depende solo del "
        "producto λT (tabla 12-2 de Cengel y Ghajar; acá con la serie de Chang y Rhee, 1984)."
    )


def blackbody_steps(result: BlackbodyResult, system: UnitSystem) -> list[ProcedureStep]:
    """Stefan–Boltzmann, Wien, Planck, las fracciones y las bandas de ε(λ)."""
    i = result.inputs
    T = n(i.T_K, _TA, system)
    sigma = _x(_sigma(system))
    steps = [
        ProcedureStep(
            "Constantes de la radiación",
            "CODATA 2018 (Tiesinga et al., 2021), en las unidades del sistema elegido.",
            constants_latex(system),
        ),
        ProcedureStep(
            "Stefan–Boltzmann y Wien",
            "La potencia total que emite el cuerpo negro va con T⁴; el máximo de la curva de "
            "Planck se corre a λ más cortas al calentarlo (Cengel y Ghajar, §12-3).",
            (
                latex_chain(
                    "E_b",
                    r"\sigma\,T^4",
                    rf"{sigma} \cdot {T}^4",
                    q(result.E_b_W_per_m2, _E, system),
                ),
                latex_chain(
                    r"\lambda_{\text{máx}}",
                    frac("C_3", "T"),
                    frac(_x(_c3(system), 5), T),
                    q(result.lambda_max_m, _LAM, system),
                ),
                _flush_chain(
                    r"E_{b\lambda,\text{máx}}",
                    frac("C_1", r"\lambda_{\text{máx}}^5\,(e^{C_2/C_3} - 1)"),
                    frac(
                        _x(_c1(system)),
                        rf"{n(result.lambda_max_m, _LAM, system)}^5\,(e^{{{_x(C2 / C3)}}} - 1)",
                    ),
                    q(result.E_blambda_max_W_per_m3, _EL, system),
                ),
            ),
        ),
    ]
    if i.lambda_point_m is not None:
        lam = n(i.lambda_point_m, _LAM, system)
        x = C2 / (i.lambda_point_m * i.T_K)
        steps.append(
            ProcedureStep(
                "Planck",
                "El poder emisivo espectral a esa longitud de onda.",
                (
                    r"E_{b\lambda} = \frac{C_1}{\lambda^5\left[\exp\left(\frac{C_2}{\lambda T}"
                    r"\right) - 1\right]}",
                    latex_chain(
                        frac("C_2", r"\lambda T"),
                        frac(_x(_c2(system), 5), times(lam, T)),
                        _x(x),
                    ),
                    latex_chain(
                        r"E_{b\lambda}",
                        frac(_x(_c1(system)), rf"{lam}^5\,(e^{{{_x(x)}}} - 1)"),
                        q(result.E_blambda_point_W_per_m3 or 0.0, _EL, system),
                    ),
                ),
            )
        )
    if result.band_fraction is not None:
        lo = i.band_lower_m or 0.0
        hi = i.band_upper_m
        lines = []
        f_lo = blackbody_fraction(lo * i.T_K)
        f_hi = 1.0 if hi is None else blackbody_fraction(hi * i.T_K)
        for sym, lam_m, f in (("1", lo, f_lo), ("2", hi, f_hi)):
            if lam_m is None:
                lines.append(rf"f_{{{sym}}} = 1\ (\lambda \to \infty)")
                continue
            lines.append(
                latex_chain(
                    rf"\lambda_{sym} T",
                    times(n(lam_m, _LAM, system), T),
                    q(lam_m * i.T_K, _LT, system),
                )
            )
            lines.append(rf"f_{{{sym}}} = f(\lambda_{sym} T) = {_x(f, 5)}")
        lines += [
            latex_chain(
                r"\Delta f",
                r"f_2 - f_1",
                sub(_x(f_hi, 5), _x(f_lo, 5)),
                _x(result.band_fraction, 5),
            ),
            latex_chain(
                r"E_{b,\text{banda}}",
                r"\Delta f\,E_b",
                times(_x(result.band_fraction, 5), n(result.E_b_W_per_m2, _E, system)),
                q(result.band_power_W_per_m2 or 0.0, _E, system),
            ),
        ]
        steps.append(ProcedureStep("Fracción en la banda", _fraction_text(), tuple(lines)))
    if result.lambda_target_m is not None and i.fraction_target is not None:
        lamT = result.lambda_target_m * i.T_K
        steps.append(
            ProcedureStep(
                "Dónde se emite esa fracción",
                "Al revés: el λT de la tabla para esa fracción, dividido por T.",
                (
                    rf"f(\lambda T) = {_x(i.fraction_target, 4)} \Rightarrow "
                    rf"\lambda T = {q(lamT, _LT, system)}",
                    latex_chain(
                        r"\lambda",
                        frac(n(lamT, _LT, system), T),
                        q(result.lambda_target_m, _LAM, system),
                    ),
                ),
            )
        )
    if result.rows:
        steps.append(_bands_step(result, system, source=False))
    if result.source_rows:
        steps.append(_bands_step(result, system, source=True))
    return numbered(steps)


def _bands_step(result: BlackbodyResult, system: UnitSystem, source: bool) -> ProcedureStep:
    i = result.inputs
    T_K = i.T_source_K if source else i.T_K
    assert T_K is not None
    rows = result.source_rows if source else result.rows
    lines: list[str] = []
    for k, row in enumerate(rows, 1):
        if row.upper_m is not None:
            lines.append(
                latex_chain(
                    rf"f(\lambda_{k} T)",
                    rf"f({q(row.upper_m * T_K, _LT, system)})",
                    _x(row.f_upper, 5),
                )
            )
    sym = r"\alpha" if source else r"\varepsilon"
    for k, row in enumerate(rows, 1):
        lines.append(
            latex_chain(
                rf"{sym}_{k}\,\Delta f_{k}",
                times(_x(row.emissivity, 3), _x(row.fraction, 5)),
                _x(row.contribution, 5),
            )
        )
    total = sum(row.contribution for row in rows)
    lines.append(rf"{sym} = \sum_i {sym}_i\,\Delta f_i = {_x(total, 4)}")
    if not source:
        lines.append(
            latex_chain(
                "E",
                r"\varepsilon\,E_b",
                times(_x(total, 4), n(result.E_b_W_per_m2, _E, system)),
                q(result.E_W_per_m2, _E, system),
            )
        )
        title = "Emisividad total"
        text = (
            "Cada banda aporta su ε por la fracción de la emisión del cuerpo negro que cae en ella "
            "(Δf_k = f_k − f_{k−1}, con f_0 = 0 y f = 1 en la última) "
            f"a la temperatura de la superficie ({_x(T_K, 4)} {unit_label(_TA, 'SI')})."
        )
    else:
        title = "Absortividad para la radiación de la fuente"
        text = (
            "Con una superficie difusa, α(λ) = ε(λ) (Kirchhoff espectral): la absortividad total "
            "pesa las bandas con la emisión de la fuente, a su temperatura "
            f"({_x(T_K, 4)} K). Por eso α ≠ ε si la fuente está a otra temperatura."
        )
    return ProcedureStep(title, text, tuple(lines))


# ---------------------------------------------------------------------
# Factores de forma
# ---------------------------------------------------------------------


def view_factor_steps(result: ViewFactorResult, system: UnitSystem) -> list[ProcedureStep]:
    """La fórmula de la geometría con los números, la reciprocidad y la suma."""
    i = result.inputs
    vf = VIEW_FACTORS[i.geometry]
    L = "length"
    dims = (i.a, i.b, i.c)[: len(vf.dims)]
    data = tuple(rf"{d} = {q(v, L, system)}" for d, v in zip(vf.dims, dims, strict=True))
    lines = list(_view_factor_lines(result))
    steps = [
        ProcedureStep(
            "Geometría",
            f"{vf.name} ({vf.source})."
            + (" Superficies muy largas: las áreas son por unidad de largo." if vf.two_d else ""),
            data,
        ),
        ProcedureStep("Factor de forma", "La fórmula cerrada con los números.", tuple(lines)),
        ProcedureStep(
            "Reciprocidad y regla de la suma",
            "A_i·F_ij = A_j·F_ji; lo que sale de i y no llega a j va al resto del recinto.",
            (
                latex_chain(
                    r"F_{ji}",
                    frac(r"A_i\,F_{ij}", "A_j"),
                    frac(
                        times(n(result.A_i, _area_kind(vf.two_d), system), _x(result.F_ij, 5)),
                        n(result.A_j, _area_kind(vf.two_d), system),
                    ),
                    _x(result.F_ji, 5),
                ),
                latex_chain(
                    r"F_{i,\text{resto}}", rf"1 - {_x(result.F_ij, 5)}", _x(result.F_i_rest, 5)
                ),
            ),
        ),
    ]
    return numbered(steps)


def _area_kind(two_d: bool) -> str:
    return "length" if two_d else "area"


def _view_factor_lines(result: ViewFactorResult) -> tuple[str, ...]:
    i = result.inputs
    g = i.geometry
    F = _x(result.F_ij, 5)
    if g == "parallel_rectangles":
        X, Y = i.a / i.c, i.b / i.c
        t1 = 0.5 * math.log((1 + X * X) * (1 + Y * Y) / (1 + X * X + Y * Y))
        t2 = X * math.sqrt(1 + Y * Y) * math.atan(X / math.sqrt(1 + Y * Y))
        t3 = Y * math.sqrt(1 + X * X) * math.atan(Y / math.sqrt(1 + X * X))
        t4 = X * math.atan(X) + Y * math.atan(Y)
        return (
            rf"\bar{{X}} = X/L = {_x(X)}",
            rf"\bar{{Y}} = Y/L = {_x(Y)}",
            latex_chain(
                "t_1",
                r"\ln\left[\frac{(1+\bar{X}^2)(1+\bar{Y}^2)}{1+\bar{X}^2+\bar{Y}^2}\right]^{1/2}",
                _x(t1),
            ),
            latex_chain(
                "t_2",
                r"\bar{X}\sqrt{1+\bar{Y}^2}\,\tan^{-1}\frac{\bar{X}}{\sqrt{1+\bar{Y}^2}}",
                _x(t2),
            ),
            latex_chain(
                "t_3",
                r"\bar{Y}\sqrt{1+\bar{X}^2}\,\tan^{-1}\frac{\bar{Y}}{\sqrt{1+\bar{X}^2}}",
                _x(t3),
            ),
            latex_chain("t_4", r"\bar{X}\tan^{-1}\bar{X} + \bar{Y}\tan^{-1}\bar{Y}", _x(t4)),
            latex_chain(
                r"F_{ij}",
                r"\frac{2}{\pi \bar{X} \bar{Y}}\,(t_1 + t_2 + t_3 - t_4)",
                F,
            ),
        )
    if g == "coaxial_disks":
        r = result.ratios
        return (
            rf"R_i = r_i/L = {_x(r['R_i'])}",
            rf"R_j = r_j/L = {_x(r['R_j'])}",
            latex_chain("S", r"1 + \frac{1 + R_j^2}{R_i^2}", _x(r["S"])),
            latex_chain(
                r"F_{ij}",
                r"\frac{1}{2}\left\{S - \left[S^2 - 4\left(\frac{r_j}{r_i}\right)^2\right]"
                r"^{1/2}\right\}",
                F,
            ),
        )
    if g == "perpendicular_rectangles":
        H, W = i.c / i.a, i.b / i.a
        s = math.sqrt(H * H + W * W)
        a = (1 + W * W) * (1 + H * H) / (1 + W * W + H * H)
        b = W * W * (1 + W * W + H * H) / ((1 + W * W) * (W * W + H * H))
        c = H * H * (1 + H * H + W * W) / ((1 + H * H) * (H * H + W * W))
        u = W * math.atan(1 / W) + H * math.atan(1 / H) - s * math.atan(1 / s)
        v = 0.25 * (math.log(a) + W * W * math.log(b) + H * H * math.log(c))
        return (
            rf"H = Z/X = {_x(H)}",
            rf"W = Y/X = {_x(W)}",
            latex_chain(
                "u",
                r"W\tan^{-1}\frac{1}{W} + H\tan^{-1}\frac{1}{H} \\ &\quad - \sqrt{H^2+W^2}\,"
                r"\tan^{-1}\frac{1}{\sqrt{H^2+W^2}}",
                _x(u),
            ),
            latex_chain(
                "v",
                r"\frac{1}{4}\ln\left\{a\,b^{W^2} c^{H^2}\right\}",
                _x(v),
            ),
            latex_chain("a", frac("(1 + W^2)(1 + H^2)", "1 + W^2 + H^2"), _x(a)),
            latex_chain("b", frac("W^2\\,(1 + W^2 + H^2)", "(1 + W^2)(W^2 + H^2)"), _x(b)),
            latex_chain("c", frac("H^2\\,(1 + H^2 + W^2)", "(1 + H^2)(H^2 + W^2)"), _x(c)),
            latex_chain(r"F_{ij}", frac("u + v", r"\pi\,W"), F),
        )
    if g == "parallel_plates_2d":
        Wi, Wj = i.a / i.c, i.b / i.c
        a = math.sqrt((Wi + Wj) ** 2 + 4.0)
        b = math.sqrt((Wj - Wi) ** 2 + 4.0)
        return (
            rf"W_i = w_i/L = {_x(Wi)}",
            rf"W_j = w_j/L = {_x(Wj)}",
            latex_chain(
                "a",
                r"\sqrt{(W_i + W_j)^2 + 4}",
                rf"\sqrt{{({_x(Wi)} + {_x(Wj)})^2 + 4}}",
                _x(a),
            ),
            latex_chain(
                "b",
                r"\sqrt{(W_j - W_i)^2 + 4}",
                rf"\sqrt{{({sub(_x(Wj), _x(Wi))})^2 + 4}}",
                _x(b),
            ),
            latex_chain(
                r"F_{ij}",
                frac("a - b", "2\\,W_i"),
                frac(sub(_x(a), _x(b)), rf"2 \cdot {_x(Wi)}"),
                F,
            ),
        )
    if g == "perpendicular_plates_2d":
        r = i.b / i.a
        return (
            latex_chain(
                r"F_{ij}",
                frac(rf"1 + {_x(r)} - \sqrt{{1 + {_x(r)}^2}}", "2"),
                F,
            ),
        )
    if g == "three_sided_2d":
        return (
            latex_chain(
                r"F_{ij}",
                frac("w_i + w_j - w_k", "2 w_i"),
                frac(f"{_x(i.a)} + {_x(i.b)} - {_x(i.c)}", rf"2 \cdot {_x(i.a)}"),
                F,
            ),
        )
    if g == "parallel_cylinders_2d":
        X = result.ratios["X"]
        root, angle = math.sqrt(X * X - 1.0), math.asin(1.0 / X)
        return (
            latex_chain("X", r"1 + \frac{s}{2r}", _x(X)),
            latex_chain(r"\sqrt{X^2 - 1}", rf"\sqrt{{{_x(X)}^2 - 1}}", _x(root)),
            latex_chain(r"\sin^{-1}\frac{1}{X}", rf"\sin^{{-1}}\frac{{1}}{{{_x(X)}}}", _x(angle)),
            latex_chain(
                r"F_{ij}",
                r"\frac{1}{\pi}\left[\sqrt{X^2 - 1} + \sin^{-1}\frac{1}{X} - X\right]",
                rf"\frac{{1}}{{\pi}}\,({_x(root)} + {_x(angle)} - {_x(X)})",
                F,
            ),
        )
    d = result.ratios["D/s"]
    return (
        latex_chain(r"\frac{D}{s}", _x(d)),
        latex_chain(
            r"F_{ij}",
            rf"1 - \sqrt{{1 - {_x(d)}^2}} \\ &\quad + {_x(d)}\,\tan^{{-1}}"
            rf"\sqrt{{\frac{{1 - {_x(d)}^2}}{{{_x(d)}^2}}}}",
            F,
        ),
    )


# ---------------------------------------------------------------------
# Dos superficies y pantallas
# ---------------------------------------------------------------------


def _inv_area(value_si: float, system: UnitSystem) -> str:
    """Una resistencia de radiación (1/m² o 1/ft²) con su unidad."""
    if value_si == 0.0:
        return "0"
    per_area = 1.0 / convert_from_si(1.0 / value_si, "area", system)
    unit = r"\mathrm{ft^{-2}}" if system == "Inglés" else r"\mathrm{m^{-2}}"
    return rf"{_x(per_area)}\ {unit}"


def _inv_area_n(value_si: float, system: UnitSystem) -> str:
    if value_si == 0.0:
        return "0"
    return _x(1.0 / convert_from_si(1.0 / value_si, "area", system))


def two_surface_steps(result: TwoSurfaceResult, system: UnitSystem) -> list[ProcedureStep]:
    """La red de resistencias (Cengel y Ghajar, §13-4 y §13-5)."""
    i = result.inputs
    T1, T2 = n(i.T1_K, _TA, system), n(i.T2_K, _TA, system)
    sigma = _x(_sigma(system))
    Eb1, Eb2 = SIGMA * i.T1_K**4, SIGMA * i.T2_K**4
    steps = [
        ProcedureStep(
            "Poderes emisivos",
            "E_b = σ·T⁴ de cada superficie, con la temperatura absoluta.",
            (
                latex_chain("E_{b1}", rf"{sigma} \cdot {T1}^4", q(Eb1, _E, system)),
                latex_chain("E_{b2}", rf"{sigma} \cdot {T2}^4", q(Eb2, _E, system)),
            ),
        )
    ]
    if i.geometry == "parallel_plates":
        terms = [r"\frac{1}{\varepsilon_1} + \frac{1}{\varepsilon_2} - 1"]
        nums = [rf"\frac{{1}}{{{_x(i.eps1, 3)}}} + \frac{{1}}{{{_x(i.eps2, 3)}}} - 1"]
        for k, s in enumerate(i.shields, 1):
            terms.append(
                rf"\frac{{1}}{{\varepsilon_{{{k},1}}}} + \frac{{1}}{{\varepsilon_{{{k},2}}}} - 1"
            )
            nums.append(rf"\frac{{1}}{{{_x(s.eps_1, 3)}}} + \frac{{1}}{{{_x(s.eps_2, 3)}}} - 1")
        denom = sum(r.R_per_m2 for r in result.resistances) * i.A1_m2
        q_flux = result.Q_W / i.A1_m2
        lines = [
            latex_chain(
                r"\Sigma",
                r" \\ &\quad + ".join(terms) if len(terms) > 1 else terms[0],
                r" \\ &\quad + ".join(nums) if len(nums) > 1 else nums[0],
                _x(denom),
            ),
            latex_chain(
                r"\frac{\dot{Q}}{A}",
                frac(r"E_{b1} - E_{b2}", r"\Sigma"),
                frac(sub(n(Eb1, _E, system), n(Eb2, _E, system)), _x(denom)),
                q(q_flux, _E, system),
            ),
            latex_chain(
                r"\dot{Q}",
                r"\frac{\dot{Q}}{A}\,A",
                times(n(q_flux, _E, system), n(i.A1_m2, "area", system)),
                q(result.Q_W, _Q, system),
            ),
        ]
        text = (
            "Placas paralelas infinitas: con A₁ = A₂ = A y F₁₂ = 1 la red da "
            "q = σ(T₁⁴ − T₂⁴)/(1/ε₁ + 1/ε₂ − 1)"
            + ("; cada pantalla suma 1/ε de cada cara − 1." if i.shields else ".")
        )
        steps.append(ProcedureStep("Calor por unidad de área", text, tuple(lines)))
    else:
        lines = []
        for r in result.resistances:
            lines.append(rf"R_{{\text{{{r.label}}}}} = {_inv_area(r.R_per_m2, system)}")
        R_tot = sum(r.R_per_m2 for r in result.resistances)
        lines += [
            latex_chain(
                r"\Sigma R",
                sum_rows([_inv_area_n(r.R_per_m2, system) for r in result.resistances], 2),
                _inv_area(R_tot, system),
            ),
            latex_chain(
                r"\dot{Q}",
                frac(r"E_{b1} - E_{b2}", r"\Sigma R"),
                frac(sub(n(Eb1, _E, system), n(Eb2, _E, system)), _inv_area_n(R_tot, system)),
                q(result.Q_W, _Q, system),
            ),
        ]
        steps.append(
            ProcedureStep(
                "Red de resistencias",
                "Superficie: (1 − ε)/(A·ε); espacio: 1/(A_i·F_ij), con F = 1 de la de adentro a "
                "la de afuera. Todas en serie.",
                tuple(lines),
            )
        )
    if i.shields:
        lines = []
        acc = 0.0
        resistances = list(result.resistances)
        for k in range(len(i.shields)):
            idx = 1 + 3 * k + 2  # hasta la cara 1 de la pantalla k (incluida)
            acc = sum(r.R_per_m2 for r in resistances[:idx])
            Ebk = Eb1 - result.Q_W * acc
            lines += [
                latex_chain(
                    rf"E_{{b,p{k + 1}}}",
                    rf"E_{{b1}} - \dot{{Q}}\,\Sigma R_{{1\to p{k + 1}}}",
                    q(Ebk, _E, system),
                ),
                latex_chain(
                    rf"T_{{p{k + 1}}}",
                    rf"\left(E_{{b,p{k + 1}}}/\sigma\right)^{{1/4}}",
                    q(result.shield_T_K[k], _TA, system),
                ),
            ]
        lines.append(
            latex_chain(
                r"\frac{\dot{Q}}{\dot{Q}_{\text{sin}}}",
                frac(n(result.Q_W, _Q, system), n(result.Q_no_shields_W, _Q, system)),
                _x(result.reduction),
            )
        )
        steps.append(
            ProcedureStep(
                "Pantallas",
                "Cada pantalla queda a la temperatura de su nodo en la red; sin pantallas el "
                "calor sería Q̇_sin.",
                tuple(lines),
            )
        )
    return numbered(steps)


# ---------------------------------------------------------------------
# Recinto
# ---------------------------------------------------------------------


def _matrix(F: tuple[tuple[float, ...], ...]) -> str:
    rows = [" & ".join(_x(f, 3) for f in row) for row in F]
    return r"F = \begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def enclosure_steps(result: EnclosureResult, system: UnitSystem) -> list[ProcedureStep]:
    """El método de las radiosidades (Cengel y Ghajar, §13-4)."""
    S = result.inputs.surfaces
    F = result.inputs.F
    nS = len(S)
    sigma = _x(_sigma(system))
    data = [_matrix(F)]
    for k, s in enumerate(S, 1):
        data.append(rf"A_{k} = {q(s.area_m2, 'area', system)}")
    steps = [
        ProcedureStep(
            "Superficies y factores de forma",
            "Las filas de F suman 1 (regla de la suma) y A_i·F_ij = A_j·F_ji (reciprocidad). "
            "Superficies: " + "; ".join(f"{k}: {s.name}" for k, s in enumerate(S, 1)) + ".",
            tuple(data),
        )
    ]
    eb_lines = []
    for k, s in enumerate(S, 1):
        if s.T_K is not None:
            eb_lines.append(
                latex_chain(
                    f"E_{{b{k}}}",
                    rf"{sigma} \cdot {n(s.T_K, _TA, system)}^4",
                    q(SIGMA * s.T_K**4, _E, system),
                )
            )
    steps.append(
        ProcedureStep("Poderes emisivos", "De las superficies a temperatura dada.", tuple(eb_lines))
    )
    eq_lines: list[str] = []
    for k, s in enumerate(S, 1):
        parts = [
            rf"{_x(F[k - 1][j], 4)}\,(J_{k} - J_{j + 1})"
            for j in range(nS)
            if j != k - 1 and F[k - 1][j] > 0.0
        ]
        if s.T_K is not None and s.emissivity >= 1.0:
            eq_lines.append(rf"J_{k} = E_{{b{k}}} = {q(SIGMA * s.T_K**4, _E, system)}")
            continue
        if s.T_K is not None:
            k_eps = s.emissivity / (1.0 - s.emissivity)
            lhs = rf"{_x(k_eps)}\,({n(SIGMA * s.T_K**4, _E, system)} - J_{k})"
        else:
            lhs = n((s.Q_W or 0.0) / s.area_m2, _E, system)
        eq_lines.append(rf"\begin{{aligned}}&{lhs} \\ &= {sum_rows(parts, 1)}\end{{aligned}}")
    J_lines = [rf"J_{k} = {q(J, _E, system)}" for k, J in enumerate(result.J_W_per_m2, 1)]
    steps.append(
        ProcedureStep(
            "Ecuaciones de las radiosidades",
            "Con T dada: ε/(1 − ε)·(E_b − J) = Σ F_ij·(J_i − J_j) (negra: J = E_b). Con el calor "
            "dado: Q̇/A = Σ F_ij·(J_i − J_j) (rerradiante: Q̇ = 0). Resolviendo el sistema:",
            tuple(eq_lines + J_lines),
        )
    )
    q_lines = []
    J = result.J_W_per_m2
    for k, s in enumerate(S, 1):
        parts = []
        for j in range(nS):
            if j == k - 1 or F[k - 1][j] <= 0.0:
                continue
            Jk, Jj = n(J[k - 1], _E, system), n(J[j], _E, system)
            if latex_is_wide(Jk, Jj):
                dJ = latex_paren(n(J[k - 1] - J[j], _E, system))
                parts.append(rf"{_x(F[k - 1][j], 4)} \cdot {dJ}")
            else:
                parts.append(rf"{_x(F[k - 1][j], 4)}\,({Jk} - {latex_paren(Jj)})")
        q_lines.append(
            latex_chain(
                rf"\dot{{Q}}_{k}",
                rf"A_{k} \sum_j F_{{{k}j}}\,(J_{k} - J_j)",
                rf"{n(s.area_m2, 'area', system)} \\ &\quad \cdot [{sum_rows(parts, 1)}]",
                q(result.Q_W[k - 1], _Q, system),
            )
        )
    T_lines = []
    for k, s in enumerate(S, 1):
        if s.T_K is not None:
            continue
        if s.reradiating or s.emissivity >= 1.0:
            T_lines.append(
                latex_chain(
                    f"T_{k}",
                    rf"(J_{k}/\sigma)^{{1/4}}",
                    q(result.T_K[k - 1], _TA, system),
                )
            )
        else:
            T_lines.append(
                latex_chain(
                    f"E_{{b{k}}}",
                    rf"J_{k} + \dot{{Q}}_{k}\,"
                    rf"\frac{{1 - \varepsilon_{k}}}{{A_{k}\,\varepsilon_{k}}}",
                    q(SIGMA * result.T_K[k - 1] ** 4, _E, system),
                )
            )
            T_lines.append(
                latex_chain(
                    f"T_{k}",
                    rf"(E_{{b{k}}}/\sigma)^{{1/4}}",
                    q(result.T_K[k - 1], _TA, system),
                )
            )
    total = sum(result.Q_W)
    q_lines.append(rf"\sum_i \dot{{Q}}_i = {_x(total, 3) if abs(total) > 1e-9 else '0'}")
    steps.append(
        ProcedureStep(
            "Calores y temperaturas",
            "Q̇_i = A_i·Σ F_ij·(J_i − J_j) sale de cada superficie (si es negativo, la "
            "superficie recibe calor). La suma es cero: el recinto está cerrado.",
            tuple(q_lines + T_lines),
        )
    )
    return numbered(steps)


# ---------------------------------------------------------------------
# Radiación y convección juntas
# ---------------------------------------------------------------------


def surface_balance_steps(result: SurfaceBalanceResult, system: UnitSystem) -> list[ProcedureStep]:
    """Convección + radiación (+ sol) de una superficie (Incropera, §13.4)."""
    i = result.inputs
    sigma = _x(_sigma(system))
    abs_lines = [
        line
        for line in (
            _abs_line("T_s", result.T_s_K, system),
            _abs_line(r"T_{\text{alr}}", i.T_surr_K, system),
        )
        if line
    ]
    A = n(i.area_m2, "area", system)
    h, Ts_, Tinf = (
        n(i.h_W_per_m2K, _H, system),
        n(result.T_s_K, _T, system),
        n(i.T_inf_K, _T, system),
    )
    diff = f"({sub(Ts_, Tinf)})"
    if latex_is_wide(h, A, Ts_, Tinf) or len(h + A + Ts_ + Tinf) > 18:
        conv = rf"{times(h, A)} \\ &\quad \cdot {diff}"
    else:
        conv = times(h, A, diff)
    lines = [
        latex_chain(
            r"\dot{Q}_{\text{conv}}",
            r"h\,A\,(T_s - T_\infty)",
            conv,
            q(result.Q_conv_W, _Q, system),
        ),
        latex_chain(
            r"\dot{Q}_{\text{rad}}",
            r"\varepsilon\,\sigma\,A\,(T_s^4 - T_{\text{alr}}^4)",
            rf"{_x(i.emissivity, 3)} \cdot {sigma} \cdot {A} \\ &\quad \cdot "
            rf"({n(result.T_s_K, _TA, system)}^4 - {n(i.T_surr_K, _TA, system)}^4)",
            q(result.Q_rad_W, _Q, system),
        ),
    ]
    if result.Q_solar_W > 0.0:
        lines.append(
            latex_chain(
                r"\dot{Q}_{\text{sol}}",
                r"\alpha_s\,G\,A",
                times(_x(i.alpha_solar, 3), n(i.G_solar_W_per_m2, _E, system), A),
                q(result.Q_solar_W, _Q, system),
            )
        )
    lines.append(
        latex_chain(
            r"\dot{Q}_{\text{neto}}",
            r"\dot{Q}_{\text{conv}} + \dot{Q}_{\text{rad}} - \dot{Q}_{\text{sol}}",
            q(result.Q_lost_W, _Q, system),
        )
    )
    Ts, Tr = n(result.T_s_K, _TA, system), n(i.T_surr_K, _TA, system)
    h_lines = [
        latex_chain(
            r"h_{\text{rad}}",
            r"\varepsilon\,\sigma\,(T_s + T_{\text{alr}})\,(T_s^2 + T_{\text{alr}}^2)",
            rf"{_x(i.emissivity, 3)} \cdot {sigma} \\ &\quad \cdot ({Ts} + {Tr}) "
            rf"\\ &\quad \cdot ({Ts}^2 + {Tr}^2)",
            q(result.h_rad_W_per_m2K, _H, system),
        )
    ]
    text = (
        "La radiación con los alrededores se suma a la convección con el aire; con T absolutas. "
        "Q̇_neto > 0 es lo que pierde la superficie."
    )
    if i.Q_in_W is not None:
        text += (
            " Con el calor dado, T_s es la que hace Q̇_neto = Q̇ entregado: una ecuación con T⁴ "
            "que se resuelve numéricamente."
        )
    steps = [ProcedureStep("Calores", text, tuple(abs_lines + lines))]
    steps.append(
        ProcedureStep(
            "Coeficiente de radiación",
            "h_rad escribe la radiación como una convección: Q̇_rad = h_rad·A·(T_s − T_alr).",
            tuple(h_lines),
        )
    )
    return numbered(steps)


def thermocouple_steps(result: ThermocoupleResult, system: UnitSystem) -> list[ProcedureStep]:
    """El balance de la junta: h·(T_f − T_th) = ε·σ·(T_th⁴ − T_w⁴) (Cengel y Ghajar, §13-5)."""
    i = result.inputs
    sigma = _x(_sigma(system))
    Tth, Tw = n(i.T_reading_K, _TA, system), n(i.T_wall_K, _TA, system)
    abs_lines = [
        line
        for line in (
            _abs_line(r"T_{\text{th}}", i.T_reading_K, system),
            _abs_line("T_w", i.T_wall_K, system),
        )
        if line
    ]
    lines = [
        latex_chain(
            r"q_{\text{rad}}",
            r"\varepsilon\,\sigma\,(T_{\text{th}}^4 - T_w^4)",
            rf"{_x(i.emissivity, 3)} \cdot {sigma} \\ &\quad \cdot ({Tth}^4 - {Tw}^4)",
            q(result.q_rad_W_per_m2, _E, system),
        ),
        latex_chain(
            "T_f",
            r"T_{\text{th}} + \frac{q_{\text{rad}}}{h}",
            rf"{n(i.T_reading_K, _T, system)} + "
            + frac(n(result.q_rad_W_per_m2, _E, system), n(i.h_W_per_m2K, _H, system)),
            q(result.T_gas_K, _T, system),
        ),
    ]
    return numbered(
        [
            ProcedureStep(
                "Balance de la junta",
                "En régimen la junta gana por convección lo que pierde por radiación con las "
                "paredes: h·(T_f − T_th) = ε·σ·(T_th⁴ − T_w⁴).",
                tuple(abs_lines + lines),
            )
        ]
    )

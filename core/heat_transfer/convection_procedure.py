"""Procedimiento didáctico de la convección (LaTeX), en el sistema de unidades activo.

Cengel y Ghajar (2015), caps. 7 a 9; Incropera et al. (2007), caps. 7 a 9:

- :func:`convection_steps`: forzada externa y natural (propiedades a la
  temperatura de película, Re o Gr y Ra, Nu con la correlación, h y Q̇);
- :func:`internal_steps`: un tubo (propiedades a la temperatura media, Re, Nu,
  h, la temperatura de salida con NTU, Q̇, ΔT_ml y la caída de presión);
- :func:`correlation_latex`: la sustitución de cada correlación, partida en
  renglones para que entre en el ancho de un celular.

Los números adimensionales entre 10³ y 10⁷ se escriben enteros con espacio fino
(42 180), como las entalpías molares de la Fase 5.
"""

from __future__ import annotations

import math

from core.fluids import fluid_with_article
from core.heat_transfer.convection import (
    _HILPERT,
    _MORGAN,
    CORRELATIONS,
    G_STANDARD,
    RE_CRITICAL_PLATE,
    RE_LAMINAR_TUBE,
    ConvectionResult,
    ExternalFlowInputs,
    InternalFlowResult,
    NaturalConvectionInputs,
    dimensionless_groups,
)
from core.heat_transfer.procedure_common import frac, n, numbered, q, sub, times
from core.latex import latex_chain, latex_number
from core.state_report import ProcedureStep
from core.units_system import UnitSystem

__all__ = ["big", "convection_steps", "correlation_latex", "internal_steps"]

_T = "temperature"
_L = "length"
_K = "thermal_conductivity"
_H = "heat_transfer_coefficient"
_W = "heat_rate"
_DT = "temperature_difference"
#: g_c del sistema inglés: 32,174 lbm·ft/(lbf·s²) (Δp en lbf/ft²).
_G_C = 32.17404855643


def big(x: float, sig: int = 4) -> str:
    r"""Un adimensional: entero entre 10³ y 10⁷, con espacio fino desde 10⁴ (``1527``,
    ``42\,180``); si no, con 4 cifras (``0.7034``, ``1.824\times 10^{7}``)."""
    if 1e3 <= abs(x) < 1e7:
        digits = sig - 1 - int(math.floor(math.log10(abs(x))))
        rounded = int(round(x, digits))
        text = (
            f"{abs(rounded):,}".replace(",", r"\,") if abs(rounded) >= 10_000 else f"{abs(rounded)}"
        )
        return ("-" if rounded < 0 else "") + text
    return latex_number(x, sig)


def _d(x: float) -> str:
    return big(x)


def _pow(base: str, exp: str) -> str:
    """``(base)^{exp}`` con paréntesis si la base no es un número simple."""
    if any(c in base for c in r"\ "):
        return rf"\left({base}\right)^{{{exp}}}"
    return f"{base}^{{{exp}}}"


# ---------------------------------------------------------------------
# Las correlaciones con los números
# ---------------------------------------------------------------------


def correlation_latex(key: str, d: dict[str, float], Nu: float) -> list[str]:
    """La correlación ``key`` con los números de ``d`` (Re, Pr, Ra…) hasta Nu.

    Las largas se parten en factores con nombre (a, b, c o ψ) para que cada
    renglón entre en el ancho de un celular.
    """
    Re, Pr = d.get("Re", 0.0), d.get("Pr", 0.0)
    Ra = d.get("Ra", 0.0)
    Re_s, Pr_s, Ra_s = _d(Re), _d(Pr), _d(Ra)
    Nu_s = _d(Nu)
    if key == "plate_laminar":
        return [
            latex_chain(
                "Nu",
                r"0.664\,Re^{1/2}\,Pr^{1/3}",
                times("0.664", _pow(Re_s, "1/2"), _pow(Pr_s, "1/3")),
                Nu_s,
            )
        ]
    if key == "plate_turbulent":
        return [
            latex_chain(
                "Nu",
                r"0.037\,Re^{0.8}\,Pr^{1/3}",
                times("0.037", _pow(Re_s, "0.8"), _pow(Pr_s, "1/3")),
                Nu_s,
            )
        ]
    if key == "plate_mixed":
        a = 0.037 * Re**0.8
        return [
            latex_chain("a", r"0.037\,Re^{0.8}", times("0.037", _pow(Re_s, "0.8")), _d(a)),
            latex_chain(
                "Nu",
                r"(a - 871)\,Pr^{1/3}",
                rf"({_d(a)} - 871) \cdot {_pow(Pr_s, '1/3')}",
                Nu_s,
            ),
        ]
    if key == "churchill_bernstein":
        a = 0.62 * Re**0.5 * Pr ** (1.0 / 3.0)
        b = (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
        c = (1.0 + (Re / 282_000.0) ** 0.625) ** 0.8
        return [
            latex_chain(
                "a",
                r"0.62\,Re^{1/2}\,Pr^{1/3}",
                times("0.62", _pow(Re_s, "1/2"), _pow(Pr_s, "1/3")),
                _d(a),
            ),
            latex_chain(
                "b",
                r"\left[1 + (0.4/Pr)^{2/3}\right]^{1/4}",
                rf"\left[1 + (0.4/{Pr_s})^{{2/3}}\right]^{{1/4}}",
                _d(b),
            ),
            latex_chain(
                "c",
                r"\left[1 + (Re/282\,000)^{5/8}\right]^{4/5}",
                rf"\left[1 + ({Re_s}/282\,000)^{{5/8}}\right]^{{4/5}}",
                _d(c),
            ),
            latex_chain(
                "Nu", r"0.3 + \frac{a\,c}{b}", "0.3 + " + frac(times(_d(a), _d(c)), _d(b)), Nu_s
            ),
        ]
    if key == "hilpert":
        C, m = _hilpert_row(Re)
        return [
            latex_chain(
                "Nu",
                r"C\,Re^{m}\,Pr^{1/3}",
                times(latex_number(C, 4), _pow(Re_s, latex_number(m, 4)), _pow(Pr_s, "1/3")),
                Nu_s,
            )
        ]
    if key == "whitaker":
        ratio = d.get("mu_ratio", 1.0)
        a = 0.4 * Re**0.5 + 0.06 * Re ** (2.0 / 3.0)
        return [
            latex_chain(
                "a",
                r"0.4\,Re^{1/2} + 0.06\,Re^{2/3}",
                times("0.4", _pow(Re_s, "1/2"))
                + r" \\ &\quad + "
                + times("0.06", _pow(Re_s, "2/3")),
                _d(a),
            ),
            latex_chain(
                "Nu",
                r"2 + a\,Pr^{0.4}\left(\frac{\mu_\infty}{\mu_s}\right)^{1/4}",
                rf"2 + {times(_d(a), _pow(Pr_s, '0.4'), _pow(_d(ratio), '1/4'))}",
                Nu_s,
            ),
        ]
    if key == "ranz_marshall":
        return [
            latex_chain(
                "Nu",
                r"2 + 0.6\,Re^{1/2}\,Pr^{1/3}",
                rf"2 + {times('0.6', _pow(Re_s, '1/2'), _pow(Pr_s, '1/3'))}",
                Nu_s,
            )
        ]
    if key == "tube_laminar":
        if d.get("constant_T", 1.0) > 0.5:
            return [r"Nu = 3.66 \quad (T_s\ \text{constante})"]
        return [r"Nu = \frac{48}{11} = 4.36 \quad (q''_s\ \text{constante})"]
    if key == "hausen":
        Gz = d["D_over_L"] * Re * Pr
        return [
            latex_chain(
                "Gz",
                r"\frac{D}{L}\,Re\,Pr",
                times(latex_number(d["D_over_L"], 4), Re_s, Pr_s),
                _d(Gz),
            ),
            latex_chain(
                "Nu",
                r"3.66 + \frac{0.0668\,Gz}{1 + 0.04\,Gz^{2/3}}",
                "3.66 + "
                + frac(times("0.0668", _d(Gz)), "1 + " + times("0.04", _pow(_d(Gz), "2/3"))),
                Nu_s,
            ),
        ]
    if key == "dittus_boelter":
        n_exp = "0.4" if d.get("heating", 1.0) > 0.5 else "0.3"
        return [
            latex_chain(
                "Nu",
                rf"0.023\,Re^{{0.8}}\,Pr^{{{n_exp}}}",
                times("0.023", _pow(Re_s, "0.8"), _pow(Pr_s, n_exp)),
                Nu_s,
            )
        ]
    if key == "sieder_tate":
        ratio = d.get("mu_ratio", 1.0)
        return [
            latex_chain(
                "Nu",
                r"0.027\,Re^{0.8}\,Pr^{1/3}\left(\frac{\mu}{\mu_s}\right)^{0.14}",
                rf"0.027 \cdot {times(_pow(Re_s, '0.8'), _pow(Pr_s, '1/3'))} \\ &\quad \cdot "
                f"{_pow(_d(ratio), '0.14')}",
                Nu_s,
            )
        ]
    if key == "gnielinski":
        f = (0.790 * math.log(Re) - 1.64) ** -2
        num = (f / 8.0) * (Re - 1000.0) * Pr
        den = 1.0 + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2.0 / 3.0) - 1.0)
        return [
            latex_chain(
                "f",
                r"(0.790\,\ln Re - 1.64)^{-2}",
                rf"(0.790\,\ln {Re_s} - 1.64)^{{-2}}",
                latex_number(f, 4),
            ),
            latex_chain(
                "a",
                r"\frac{f}{8}\,(Re - 1000)\,Pr",
                rf"{frac(latex_number(f, 4), '8')} \cdot ({Re_s} - 1000) \cdot {Pr_s}",
                _d(num),
            ),
            latex_chain(
                "b",
                r"1 + 12.7\,\sqrt{f/8}\,(Pr^{2/3} - 1)",
                rf"1 + 12.7\,\sqrt{{{latex_number(f, 4)}/8}} \\ &\quad \cdot "
                f"({_pow(Pr_s, '2/3')} - 1)",
                latex_number(den, 4),
            ),
            latex_chain("Nu", r"\frac{a}{b}", frac(_d(num), latex_number(den, 4)), Nu_s),
        ]
    if key in ("vertical_churchill_chu", "horizontal_cylinder_churchill_chu"):
        c0, c1 = ("0.825", "0.492") if key == "vertical_churchill_chu" else ("0.60", "0.559")
        psi = (1.0 + (float(c1) / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
        root = Ra ** (1.0 / 6.0)
        return [
            latex_chain("Ra^{1/6}", _pow(Ra_s, "1/6"), latex_number(root, 4)),
            latex_chain(
                r"\psi",
                rf"\left[1 + ({c1}/Pr)^{{9/16}}\right]^{{8/27}}",
                rf"\left[1 + ({c1}/{Pr_s})^{{9/16}}\right]^{{8/27}}",
                latex_number(psi, 4),
            ),
            latex_chain(
                "Nu",
                rf"\left({c0} + \frac{{0.387\,Ra^{{1/6}}}}{{\psi}}\right)^2",
                rf"\left({c0} + {frac(times('0.387', latex_number(root, 4)), latex_number(psi, 4))}"
                r"\right)^2",
                Nu_s,
            ),
        ]
    if key == "vertical_mcadams":
        if Ra <= 1e9:
            formula, numbers = r"0.59\,Ra^{1/4}", times("0.59", _pow(Ra_s, "1/4"))
        else:
            formula, numbers = r"0.10\,Ra^{1/3}", times("0.10", _pow(Ra_s, "1/3"))
        return [latex_chain("Nu", formula, numbers, Nu_s)]
    if key == "horizontal_upper":
        if Ra <= 1e7:
            formula, numbers = r"0.54\,Ra^{1/4}", times("0.54", _pow(Ra_s, "1/4"))
        else:
            formula, numbers = r"0.15\,Ra^{1/3}", times("0.15", _pow(Ra_s, "1/3"))
        return [latex_chain("Nu", formula, numbers, Nu_s)]
    if key == "horizontal_lower":
        return [latex_chain("Nu", r"0.27\,Ra^{1/4}", times("0.27", _pow(Ra_s, "1/4")), Nu_s)]
    if key == "morgan":
        C, m = _morgan_row(Ra)
        return [
            latex_chain(
                "Nu",
                r"C\,Ra^{n}",
                times(latex_number(C, 4), _pow(Ra_s, latex_number(m, 4))),
                Nu_s,
            )
        ]
    if key == "sphere_churchill":
        psi = (1.0 + (0.469 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)
        root = Ra**0.25
        return [
            latex_chain("Ra^{1/4}", _pow(Ra_s, "1/4"), latex_number(root, 4)),
            latex_chain(
                r"\psi",
                r"\left[1 + (0.469/Pr)^{9/16}\right]^{4/9}",
                rf"\left[1 + (0.469/{Pr_s})^{{9/16}}\right]^{{4/9}}",
                latex_number(psi, 4),
            ),
            latex_chain(
                "Nu",
                r"2 + \frac{0.589\,Ra^{1/4}}{\psi}",
                rf"2 + {frac(times('0.589', latex_number(root, 4)), latex_number(psi, 4))}",
                Nu_s,
            ),
        ]
    raise ValueError(f"Correlación desconocida: {key!r}.")  # pragma: no cover


def _hilpert_row(Re: float) -> tuple[float, float]:
    C, m = _HILPERT[-1][2:]
    for _lo, hi, C_k, m_k in _HILPERT:
        if Re < hi:
            return C_k, m_k
    return C, m


def _morgan_row(Ra: float) -> tuple[float, float]:
    C, m = _MORGAN[-1][1:]
    for hi, C_k, m_k in _MORGAN:
        if Ra <= hi:
            return C_k, m_k
    return C, m


# ---------------------------------------------------------------------
# Propiedades
# ---------------------------------------------------------------------


def _props_latex(r: ConvectionResult | InternalFlowResult, system: UnitSystem) -> list[str]:
    p = r.props
    latex = [
        rf"\rho = {q(p.rho_kg_per_m3, 'density', system)}",
        rf"\mu = {q(p.mu_Pa_s, 'dynamic_viscosity', system)}",
        latex_chain(
            r"\nu",
            frac(r"\mu", r"\rho"),
            frac(n(p.mu_Pa_s, "dynamic_viscosity", system), n(p.rho_kg_per_m3, "density", system)),
            q(p.nu_m2_per_s, "diffusivity", system),
        ),
        rf"k = {q(p.k_W_per_mK, _K, system)}",
        rf"c_p = {q(p.cp_J_per_kgK, 'specific_heat', system)}",
        rf"Pr = \frac{{\mu\,c_p}}{{k}} = {latex_number(p.Pr, 4)}",
    ]
    return latex


# ---------------------------------------------------------------------
# Externa y natural
# ---------------------------------------------------------------------


def _film_step(r: ConvectionResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    Ts, Tinf = n(i.T_s_K, _T, system), n(i.T_inf_K, _T, system)
    sphere = isinstance(i, ExternalFlowInputs) and i.geometry == "sphere"
    latex: list[str] = []
    if sphere:
        latex.append(rf"T_\infty = {q(i.T_inf_K, _T, system)}")
        text = (
            "Whitaker usa las propiedades a la temperatura del fluido T∞ y corrige con la "
            "viscosidad en la superficie, μ_s a T_s (Incropera, §7.5). CoolProp da las "
            f"propiedades del {_fluid(r)} a esa temperatura y presión."
        )
    else:
        latex.append(
            latex_chain(
                "T_f",
                frac(r"T_s + T_\infty", "2"),
                frac(f"{Ts} + {Tinf}", "2"),
                q(r.T_film_K, _T, system),
            )
        )
        text = (
            "Las propiedades se evalúan a la **temperatura de película** T_f, el promedio entre "
            f"la superficie y el fluido: CoolProp da las del {_fluid(r)} a T_f y a la presión."
        )
    latex += _props_latex(r, system)
    if sphere and r.mu_s_Pa_s is not None:
        latex.append(rf"\mu_s = {q(r.mu_s_Pa_s, 'dynamic_viscosity', system)}")
    if r.is_natural:
        p = r.props
        latex.append(rf"\beta = {q(p.beta_per_K, 'expansion_coefficient', system)}")
        if not p.is_liquid:
            inv = f"{1.0 / r.T_film_K:.3e}".replace(".", ",").replace("e-0", "·10⁻")
            text += f" En un gas ideal β = 1/T: 1/T_f = {inv} 1/K, casi lo mismo que da CoolProp."
    return ProcedureStep("Propiedades", text, tuple(latex))


def _fluid(r: ConvectionResult | InternalFlowResult) -> str:
    return fluid_with_article(r.props.fluid)[3:]


def _reynolds_step(r: ConvectionResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    assert isinstance(i, ExternalFlowInputs)
    L_sym = "L" if i.geometry == "plate" else "D"
    Re = r.Re or 0.0
    latex = [
        latex_chain(
            "Re",
            frac(rf"V\,{L_sym}", r"\nu"),
            frac(
                times(n(i.V_m_per_s, "speed", system), n(i.length_m, _L, system)),
                n(r.props.nu_m2_per_s, "diffusivity", system),
            ),
            big(Re),
        )
    ]
    if i.geometry == "plate":
        text = f"El número de Reynolds con el largo de la placa: el régimen es **{r.regime}**."
        if not i.tripped and Re >= RE_CRITICAL_PLATE:
            x_cr = RE_CRITICAL_PLATE * r.props.nu_m2_per_s / i.V_m_per_s
            latex.append(
                latex_chain(
                    r"x_{\text{cr}}",
                    frac(r"Re_{\text{cr}}\,\nu", "V"),
                    frac(
                        times(r"5\times 10^{5}", n(r.props.nu_m2_per_s, "diffusivity", system)),
                        n(i.V_m_per_s, "speed", system),
                    ),
                    q(x_cr, _L, system),
                )
            )
            text += (
                " La capa límite es laminar hasta x_cr (Re_cr = 5·10⁵) y turbulenta después: "
                "Nu promedia los dos tramos (Incropera, §7.2.3)."
            )
        elif i.tripped:
            text += " Un alambre o una rugosidad en el borde la vuelve turbulenta desde x = 0."
    else:
        text = (
            f"El número de Reynolds con el diámetro: **{r.regime}** (la capa límite se vuelve "
            "turbulenta antes de separarse con Re ≳ 2·10⁵)."
        )
    return ProcedureStep("Número de Reynolds", text, tuple(latex))


def _rayleigh_step(r: ConvectionResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    assert isinstance(i, NaturalConvectionInputs)
    p = r.props
    latex: list[str] = []
    horizontal = i.geometry in ("horizontal_plate_up", "horizontal_plate_down")
    if horizontal:
        a, b = n(i.length_m, _L, system), n(i.width_m, _L, system)
        latex.append(
            latex_chain(
                "L",
                frac("A", "P"),
                frac(times(a, b), f"2 \\cdot ({a} + {b})"),
                q(r.L_char_m, _L, system),
            )
        )
    ts, tinf = n(i.T_s_K, _T, system), n(i.T_inf_K, _T, system)
    dT = abs(i.T_s_K - i.T_inf_K)
    hot = i.T_s_K > i.T_inf_K
    latex.append(
        latex_chain(
            r"\Delta T",
            r"T_s - T_\infty" if hot else r"T_\infty - T_s",
            sub(ts, tinf) if hot else sub(tinf, ts),
            q(dT, _DT, system),
        )
    )
    num = times(
        n(G_STANDARD, "acceleration", system, 4),
        n(p.beta_per_K, "expansion_coefficient", system, 4),
        n(dT, _DT, system, 4),
        _pow(n(r.L_char_m, _L, system, 4), "3"),
    )
    latex.append(
        latex_chain(
            "Gr",
            frac(r"g\,\beta\,\Delta T\,L^3", r"\nu^2"),
            frac(num, _pow(n(p.nu_m2_per_s, "diffusivity", system), "2")),
            big(r.Gr or 0.0),
        )
    )
    latex.append(
        latex_chain(
            "Ra", r"Gr\,Pr", times(big(r.Gr or 0.0), latex_number(p.Pr, 4)), big(r.Ra or 0.0)
        )
    )
    text = (
        "En la convección natural el fluido se mueve por el **empuje**: el número de Grashof "
        "compara el empuje con la viscosidad, y Ra = Gr·Pr ordena las correlaciones como Re en la "
        "forzada (Incropera, §9.1 y §9.6)."
    )
    if horizontal:
        text += (
            " En una placa horizontal la longitud característica es L = A/P (Goldstein et al., "
            "1973)."
        )
    return ProcedureStep("Números de Grashof y Rayleigh", text, tuple(latex))


def _nusselt_step(r: ConvectionResult | InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    c = CORRELATIONS[r.correlation]
    d = dimensionless_groups(r)
    latex = correlation_latex(r.correlation, d, r.Nu)
    if r.correlation == "whitaker" and r.mu_s_Pa_s is not None:
        latex.insert(
            0,
            latex_chain(
                r"\frac{\mu_\infty}{\mu_s}",
                frac(
                    n(r.props.mu_Pa_s, "dynamic_viscosity", system),
                    n(r.mu_s_Pa_s, "dynamic_viscosity", system),
                ),
                latex_number(d["mu_ratio"], 4),
            ),
        )
    if r.correlation == "sieder_tate" and r.mu_s_Pa_s is not None:
        latex.insert(0, rf"\mu_s = {q(r.mu_s_Pa_s, 'dynamic_viscosity', system)}")
    text = f"**{c.name}** (válida para {c.validity})."
    if not c.in_range(d):
        text += " Con estos datos está **fuera de su rango**: el resultado es una extrapolación."
    return ProcedureStep("Número de Nusselt", text, tuple(latex))


def _h_and_heat_steps(r: ConvectionResult, system: UnitSystem) -> list[ProcedureStep]:
    i = r.inputs
    L_sym = "D" if i.geometry in ("cylinder", "sphere", "horizontal_cylinder") else "L"
    h_latex = latex_chain(
        "h",
        frac(r"Nu\,k", L_sym),
        frac(times(big(r.Nu), n(r.props.k_W_per_mK, _K, system)), n(r.L_char_m, _L, system)),
        q(r.h_W_per_m2K, _H, system),
    )
    h_text = (
        "El número de Nusselt es el gradiente de temperatura adimensional en la superficie: "
        "h = Nu·k/L."
    )
    geo = i.geometry
    L, W = n(i.length_m, _L, system), n(i.width_m, _L, system)
    if geo in ("plate", "vertical_plate", "horizontal_plate_up", "horizontal_plate_down"):
        area = latex_chain("A", r"L\,w", times(L, W), q(r.area_m2, "area", system))
    elif geo in ("cylinder", "horizontal_cylinder"):
        area = latex_chain(
            "A", r"\pi\,D\,L", rf"\pi \cdot {times(L, W)}", q(r.area_m2, "area", system)
        )
    else:
        area = latex_chain(
            "A", r"\pi\,D^2", rf"\pi \cdot {_pow(L, '2')}", q(r.area_m2, "area", system)
        )
    hA = times(n(r.h_W_per_m2K, _H, system), n(r.area_m2, "area", system))
    dT = f"({sub(n(i.T_s_K, _T, system), n(i.T_inf_K, _T, system))})"
    Q_latex = latex_chain(
        r"\dot{Q}",
        r"h\,A\,(T_s - T_\infty)",
        rf"{hA} \\ &\quad \cdot {dT}" if system == "SI" else times(hA, dT),
        q(r.Q_W, _W, system),
    )
    Q_text = "La ley de enfriamiento de Newton: Q̇ = h·A·(T_s − T∞)."
    if r.Q_W < 0.0:
        Q_text += " Sale negativo: la superficie está más fría que el fluido y gana calor."
    return [
        ProcedureStep("Coeficiente de convección", h_text, (h_latex,)),
        ProcedureStep("Calor", Q_text, (area, Q_latex)),
    ]


def convection_steps(result: ConvectionResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de una convección externa o natural, con los valores en ``system``."""
    steps = [_film_step(result, system)]
    if result.is_natural:
        steps.append(_rayleigh_step(result, system))
    else:
        steps.append(_reynolds_step(result, system))
    steps.append(_nusselt_step(result, system))
    steps += _h_and_heat_steps(result, system)
    return numbered(steps)


# ---------------------------------------------------------------------
# Flujo interno
# ---------------------------------------------------------------------


def _mean_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    latex = [
        latex_chain(
            r"T_m",
            frac(r"T_{\text{ent}} + T_{\text{sal}}", "2"),
            frac(f"{n(i.T_in_K, _T, system)} + {n(r.T_out_K, _T, system)}", "2"),
            q(r.T_mean_K, _T, system),
        ),
        *_props_latex(r, system),
    ]
    text = (
        "Las propiedades se evalúan a la **temperatura media** del fluido, el promedio entre la "
        "entrada y la salida (Incropera, §8.3). La salida depende de h, así que se itera: con las "
        "propiedades de la entrada sale una T_sal, con ella las propiedades a T_m, y así hasta "
        f"que T_sal no cambia ({r.iterations} vueltas)."
    )
    return ProcedureStep("Temperatura media y propiedades", text, tuple(latex))


def _tube_reynolds_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    latex = [
        latex_chain(
            "Re",
            frac(r"4\,\dot{m}", r"\pi\,D\,\mu"),
            frac(
                times("4", n(i.m_dot_kg_s, "mass_flow", system)),
                r"\pi \cdot "
                + times(n(i.D_m, _L, system), n(r.props.mu_Pa_s, "dynamic_viscosity", system)),
            ),
            big(r.Re),
        )
    ]
    text = (
        f"Re = ρ·V·D/μ = 4ṁ/(π·D·μ): el flujo es **{r.regime}** (laminar si Re < 2300, "
        "turbulento desde ~10⁴)."
    )
    return ProcedureStep("Número de Reynolds", text, tuple(latex))


def _tube_h_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    latex = [
        latex_chain(
            "h",
            frac(r"Nu\,k", "D"),
            frac(times(big(r.Nu), n(r.props.k_W_per_mK, _K, system)), n(r.inputs.D_m, _L, system)),
            q(r.h_W_per_m2K, _H, system),
        )
    ]
    return ProcedureStep("Coeficiente de convección", "h = Nu·k/D.", tuple(latex))


def _outlet_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    C = i.m_dot_kg_s * r.props.cp_J_per_kgK
    A_n = n(r.area_m2, "area", system)
    C_n = n(C, "heat_capacity_rate", system)
    latex = [
        latex_chain(
            "A",
            r"\pi\,D\,L",
            rf"\pi \cdot {times(n(i.D_m, _L, system), n(i.L_m, _L, system))}",
            q(r.area_m2, "area", system),
        ),
        latex_chain(
            r"\dot{m}\,c_p",
            rf"{q(i.m_dot_kg_s, 'mass_flow', system)} \\ &\quad \cdot "
            f"{q(r.props.cp_J_per_kgK, 'specific_heat', system)}",
            q(C, "heat_capacity_rate", system),
        ),
    ]
    T_in = n(i.T_in_K, _T, system)
    if i.condition == "constant_T":
        NTU = r.h_W_per_m2K * r.area_m2 / C
        Ts = n(i.T_s_K or 0.0, _T, system)
        latex += [
            latex_chain(
                "NTU",
                frac(r"h\,A", r"\dot{m}\,c_p"),
                frac(times(n(r.h_W_per_m2K, _H, system), A_n), C_n),
                latex_number(NTU, 4),
            ),
            latex_chain(
                r"T_{\text{sal}}",
                r"T_s - (T_s - T_{\text{ent}})\,e^{-NTU}",
                sub(Ts, f"({sub(Ts, T_in)})")
                + (r" \\ &\quad" if system == "SI" else "")
                + rf" \cdot e^{{-{latex_number(NTU, 4)}}}",
                q(r.T_out_K, _T, system),
            ),
        ]
        text = (
            "Con la **pared a temperatura constante** la diferencia T_s − T_m cae en forma "
            "exponencial a lo largo del tubo (Incropera, §8.3.3): NTU = h·A/(ṁ·c_p) dice cuánto "
            "se acerca el fluido a la pared."
        ) + _capacity_units_note(system)
    else:
        q_n = n(i.q_W_per_m2 or 0.0, "heat_flux", system)
        latex.append(
            latex_chain(
                r"T_{\text{sal}}",
                r"T_{\text{ent}} + \frac{q''_s\,A}{\dot{m}\,c_p}",
                f"{T_in} + {frac(times(q_n, A_n), C_n)}",
                q(r.T_out_K, _T, system),
            )
        )
        text = (
            "Con **flujo de calor constante** por la pared, el fluido se calienta linealmente: "
            "toda la energía que entra por la pared la lleva el fluido (Incropera, §8.3.2)."
        ) + _capacity_units_note(system)
    return ProcedureStep("Temperatura de salida", text, tuple(latex))


def _capacity_units_note(system: UnitSystem) -> str:
    """ṁ·c_p en las unidades de h·A (W/K o Btu/(h·°F))."""
    if system == "Técnico":
        return " Con c_p en kJ/(kg·K), ṁ·c_p sale en kW/K: por 1000, en W/K, las unidades de h·A."
    if system == "Inglés":
        return " ṁ·c_p sale en Btu/(s·°F): por 3600 s/h, en Btu/(h·°F), las unidades de h·A."
    return ""


def _tube_heat_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    C = i.m_dot_kg_s * r.props.cp_J_per_kgK
    T_in, T_out = n(i.T_in_K, _T, system), n(r.T_out_K, _T, system)
    latex = [
        latex_chain(
            r"\dot{Q}",
            r"\dot{m}\,c_p\,(T_{\text{sal}} - T_{\text{ent}})",
            times(n(C, "heat_capacity_rate", system), f"({sub(T_out, T_in)})"),
            q(r.Q_W, _W, system),
        )
    ]
    if i.condition == "constant_T":
        Ts = i.T_s_K or 0.0
        dTi, dTo = Ts - i.T_in_K, Ts - r.T_out_K
        dTi_n, dTo_n = n(dTi, _DT, system), n(dTo, _DT, system)
        latex += [
            latex_chain(
                r"\Delta T_{\text{ml}}",
                frac(
                    r"\Delta T_{\text{ent}} - \Delta T_{\text{sal}}",
                    r"\ln(\Delta T_{\text{ent}}/\Delta T_{\text{sal}})",
                ),
                # el cociente es el mismo con los dos ΔT negativos (el fluido se enfría)
                frac(sub(dTi_n, dTo_n), rf"\ln({abs_n(dTi_n)}/{abs_n(dTo_n)})"),
                q(r.dT_lm_K or 0.0, _DT, system),
            ),
            latex_chain(
                r"\dot{Q}",
                r"h\,A\,\Delta T_{\text{ml}}",
                times(
                    n(r.h_W_per_m2K, _H, system),
                    n(r.area_m2, "area", system),
                    n(r.dT_lm_K or 0.0, _DT, system),
                ),
                q(r.Q_W, _W, system),
            ),
        ]
        text = (
            "El calor del balance de energía del fluido coincide con h·A·ΔT_ml, la diferencia "
            "media logarítmica entre la pared y el fluido (ΔT = T_s − T_m)."
        )
    else:
        q_n = n(i.q_W_per_m2 or 0.0, "heat_flux", system)
        latex += [
            latex_chain(
                r"\dot{Q}",
                r"q''_s\,A",
                times(q_n, n(r.area_m2, "area", system)),
                q(r.Q_W, _W, system),
            ),
            latex_chain(
                r"T_{s,\text{sal}}",
                r"T_{\text{sal}} + \frac{q''_s}{h}",
                f"{T_out} + {frac(q_n, n(r.h_W_per_m2K, _H, system))}",
                q(r.T_s_out_K or 0.0, _T, system),
            ),
        ]
        text = (
            "Con flujo constante, la pared está q''/h por encima del fluido en todo el tubo "
            "desarrollado: la más caliente es la de la salida."
        )
    return ProcedureStep("Calor", text, tuple(latex))


def abs_n(tex: str) -> str:
    """El valor absoluto de un número ya escrito en LaTeX."""
    return tex[1:] if tex.startswith("-") else tex


def _pressure_step(r: InternalFlowResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    rho_n = n(r.props.rho_kg_per_m3, "density", system)
    D_n = n(i.D_m, _L, system)
    latex = [
        latex_chain(
            "V",
            frac(r"\dot{m}", r"\rho\,\pi\,D^2/4"),
            frac(
                n(i.m_dot_kg_s, "mass_flow", system), rf"{rho_n} \cdot \pi \cdot {_pow(D_n, '2')}/4"
            ),
            q(r.V_m_per_s, "speed", system),
        )
    ]
    if r.Re < RE_LAMINAR_TUBE:
        latex.append(
            latex_chain("f", frac("64", "Re"), frac("64", big(r.Re)), latex_number(r.f, 4))
        )
        f_text = "En laminar f = 64/Re (Hagen–Poiseuille)."
    else:
        Re_f = max(r.Re, 3000.0)
        latex.append(
            latex_chain(
                "f",
                r"(0.790\,\ln Re - 1.64)^{-2}",
                rf"(0.790\,\ln {big(Re_f)} - 1.64)^{{-2}}",
                latex_number(r.f, 4),
            )
        )
        f_text = "En turbulento, f de Petukhov para un tubo liso (Incropera, §8.1)."
    english = system == "Inglés"
    dp_formula = (
        r"f\,\frac{L}{D}\,\frac{\rho\,V^2}{2\,g_c}"
        if english
        else r"f\,\frac{L}{D}\,\frac{\rho\,V^2}{2}"
    )
    V2 = _pow(n(r.V_m_per_s, "speed", system), "2")
    den = r"2 \cdot 32.174" if english else "2"
    latex.append(
        latex_chain(
            r"\Delta p",
            dp_formula,
            rf"{times(latex_number(r.f, 4), frac(n(i.L_m, _L, system), D_n))} \\ &\quad \cdot "
            f"{frac(times(rho_n, V2), den)}",
            q(r.dp_Pa, "pressure_drop", system),
        )
    )
    text = f"La caída de presión por fricción (Darcy–Weisbach). {f_text}"
    if english:
        text += " En el sistema inglés, g_c = 32,174 lbm·ft/(lbf·s²) pasa a lbf/ft²."
    return ProcedureStep("Caída de presión", text, tuple(latex))


def internal_steps(result: InternalFlowResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del flujo en un tubo, con los valores en ``system``."""
    return numbered(
        [
            _mean_step(result, system),
            _tube_reynolds_step(result, system),
            _nusselt_step(result, system),
            _tube_h_step(result, system),
            _outlet_step(result, system),
            _tube_heat_step(result, system),
            _pressure_step(result, system),
        ]
    )

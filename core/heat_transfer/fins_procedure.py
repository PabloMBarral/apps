"""Procedimiento didáctico de las aletas (LaTeX), en el sistema de unidades activo.

Incropera et al. (2007), §3.6 (tabla 3.4 y §3.6.4); Cengel y Ghajar (2015), §3-6:
sección y perímetro, m y M, el calor según la punta, la temperatura de la punta,
la eficiencia, la efectividad, el número de Biot y, si hay, el arreglo.

La aleta anular se escribe con una constante γ que resume la condición de la
punta (θ = C₁·I₀(mr) + C₂·K₀(mr), con C₁ = γ·C₂):

    q = 2π·k·r₁·t·m·θ_b·[K₁(mr₁) − γ·I₁(mr₁)] / [K₀(mr₁) + γ·I₀(mr₁)],

γ = K₁(mr₂c)/I₁(mr₂c) con la punta adiabática (la forma de la tabla 3.5 de
Incropera), γ = [K₁(mr₂) − β·K₀(mr₂)]/[I₁(mr₂) + β·I₀(mr₂)] con la convectiva
(β = h/(m·k)) y γ = 0 si es infinita.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.special import i0, i1, k0, k1

from core.heat_transfer.fins import FinArrayResult, FinResult
from core.heat_transfer.procedure_common import (
    dimensionless,
    frac,
    n,
    numbered,
    q,
    sub,
    times,
)
from core.latex import latex_chain
from core.state_report import ProcedureStep
from core.units_system import UnitSystem

__all__ = ["fin_steps"]

_L = "length"
_K = "thermal_conductivity"
_H = "heat_transfer_coefficient"
_W = "heat_rate"
_DT = "temperature_difference"
_A = "area"
#: Más allá de este m·r las funciones de Bessel sin escalar desbordan (I₀(700) ~ 10³⁰²).
_BESSEL_MAX = 600.0


def _theta_step(r: FinResult, system: UnitSystem) -> list[str]:
    i = r.inputs
    return [
        latex_chain(
            r"\theta_b",
            r"T_b - T_\infty",
            sub(n(i.T_base_K, "temperature", system), n(i.T_inf_K, "temperature", system)),
            q(i.theta_b, _DT, system),
        )
    ]


def _geometry_step(r: FinResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    latex = _theta_step(r, system)
    if i.shape == "straight":
        w, t = n(i.width_m, _L, system), n(i.thickness_m, _L, system)
        latex += [
            latex_chain("A_c", r"w\,t", times(w, t), q(r.A_base_m2, _A, system)),
            latex_chain(
                "P",
                "2\\,(w + t)",
                f"2 \\cdot ({w} + {t})",
                q(2.0 * (i.width_m + i.thickness_m), _L, system),
            ),
        ]
        text = "Aleta **recta** de sección rectangular w × t: el perímetro incluye los cantos."
    elif i.shape == "pin":
        D = n(i.diameter_m, _L, system)
        latex += [
            latex_chain("A_c", frac(r"\pi\,D^2", "4"), frac(rf"\pi \cdot {D}^2", "4"),
                        q(r.A_base_m2, _A, system)),
            latex_chain("P", r"\pi\,D", rf"\pi \cdot {D}", q(math.pi * i.diameter_m, _L, system)),
        ]  # fmt: skip
        text = "Aleta **de aguja** (cilíndrica) de diámetro D."
    else:
        r1, t = n(i.r_base_m, _L, system), n(i.thickness_m, _L, system)
        latex += [
            rf"r_1 = {q(i.r_base_m, _L, system)}",
            latex_chain(
                "r_2",
                "r_1 + L",
                f"{r1} + {n(i.length_m, _L, system)}",
                q(i.r_base_m + i.length_m, _L, system),
            ),
            latex_chain(
                "A_c", r"2\pi\,r_1\,t", rf"2\pi \cdot {times(r1, t)}", q(r.A_base_m2, _A, system)
            ),
        ]
        text = (
            "Aleta **anular** (un disco alrededor de un caño de radio r₁): la sección por la que "
            "entra el calor es la de la base, A_c = 2π·r₁·t, y crece con el radio."
        )
    if i.tip == "corrected":
        if i.shape == "pin":
            corr, corr_n = (
                r"L + D/4",
                f"{n(i.length_m, _L, system)} + {n(i.diameter_m, _L, system)}/4",
            )
        else:
            corr, corr_n = (
                r"L + t/2",
                f"{n(i.length_m, _L, system)} + {n(i.thickness_m, _L, system)}/2",
            )
        latex.append(latex_chain("L_c", corr, corr_n, q(r.L_c_m, _L, system)))
        text += (
            " Con la **longitud corregida** L_c, la punta convectiva se reemplaza por una "
            "adiabática con el área de la punta repartida en el costado (Cengel y Ghajar, §3-6)."
        )
    return ProcedureStep("Geometría", text, tuple(latex))


def _m_step(r: FinResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    h, k = n(i.h_W_per_m2K, _H, system), n(i.k_W_per_mK, _K, system)
    L_sym = "L_c" if i.tip == "corrected" else "L"
    if i.shape == "annular":
        two_h = times("2", h)
        root = frac(two_h, times(k, n(i.thickness_m, _L, system)))
        latex = [
            latex_chain(
                "m",
                r"\sqrt{\frac{2h}{k\,t}}",
                rf"\sqrt{{{root}}}",
                q(r.m_per_m, "inverse_length", system),
            )
        ]
        text = (
            "En la aleta anular el parámetro es m = √(2h/(k·t)) (las dos caras del disco "
            "convectan): la ecuación es la de Bessel modificada de orden cero (Incropera, §3.6.4)."
        )
    else:
        A_c = r.A_base_m2
        P = math.pi * i.diameter_m if i.shape == "pin" else 2.0 * (i.width_m + i.thickness_m)
        P_n, A_n = n(P, _L, system), n(A_c, _A, system)
        latex = [
            latex_chain(
                "m",
                r"\sqrt{\frac{h\,P}{k\,A_c}}",
                rf"\sqrt{{{frac(times(h, P_n), times(k, A_n))}}}",
                q(r.m_per_m, "inverse_length", system),
            ),
            latex_chain(
                "M",
                r"\sqrt{h\,P\,k\,A_c}\;\theta_b",
                r"k\,A_c\,m\,\theta_b",
                rf"{times(k, A_n)} \\ &\quad \cdot "
                + times(n(r.m_per_m, "inverse_length", system), n(i.theta_b, _DT, system)),
                q(r.M_W, _W, system),
            ),
        ]
        text = (
            "Con la temperatura uniforme en cada sección, θ = T − T∞ cumple d²θ/dx² = m²·θ "
            "(Incropera, §3.6.2): m dice qué tan rápido cae la temperatura a lo largo de la "
            "aleta, y M es el calor de una aleta infinita."
        )
    latex.append(
        latex_chain(
            rf"m\,{L_sym}",
            times(n(r.m_per_m, "inverse_length", system), n(r.L_c_m, _L, system)),
            dimensionless(r.mL),
        )
    )
    return ProcedureStep("Parámetro m", text, tuple(latex))


def _uniform_heat_latex(r: FinResult, system: UnitSystem) -> tuple[list[str], str]:
    """El calor y la temperatura de la punta de una aleta de sección constante."""
    i = r.inputs
    M, mL = n(r.M_W, _W, system), dimensionless(r.mL)
    Q = q(r.Q_W, _W, system)
    tip = i.tip
    latex: list[str] = []
    if tip == "convective":
        beta = i.h_W_per_m2K / (r.m_per_m * i.k_W_per_mK)
        b = dimensionless(beta)
        t = dimensionless(math.tanh(r.mL))
        latex += [
            latex_chain(
                r"\beta",
                frac("h", "m\\,k"),
                frac(
                    n(i.h_W_per_m2K, _H, system),
                    times(n(r.m_per_m, "inverse_length", system), n(i.k_W_per_mK, _K, system)),
                ),
                b,
            ),
            latex_chain(
                r"\dot{Q}",
                r"M\,\frac{\tanh mL + \beta}{1 + \beta\,\tanh mL}",
                rf"{M} \cdot {frac(f'{t} + {b}', '1 + ' + times(b, t))}",
                Q,
            ),
        ]
        ratio = 1.0 / (math.cosh(r.mL) + beta * math.sinh(r.mL)) if r.mL < 700 else 0.0
        theta_formula = r"\frac{1}{\cosh mL + \beta\,\sinh mL}"
        text = (
            "Punta **convectiva** (la exacta, Incropera, tabla 3.4): la punta también pierde calor "
            "por convección, −k·dθ/dx = h·θ en x = L."
        )
    elif tip in ("adiabatic", "corrected"):
        sym = "mL_c" if tip == "corrected" else "mL"
        latex.append(
            latex_chain(
                r"\dot{Q}",
                rf"M\,\tanh {sym}",
                rf"{M} \cdot \tanh {mL}",
                Q,
            )
        )
        ratio = 1.0 / math.cosh(r.mL) if r.mL < 700 else 0.0
        theta_formula = rf"\frac{{1}}{{\cosh {sym}}}"
        text = (
            "Punta **adiabática** (dθ/dx = 0 en la punta): q = M·tanh(mL)."
            if tip == "adiabatic"
            else "Con la longitud corregida, la punta es adiabática en L_c: q = M·tanh(m·L_c)."
        )
    elif tip == "temperature":
        ratio = (i.T_tip_K - i.T_inf_K) / i.theta_b  # type: ignore[operator]
        d = dimensionless
        fraction = frac(sub(d(math.cosh(r.mL)), d(ratio)), d(math.sinh(r.mL)))
        latex += [
            latex_chain(
                r"\frac{\theta_L}{\theta_b}",
                frac(r"T_L - T_\infty", r"\theta_b"),
                dimensionless(ratio),
            ),
            latex_chain(
                r"\dot{Q}",
                r"M\,\frac{\cosh mL - \theta_L/\theta_b}{\sinh mL}",
                rf"{M} \cdot {fraction}",
                Q,
            ),
        ]
        theta_formula = ""
        text = "Punta a **temperatura dada** T_L (Incropera, tabla 3.4)."
    else:
        latex.append(latex_chain(r"\dot{Q}", "M", Q))
        ratio = math.exp(-r.mL)
        theta_formula = r"e^{-mL}"
        text = (
            "Aleta **infinita**: la punta llega a la temperatura del fluido y q = M. Vale si "
            "mL > 2,65 (tanh 2,65 = 0,99)."
        )
    if theta_formula:
        latex.append(latex_chain(r"\frac{\theta_L}{\theta_b}", theta_formula, dimensionless(ratio)))
    return latex, text


def _values_block(values: list[tuple[str, float]], fmt: Callable[[float], str]) -> str:
    """Varios valores alineados en el signo igual, uno por renglón."""
    rows = r" \\ ".join(f"{sym} &= {fmt(v)}" for sym, v in values)
    return rf"\begin{{aligned}}{rows}\end{{aligned}}"


def _bessel(x: float) -> tuple[float, float, float, float]:
    return float(i0(x)), float(i1(x)), float(k0(x)), float(k1(x))


def _annular_heat_latex(r: FinResult, system: UnitSystem) -> tuple[list[str], str]:
    i = r.inputs
    m = r.m_per_m
    r1 = i.r_base_m
    rE = r1 + r.L_c_m
    x1, xE = m * r1, m * rE
    if max(x1, xE) > _BESSEL_MAX:  # pragma: no cover — aletas irreales
        return [rf"\dot{{Q}} = {q(r.Q_W, _W, system)}"], "La aleta anular (funciones de Bessel)."
    I0a, I1a, K0a, K1a = _bessel(x1)
    I0b, I1b, K0b, K1b = _bessel(xE)

    def d(x: float) -> str:
        # 5 cifras: la resta K₁ − γ·I₁ pierde precisión
        return dimensionless(x, 5)

    r2 = "r_{2c}" if i.tip == "corrected" else "r_2"
    m_n = n(m, "inverse_length", system)
    latex = [latex_chain("m\\,r_1", times(m_n, n(r1, _L, system)), d(x1))]
    if i.tip != "infinite":
        latex.append(latex_chain(f"m\\,{r2}", times(m_n, n(rE, _L, system)), d(xE)))
    latex.append(
        _values_block(
            [("I_0(mr_1)", I0a), ("I_1(mr_1)", I1a), ("K_0(mr_1)", K0a), ("K_1(mr_1)", K1a)], d
        )
    )
    tip = i.tip
    beta = i.h_W_per_m2K / (m * i.k_W_per_mK)
    if tip in ("adiabatic", "corrected"):
        latex.append(_values_block([(f"I_1(m{r2})", I1b), (f"K_1(m{r2})", K1b)], d))
        gamma = K1b / I1b
        latex.append(latex_chain(r"\gamma", frac(f"K_1(m{r2})", f"I_1(m{r2})"), d(gamma)))
        text = (
            "La aleta anular con la punta adiabática (en r₂c si la longitud es corregida): la "
            "forma de la tabla 3.5 de Incropera, con γ = C₁/C₂."
        )
    elif tip == "convective":
        latex += [
            _values_block(
                [("I_0(mr_2)", I0b), ("I_1(mr_2)", I1b), ("K_0(mr_2)", K0b), ("K_1(mr_2)", K1b)], d
            ),
            latex_chain(
                r"\beta",
                frac("h", "m\\,k"),
                frac(n(i.h_W_per_m2K, _H, system), times(m_n, n(i.k_W_per_mK, _K, system))),
                d(beta),
            ),
        ]
        gamma = (K1b - beta * K0b) / (I1b + beta * I0b)
        latex.append(
            latex_chain(
                r"\gamma",
                frac(r"K_1(mr_2) - \beta\,K_0(mr_2)", r"I_1(mr_2) + \beta\,I_0(mr_2)"),
                d(gamma),
            )
        )
        text = (
            "La aleta anular con la punta convectiva: θ = C₁·I₀(mr) + C₂·K₀(mr), con "
            "−k·dθ/dr = h·θ en r₂, que fija γ = C₁/C₂ (Incropera, §3.6.4)."
        )
    elif tip == "infinite":
        gamma = 0.0
        latex.append(r"\gamma = 0")
        text = "La aleta anular infinita: θ = C₂·K₀(mr) (I₀ crece sin límite, así que C₁ = 0)."
    else:
        theta_L = (i.T_tip_K or i.T_inf_K) - i.T_inf_K
        det = I0a * K0b - K0a * I0b
        C1 = (i.theta_b * K0b - theta_L * K0a) / det
        C2 = (I0a * theta_L - I0b * i.theta_b) / det
        latex += [
            _values_block([("I_0(mr_2)", I0b), ("K_0(mr_2)", K0b)], d),
            r"C_1\,I_0(mr_1) + C_2\,K_0(mr_1) = \theta_b",
            r"C_1\,I_0(mr_2) + C_2\,K_0(mr_2) = \theta_L",
            rf"C_1 = {q(C1, _DT, system)},\quad C_2 = {q(C2, _DT, system)}",
            latex_chain(
                r"\dot{Q}",
                r"2\pi\,k\,r_1\,t\,m \\ &\quad \cdot \left[C_2\,K_1(mr_1) - C_1\,I_1(mr_1)\right]",
                q(r.Q_W, _W, system),
            ),
        ]
        text = (
            "La aleta anular con la punta a temperatura dada: θ = C₁·I₀(mr) + C₂·K₀(mr) con "
            "las dos temperaturas, un sistema de 2 × 2 (Incropera, §3.6.4)."
        )
        return latex, text
    num = sub(d(K1a), times(d(gamma), d(I1a)))
    den = f"{d(K0a)} + {times(d(gamma), d(I0a))}"
    phi = (K1a - gamma * I1a) / (K0a + gamma * I0a)
    # M = k·A_c·m·θ_b = 2π·k·r₁·t·m·θ_b: el calor de la aleta infinita recta equivalente
    pre = times(
        n(i.k_W_per_mK, _K, system),
        n(r.A_base_m2, _A, system),
        m_n,
        n(i.theta_b, _DT, system),
    )
    if tip != "infinite":
        ratio = (gamma * I0b + K0b) / (gamma * I0a + K0a)
        latex.append(
            latex_chain(
                r"\frac{\theta_L}{\theta_b}",
                frac(rf"\gamma\,I_0(m{r2}) + K_0(m{r2})", r"\gamma\,I_0(mr_1) + K_0(mr_1)"),
                d(ratio),
            )
        )
    latex += [
        latex_chain("M", r"k\,A_c\,m\,\theta_b", pre, q(r.M_W, _W, system)),
        latex_chain(
            r"\phi",
            r"\frac{K_1(mr_1) - \gamma\,I_1(mr_1)}{K_0(mr_1) + \gamma\,I_0(mr_1)}",
            frac(num, den),
            d(phi),
        ),
        latex_chain(
            r"\dot{Q}",
            r"M\,\phi",
            times(n(r.M_W, _W, system), d(phi)),
            q(r.Q_W, _W, system),
        ),
    ]
    return latex, text


def _heat_step(r: FinResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    if i.shape == "annular":
        latex, text = _annular_heat_latex(r, system)
    else:
        latex, text = _uniform_heat_latex(r, system)
    T_inf = n(i.T_inf_K, "temperature", system)
    if i.tip == "temperature":
        latex.append(rf"T_L = {q(r.T_tip_K, 'temperature', system)}")
    else:
        ratio = (r.T_tip_K - i.T_inf_K) / i.theta_b
        latex.append(
            latex_chain(
                "T_L",
                r"T_\infty + \theta_b\,\frac{\theta_L}{\theta_b}",
                f"{T_inf} + {times(n(i.theta_b, _DT, system), dimensionless(ratio))}",
                q(r.T_tip_K, "temperature", system),
            )
        )
    return ProcedureStep("Calor de la aleta", text, tuple(latex))


def _area_latex(r: FinResult, system: UnitSystem) -> str:
    """El área de la aleta (la de la eficiencia)."""
    i = r.inputs
    A = q(r.A_fin_m2, _A, system)
    if i.shape == "annular":
        rE = n(i.r_base_m + r.L_c_m, _L, system)
        r1 = n(i.r_base_m, _L, system)
        r2 = "r_{2c}" if i.tip == "corrected" else "r_2"
        if i.tip == "convective":
            formula = r"2\pi\,(r_2^2 - r_1^2) + 2\pi\,r_2\,t"
            numbers = (
                rf"2\pi \cdot ({rE}^2 - {r1}^2) \\ &\quad + 2\pi \cdot "
                f"{times(rE, n(i.thickness_m, _L, system))}"
            )
        else:
            formula = rf"2\pi\,({r2}^2 - r_1^2)"
            numbers = rf"2\pi \cdot ({rE}^2 - {r1}^2)"
        return latex_chain("A_f", formula, numbers, A)
    P = math.pi * i.diameter_m if i.shape == "pin" else 2.0 * (i.width_m + i.thickness_m)
    P_n, L_n = n(P, _L, system), n(r.L_c_m, _L, system)
    if i.tip == "convective":
        numbers = rf"{times(P_n, L_n)} \\ &\quad + {n(r.A_base_m2, _A, system)}"
        return latex_chain("A_f", r"P\,L + A_c", numbers, A)
    L_sym = "L_c" if i.tip == "corrected" else "L"
    return latex_chain("A_f", rf"P\,{L_sym}", times(P_n, L_n), A)


def _performance_step(r: FinResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    h, tb = n(i.h_W_per_m2K, _H, system), n(i.theta_b, _DT, system)
    Q = n(r.Q_W, _W, system)
    latex: list[str] = []
    if r.efficiency is not None:
        latex.append(_area_latex(r, system))
        latex.append(
            latex_chain(
                r"\eta_f",
                frac(r"\dot{Q}", r"h\,A_f\,\theta_b"),
                frac(Q, times(h, n(r.A_fin_m2, _A, system), tb)),
                dimensionless(r.efficiency),
            )
        )
    latex.append(
        latex_chain(
            r"\varepsilon_f",
            frac(r"\dot{Q}", r"h\,A_c\,\theta_b"),
            frac(Q, times(h, n(r.A_base_m2, _A, system), tb)),
            dimensionless(r.effectiveness),
        )
    )
    delta_sym = "D/4" if i.shape == "pin" else "t/2"
    delta = i.diameter_m / 4.0 if i.shape == "pin" else i.thickness_m / 2.0
    latex.append(
        latex_chain(
            "Bi",
            frac(rf"h\,({delta_sym})", "k"),
            frac(times(h, n(delta, _L, system)), n(i.k_W_per_mK, _K, system)),
            dimensionless(r.biot),
        )
    )
    text = (
        "La **eficiencia** η compara el calor con el de una aleta ideal toda a T_b (sin "
        "resistencia de conducción); la **efectividad** ε, con el que saldría por la base sin la "
        "aleta: con ε < 2 no se justifica (Incropera, §3.6.3). El modelo unidimensional pide "
        "Bi = h·δ/k < 0,1."
    )
    if r.efficiency is None:
        text = (
            "La aleta infinita no tiene eficiencia (su área es infinita). La **efectividad** ε "
            "compara el calor con el que saldría por la base sin la aleta: con ε < 2 no se "
            "justifica (Incropera, §3.6.3). El modelo unidimensional pide Bi = h·δ/k < 0,1."
        )
    return ProcedureStep("Eficiencia, efectividad y Biot", text, tuple(latex))


def _array_step(a: FinArrayResult, system: UnitSystem) -> ProcedureStep:
    f = a.fin
    i = f.inputs
    N = str(a.inputs.n_fins)
    h, tb = n(i.h_W_per_m2K, _H, system), n(i.theta_b, _DT, system)
    A_b = n(a.inputs.base_area_m2, _A, system)
    A_sin, A_f = n(a.A_unfinned_m2, _A, system), n(f.A_fin_m2, _A, system)
    eta = dimensionless(f.efficiency or 0.0)
    latex = [
        latex_chain(
            r"A_{\text{sin}}",
            r"A_{\text{base}} - N\,A_c",
            sub(A_b, times(N, n(f.A_base_m2, _A, system))),
            q(a.A_unfinned_m2, _A, system),
        ),
        latex_chain(
            "A_t",
            r"N\,A_f + A_{\text{sin}}",
            f"{times(N, A_f)} + {A_sin}",
            q(a.A_total_m2, _A, system),
        ),
        latex_chain(
            r"\dot{Q}_{\text{total}}",
            r"N\,\dot{Q} + h\,A_{\text{sin}}\,\theta_b",
            f"{times(N, n(f.Q_W, _W, system))} \\\\ &\\quad + {times(h, A_sin, tb)}",
            q(a.Q_total_W, _W, system),
        ),
        latex_chain(
            r"\eta_o",
            r"1 - \frac{N\,A_f}{A_t}\,(1 - \eta_f)",
            rf"1 - {frac(times(N, A_f), n(a.A_total_m2, _A, system))} \\ &\quad \cdot (1 - {eta})",
            dimensionless(a.overall_efficiency),
        ),
        latex_chain(
            r"\dot{Q}_{\text{sin aletas}}",
            r"h\,A_{\text{base}}\,\theta_b",
            times(h, A_b, tb),
            q(a.Q_no_fins_W, _W, system),
        ),
        latex_chain(
            r"\varepsilon",
            frac(r"\dot{Q}_{\text{total}}", r"\dot{Q}_{\text{sin aletas}}"),
            frac(n(a.Q_total_W, _W, system), n(a.Q_no_fins_W, _W, system)),
            dimensionless(a.effectiveness),
        ),
    ]
    text = (
        f"El **arreglo** de {N} aletas: la base que queda sin aletas sigue convectando a T_b. La "
        "eficiencia global η_o pondera la de las aletas con la del área sin aletas (que es 1) "
        "(Incropera, §3.6.5)."
    )
    return ProcedureStep("Arreglo de aletas", text, tuple(latex))


def fin_steps(
    result: FinResult, system: UnitSystem, array: FinArrayResult | None = None
) -> list[ProcedureStep]:
    """Los pasos de una aleta (y de su arreglo), con los valores reemplazados en ``system``."""
    steps = [
        _geometry_step(result, system),
        _m_step(result, system),
        _heat_step(result, system),
        _performance_step(result, system),
    ]
    if array is not None:
        steps.append(_array_step(array, system))
    return numbered(steps)

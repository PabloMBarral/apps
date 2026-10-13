"""Tests del procedimiento LaTeX de la conducción transitoria (Fase 11.1).

Que cada paso esté bien formado en los tres sistemas de unidades, que los números que se
muestran sean los del resultado y que los factores de unidades (1000 en el Técnico, 3600 en
el Inglés) estén a la vista. El ancho y KaTeX se verifican aparte, con un navegador.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from core.heat_transfer import transient as tr
from core.heat_transfer import transient_procedure as tp
from core.latex import latex_number, latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")
C = 273.15


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        assert "None" not in st.text and not re.search(r"\be[+-]0", st.text), st.text
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert tex.count(r"\begin{aligned}") == tex.count(r"\end{aligned}"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex  # «+ -3» sin paréntesis
            assert "nan" not in tex and "inf" not in tex.replace(r"\infty", ""), tex
            assert "_Pack" not in tex, tex
    return "\n".join(t for st in steps for t in st.latex)


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Helpers de ancho
# ---------------------------------------------------------------------


def test_vis_counts_fractions_by_their_widest_part() -> None:
    assert tp._vis(r"\frac{1}{2}") == pytest.approx(tp._vis("1") + 1.0)
    assert tp._vis(r"\frac{123456}{1}") > tp._vis("12345")
    assert tp._vis(r"1.5\times 10^{-7}") > tp._vis("1.5")
    assert tp._vis("1 + 2") > tp._vis("12")


def test_pack_breaks_rows_that_do_not_fit() -> None:
    short = tp._pack(tp._prod("2", "3"), 30.0)
    assert short == r"2 \cdot 3"
    long = tp._pack(tp._prod(*["1.2345\\times 10^{-7}"] * 4), 20.0)
    assert long.count(r"\\ &\quad") >= 2
    # los negativos van entre paréntesis después del primero
    assert tp._pack(tp._prod("-2", "-3"), 30.0) == r"-2 \cdot (-3)"


def test_exp_term_survives_underflow() -> None:
    assert tp._exp_term(2.0, 1.0) == latex_number(2.0 * math.exp(-1.0), 3)
    # e^(−1000) es menor que el menor float: el número sale de los logaritmos
    tex = tp._exp_term(-3.0, 1000.0)
    assert tex.startswith("-") and r"\times 10^{-434}" in tex
    assert tp._exp_term(0.5, 5000.0).endswith(r"10^{-2172}")


# ---------------------------------------------------------------------
# Capacidad concentrada
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.LUMPED_EXAMPLES))
def test_lumped_steps(name: str, system: str) -> None:
    r = tr.solve_lumped(tr.LUMPED_EXAMPLES[name].inputs)
    steps = tp.lumped_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[:3] == ["Longitud característica", "Número de Biot", "Constante de tiempo"]
    assert titles[3] == ("Tiempo" if r.inputs.T_target_K is not None else "Temperatura")
    assert titles[4] == "Calor"
    assert _q(r.Lc, "length", system) in tex
    assert latex_number(r.b, 5) in tex
    assert _q(r.time_constant_s, "time", system) in tex
    if r.inputs.T_target_K is not None:
        assert _q(r.time_s, "time", system) in tex
    else:
        assert _q(r.T_K, "temperature", system) in tex
    assert _q(abs(r.Q_J), "energy", system) in tex


@pytest.mark.parametrize("body", list(tr.BODIES))
def test_lumped_bodies_have_their_geometry(body: str) -> None:
    dims = {
        "sphere": (0.02,),
        "cylinder": (0.02, 0.1),
        "long_cylinder": (0.02,),
        "plate": (0.01,),
        "cube": (0.02,),
        "box": (0.02, 0.03, 0.04),
        "custom": (1e-5, 0.006),
    }[body]
    r = tr.solve_lumped(
        tr.LumpedInputs(body, dims, tr.SOLIDS["aluminum"], 50.0, 400.0, 300.0, time_s=60.0)  # type: ignore[arg-type]
    )
    for system in SYSTEMS:
        tex = _check_steps(tp.lumped_steps(r, system))  # type: ignore[arg-type]
        assert _q(r.Lc, "length", system) in tex
        if body == "plate":
            assert r"A_{\text{cara}}" in tex
            assert _q(abs(r.Q_J), "energy_per_area", system) in tex
        elif body == "long_cylinder":
            assert r"\frac{Q}{L}" in tex
            assert _q(abs(r.Q_J), "energy_per_length", system) in tex
        else:
            assert _q(r.mass, "mass", system) in tex


def test_lumped_factor_is_shown() -> None:
    r = tr.solve_lumped(tr.LUMPED_EXAMPLES["Bolas de acero en un horno (ejemplo propio)"].inputs)
    si = _check_steps(tp.lumped_steps(r, "SI"))
    tec = _check_steps(tp.lumped_steps(r, "Técnico"))
    eng = _check_steps(tp.lumped_steps(r, "Inglés"))
    assert r"\frac{100}{7832" in si
    assert r"\frac{100}{1000 \cdot 7832" in tec
    assert "3600" in eng
    # b es el mismo en los tres sistemas (1/s)
    for tex in (si, tec, eng):
        assert latex_number(r.b, 5) + r"\ \mathrm{1/s}" in tex


def test_lumped_text_follows_the_direction_of_the_heat() -> None:
    heating = tr.solve_lumped(
        tr.LUMPED_EXAMPLES["Bolas de acero en un horno (ejemplo propio)"].inputs
    )
    cooling = tr.solve_lumped(
        tr.LUMPED_EXAMPLES["La hora de la muerte (Çengel y Ghajar §4-1)"].inputs
    )
    assert "recibe" in tp.lumped_steps(heating, "Técnico")[-1].text
    assert r"T_\infty - T_i" in "".join(tp.lumped_steps(heating, "Técnico")[-1].latex)
    assert "entrega" in tp.lumped_steps(cooling, "Técnico")[-1].text
    assert r"T_i - T_\infty" in "".join(tp.lumped_steps(cooling, "Técnico")[-1].latex)
    # el Biot de la hora de la muerte pasa de 0,1 y el texto lo dice
    assert "no** es uniforme" in tp.lumped_steps(cooling, "Técnico")[1].text


# ---------------------------------------------------------------------
# Pared, cilindro y esfera
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.ONE_DIM_EXAMPLES))
def test_one_dim_steps(name: str, system: str) -> None:
    r = tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES[name].inputs)
    steps = tp.one_dim_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[:2] == ["Difusividad, Biot y Fourier", "λ₁ y A₁"]
    assert titles[-1] == "Serie completa"
    assert latex_number(r.lambda1, 5) in tex and latex_number(r.A1, 5) in tex
    # la serie da el resultado de la página
    last = "".join(steps[-1].latex)
    assert latex_number(r.theta0, 4) in last
    assert latex_number(r.heat_ratio, 4) in last
    if r.inputs.T_center_target_K is not None:
        assert titles[2] == "Tiempo"
        assert _q(r.time_s, "time", system) in last
        assert _q(r.time_one_term_s or 0.0, "time", system) in tex
    else:
        assert titles[2] == "Temperatura del centro"
        assert _q(r.T_center_K, "temperature", system) in last
    if r.inputs.position > 0.0:
        assert "Temperatura en la posición" in titles
        assert _q(r.T_K, "temperature", system) in last


def test_one_dim_book_numbers_in_the_procedure() -> None:
    egg = tr.solve_one_dimensional(
        tr.ONE_DIM_EXAMPLES["Huevo hervido (Çengel y Ghajar §4-2)"].inputs
    )
    tex = _check_steps(tp.one_dim_steps(egg, "Técnico"))
    # tabla 4-2 de Çengel y Ghajar con Bi = 50: λ₁ = 3.0788, A₁ = 1.9962
    assert r"\lambda_1 &= 3.0788" in tex and "&= 1.9962" in tex
    shaft = tr.solve_one_dimensional(
        tr.ONE_DIM_EXAMPLES["Eje de acero inoxidable (Çengel y Ghajar §4-2)"].inputs
    )
    tex = _check_steps(tp.one_dim_steps(shaft, "Técnico"))
    assert r"364.13\ \mathrm{{}^{\circ}C}" in tex
    assert r"30124\ \mathrm{kJ/m}" in tex  # el libro: 30 120 kJ por metro


@pytest.mark.parametrize("geometry", list(tr.GEOMETRIES))
def test_one_dim_short_time_warns_and_series_differs(geometry: str) -> None:
    r = tr.solve_one_dimensional(
        tr.OneDimInputs(
            geometry, 0.05, tr.SOLIDS["stainless"], 500.0, 600.0 + C, 20.0 + C, time_s=100.0
        )  # type: ignore[arg-type]
    )
    assert r.tau < tr.TAU_ONE_TERM
    steps = tp.one_dim_steps(r, "Técnico")
    _check_steps(steps)
    assert "τ < 0,2" in steps[2].text
    # el segundo término todavía pesa
    series = "".join(steps[-1].latex)
    assert "A_2" in series


def test_one_dim_sphere_uses_s1() -> None:
    r = tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES["Huevo hervido (Çengel y Ghajar §4-2)"].inputs)
    tex = _check_steps(tp.one_dim_steps(r, "SI"))
    assert r"s_1 &= \sin\lambda_1 - \lambda_1\cos\lambda_1" in tex
    assert r"\frac{s_1}{\lambda_1^3}" in tex
    # s₁ sin perder cifras con λ chico (≈ λ³/3)
    assert tp._sphere_s1(1e-4) == pytest.approx((1e-4) ** 3 / 3.0, rel=1e-6)
    assert tp._sphere_s1(0.5) == pytest.approx(math.sin(0.5) - 0.5 * math.cos(0.5))


def test_one_dim_factor_is_shown() -> None:
    r = tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES["Huevo hervido (Çengel y Ghajar §4-2)"].inputs)
    tec = _check_steps(tp.one_dim_steps(r, "Técnico"))
    eng = _check_steps(tp.one_dim_steps(r, "Inglés"))
    assert r"\frac{0.6}{1000 \cdot 1000" in tec
    assert r"{3600 \cdot " in eng
    for system in SYSTEMS:
        tex = _check_steps(tp.one_dim_steps(r, system))  # type: ignore[arg-type]
        assert _q(r.inputs.solid.alpha, "diffusivity", system) in tex


# ---------------------------------------------------------------------
# Semi-infinito y contacto
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.SEMI_INFINITE_EXAMPLES))
def test_semi_infinite_steps(name: str, system: str) -> None:
    r = tr.solve_semi_infinite(tr.SEMI_INFINITE_EXAMPLES[name].inputs)
    steps = tp.semi_infinite_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[1:] == ["Flujo en la superficie y energía", "Profundidad de penetración"]
    assert _q(r.penetration_m, "length", system) in tex
    assert _q(r.energy_per_area, "energy_per_area", system) in tex
    i = r.inputs
    if i.T_target_K is not None and i.depth_m is None:
        assert _q(r.depth_m, "length", system) in tex
    elif i.T_target_K is not None:
        assert _q(r.time_s, "time", system) in tex
    else:
        assert _q(r.T_K, "temperature", system) in tex


def test_semi_infinite_pipes_like_the_book() -> None:
    r = tr.solve_semi_infinite(
        tr.SEMI_INFINITE_EXAMPLES["Caños de agua enterrados (Çengel y Ghajar §4-3)"].inputs
    )
    steps = tp.semi_infinite_steps(r, "Técnico")
    tex = _check_steps(steps)
    # la inversa de erfc en cerrado: η = 0,37 y x = 0,80 m
    assert r"\operatorname{erfc}^{-1}(0.6)" in tex
    assert r"0.80094\ \mathrm{m}" in tex
    assert steps[0].title.endswith("la profundidad")


@pytest.mark.parametrize("boundary", list(tr.SEMI_BOUNDARIES))
@pytest.mark.parametrize("unknown", ["T", "x", "t"])
def test_semi_infinite_every_boundary_and_unknown(boundary: str, unknown: str) -> None:
    kw: dict[str, float] = {"depth_m": 0.03, "time_s": 1800.0, "T_target_K": 60.0 + C}
    kw.pop({"T": "T_target_K", "x": "depth_m", "t": "time_s"}[unknown])
    if boundary == "temperature":
        kw["T_s"] = 200.0 + C
    elif boundary == "flux":
        kw["q_s"] = 3000.0
    else:
        kw.update(h=40.0, T_inf=200.0 + C)
    r = tr.solve_semi_infinite(
        tr.SemiInfiniteInputs(boundary, tr.SOLIDS["concrete"], 20.0 + C, **kw)
    )  # type: ignore[arg-type]
    for system in SYSTEMS:
        steps = tp.semi_infinite_steps(r, system)  # type: ignore[arg-type]
        tex = _check_steps(steps)
        if unknown != "T" and boundary != "temperature":
            assert "numéricamente" in steps[0].text
        assert _q(r.T_K, "temperature", system) in tex or unknown != "T"


def test_semi_infinite_energy_factor() -> None:
    r = tr.solve_semi_infinite(
        tr.SEMI_INFINITE_EXAMPLES["Bloque de madera al sol (Çengel y Ghajar §4-3)"].inputs
    )
    assert r"\frac{1250 \cdot 1200}{1000}" in _check_steps(tp.semi_infinite_steps(r, "Técnico"))
    assert "{3600}" in _check_steps(tp.semi_infinite_steps(r, "Inglés"))
    assert r"1250 \cdot 1200 \\" in _check_steps(tp.semi_infinite_steps(r, "SI"))


def test_semi_infinite_convection_with_huge_exponent() -> None:
    """Con h enorme e^(2ηβ + β²) desborda: el producto se muestra calculado junto."""
    r = tr.solve_semi_infinite(
        tr.SemiInfiniteInputs(
            "convection",
            tr.SOLIDS["concrete"],
            15.0 + C,
            h=1e6,
            T_inf=300.0 + C,
            depth_m=0.5,
            time_s=3600.0,
        )
    )
    for system in SYSTEMS:
        _check_steps(tp.semi_infinite_steps(r, system))  # type: ignore[arg-type]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.CONTACT_EXAMPLES))
def test_contact_steps(name: str, system: str) -> None:
    r = tr.solve_contact(tr.CONTACT_EXAMPLES[name].inputs)
    steps = tp.contact_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Temperatura de contacto", "Flujo por el contacto"]
    assert _q(r.T_contact_K, "temperature", system) in tex
    assert latex_number(r.share_a, 4) in tex
    assert _q(r.flux_W_m2 or 0.0, "heat_flux", system) in tex
    no_time = tr.solve_contact(replace(r.inputs, time_s=None))
    assert _titles(tp.contact_steps(no_time, system)) == ["Temperatura de contacto"]  # type: ignore[arg-type]


def test_contact_effusivity_units() -> None:
    r = tr.solve_contact(tr.CONTACT_EXAMPLES["La mano en madera (ejemplo propio)"].inputs)
    si = _check_steps(tp.contact_steps(r, "SI"))
    tec = _check_steps(tp.contact_steps(r, "Técnico"))
    eng = _check_steps(tp.contact_steps(r, "Inglés"))
    # el SI y el Técnico dan la misma e (con el 1000 a la vista en el Técnico)
    e_a = latex_number(r.inputs.solid_a.effusivity, 5)
    assert e_a in si and e_a in tec and r"\sqrt{1000 \cdot" in tec
    assert "h^{1/2}" in eng
    # la temperatura de contacto no depende de las unidades de e
    assert _q(r.T_contact_K, "temperature", "Inglés") in eng


# ---------------------------------------------------------------------
# Producto
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.PRODUCT_EXAMPLES))
def test_product_steps(name: str, system: str) -> None:
    r = tr.solve_product(tr.PRODUCT_EXAMPLES[name].inputs)
    steps = tp.product_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[0] == "Difusividad"
    assert sum(t.startswith("Factor") for t in titles) == len(r.factors)
    assert "Producto" in titles
    assert latex_number(r.theta, 4) in tex
    assert _q(r.T_K, "temperature", system) in tex
    if r.heat_ratio is None:
        assert titles[-1] == "Producto"
    else:
        assert titles[-1] == "Calor"
        assert latex_number(r.heat_ratio, 4) in tex
        assert _q(abs(r.Q or 0.0), "energy", system) in tex


@pytest.mark.parametrize("shape", list(tr.PRODUCT_SHAPES))
def test_product_every_shape(shape: str) -> None:
    dims = {
        "short_cylinder": (0.1, 0.12),
        "semi_cylinder": (0.2,),
        "bar": (0.1, 0.05),
        "box": (0.2, 0.1, 0.06),
        "semi_plate": (0.1,),
        "quarter": (),
        "corner": (),
    }[shape]
    point = {
        "short_cylinder": (0.5, 1.0),
        "semi_cylinder": (1.0, 0.05),
        "bar": (1.0, 0.0),
        "box": (1.0, 1.0, 0.0),
        "semi_plate": (0.0, 0.02),
        "quarter": (0.01, 0.0),
        "corner": (0.0, 0.01, 0.02),
    }[shape]
    inp = tr.ProductInputs(
        shape,  # type: ignore[arg-type]
        tr.product_factors(shape, dims, point),  # type: ignore[arg-type]
        tr.SOLIDS["brass"],
        60.0,
        120.0 + C,
        25.0 + C,
        900.0,
    )
    r = tr.solve_product(inp)
    for system in SYSTEMS:
        tex = _check_steps(tp.product_steps(r, system))  # type: ignore[arg-type]
        assert latex_number(r.theta, 4) in tex
    # los factores repetidos se numeran
    symbols = tp._factor_symbols(inp.factors)
    assert len(set(symbols)) == len(symbols)


def test_product_langston_rows() -> None:
    r = tr.solve_product(tr.PRODUCT_EXAMPLES["Ladrillo que se enfría (ejemplo propio)"].inputs)
    tex = _check_steps(tp.product_steps(r, "Técnico"))
    assert r"q_3\,(1 - q_1)(1 - q_2)" in tex
    # la solución de un término se aparta de la serie con τ < 0,2: se muestra la serie
    assert r"(\text{serie})" in tex
    ratios = [f.heat_ratio or 0.0 for f in r.factors]
    q1, q2, q3 = ratios
    assert r.heat_ratio == pytest.approx(q1 + q2 * (1 - q1) + q3 * (1 - q1) * (1 - q2))

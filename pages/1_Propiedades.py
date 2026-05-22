"""Página 1 — Calculador de propiedades de un fluido puro.

Calcula el estado completo de un fluido (de la lista de
:data:`core.fluids.SUPPORTED_FLUIDS`) a partir de cualquier par de
variables independientes (T-p, p-h, h-s, p-x, T-x, p-s, T-s). El
sistema de unidades es el seleccionado en el sidebar (SI / Técnico /
Inglés) y se aplica a inputs y outputs por igual; internamente todo
viaja en SI a :func:`core.fluids.state_from_pair`.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Expansor `📖 Fórmulas teóricas` con link al apartado correspondiente
  de vademecum-termo.
- Expansor `🔬 Procedimiento` con las ecuaciones aplicadas en LaTeX y
  los valores reemplazados (modo didáctico).
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

import streamlit as st

from core.fluids import SUPPORTED_FLUIDS, state_from_pair
from ui.branding import SUBJECT, sidebar_credits
from ui.units_ui import number_input_si, quantity_label, render_units_selector

PAGE_VERSION = "0.7.0"

_NO_COHERENT = "Revisá que sean coherentes los valores ingresados, y volvé a intentarlo."


@st.cache_data(show_spinner=False)
def _state_cached(
    pair: str, fluid: str, **kwargs_si: float
) -> tuple[float, float, float, float, float]:
    """Wrapper cacheado sobre :func:`core.fluids.state_from_pair`.

    Devuelve la tupla ``(T_K, P_Pa, h_J_per_kg, s_J_per_kg_K, x)``
    en SI — la UI se encarga de la conversión al sistema del usuario.
    """
    sp = state_from_pair(fluid, pair, **kwargs_si)  # type: ignore[arg-type]
    return sp.T_K, sp.P_Pa, sp.h_J_per_kg, sp.s_J_per_kg_K, sp.x


def _compute_si(
    pair: str, fluid: str, **kwargs_si: float
) -> tuple[float, float, float, float, float] | None:
    """Llama al cache. Si CoolProp falla, muestra el error y devuelve None."""
    try:
        return _state_cached(pair, fluid, **kwargs_si)
    except ValueError as exc:
        st.error(f"Error en el cálculo: {exc}")
        return None


def _render_result_table(
    *,
    state_si: tuple[float, float, float, float, float],
    independent_vars: tuple[str, str],
) -> None:
    """Muestra el estado completo en el sistema actual. ``independent_vars``
    son los dos símbolos del par usado para identificar la fila destacada."""
    T_K, P_Pa, h_J, s_J, x = state_si
    label_T = quantity_label(T_K, "temperature", precision=4)
    label_P = quantity_label(P_Pa, "pressure", precision=4)
    label_h = quantity_label(h_J, "specific_enthalpy", precision=6)
    label_s = quantity_label(s_J, "specific_entropy", precision=6)
    head_a, head_b = independent_vars
    st.success(f"**Resultados** a {head_a} y {head_b}:")
    st.write(f"- Temperatura: **{label_T}**")
    st.write(f"- Presión: **{label_P}**")
    st.write(f"- Entalpía: **{label_h}**")
    st.write(f"- Entropía: **{label_s}**")
    st.write(f"- Título (calidad x): **{x:.4g}** (-1 = fuera de campana)")


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

st.subheader(SUBJECT)

# Selector de fluido top-level (multifluido habilitado en Fase 1.4).
fluid = st.selectbox(
    "Fluido (CoolProp)",
    SUPPORTED_FLUIDS,
    index=0,
    key="prop_fluid",
    help=(
        "El módulo `core.fluids` opera con cualquier fluido reconocido por "
        "CoolProp. Para esta página ofrecemos una selección curada de la "
        "lista canónica del proyecto."
    ),
)

st.title(f"💧 Propiedades de {fluid}")
st.markdown("---")

st.sidebar.title("Seleccioná un par de variables:")
option = st.sidebar.radio(
    "Modo de cálculo",
    (
        "t y p",
        "p y h",
        "h y s",
        "p y x",
        "t y x",
        "p y s",
        "t y s",
    ),
    label_visibility="collapsed",
)

sidebar_credits(version=PAGE_VERSION, page_name="Propiedades")
render_units_selector()


# ---------------------------------------------------------------------
# Modos (7 pares independientes)
# ---------------------------------------------------------------------

if option == "t y p":
    st.write("### Temperatura y Presión")
    with st.form(key="tp_form"):
        t_K = number_input_si(
            label="Ingrese la temperatura",
            kind="temperature",
            default_si=273.15,
            key="tp_t",
            format="%.4f",
        )
        p_Pa = number_input_si(
            label="Ingrese la presión",
            kind="pressure",
            default_si=1.0e5,
            key="tp_p",
            format="%.4f",
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Presión")

    if submit:
        result = _compute_si("TP", fluid, t=t_K, p=p_Pa)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(t_K, "temperature", precision=4),
                    quantity_label(p_Pa, "pressure", precision=4),
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "p y h":
    st.write("### Presión y Entalpía")
    with st.form(key="ph_form"):
        h_J = number_input_si(
            label="Ingrese la entalpía",
            kind="specific_enthalpy",
            default_si=0.0,
            key="ph_h",
            format="%.4f",
        )
        p_Pa = number_input_si(
            label="Ingrese la presión",
            kind="pressure",
            default_si=1.0e5,
            key="ph_p",
            format="%.4f",
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Entalpía")

    if submit:
        result = _compute_si("PH", fluid, p=p_Pa, h=h_J)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(h_J, "specific_enthalpy", precision=6),
                    quantity_label(p_Pa, "pressure", precision=4),
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "h y s":
    st.write("### Entalpía y Entropía")
    with st.form(key="hs_form"):
        h_J = number_input_si(
            label="Ingrese la entalpía",
            kind="specific_enthalpy",
            default_si=0.0,
            key="hs_h",
            format="%.4f",
        )
        s_J = number_input_si(
            label="Ingrese la entropía",
            kind="specific_entropy",
            default_si=0.0,
            key="hs_s",
            format="%.6f",
        )
        submit = st.form_submit_button(label="Calcular desde Entalpía y Entropía")

    if submit:
        result = _compute_si("HS", fluid, h=h_J, s=s_J)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(h_J, "specific_enthalpy", precision=6),
                    quantity_label(s_J, "specific_entropy", precision=6),
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "p y x":
    st.write("### Presión y Título")
    with st.form(key="px_form"):
        p_Pa = number_input_si(
            label="Ingrese la presión",
            kind="pressure",
            default_si=1.0e5,
            key="px_p",
            format="%.4f",
        )
        x = st.number_input(
            "Ingrese el título (calidad del vapor) [0-1]",
            value=0.0,
            step=0.01,
            format="%.4f",
            min_value=0.0,
            max_value=1.0,
            key="px_x",
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Título")

    if submit:
        result = _compute_si("PX", fluid, p=p_Pa, x=x)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(p_Pa, "pressure", precision=4),
                    f"x = {x:.4f}",
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "t y x":
    st.write("### Temperatura y Título")
    with st.form(key="tx_form"):
        t_K = number_input_si(
            label="Ingrese la temperatura",
            kind="temperature",
            default_si=273.15,
            key="tx_t",
            format="%.4f",
        )
        x = st.number_input(
            "Ingrese el título (calidad del vapor) [0-1]",
            value=0.0,
            step=0.01,
            format="%.4f",
            min_value=0.0,
            max_value=1.0,
            key="tx_x",
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Título")

    if submit:
        result = _compute_si("TX", fluid, t=t_K, x=x)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(t_K, "temperature", precision=4),
                    f"x = {x:.4f}",
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "p y s":
    st.write("### Presión y Entropía")
    with st.form(key="ps_form"):
        p_Pa = number_input_si(
            label="Ingrese la presión",
            kind="pressure",
            default_si=1.0e5,
            key="ps_p",
            format="%.4f",
        )
        s_J = number_input_si(
            label="Ingrese la entropía",
            kind="specific_entropy",
            default_si=0.0,
            key="ps_s",
            format="%.6f",
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Entropía")

    if submit:
        result = _compute_si("PS", fluid, p=p_Pa, s=s_J)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(p_Pa, "pressure", precision=4),
                    quantity_label(s_J, "specific_entropy", precision=6),
                ),
            )
        else:
            st.write(_NO_COHERENT)


elif option == "t y s":
    st.write("### Temperatura y Entropía")
    with st.form(key="ts_form"):
        t_K = number_input_si(
            label="Ingrese la temperatura",
            kind="temperature",
            default_si=273.15,
            key="ts_t",
            format="%.4f",
        )
        s_J = number_input_si(
            label="Ingrese la entropía",
            kind="specific_entropy",
            default_si=0.0,
            key="ts_s",
            format="%.6f",
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Entropía")

    if submit:
        result = _compute_si("TS", fluid, t=t_K, s=s_J)
        if result is not None:
            _render_result_table(
                state_si=result,
                independent_vars=(
                    quantity_label(t_K, "temperature", precision=4),
                    quantity_label(s_J, "specific_entropy", precision=6),
                ),
            )
        else:
            st.write(_NO_COHERENT)

"""Página 1 — Calculador de propiedades del agua/vapor.

Migración 1:1 de la calculadora monolítica original a la arquitectura
multipágina. Por ahora solo soporta agua; el módulo `core.fluids` ya está
parametrizado por fluido, así que multifluido entra en Fase 1 cambiando
solo la UI.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Expansor `📖 Fórmulas teóricas` con link al apartado correspondiente
  de vademecum-termo.
- Expansor `🔬 Procedimiento` con las ecuaciones aplicadas en LaTeX y
  los valores reemplazados (modo didáctico).
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

import streamlit as st

from core.fluids import state_from_pair
from core.units import bar_to_pa, c_to_k, j_to_kj, k_to_c, kj_to_j, pa_to_bar
from ui.branding import SUBJECT, sidebar_credits

PAGE_VERSION = "0.6.0"
FLUID = "Water"


@st.cache_data(show_spinner=False)
def _state_cached(
    pair: str, fluid: str, **kwargs_si: float
) -> tuple[float, float, float, float, float]:
    """Wrapper cacheado sobre :func:`core.fluids.state_from_pair`.

    Devuelve una tupla ``(T_K, P_Pa, h_J_per_kg, s_J_per_kg_K, x)`` —
    una tupla en vez del dataclass para que el cache de Streamlit pueda
    serializarla sin sobresaltos.
    """
    sp = state_from_pair(fluid, pair, **kwargs_si)  # type: ignore[arg-type]
    return sp.T_K, sp.P_Pa, sp.h_J_per_kg, sp.s_J_per_kg_K, sp.x


def _compute_in_didactic_units(
    pair: str, **kwargs_si: float
) -> tuple[float, float, float, float, float] | None:
    """Llama al cache y devuelve los resultados en unidades didácticas.

    Convierte de SI a (°C, bar, kJ/kg, kJ/(kg·K), -). Si CoolProp falla,
    muestra el error en pantalla y devuelve ``None`` — la página puede
    así emitir el mensaje genérico de revisión.
    """
    try:
        T_K, P_Pa, h_J, s_J, x = _state_cached(pair, FLUID, **kwargs_si)
    except ValueError as exc:
        st.error(f"Error en el cálculo: {exc}")
        return None
    return k_to_c(T_K), pa_to_bar(P_Pa), j_to_kj(h_J), j_to_kj(s_J), x


st.subheader(SUBJECT)
st.title("💧 Calculador de propiedades del agua")
st.markdown("---")

st.sidebar.title("Seleccioná una opción:")
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


_NO_COHERENT = "Revisá que sean coherentes los valores ingresados, y volvé a intentarlo."


if option == "t y p":
    st.write("### Temperatura y Presión")
    with st.form(key="tp_form"):
        t = st.number_input(
            "Ingrese la temperatura [°C]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        p = st.number_input(
            "Ingrese la presión [bar(a)]",
            value=1.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Presión")

    if submit:
        result = _compute_in_didactic_units("TP", t=c_to_k(t), p=bar_to_pa(p))
        if result is not None:
            T_C, P_bar, h, s, x = result
            st.write(f"Resultados a {T_C:.2f} °C y {P_bar:.2f} bar(a):")
            st.write(f"Entalpía: {h:.2f} kJ/kg")
            st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
            st.write(f"Título: {x:.2f}")
        else:
            st.write(_NO_COHERENT)


elif option == "p y h":
    st.write("### Presión y Entalpía")
    with st.form(key="ph_form"):
        h = st.number_input(
            "Ingrese la entalpía [kJ/kg]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        p = st.number_input(
            "Ingrese la presión [bar(a)]",
            value=1.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Entalpía")

    if submit:
        result = _compute_in_didactic_units("PH", p=bar_to_pa(p), h=kj_to_j(h))
        if result is not None:
            T_C, P_bar, h_out, s, x = result
            st.write(f"Resultados a {h:.2f} kJ/kg y {P_bar:.2f} bar(a):")
            st.write(f"Temperatura: {T_C:.2f} °C")
            st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
            st.write(f"Título: {x:.2f}")
        else:
            st.write(_NO_COHERENT)


elif option == "h y s":
    st.write("### Entalpía y Entropía")
    with st.form(key="hs_form"):
        h = st.number_input(
            "Ingrese la entalpía [kJ/kg]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        s = st.number_input(
            "Ingrese la entropía [kJ/(kg·K)]",
            value=0.0,
            step=0.01,
            format="%.4f",
            min_value=0.0,
        )
        submit = st.form_submit_button(label="Calcular desde Entalpía y Entropía")

    if submit:
        result = _compute_in_didactic_units("HS", h=kj_to_j(h), s=kj_to_j(s))
        if result is not None:
            T_C, P_bar, h_out, s_out, x = result
            st.write(f"Resultados a {h:.2f} kJ/kg y {s:.4f} kJ/(kg·K):")
            st.write(f"Temperatura: {T_C:.2f} °C")
            st.write(f"Presión: {P_bar:.2f} bar(a)")
            st.write(f"Título: {x:.2f}")
        else:
            st.write(_NO_COHERENT)


elif option == "p y x":
    st.write("### Presión y Título")
    with st.form(key="px_form"):
        p = st.number_input(
            "Ingrese la presión [bar(a)]",
            value=1.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        x = st.number_input(
            "Ingrese el título (calidad del vapor) [0-1]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Título")

    if submit:
        result = _compute_in_didactic_units("PX", p=bar_to_pa(p), x=x)
        if result is not None:
            T_C, P_bar, h, s, x_out = result
            st.write(f"Resultados a {P_bar:.2f} bar(a) y {x:.2f}:")
            st.write(f"Temperatura: {T_C:.2f} °C")
            st.write(f"Entalpía: {h:.2f} kJ/kg")
            st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
        else:
            st.write(_NO_COHERENT)


elif option == "t y x":
    st.write("### Temperatura y Título")
    with st.form(key="tx_form"):
        t = st.number_input(
            "Ingrese la temperatura [°C]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        x = st.number_input(
            "Ingrese el título (calidad del vapor) [0-1]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Título")

    if submit:
        result = _compute_in_didactic_units("TX", t=c_to_k(t), x=x)
        if result is not None:
            T_C, P_bar, h, s, x_out = result
            st.write(f"Resultados a {T_C:.2f} °C y {x:.2f}:")
            st.write(f"Presión: {P_bar:.2f} bar(a)")
            st.write(f"Entalpía: {h:.2f} kJ/kg")
            st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
        else:
            st.write(_NO_COHERENT)


elif option == "p y s":
    st.write("### Presión y Entropía")
    with st.form(key="ps_form"):
        p = st.number_input(
            "Ingrese la presión [bar(a)]",
            value=1.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        s = st.number_input(
            "Ingrese la entropía [kJ/(kg·K)]",
            value=0.0,
            step=0.01,
            format="%.4f",
            min_value=0.0,
        )
        submit = st.form_submit_button(label="Calcular desde Presión y Entropía")

    if submit:
        result = _compute_in_didactic_units("PS", p=bar_to_pa(p), s=kj_to_j(s))
        if result is not None:
            T_C, P_bar, h, s_out, x = result
            st.write(f"Resultados a {P_bar:.2f} bar(a) y {s:.4f} kJ/(kg·K):")
            st.write(f"Temperatura: {T_C:.2f} °C")
            st.write(f"Entalpía: {h:.2f} kJ/kg")
            st.write(f"Título: {x:.2f}")
        else:
            st.write(_NO_COHERENT)


elif option == "t y s":
    st.write("### Temperatura y Entropía")
    with st.form(key="ts_form"):
        t = st.number_input(
            "Ingrese la temperatura [°C]",
            value=0.0,
            step=0.01,
            format="%.2f",
            min_value=0.0,
        )
        s = st.number_input(
            "Ingrese la entropía [kJ/(kg·K)]",
            value=0.0,
            step=0.01,
            format="%.4f",
            min_value=0.0,
        )
        submit = st.form_submit_button(label="Calcular desde Temperatura y Entropía")

    if submit:
        result = _compute_in_didactic_units("TS", t=c_to_k(t), s=kj_to_j(s))
        if result is not None:
            T_C, P_bar, h, s_out, x = result
            st.write(f"Resultados a {T_C:.2f} °C y {s:.4f} kJ/(kg·K):")
            st.write(f"Presión: {P_bar:.2f} bar(a)")
            st.write(f"Entalpía: {h:.2f} kJ/kg")
            st.write(f"Título: {x:.2f}")
        else:
            st.write(_NO_COHERENT)

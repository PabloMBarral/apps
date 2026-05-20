"""Entry point — TA216 Apps.

Define la navegación explícita con :func:`st.navigation` y :class:`st.Page`
(API introducida en Streamlit 1.36). Los archivos físicos en ``pages/``
no se renombran: se referencian por path con un label e ícono custom.
"""

from __future__ import annotations

import streamlit as st

from ui.branding import SUBJECT, sidebar_credits

PAGE_VERSION = "0.4.0"


def _home_page() -> None:
    st.set_page_config(
        page_title="TA216 Apps",
        page_icon="🏠",
        layout="centered",
    )
    st.subheader("Facultad de Ingeniería — UBA")
    st.title("🏠 TA216 — Apps")
    st.markdown("---")

    st.markdown(
        f"""
        Suite de herramientas didácticas de ingeniería térmica para la
        materia **{SUBJECT}** (FIUBA).

        👈 Elegí una página del menú lateral para empezar.

        Las **fórmulas teóricas** que sustentan estos cálculos viven en el
        repo hermano
        [**vademecum-termo**](https://github.com/PabloMBarral/vademecum-termo).
        Pensalos como un único material: el vademecum explica, este repo
        calcula.

        ---

        ### Módulos disponibles

        - 💧 **Propiedades** — agua/vapor a partir de cualquier par de
          variables independientes (T-p, p-h, h-s, p-x, T-x, p-s, T-s).
        - 📐 **Interpolación** — lineal simple y doble entrada (bilineal)
          sobre tablas, con procedimiento didáctico paso a paso y
          comparación opcional contra CoolProp.

        ### En desarrollo

        Isoentrópicos, diagramas, ciclos Rankine / refrigeración /
        Brayton / combinado, psicrometría, combustión, poder calorífico,
        ISO 6976, exergía. Ver el
        [README](https://github.com/PabloMBarral/apps#m%C3%B3dulos) para
        el roadmap completo.
        """
    )

    sidebar_credits(version=PAGE_VERSION, page_name="Home")


pages = [
    st.Page(_home_page, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Propiedades.py", title="Propiedades", icon="💧"),
    st.Page("pages/2_Interpolacion.py", title="Interpolación", icon="📐"),
]
pg = st.navigation(pages)
pg.run()

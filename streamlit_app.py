"""Home / landing — Tecnología del Calor — Apps."""

import streamlit as st

st.set_page_config(
    page_title="Tecnología del Calor",
    page_icon="🔥",
    layout="centered",
)

st.subheader("Facultad de Ingeniería — UBA")
st.title("🔥 Tecnología del Calor — Apps")
st.markdown("---")

st.markdown(
    """
    Suite de herramientas didácticas de ingeniería térmica para la
    materia **Tecnología del Calor** (FIUBA).

    👈 Elegí una página del menú lateral para empezar.

    Las **fórmulas teóricas** que sustentan estos cálculos viven en el repo
    hermano [**vademecum-termo**](https://github.com/PabloMBarral/vademecum-termo).
    Pensalos como un único material: el vademecum explica, este repo calcula.

    ---

    ### Módulos disponibles

    - **1 · Propiedades** — agua/vapor a partir de cualquier par de variables
      independientes (T-p, p-h, h-s, p-x, T-x, p-s, T-s).

    ### En desarrollo

    Interpolación, isoentrópicos, diagramas, ciclos Rankine /
    refrigeración / Brayton / combinado, psicrometría, combustión,
    poder calorífico, ISO 6976, exergía. Ver el
    [README](https://github.com/PabloMBarral/apps#m%C3%B3dulos) para el
    roadmap completo.
    """
)

st.sidebar.markdown("### Acerca")
st.sidebar.write("Desarrollado por **Pablo M. Barral** para Tecnología del Calor — FIUBA.")
st.sidebar.write("Contacto: pbarral@fi.uba.ar")
st.sidebar.write("ORCID 0000-0003-1125-4199")
st.sidebar.write("Versión: 0.2.0 — Fase 0")
st.sidebar.markdown("[Repo en GitHub](https://github.com/PabloMBarral/apps)")

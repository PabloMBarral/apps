"""Bloque de créditos compartido para el sidebar de todas las páginas.

Centraliza autor, materia, contacto, ORCID, LinkedIn y versión. Se llama
una vez por página desde su top-level. La idea es que cambios futuros
(nuevo link, nuevo perfil, bump de versión) toquen un solo archivo.
"""

from __future__ import annotations

import streamlit as st

AUTHOR_NAME = "Pablo M. Barral"
SUBJECT = "TA216 — Tecnología de Calor Avanzada"
CONTACT_EMAIL = "pbarral@fi.uba.ar"
ORCID_URL = "https://orcid.org/0000-0003-1125-4199"
ORCID_ID = "0000-0003-1125-4199"
LINKEDIN_URL = "https://www.linkedin.com/in/pablo-barral/"
LINKEDIN_HANDLE = "pablo-barral"
GITHUB_REPO_URL = "https://github.com/PabloMBarral/apps"
README_URL = "https://github.com/PabloMBarral/apps/blob/main/README.md"


def sidebar_credits(*, version: str, page_name: str | None = None) -> None:
    """Renderiza el bloque de créditos en ``st.sidebar``.

    Parámetros
    ----------
    version : str
        Etiqueta de versión a mostrar (ej. ``"0.4.0"``).
    page_name : str, opcional
        Nombre de la página actual; se concatena después de la versión
        para que el alumno sepa dónde está parado.
    """
    st.sidebar.markdown("### Acerca")
    st.sidebar.markdown(f"Desarrollado por **{AUTHOR_NAME}** para **{SUBJECT}**.")
    st.sidebar.markdown(
        f"[ORCID {ORCID_ID}]({ORCID_URL})  ·  [LinkedIn @{LINKEDIN_HANDLE}]({LINKEDIN_URL})"
    )
    st.sidebar.markdown(f"Contacto: [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})")
    version_label = f"Versión: {version}"
    if page_name:
        version_label = f"{version_label} — {page_name}"
    st.sidebar.markdown(version_label)
    st.sidebar.markdown(f"[Repo en GitHub]({GITHUB_REPO_URL})  ·  [README]({README_URL})")
    st.sidebar.caption("Powered by CoolProp.")

"""Página Acerca de: créditos, cómo citar la app, sus fuentes y sus licencias.

Todo sale del ``CITATION.cff`` del repositorio (``core.citation``): la versión,
los autores y las referencias, así que actualizar el archivo actualiza la
página. Muestra también las licencias (la de la app y las de los datos de
terceros) y las versiones instaladas de las librerías, para reportar un error.

No es un módulo de cálculo: no lleva fórmulas teóricas ni procedimiento, y lo
que se descarga es el ``CITATION.cff`` y la bibliografía en BibTeX.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core import citation as ct
from ui.branding import (
    GITHUB_REPO_URL,
    LINKEDIN_URL,
    README_URL,
    SUBJECT,
    VADEMECUM_DOI_URL,
    VADEMECUM_PDF_URL,
    VADEMECUM_REPO_URL,
    sidebar_credits,
)
from ui.units_ui import render_units_selector

PAGE_VERSION = "0.25.1"

ROOT = Path(__file__).resolve().parents[1]
ALL_KINDS = "Todas"


# ---------------------------------------------------------------------
# Datos (cacheados: el CITATION.cff tarda ~60 ms en leerse)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _citation(mtime_ns: int) -> ct.Citation:
    """El CITATION.cff; la fecha de modificación invalida el caché si cambia."""
    del mtime_ns
    return ct.load_citation()


@st.cache_data(show_spinner=False)
def _bibliography(mtime_ns: int) -> str:
    return ct.bibliography_bibtex(_citation(mtime_ns))


@st.cache_data(show_spinner=False)
def _versions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "Librería": "Python",
            "Versión": ct.python_version(),
            "Licencia": "PSF-2.0",
            "Para qué": "El intérprete",
        }
    ]
    for name, use in ct.LIBRARIES:
        info = ct.package_info(name)
        rows.append(
            {
                "Librería": name,
                "Versión": info.version or "no instalada",
                "Licencia": info.license,
                "Para qué": use,
            }
        )
    return rows


def _date(iso: str) -> str:
    """«2026-10-10» → «10/10/2026»."""
    try:
        return date.fromisoformat(iso).strftime("%d/%m/%Y")
    except ValueError:
        return iso or "—"


# ---------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------


def _render_app(cit: ct.Citation) -> None:
    st.markdown("### La app")
    if cit.abstract:
        st.markdown(ct.markdown_escape(cit.abstract))
    left, right = st.columns(2)
    left.metric("Versión", cit.version)
    right.metric("Publicada", _date(cit.date_released))
    left, right = st.columns(2)
    left.metric("Licencia", cit.license or "—")
    right.metric("Fuentes citadas", len(cit.references))

    for person in cit.authors:
        line = f"**{ct.markdown_escape(person.full_name())}**"
        if person.affiliation:
            line += f" — {ct.markdown_escape(person.affiliation)}"
        links = []
        if person.orcid:
            orcid_id = person.orcid.rstrip("/").rsplit("/", 1)[-1]
            links.append(f"[ORCID {orcid_id}]({person.orcid})")
        if person.email:
            links.append(f"[{person.email}](mailto:{person.email})")
        st.markdown(line + ("  \n" + " · ".join(links) if links else ""))

    doi = VADEMECUM_DOI_URL.removeprefix("https://doi.org/")
    st.markdown(
        f"El código está en [GitHub]({GITHUB_REPO_URL}) (con el [README]({README_URL})) y "
        f"las fórmulas teóricas, en el **vademecum** ([PDF]({VADEMECUM_PDF_URL}), "
        f"[repositorio]({VADEMECUM_REPO_URL}), DOI [{doi}]({VADEMECUM_DOI_URL})). "
        f"El autor también está en [LinkedIn]({LINKEDIN_URL})."
    )


def _render_how_to_cite(cit: ct.Citation, bibliography: str) -> None:
    st.markdown("### Cómo citar la app")
    st.markdown(
        "Si usás la app en investigación, docencia o publicaciones, citala así, y citá "
        "también las fuentes que correspondan (están más abajo):"
    )
    st.markdown("> " + ct.citation_apa(cit, markdown=True))
    st.caption("Para copiar (APA, 7.ª edición):")
    st.code(ct.citation_apa(cit), language=None, wrap_lines=True)
    st.caption("En BibTeX:")
    st.code(ct.citation_bibtex(cit), language="latex")
    left, right = st.columns(2)
    left.download_button(
        "Descargar CITATION.cff",
        data=ct.CITATION_PATH.read_bytes(),
        file_name="CITATION.cff",
        mime="text/plain",
        key="ab_cff",
    )
    right.download_button(
        "Descargar la bibliografía (.bib)",
        data=bibliography,
        file_name="ta216-apps.bib",
        mime="text/plain",
        key="ab_bib",
    )
    st.caption(
        "La bibliografía trae la app y todas las fuentes de abajo. En GitHub, el botón "
        "«Cite this repository» (a la derecha del repositorio) da la misma cita."
    )


def _render_references(cit: ct.Citation) -> None:
    st.markdown("### Las fuentes")
    st.markdown(
        f"La app se apoya en **{len(cit.references)} fuentes**: las librerías con las que "
        "calcula, los artículos de las correlaciones, los libros de los ejemplos y las "
        "normas. Si publicás resultados hechos con la app, citá también las que "
        "correspondan."
    )
    groups = ct.group_references(cit.references)
    labels = {ALL_KINDS: f"Todas ({len(cit.references)})"}
    labels.update({label: f"{label} ({len(refs)})" for label, refs in groups.items()})
    kind = st.selectbox("Tipo", list(labels), format_func=labels.get, key="ab_kind")
    query = st.text_input(
        "Buscar",
        key="ab_query",
        placeholder="Un autor, una palabra del título o para qué se usa",
    )
    refs = [
        r
        for r in cit.references
        if (kind == ALL_KINDS or r.kind_label == kind) and ct.reference_matches(r, query)
    ]
    if not refs:
        st.info(f"Ninguna fuente coincide con «{query.strip()}». Probá con otra palabra.")
        return
    if len(refs) < len(cit.references):
        st.caption(f"{len(refs)} de {len(cit.references)} fuentes.")
    for label, items in ct.group_references(refs).items():
        st.markdown(f"#### {label} ({len(items)})")
        for ref in items:
            text = ct.reference_apa(ref, markdown=True)
            if ref.scope:
                text += f"  \n**Para qué:** {ct.markdown_escape(ref.scope)}"
            st.markdown(text)


def _render_licenses(cit: ct.Citation) -> None:
    st.markdown("### Licencias")
    st.markdown(
        f"La app se distribuye con la licencia **{cit.license or 'MIT'}**: se puede usar, "
        "copiar, modificar y distribuir libremente, siempre que se conserve el aviso de "
        "copyright y de la licencia, y se ofrece sin garantía. Las librerías conservan las "
        "suyas (están en las versiones instaladas, abajo) y los datos de terceros que trae "
        "el repositorio, también:"
    )
    st.markdown(
        "\n".join(
            f"- **{ct.markdown_escape(s.path.removeprefix('data/'))}** — "
            f"{ct.markdown_escape(s.source)}. *{ct.markdown_escape(s.license)}*"
            for s in ct.DATA_SOURCES
        )
    )
    with st.expander("📄 Licencia de la app (MIT)"):
        st.code((ROOT / "LICENSE").read_text(encoding="utf-8"), language=None, wrap_lines=True)
    with st.expander("📄 Licencia de los datos de las biomasas (modeldata, MIT)"):
        st.code(
            (ROOT / "data" / "LICENSE-modeldata.txt").read_text(encoding="utf-8"),
            language=None,
            wrap_lines=True,
        )


def _render_versions() -> None:
    st.markdown("### Versiones instaladas")
    rows = _versions()
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * (len(rows) + 1) + 3
    )
    st.caption(
        f"Si encontrás un error, contalo en un [issue del repositorio]({GITHUB_REPO_URL}/issues) "
        "con esta tabla: algunos resultados cambian con la versión de CoolProp o de TESPy."
    )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Acerca de", page_icon="ℹ️", layout="centered")

st.subheader(SUBJECT)
st.title("ℹ️ Acerca de")
st.markdown(
    "Quién hizo la app, cómo citarla, en qué fuentes se apoya y con qué licencias se "
    "distribuye. Todo sale del `CITATION.cff` del repositorio."
)
st.markdown("---")

_mtime = ct.CITATION_PATH.stat().st_mtime_ns
_cit = _citation(_mtime)
_render_app(_cit)
st.markdown("---")
_render_how_to_cite(_cit, _bibliography(_mtime))
st.markdown("---")
_render_references(_cit)
st.markdown("---")
_render_licenses(_cit)
st.markdown("---")
_render_versions()

sidebar_credits(version=PAGE_VERSION, page_name="Acerca de")
render_units_selector()

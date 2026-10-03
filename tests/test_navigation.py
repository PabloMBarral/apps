"""Navegación de la app (``streamlit_app.py`` + ``st.navigation``).

Regresión: con una carpeta ``pages/`` en la raíz, Streamlit arrancaba en
el modo multipágina viejo hasta que corría ``st.navigation``; un link
directo a una página justo después de un reinicio mostraba el menú con
nombres de archivo ("streamlit app", "Interpolacion"). Las páginas viven
ahora en ``app_pages/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "streamlit_app.py"
PAGE_FILES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app_pages").glob("*.py"))


def test_no_legacy_pages_directory() -> None:
    assert not (ROOT / "pages").exists(), (
        "No crear una carpeta pages/: activa el modo multipágina viejo de Streamlit "
        "(ver CLAUDE.md, sección Navegación)."
    )


def test_every_page_file_is_registered_in_navigation() -> None:
    registered = re.findall(r'st\.Page\("([^"]+)"', MAIN.read_text(encoding="utf-8"))
    assert sorted(registered) == PAGE_FILES


def test_home_runs() -> None:
    at = AppTest.from_file(str(MAIN), default_timeout=120).run()
    assert not at.exception
    assert any("TA216" in t.value for t in at.title)


@pytest.mark.parametrize("page", PAGE_FILES)
def test_every_page_runs_through_navigation(page: str) -> None:
    at = AppTest.from_file(str(MAIN), default_timeout=120).run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]

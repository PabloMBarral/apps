"""Tests mínimos para :mod:`ui.branding`.

No testeamos rendering (Streamlit no se ejecuta en pytest sin AppTest);
solo verificamos que las constantes están y que la función es
invocable con la signatura esperada.
"""

from __future__ import annotations

import inspect

from ui import branding


def test_author_metadata_present() -> None:
    assert branding.AUTHOR_NAME == "Pablo M. Barral"
    assert branding.SUBJECT == "TA216 — Tecnología de Calor Avanzada"
    assert "pbarral" in branding.CONTACT_EMAIL
    assert branding.ORCID_URL.startswith("https://orcid.org/")
    assert "linkedin.com" in branding.LINKEDIN_URL
    assert branding.GITHUB_REPO_URL.startswith("https://github.com/")


def test_sidebar_credits_is_callable() -> None:
    assert callable(branding.sidebar_credits)


def test_sidebar_credits_signature() -> None:
    sig = inspect.signature(branding.sidebar_credits)
    assert "version" in sig.parameters
    assert "page_name" in sig.parameters
    # Ambos parámetros deben ser keyword-only.
    assert sig.parameters["version"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["page_name"].kind == inspect.Parameter.KEYWORD_ONLY
    # page_name debe tener default (= None).
    assert sig.parameters["page_name"].default is None

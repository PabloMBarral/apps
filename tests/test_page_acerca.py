"""Tests de /Acerca (AppTest): la cita, las fuentes, las licencias y las versiones."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core import citation as ct

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "99_Acerca.py")


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _references(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if "**Para qué:**" in m.value]


def test_shows_the_citation_from_the_cff() -> None:
    at = _app()
    cit = ct.load_citation()
    assert at.title[0].value == "ℹ️ Acerca de"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Versión"] == cit.version
    assert metrics["Fuentes citadas"] == str(len(cit.references))
    assert at.code[0].value == ct.citation_apa(cit)
    assert at.code[1].value == ct.citation_bibtex(cit)
    assert len(at.get("download_button")) == 2


def test_lists_every_reference_grouped_by_kind() -> None:
    at = _app()
    cit = ct.load_citation()
    assert len(_references(at)) == len(cit.references)
    headings = [m.value for m in at.markdown if m.value.startswith("#### ")]
    expected = [f"#### {k} ({len(v)})" for k, v in ct.group_references(cit.references).items()]
    assert headings == expected


def test_filter_by_kind_and_search() -> None:
    at = _app()
    at.selectbox(key="ab_kind").set_value("Normas").run()
    _no_problems(at)
    assert len(_references(at)) == 5
    at.text_input(key="ab_query").input("ASTM").run()
    assert len(_references(at)) == 3
    at.selectbox(key="ab_kind").set_value("Todas").run()
    at.text_input(key="ab_query").input("cengel ghajar").run()
    found = _references(at)
    assert len(found) == 1 and "Çengel" in found[0]
    at.text_input(key="ab_query").input("palabra-que-no-esta").run()
    _no_problems(at)
    assert not _references(at)
    assert "Ninguna fuente coincide" in at.info[0].value


def test_licenses_and_versions() -> None:
    at = _app()
    assert [e.label for e in at.expander] == [
        "📄 Licencia de la app (MIT)",
        "📄 Licencia de los datos de las biomasas (modeldata, MIT)",
    ]
    assert at.code[2].value.startswith("MIT License")
    table = at.dataframe[0].value
    assert list(table["Librería"])[:3] == ["Python", "streamlit", "CoolProp"]
    assert "no instalada" not in set(table["Versión"])
    data_list = next(m.value for m in at.markdown if "nasa9\\_thermo.csv" in m.value)
    assert data_list.count("\n- ") == len(ct.DATA_SOURCES) - 1

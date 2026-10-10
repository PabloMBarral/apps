"""Tests de core.citation (página /Acerca): el CITATION.cff, las citas y los datos."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import citation as ct

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cit() -> ct.Citation:
    return ct.load_citation()


def _ref(cit: ct.Citation, fragment: str) -> ct.Reference:
    return next(r for r in cit.references if fragment in r.title)


# ---------------------------------------------------------------------
# Lectura del CITATION.cff
# ---------------------------------------------------------------------


def test_reads_the_repository_citation(cit: ct.Citation) -> None:
    assert cit.title == "TA216 — Tecnología de Calor Avanzada — Apps"
    assert cit.license == "MIT"
    assert cit.year == int(cit.date_released[:4])
    assert cit.url == "https://github.com/PabloMBarral/apps"
    author = cit.authors[0]
    assert (author.family, author.given) == ("Barral", "Pablo M.")
    assert author.orcid.endswith("0000-0003-1125-4199")
    assert len(cit.references) >= 62


def test_version_matches_the_home_page(cit: ct.Citation) -> None:
    text = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    match = re.search(r'PAGE_VERSION = "([^"]+)"', text)
    assert match is not None
    assert cit.version == match.group(1)


def test_every_reference_is_complete(cit: ct.Citation) -> None:
    for ref in cit.references:
        assert ref.title and ref.scope and ref.authors, ref
        assert ref.kind in ct.REFERENCE_KINDS, ref.kind


def test_no_unquoted_numbers_with_a_leading_zero() -> None:
    """YAML lee 025010 en octal (10760): un número con cero adelante va entre comillas."""
    text = ct.CITATION_PATH.read_text(encoding="utf-8")
    assert not re.search(r"^\s+[\w-]+:\s+0\d+\s*$", text, re.M)


def test_codata_article_number_keeps_its_zero(cit: ct.Citation) -> None:
    assert _ref(cit, "CODATA").start == "025010"


def test_parse_errors_are_explained() -> None:
    with pytest.raises(ValueError, match="le falta: authors, version"):
        ct.parse_citation("title: Algo\n")
    with pytest.raises(ValueError, match="mapa YAML"):
        ct.parse_citation("- una\n- lista\n")
    with pytest.raises(ValueError, match="YAML válido"):
        ct.parse_citation("title: [sin cerrar\n")


def test_minimal_citation_without_references() -> None:
    c = ct.parse_citation(
        'title: "Mi app"\nversion: 1.0\nauthors:\n  - family-names: Pérez\n    given-names: Ana\n'
    )
    assert c.version == "1.0" and c.references == () and c.year is None
    assert ct.citation_apa(c) == "Pérez, A. (s. f.). Mi app (Versión 1.0) [Software]."


# ---------------------------------------------------------------------
# Autores
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "initials"),
    [("Ian H.", "I. H."), ("Jean-Pierre", "J.-P."), ("Wolfgang", "W."), ("", "")],
)
def test_initials(given: str, initials: str) -> None:
    assert ct.Person(family="X", given=given).initials == initials


def test_person_forms() -> None:
    person = ct.Person(family="Chase", given="Malcolm W.", suffix="Jr.")
    assert person.apa() == "Chase, M. W., Jr."
    assert person.bibtex() == "Chase, Jr., Malcolm W."
    assert person.full_name() == "Malcolm W. Chase Jr."
    entity = ct.Person(name="ASTM International")
    assert entity.is_entity
    assert entity.apa() == entity.full_name() == "ASTM International"
    assert entity.bibtex() == "{ASTM International}"


def test_author_lists_use_y() -> None:
    a, b, c = (ct.Person(family=f, given="Ana") for f in ("Uno", "Dos", "Tres"))
    assert ct.authors_apa([]) == ""
    assert ct.authors_apa([a]) == "Uno, A."
    assert ct.authors_apa([a, b]) == "Uno, A. y Dos, A."
    assert ct.authors_apa([a, b, c]) == "Uno, A., Dos, A. y Tres, A."


# ---------------------------------------------------------------------
# APA
# ---------------------------------------------------------------------


def test_app_citation_matches_the_readme(cit: ct.Citation) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Cita sugerida:\n\n((?:> .*\n)+)", readme)
    assert match is not None
    quoted = " ".join(line.removeprefix("> ").strip() for line in match.group(1).splitlines())
    assert quoted.replace("*", "") == ct.citation_apa(cit)
    assert ct.citation_apa(cit, markdown=True).startswith(
        "Barral, P. M. (" + str(cit.year) + "). *TA216 — Tecnología de Calor Avanzada — Apps*"
    )


def test_article(cit: ct.Citation) -> None:
    assert ct.reference_apa(_ref(cit, "CoolProp")) == (
        "Bell, I. H., Wronski, J., Quoilin, S. y Lemort, V. (2014). Pure and Pseudo-pure Fluid "
        "Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library "
        "CoolProp. Industrial & Engineering Chemistry Research, 53(6), 2498–2508. "
        "https://doi.org/10.1021/ie4033999"
    )
    # un número de artículo en vez de páginas
    assert "Reviews of Modern Physics, 93(2), 025010." in ct.reference_apa(_ref(cit, "CODATA"))


def test_article_in_markdown(cit: ct.Citation) -> None:
    md = ct.reference_apa(_ref(cit, "CoolProp"), markdown=True)
    assert "*Industrial & Engineering Chemistry Research*, *53*(6), 2498–2508." in md
    assert md.endswith("[https://doi.org/10.1021/ie4033999](https://doi.org/10.1021/ie4033999)")


def test_books(cit: ct.Citation) -> None:
    assert ct.reference_apa(_ref(cit, "Heat and Mass Transfer: Fundamentals")) == (
        "Çengel, Y. A. y Ghajar, A. J. (2015). Heat and Mass Transfer: Fundamentals and "
        "Applications (5.ª ed.). McGraw-Hill Education."
    )
    janaf = ct.reference_apa(_ref(cit, "NIST-JANAF"))
    assert janaf.startswith("Chase, M. W., Jr. (1998). NIST-JANAF Thermochemical Tables (4.ª ed.).")


def test_standard_by_an_entity(cit: ct.Citation) -> None:
    iso = ct.reference_apa(_ref(cit, "Natural gas"))
    assert iso.startswith("International Organization for Standardization. (2016). Natural gas")
    # la entidad que firma no se repite como editora
    assert iso.endswith("(ISO 6976:2016).")


def test_report_conference_and_software(cit: ct.Citation) -> None:
    assert ct.reference_apa(_ref(cit, "NASA Glenn Coefficients")).endswith(
        "(NASA/TP-2002-211556). NASA Glenn Research Center."
    )
    assert ct.reference_apa(_ref(cit, "cross-flow")) == (
        "Mason, J. L. (1955). Heat transfer in cross-flow. En Proceedings of the Second U.S. "
        "National Congress of Applied Mechanics (pp. 801–803). ASME."
    )
    assert ct.reference_apa(_ref(cit, "Streamlit")) == (
        "Streamlit Inc. (s. f.). Streamlit [Software]. https://streamlit.io"
    )
    assert ct.reference_apa(_ref(cit, "vademecum")).endswith(
        "https://doi.org/10.5281/zenodo.20092635"
    )


def test_markdown_escapes_what_streamlit_would_interpret() -> None:
    ref = ct.Reference(
        kind="article",
        title="Costo en $ y *énfasis* [corchetes] _x_",
        authors=(ct.Person(family="Autor", given="Ana"),),
        year=2000,
        journal="Revista",
        doi="10.1/a(b)c",
    )
    md = ct.reference_apa(ref, markdown=True)
    for escaped in (r"\$", r"\*énfasis\*", r"\[corchetes\]", r"\_x\_"):
        assert escaped in md
    assert "(https://doi.org/10.1/a%28b%29c)" in md


# ---------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------


def test_bibliography_has_unique_keys_and_balanced_entries(cit: ct.Citation) -> None:
    bib = ct.bibliography_bibtex(cit)
    entries = re.findall(r"^@(\w+)\{([^,]+),$", bib, re.M)
    keys = [key for _, key in entries]
    assert len(entries) == len(cit.references) + 1
    assert len(set(keys)) == len(keys)
    assert keys[0].startswith("barral") and keys[0].endswith("ta216")
    for chunk in re.split(r"\n(?=@)", bib):
        if chunk.startswith("@"):
            assert chunk.count("{") == chunk.count("}"), chunk
            assert "  title" in chunk, chunk
    articles = [c for c in re.split(r"\n(?=@)", bib) if c.startswith("@article")]
    assert all("journal" in c and "year" in c for c in articles)


def test_bibtex_escapes_and_entities(cit: ct.Citation) -> None:
    entry = ct.reference_bibtex(_ref(cit, "CoolProp"), "k")
    assert entry.startswith("@article{k,")
    assert "journal = {Industrial \\& Engineering Chemistry Research}" in entry
    assert "pages   = {2498--2508}" in entry
    iso = ct.reference_bibtex(_ref(cit, "Natural gas"))
    assert iso.startswith("@techreport{international2016natural,")
    assert "author      = {{International Organization for Standardization}}" in iso
    assert "number      = {ISO 6976:2016}" in iso


def test_app_bibtex(cit: ct.Citation) -> None:
    entry = ct.citation_bibtex(cit)
    assert entry.startswith("@software{barral")
    assert "author  = {Barral, Pablo M.}" in entry
    assert f"version = {{{cit.version}}}" in entry
    assert "license = {MIT}" in entry


# ---------------------------------------------------------------------
# Agrupar y buscar
# ---------------------------------------------------------------------


def test_groups_follow_the_kind_order(cit: ct.Citation) -> None:
    groups = ct.group_references(cit.references)
    assert list(groups) == [label for label in ct.REFERENCE_KINDS.values() if label in groups]
    assert sum(len(refs) for refs in groups.values()) == len(cit.references)
    assert len(groups["Normas"]) == 5


def test_unknown_kinds_go_last() -> None:
    refs = [
        ct.Reference(kind="thesis", title="T", authors=()),
        ct.Reference(kind="book", title="B", authors=()),
    ]
    assert list(ct.group_references(refs)) == ["Libros", ct.OTHER_KIND]


def test_search_ignores_accents_and_case(cit: ct.Citation) -> None:
    cengel = _ref(cit, "Heat and Mass Transfer: Fundamentals")
    assert ct.reference_matches(cengel, "")
    assert ct.reference_matches(cengel, "cengel ghajar")
    assert ct.reference_matches(cengel, "ÇENGEL")
    assert not ct.reference_matches(cengel, "cengel coolprop")
    # también busca en para qué se usa y en el tipo
    assert ct.reference_matches(_ref(cit, "CoolProp"), "propiedades termofisicas")
    assert ct.reference_matches(cengel, "libros")


# ---------------------------------------------------------------------
# Datos de terceros y librerías
# ---------------------------------------------------------------------


def test_every_data_file_has_its_source() -> None:
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "data").glob("*.csv"))
    assert sorted(s.path for s in ct.DATA_SOURCES) == files
    assert all(s.source and s.license for s in ct.DATA_SOURCES)


def _runtime_requirements() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    runtime = text.split("# --- Desarrollo")[0]
    names = set()
    for line in runtime.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=!~\[ ;]", line, maxsplit=1)[0].lower())
    return names


def test_libraries_match_the_requirements() -> None:
    assert {name.lower() for name, _ in ct.LIBRARIES} == _runtime_requirements()


def test_package_info() -> None:
    import numpy

    info = ct.package_info("numpy")
    assert info.version == numpy.__version__
    assert info.license != "—"
    missing = ct.package_info("paquete-que-no-existe-ta216")
    assert missing.version is None and missing.license == "—"
    assert ct.python_version().count(".") == 2

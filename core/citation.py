"""Cómo citar la app y sus fuentes, leído del ``CITATION.cff`` (página /Acerca).

El ``CITATION.cff`` (Citation File Format 1.2.0; Druskat et al., 2021) es la
única fuente de la versión, los autores y las referencias: la página los lee
de acá, así que actualizar el archivo actualiza la página. Las citas siguen las
normas APA (7.ª edición) en castellano: «y» entre los dos últimos autores,
«s. f.» si no hay año, «5.ª ed.», «En» para un trabajo dentro de un libro y el
DOI como URL. El BibTeX usa los tipos clásicos (``@techreport`` para normas e
informes) y ``@software``, como el botón «Cite this repository» de GitHub.

Sin Streamlit: la página (``app_pages/99_Acerca.py``) arma la vista.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CITATION_PATH",
    "DATA_SOURCES",
    "LIBRARIES",
    "OTHER_KIND",
    "REFERENCE_KINDS",
    "Citation",
    "DataSource",
    "PackageInfo",
    "Person",
    "Reference",
    "authors_apa",
    "bibliography_bibtex",
    "citation_apa",
    "citation_bibtex",
    "group_references",
    "load_citation",
    "markdown_escape",
    "package_info",
    "parse_citation",
    "python_version",
    "reference_apa",
    "reference_bibtex",
    "reference_matches",
]

ROOT = Path(__file__).resolve().parents[1]
CITATION_PATH = ROOT / "CITATION.cff"

#: Tipo de referencia del CFF → título de su grupo en la página, en este orden.
REFERENCE_KINDS: dict[str, str] = {
    "software": "Software",
    "article": "Artículos",
    "book": "Libros",
    "standard": "Normas",
    "report": "Informes técnicos",
    "conference-paper": "Trabajos de congresos",
}
#: Grupo de los tipos que no están en :data:`REFERENCE_KINDS`.
OTHER_KIND = "Otras"

_BIBTEX_TYPES: dict[str, str] = {
    "article": "article",
    "book": "book",
    "software": "software",
    "standard": "techreport",
    "report": "techreport",
    "conference-paper": "inproceedings",
}


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """Un autor del CFF: una persona (apellido y nombres) o una entidad (``name``)."""

    family: str = ""
    given: str = ""
    suffix: str = ""
    name: str = ""
    affiliation: str = ""
    orcid: str = ""
    email: str = ""

    @property
    def is_entity(self) -> bool:
        """Una institución (ISO, ASHRAE…): se cita con su nombre completo."""
        return not self.family

    @property
    def initials(self) -> str:
        """Las iniciales de los nombres: «Ian H.» → «I. H.», «Jean-Pierre» → «J.-P.»."""
        words = []
        for word in self.given.split():
            pieces = [p for p in word.split("-") if p]
            words.append("-".join(f"{p[0]}." for p in pieces))
        return " ".join(words)

    def full_name(self) -> str:
        """Nombres y apellido, como se lee: «Pablo M. Barral»."""
        if self.is_entity:
            return self.name
        return " ".join(p for p in (self.given, self.family, self.suffix) if p)

    def apa(self) -> str:
        """Como lo pide APA: «Bell, I. H.», «Chase, M. W., Jr.» o la entidad."""
        if self.is_entity:
            return self.name
        text = self.family
        if self.initials:
            text += f", {self.initials}"
        if self.suffix:
            text += f", {self.suffix}"
        return text

    def bibtex(self) -> str:
        """Como lo pide BibTeX: «Bell, Ian H.»; una entidad, entre llaves."""
        if self.is_entity:
            return "{" + _bibtex_escape(self.name) + "}"
        text = _bibtex_escape(self.family)
        if self.suffix:
            text += ", " + _bibtex_escape(self.suffix)
        if self.given:
            text += ", " + _bibtex_escape(self.given)
        return text


@dataclass(frozen=True)
class Reference:
    """Una referencia del CFF (``references``) con los campos que usan las citas."""

    kind: str
    title: str
    authors: tuple[Person, ...]
    scope: str = ""
    year: int | None = None
    journal: str = ""
    volume: str = ""
    issue: str = ""
    start: str = ""
    end: str = ""
    edition: str = ""
    collection_title: str = ""
    publisher: str = ""
    institution: str = ""
    number: str = ""
    version: str = ""
    isbn: str = ""
    doi: str = ""
    url: str = ""
    license: str = ""
    notes: str = ""

    @property
    def kind_label(self) -> str:
        """El título del grupo de la página («Artículos», «Normas»…)."""
        return REFERENCE_KINDS.get(self.kind, OTHER_KIND)

    @property
    def link(self) -> str:
        """El DOI como URL (APA 7) o, si no tiene, la URL o el repositorio."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url

    @property
    def pages(self) -> str:
        """«2498–2508» (raya corta), o el número de artículo si es uno solo."""
        if self.start and self.end:
            return f"{self.start}–{self.end}"
        return self.start


@dataclass(frozen=True)
class Citation:
    """El ``CITATION.cff``: la app (título, versión, autores…) y sus referencias."""

    title: str
    version: str
    date_released: str
    authors: tuple[Person, ...]
    abstract: str = ""
    message: str = ""
    keywords: tuple[str, ...] = ()
    license: str = ""
    url: str = ""
    doi: str = ""
    references: tuple[Reference, ...] = ()

    @property
    def year(self) -> int | None:
        """El año de ``date-released`` (el de la cita)."""
        match = re.match(r"\d{4}", self.date_released)
        return int(match.group()) if match else None


@dataclass(frozen=True)
class DataSource:
    """Un archivo de ``data/`` con su origen y la licencia con la que se usa."""

    path: str
    source: str
    license: str


#: Los datos de terceros que trae el repositorio (un test vigila que estén todos).
DATA_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        "data/iso6976_components.csv",
        "Masas molares y fórmulas de los componentes (ISO 6976:2016)",
        "Valores de la norma (se cita)",
    ),
    DataSource(
        "data/iso6976_calorific_values.csv",
        "Poder calorífico molar de los componentes (ISO 6976:2016)",
        "Valores de la norma (se cita)",
    ),
    DataSource(
        "data/iso6976_summation_factors.csv",
        "Factores de suma (ISO 6976:2016)",
        "Valores de la norma (se cita)",
    ),
    DataSource(
        "data/iso6976_constants.csv",
        "Constante de los gases y masas atómicas (ISO 6976:2016)",
        "Valores de la norma (se cita)",
    ),
    DataSource(
        "data/nasa9_thermo.csv",
        "Polinomios NASA de McBride, Zehe y Gordon (2002), del thermo.inp de NASA CEA "
        "(scripts/extract_nasa9.py)",
        "Apache-2.0",
    ),
    DataSource(
        "data/ghugare2014_biomass.csv",
        "Ghugare et al. (2014), distribuido por el paquete de R modeldata",
        "MIT (data/LICENSE-modeldata.txt)",
    ),
    DataSource(
        "data/argonne_premium_coals.csv",
        "The Argonne Premium Coal Sample Program (Vorres, 1990)",
        "Datos publicados (se cita)",
    ),
    DataSource(
        "data/szargut_chemical_exergy.csv",
        "Szargut, Morris y Steward (1988) y Ahrendts (1980), de los datos de TESPy 0.7.9 "
        "(data/ChemEx) y la tabla A-26 de Moran y Shapiro",
        "MIT (TESPy)",
    ),
)

#: Las librerías de ``requirements.txt`` (sin las de desarrollo) y para qué se usan.
LIBRARIES: tuple[tuple[str, str], ...] = (
    ("streamlit", "Interfaz web"),
    ("CoolProp", "Propiedades termofísicas"),
    ("tespy", "Ciclos termodinámicos (y el control cruzado)"),
    ("fluprodia", "Diagramas de propiedades"),
    ("plotly", "Gráficos interactivos"),
    ("numpy", "Cálculo numérico"),
    ("scipy", "Raíces, integrales, ecuaciones diferenciales y funciones especiales"),
    ("pandas", "Tablas"),
    ("matplotlib", "Gráficos base de fluprodia"),
    ("PyYAML", "Lectura del CITATION.cff (esta página)"),
)


@dataclass(frozen=True)
class PackageInfo:
    """Una librería instalada: versión (``None`` si falta) y licencia declarada."""

    name: str
    version: str | None
    license: str


# ---------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------


def _text(value: Any) -> str:
    """Un campo del CFF como texto: los volúmenes y páginas pueden venir como
    números, y el editor o la institución, como ``{name: …}``."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return _text(value.get("name"))
    return str(value).strip()


def _person(data: Mapping[str, Any]) -> Person:
    return Person(
        family=_text(data.get("family-names")),
        given=_text(data.get("given-names")),
        suffix=_text(data.get("name-suffix")),
        name=_text(data.get("name")),
        affiliation=_text(data.get("affiliation")),
        orcid=_text(data.get("orcid")),
        email=_text(data.get("email")),
    )


def _people(items: Any) -> tuple[Person, ...]:
    return tuple(_person(a) for a in items or () if isinstance(a, Mapping))


def _reference(data: Mapping[str, Any]) -> Reference:
    year = _text(data.get("year"))
    return Reference(
        kind=_text(data.get("type")) or "generic",
        title=_text(data.get("title")),
        authors=_people(data.get("authors")),
        scope=_text(data.get("scope")),
        year=int(year) if year.isdigit() else None,
        journal=_text(data.get("journal")),
        volume=_text(data.get("volume")),
        issue=_text(data.get("issue")),
        start=_text(data.get("start")),
        end=_text(data.get("end")),
        edition=_text(data.get("edition")),
        collection_title=_text(data.get("collection-title")),
        publisher=_text(data.get("publisher")),
        institution=_text(data.get("institution")),
        number=_text(data.get("number")),
        version=_text(data.get("version")),
        isbn=_text(data.get("isbn")),
        doi=_text(data.get("doi")),
        url=_text(data.get("url")) or _text(data.get("repository-code")),
        license=_text(data.get("license")),
        notes=_text(data.get("notes")),
    )


def parse_citation(text: str) -> Citation:
    """Lee el texto de un ``CITATION.cff`` (YAML).

    Raises
    ------
    ValueError
        Si no es un mapa YAML o le falta el título, los autores o la versión.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"El CITATION.cff no es YAML válido: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("El CITATION.cff tiene que ser un mapa YAML (title:, authors:, …).")
    missing = [key for key in ("title", "authors", "version") if not data.get(key)]
    if missing:
        raise ValueError(f"Al CITATION.cff le falta: {', '.join(missing)}.")
    return Citation(
        title=_text(data.get("title")),
        version=_text(data.get("version")),
        date_released=_text(data.get("date-released")),
        authors=_people(data.get("authors")),
        abstract=" ".join(_text(data.get("abstract")).split()),
        message=" ".join(_text(data.get("message")).split()),
        keywords=tuple(_text(k) for k in data.get("keywords") or ()),
        license=_text(data.get("license")),
        url=_text(data.get("repository-code")) or _text(data.get("url")),
        doi=_text(data.get("doi")),
        references=tuple(
            _reference(r) for r in data.get("references") or () if isinstance(r, Mapping)
        ),
    )


def load_citation(path: Path | str = CITATION_PATH) -> Citation:
    """Lee el ``CITATION.cff`` del repositorio (u otro)."""
    return parse_citation(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# APA
# ---------------------------------------------------------------------

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~$])")


def markdown_escape(text: str) -> str:
    """Escapa lo que Streamlit interpretaría en Markdown: cursivas (``*``, ``_``),
    LaTeX (``$``), links (``[ ]``), código, tablas y tachado."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


class _Format:
    """Texto plano o Markdown (con cursivas, escapes y links)."""

    def __init__(self, markdown: bool) -> None:
        self.markdown = markdown

    def text(self, value: str) -> str:
        return markdown_escape(value) if self.markdown else value

    def italic(self, value: str) -> str:
        return f"*{self.text(value)}*" if self.markdown and value else value

    def link(self, url: str) -> str:
        if not self.markdown:
            return url
        href = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        return f"[{markdown_escape(url)}]({href})"


def _period(raw: str, formatted: str) -> str:
    """Cierra un elemento con punto, salvo que el texto ya termine en uno (o en ?, !)."""
    return formatted if raw.rstrip().endswith((".", "?", "!")) else f"{formatted}."


def _edition(edition: str) -> str:
    """«5» → «5.ª ed.» (APA en castellano)."""
    return f"{edition}.ª ed." if edition.isdigit() else f"{edition} ed."


def authors_apa(persons: Iterable[Person]) -> str:
    """«Bell, I. H., Wronski, J. y Lemort, V.» («y» entre los dos últimos)."""
    names = [p.apa() for p in persons]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " y " + names[-1]


def _same_as_author(issuer: str, authors: tuple[Person, ...]) -> bool:
    """APA no repite el editor cuando es la misma entidad que firma."""
    return len(authors) == 1 and authors[0].is_entity and authors[0].name == issuer


def reference_apa(ref: Reference, *, markdown: bool = False) -> str:
    """La cita de una referencia según APA 7 (en castellano).

    Con ``markdown=True`` los títulos de libros, revistas y normas van en
    cursiva, el texto se escapa y el DOI es un link.
    """
    f = _Format(markdown)
    who = authors_apa(ref.authors)
    year = f"({ref.year})" if ref.year else "(s. f.)"
    parts = [f"{_period(who, f.text(who))} {year}." if who else f"{year}."]

    issuer = ref.publisher or ref.institution
    if _same_as_author(issuer, ref.authors):
        issuer = ""

    if ref.kind == "article":
        parts.append(_period(ref.title, f.text(ref.title)))
        source = f.italic(ref.journal)
        if ref.volume:
            source += f", {f.italic(ref.volume)}"
            if ref.issue:
                source += f"({f.text(ref.issue)})"
        if ref.pages:
            source += f", {f.text(ref.pages)}" if source else f.text(ref.pages)
        if source:
            parts.append(f"{source}.")
    elif ref.kind == "conference-paper":
        parts.append(_period(ref.title, f.text(ref.title)))
        if ref.collection_title:
            within = f"En {f.italic(ref.collection_title)}"
            if ref.pages:
                within += f" (pp. {f.text(ref.pages)})"
            parts.append(f"{within}.")
        if issuer:
            parts.append(_period(issuer, f.text(issuer)))
    else:
        title = f.italic(ref.title)
        details = []
        if ref.number and ref.number not in ref.title:
            details.append(f.text(ref.number))
        if ref.version:
            details.append(f"Versión {f.text(ref.version)}")
        if ref.edition:
            details.append(_edition(ref.edition))
        if details:
            title += f" ({', '.join(details)})"
        if ref.kind == "software":
            title += " [Software]"
            parts.append(f"{title}.")
        else:
            parts.append(_period(ref.title, title) if not details else f"{title}.")
        if ref.collection_title:
            parts.append(_period(ref.collection_title, f.text(ref.collection_title)))
        if issuer:
            parts.append(_period(issuer, f.text(issuer)))

    if ref.notes:
        parts.append(_period(ref.notes, f.text(ref.notes)))
    if ref.link:
        parts.append(f.link(ref.link))
    return " ".join(parts)


def citation_apa(citation: Citation, *, markdown: bool = False) -> str:
    """La cita sugerida de la app: autores (año). *Título* (Versión x) [Software]. URL."""
    f = _Format(markdown)
    who = authors_apa(citation.authors)
    year = f"({citation.year})" if citation.year else "(s. f.)"
    text = (
        f"{_period(who, f.text(who))} {year}. {f.italic(citation.title)} "
        f"(Versión {f.text(citation.version)}) [Software]."
    )
    link = f"https://doi.org/{citation.doi}" if citation.doi else citation.url
    return f"{text} {f.link(link)}" if link else text


# ---------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------

_BIBTEX_SPECIAL = re.compile(r"([&%$#_])")
_KEY_STOPWORDS = frozenset(
    "a an the of on and for in to with by from el la los las de del y en un una "
    "die der das des und von zur zum fur".split()
)


def _bibtex_escape(text: str) -> str:
    """Escapa los caracteres que LaTeX interpreta (& % $ # _)."""
    return _BIBTEX_SPECIAL.sub(r"\\\1", text)


def _ascii(text: str) -> str:
    """Solo letras y dígitos ASCII, en minúscula («Çengel» → «cengel»)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isascii() and c.isalnum()).lower()


def _key_base(authors: tuple[Person, ...], year: int | None, title: str) -> str:
    """La clave «apellido + año + primera palabra del título» (bell2014pure)."""
    first = authors[0] if authors else Person(name="anon")
    who = _ascii(first.family) if first.family else _ascii(first.name.split()[0])
    word = next(
        (w for w in (_ascii(t) for t in title.split()) if w and w not in _KEY_STOPWORDS),
        "",
    )
    return f"{who or 'anon'}{year or ''}{word}"


def _unique(base: str, used: set[str]) -> str:
    key, suffix = base, ord("b")
    while key in used:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used.add(key)
    return key


def _entry(entry_type: str, key: str, fields: list[tuple[str, str]]) -> str:
    width = max(len(name) for name, _ in fields)
    lines = [f"@{entry_type}{{{key},"]
    for k, (name, value) in enumerate(fields):
        comma = "," if k < len(fields) - 1 else ""
        lines.append(f"  {name.ljust(width)} = {{{value}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def reference_bibtex(ref: Reference, key: str | None = None) -> str:
    """La entrada BibTeX de una referencia (la clave sale del autor, el año y el título)."""
    entry_type = _BIBTEX_TYPES.get(ref.kind, "misc")
    fields: list[tuple[str, str]] = []
    if ref.authors:
        fields.append(("author", " and ".join(p.bibtex() for p in ref.authors)))
    fields.append(("title", "{" + _bibtex_escape(ref.title) + "}"))
    issuer = ref.publisher or ref.institution
    if entry_type == "article":
        fields += [("journal", _bibtex_escape(ref.journal))] if ref.journal else []
        fields += [("volume", ref.volume)] if ref.volume else []
        fields += [("number", ref.issue)] if ref.issue else []
    elif entry_type == "inproceedings":
        if ref.collection_title:
            fields.append(("booktitle", _bibtex_escape(ref.collection_title)))
    elif entry_type == "techreport":
        institution = ref.institution or ref.publisher
        if not institution and len(ref.authors) == 1 and ref.authors[0].is_entity:
            institution = ref.authors[0].name
        fields += [("institution", _bibtex_escape(institution))] if institution else []
        fields += [("number", _bibtex_escape(ref.number))] if ref.number else []
    elif entry_type == "book" and ref.collection_title:
        fields.append(("series", _bibtex_escape(ref.collection_title)))
    if ref.pages:
        fields.append(("pages", ref.pages.replace("–", "--")))
    if entry_type in ("book", "inproceedings", "misc", "software") and issuer:
        fields.append(("publisher", _bibtex_escape(issuer)))
    if ref.edition:
        fields.append(("edition", ref.edition))
    if ref.version:
        fields.append(("version", ref.version))
    if ref.year:
        fields.append(("year", str(ref.year)))
    if ref.isbn:
        fields.append(("isbn", ref.isbn))
    if ref.doi:
        fields.append(("doi", ref.doi))
    elif ref.url:
        fields.append(("url", ref.url))
    if ref.notes:
        fields.append(("note", _bibtex_escape(ref.notes)))
    return _entry(entry_type, key or _key_base(ref.authors, ref.year, ref.title), fields)


def citation_bibtex(citation: Citation, key: str | None = None) -> str:
    """La entrada ``@software`` de la app (como el «Cite this repository» de GitHub)."""
    fields = [
        ("author", " and ".join(p.bibtex() for p in citation.authors)),
        ("title", "{" + _bibtex_escape(citation.title) + "}"),
    ]
    if citation.year:
        fields.append(("year", str(citation.year)))
    fields.append(("version", citation.version))
    if citation.doi:
        fields.append(("doi", citation.doi))
    if citation.url:
        fields.append(("url", citation.url))
    if citation.license:
        fields.append(("license", citation.license))
    base = _key_base(citation.authors, citation.year, citation.title)
    return _entry("software", key or base, fields)


def bibliography_bibtex(citation: Citation) -> str:
    """El archivo ``.bib`` completo: la app y todas sus referencias, con claves únicas."""
    used: set[str] = set()
    app_key = _unique(_key_base(citation.authors, citation.year, citation.title), used)
    entries = [citation_bibtex(citation, app_key)]
    for ref in citation.references:
        key = _unique(_key_base(ref.authors, ref.year, ref.title), used)
        entries.append(reference_bibtex(ref, key))
    header = (
        f"% {citation.title}, versión {citation.version}: la app y las fuentes en las que "
        "se apoya.\n% Generado a partir del CITATION.cff del repositorio.\n"
    )
    return header + "\n" + "\n\n".join(entries) + "\n"


# ---------------------------------------------------------------------
# Agrupar y buscar
# ---------------------------------------------------------------------


def group_references(references: Iterable[Reference]) -> dict[str, list[Reference]]:
    """Las referencias por grupo, en el orden de :data:`REFERENCE_KINDS` (y «Otras»)."""
    order = [*REFERENCE_KINDS.values(), OTHER_KIND]
    groups: dict[str, list[Reference]] = {label: [] for label in order}
    for ref in references:
        groups[ref.kind_label].append(ref)
    return {label: refs for label, refs in groups.items() if refs}


def _fold(text: str) -> str:
    """Sin tildes ni mayúsculas, para buscar («Çengel» ≈ «cengel»)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def reference_matches(ref: Reference, query: str) -> bool:
    """Si todas las palabras de la búsqueda están en la cita o en para qué se usa."""
    haystack = _fold(f"{reference_apa(ref)} {ref.scope} {ref.kind_label}")
    return all(word in haystack for word in _fold(query).split())


# ---------------------------------------------------------------------
# Librerías instaladas
# ---------------------------------------------------------------------


def _license(meta: Any) -> str:
    """La licencia declarada en los metadatos del paquete (SPDX si la hay)."""
    expression = meta.get("License-Expression")
    if expression:
        return str(expression)
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License :: OSI Approved :: "):
            return classifier.rsplit("::", 1)[1].strip().removesuffix(" License")
    text = str(meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 40:
        return text
    return "—"


def package_info(name: str) -> PackageInfo:
    """Versión instalada y licencia de una librería (``None`` si no está instalada)."""
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return PackageInfo(name, None, "—")
    return PackageInfo(name, metadata.version(name), _license(meta))


def python_version() -> str:
    """La versión de Python que corre la app («3.11.9»)."""
    return ".".join(str(v) for v in sys.version_info[:3])

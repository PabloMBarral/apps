# CLAUDE.md

> Este archivo da contexto al agente Claude Code. Mantenelo actualizado:
> cualquier cambio de arquitectura, dependencia o convención debería
> reflejarse acá.

## Proyecto

Suite de herramientas didácticas de ingeniería térmica para la materia
**Tecnología del Calor** (FIUBA). Streamlit + CoolProp + TESPy + fluprodia.

Repo hermano de fórmulas teóricas:
[PabloMBarral/vademecum-termo](https://github.com/PabloMBarral/vademecum-termo).
Cada página de la app debe linkear la sección correspondiente del vademecum.

Autor: Pablo M. Barral (pbarral@fi.uba.ar, ORCID 0000-0003-1125-4199).
Licencia: MIT.

## Stack

- Python 3.11+
- **Streamlit** — UI (multipágina con carpeta `pages/`)
- **CoolProp** — propiedades termofísicas punto a punto
- **TESPy** — simulación de ciclos termodinámicos
- **fluprodia** — diagramas de propiedades de fluidos
- **NumPy, SciPy, pandas, matplotlib** — utilitarios numéricos y plots base
- **pytest** — tests
- **ruff** — lint y format

Toda dependencia nueva debe agregarse a `requirements.txt` Y a
`CITATION.cff` (con autoría/cita correspondiente si es académica).

## Arquitectura

Separación estricta entre lógica de cálculo (testeable, sin Streamlit)
y UI (Streamlit).

```
apps/
├── streamlit_app.py           # Home / landing
├── pages/                     # Una página por módulo (numeradas)
│   ├── 1_Propiedades.py
│   ├── 2_Interpolacion.py
│   ├── 3_Isoentropicos.py
│   ├── 4_Diagramas.py
│   ├── 5_Rankine.py
│   ├── 6_Refrigeracion.py
│   ├── 7_Psicrometria.py
│   ├── 8_Combustion.py
│   ├── 9_Poder_Calorifico.py
│   ├── 10_ISO_6976.py
│   ├── 11_Exergia.py
│   └── 99_Acerca.py           # Créditos, licencias, citas
├── core/                      # Lógica pura, sin dependencia de Streamlit
│   ├── __init__.py
│   ├── units.py               # Selector global de unidades
│   ├── fluids.py              # Wrappers cacheados sobre CoolProp
│   ├── interpolation.py       # Interpolación lineal y doble entrada
│   ├── isentropic.py          # Turbina / compresor / bomba; politrópico
│   ├── exergy.py              # Exergía física y química
│   ├── combustion/
│   │   ├── __init__.py
│   │   ├── fuels.py           # Modelos de combustible (sólido/líquido/gas)
│   │   ├── stoichiometry.py   # Reacciones, exceso aire, productos
│   │   ├── heating_value.py   # PCI/PCS por correlaciones (último/próximo)
│   │   └── iso6976.py         # ISO 6976:2016 — gases combustibles
│   ├── cycles/
│   │   ├── rankine.py
│   │   ├── refrigeration.py
│   │   ├── brayton.py
│   │   └── combined.py
│   └── plots.py               # fluprodia + matplotlib helpers
├── tests/                     # pytest: tests/test_<modulo>.py
├── data/                      # Tablas, propiedades por componente, etc.
│   ├── iso6976_components.csv # Valores tabulados por componente puro
│   └── szargut_chemical_exergy.csv
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── README.md
└── CLAUDE.md
```

## Convenciones

### Unidades

- Por defecto: **bar(a)**, **°C**, **kJ/kg**, **kJ/(kg·K)**, título
  adimensional, fracciones másicas/molares como decimales (no porcentajes).
- Cada función pura en `core/` recibe valores en **SI** internamente
  (Pa, K, J/kg). La conversión vive en `core/units.py` y en la UI.
- Hay un selector global de sistema de unidades (SI / técnico / inglés)
  en sidebar; las páginas leen del `st.session_state`.

### Estilo

- **Type hints obligatorios** en todo `core/`.
- Funciones puras, sin estado global. Resultados como `dataclass`
  cuando hay varios valores (`StatePoint`, `CycleResult`, etc.).
- Cache de CoolProp con `@st.cache_data` en los wrappers de `fluids.py`.
- Idioma de la UI: **español rioplatense**. Los identificadores de
  código en inglés, comentarios y docstrings en español.
- Cada función académicamente relevante incluye en su docstring una
  cita corta a la fuente (libro de texto, paper, norma).

### Páginas Streamlit

Toda página debe tener, mínimo:

1. Título y descripción breve del módulo.
2. Un expansor `📖 Fórmulas teóricas` con link al apartado del vademecum.
3. Inputs validados con rangos razonables y mensajes claros.
4. Resultado principal destacado + tabla con todos los estados.
5. Si aplica, un diagrama con fluprodia / matplotlib.
6. Un expansor `🔬 Procedimiento` con las ecuaciones aplicadas en LaTeX
   y los valores reemplazados (modo didáctico).
7. Botón de exportar resultados (CSV / JSON).

### Citas y licencias

- **Toda librería externa de cálculo** (no UI) que se sume al proyecto
  exige actualizar `CITATION.cff` con su referencia formal y `README.md`
  con su BibTeX.
- **Normas técnicas** (ISO, ASHRAE, IRAM): citar siempre versión y año.
- En la página `99_Acerca.py` se muestra dinámicamente el contenido de
  `CITATION.cff` y la licencia.

## Comandos

```bash
# Correr localmente
streamlit run streamlit_app.py

# Tests
pytest
pytest tests/test_combustion.py -v

# Lint + format
ruff check .
ruff format .

# Cobertura
pytest --cov=core --cov-report=term-missing
```

## Reglas para Claude Code

- **No** mezclar UI (Streamlit) con cálculo en el mismo archivo. Si
  encontrás `st.*` dentro de `core/`, refactorizá.
- **No** hardcodear unidades dentro de funciones de cálculo en `core/`.
- **No** agregar dependencias sin actualizar `requirements.txt` Y
  `CITATION.cff`.
- **No** romper la API pública existente sin migración explícita y
  tests que cubran el caso viejo.
- Antes de implementar un módulo nuevo, **proponer un plan**
  (estructura de archivos, firmas de funciones, casos de test) y
  esperar OK. Usar Plan Mode (`Shift+Tab`) si la tarea es grande.
- Cada PR / commit nuevo debe incluir: código + tests + actualización
  de README si es módulo nuevo + actualización de `CITATION.cff` si
  cambian dependencias.
- Cuando integres TESPy, basarse en los ejemplos canónicos de la doc
  oficial (`tespy.readthedocs.io`), no inventar la API.
- Para ISO 6976, los valores por componente puro deben venir de
  `data/iso6976_components.csv` extraídos de la norma; no hardcodear
  en código. Tests obligatorios contra los ejemplos del anexo de la norma.
- Para correlaciones de PCI (Dulong, Boie, Channiwala-Parikh), cada
  función lleva en el docstring la referencia exacta al paper original
  y el rango de validez (tipo de combustible).
- Mensajes de error orientados al alumno: explicar qué entrada está
  fuera de rango y por qué, no solo "ValueError".

## Notas didácticas

Este software apunta a estudiantes de grado de ingeniería. Priorizar
**transparencia del cálculo** sobre performance:

- Mostrar pasos intermedios siempre que sea pedagógicamente útil.
- Permitir que el alumno vea la diferencia entre interpolar tablas y
  usar la ecuación de estado de CoolProp.
- Permitir comparar correlaciones de PCI entre sí para el mismo
  combustible.
- Mostrar destrucción exergética con interpretación física, no solo
  número.

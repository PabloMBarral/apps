# TA216 — Tecnología de Calor Avanzada — Apps

[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org)
[![CoolProp](https://img.shields.io/badge/powered%20by-CoolProp-1f6feb)](http://www.coolprop.org)
[![TESPy](https://img.shields.io/badge/powered%20by-TESPy-2ea44f)](https://tespy.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/cite-CITATION.cff-orange)](CITATION.cff)

Suite de herramientas didácticas en Python/Streamlit para la materia
**TA216 — Tecnología de Calor Avanzada** (Facultad de Ingeniería, Universidad de Buenos Aires).

> 📖 Las **fórmulas teóricas** que sustentan estos cálculos están en el repo
> hermano [vademecum-termo](https://github.com/PabloMBarral/vademecum-termo).
> Pensalos como un único material: el vademecum explica, este repo calcula.

---

## Módulos

### Estables
- ✅ **Estado termodinámico del agua** (y de los otros fluidos del
  proyecto) a partir de cualquier par de propiedades independientes: T-p,
  p-x, T-x, p-h, p-s, T-s, h-s, **T-v, p-v** (tanques rígidos) y **p-u**
  (sistemas cerrados). Muestra la **región** (líquido comprimido, vapor
  húmedo, vapor sobrecalentado, supercrítico) con su sobrecalentamiento,
  subenfriamiento o título; todas las propiedades (v, ρ, u, h, s, cp, cv,
  γ, velocidad del sonido, μ, k, ν, α, Pr y factor de compresibilidad Z);
  la fila de las tablas de saturación a la p y a la T del estado (como en
  Cengel A-4/A-5); advertencias didácticas (por ejemplo, T-p sobre la
  curva de saturación); el **procedimiento para resolverlo con las
  tablas**, en LaTeX y en el sistema de unidades activo (saturación,
  comparación con f y g, regla de la palanca, aproximación de líquido
  incompresible, comparación con gas ideal, verificación h = u + p·v); el
  punto sobre el diagrama y exportación CSV/JSON. Una **tabla de estados**
  permite armar un ciclo estado por estado y verlo sobre el diagrama, con
  las isobáricas e isoentrópicas reales entre estados consecutivos.
  Validación con mensajes en castellano (punto triple, título fuera de la
  campana, estados fuera del rango de la ecuación de estado). _Fase 1.6
  cerrada._
- ✅ **Interpolación lineal y doble entrada** sobre tablas, con procedimiento
  paso a paso (LaTeX + explicación en español) y comparación contra el valor
  exacto de CoolProp cuando la tabla es reconocida como saturación o vapor
  sobrecalentado. Tabla editable in-place (`st.data_editor`), comparación
  CoolProp como opt-in con selector de fluido (Water, R134a, R410A, R1234yf,
  NH₃, CO₂, Air), normalizador de unidades tolerante a notación variada
  (`kJ/(kg·K)`, `kJ/kg-K`, `kJ kg^-1 K^-1`, etc.). Expansor con las
  fórmulas del vademecum (§2.1 y §2.2) y descarga del resultado en
  CSV/JSON (nodos o vértices, intermedios y procedimiento). _Fase 1.2
  cerrada; teoría y export en la 1.7._
- ✅ **Rendimientos isoentrópicos** de turbinas, compresores y bombas:
  modo directo (calcular el estado real dado η_s) e inverso (recuperar η_s
  a partir de los dos estados). Compresor multietapa con η_s por etapa,
  intercooler opcional entre etapas, y benchmark built-in contra una sola
  etapa equivalente. Valores iniciales calculables para los 7 fluidos y
  validaciones con mensajes en castellano (entrada líquida a un compresor,
  vapor en la bomba, interenfriador que condensaría, estados fuera del
  rango de la ecuación de estado). Tabla de estados con la región de cada
  uno, procedimiento paso a paso en LaTeX **en el sistema de unidades
  activo**, expansor con las fórmulas del vademecum (§10.4, §3.3, §6.3) y
  descarga CSV/JSON. Comparación opt-in en bomba contra el modelo de
  líquido incompresible `w_p ≈ v_1·Δp/η_s`. _Fase 1.3 cerrada; 1.5b y
  validaciones en la 1.7._
- ✅ **Multifluido + selector global de unidades**: la página de
  Propiedades trabaja con cualquier fluido de `core.fluids.SUPPORTED_FLUIDS`
  (Water, R134a, R410A, R1234yf, NH₃, CO₂, Air). Selector global en
  sidebar entre **SI** (K, Pa, J/kg, J/(kg·K)), **Técnico** (°C, bar,
  kJ/kg, kJ/(kg·K)) — default — e **Inglés** (°F, psia, Btu/lb,
  Btu/(lb·°R)). El sistema se persiste vía `st.session_state` y aplica
  a Propiedades e Isoentrópicos. Interpolación respeta las unidades del
  CSV original; ISO 6976 usa las unidades de la norma. Los procedimientos
  en LaTeX de Propiedades e Isoentrópicos siguen el sistema activo.
  _Fase 1.4 cerrada._
- ✅ **ISO 6976:2016 — Poder calorífico de gases combustibles**:
  poder calorífico bruto y neto (molar, másico, volumétrico), densidad,
  densidad relativa al aire e índices de Wobbe G y N, con propagación
  de incertidumbre estándar (matriz identidad; la matriz de normalización
  queda deferida a fase futura por requerir ISO 14912:2003). Composición
  editable in-place con autocompletado de los 60 componentes de la norma.
  Ejemplos del Annex D (D.2, D.3, D.4) precargados y verificados contra
  los valores tabulados de la norma. Expansor con la relación PCS − PCI
  del vademecum (§16.9) y las fórmulas de la norma; descarga CSV/JSON con
  u (k = 1) y U (k = 2) de cada magnitud. _Fase 2.3 cerrada; teoría y
  export en la 1.7._
- ✅ **Diagramas de propiedades** (log p–h, T–s, h–s, p–log v) con
  [fluprodia](https://github.com/fwitte/fluprodia) + plotly. La página de
  **Propiedades** dibuja el estado calculado sobre el diagrama elegido
  para los 7 fluidos del proyecto. La página de **Isoentrópicos** dibuja
  el proceso completo: la isoentrópica real 1→2s vía CoolProp, los
  segmentos 2s→2 y 1→2 como referencia visual (línea recta que **no**
  representa la trayectoria termodinámica real), y para el compresor
  multietapa cada etapa con su intercooler isobárico. Diagrama por
  ``(fluido, sistema de unidades)`` cacheado con `@st.cache_resource`
  para amortizar el cálculo de isolíneas, con isotermas e isobaras
  redondas en las unidades de cada sistema. _Fase 1.5a cerrada._

Todas las páginas están pensadas para usarse **desde el celular**: las
ecuaciones del procedimiento se escriben en renglones cortos (una
igualdad por renglón) para que entren en el ancho de la pantalla.

### En desarrollo (roadmap)
- **Ciclos termodinámicos** con [TESPy](https://tespy.readthedocs.io):
  Rankine (simple, con recalentamiento, con regeneración),
  refrigeración por compresión de vapor (simple, con economizador, cascada),
  Brayton (simple, con regeneración, intercooling, recalentamiento),
  ciclo combinado, cogeneración.
- **Psicrometría** y procesos HVAC sobre carta psicrométrica interactiva.
- **Estequiometría** de combustión: combustibles puros y mezclas,
  exceso de aire, composición de humos en base seca y húmeda,
  temperatura adiabática de llama.
- **Poder calorífico** a partir de:
  - composición **última** (Dulong, Boie, Channiwala-Parikh),
  - composición **próxima** (correlaciones de Parikh y similares),
  - composición molar según **ISO 6976:2016** para gases combustibles.
- **Exergía**: exergía física de cualquier estado, exergía química
  (tablas estándar), destrucción exergética por componente y diagrama
  de Grassmann para los ciclos.
- **Transferencia de calor**: conducción multicapa, aletas, correlaciones
  de convección, radiación entre superficies, intercambiadores
  por LMTD y ε-NTU.

---

## Instalación local

Requiere Python 3.11+.

```bash
git clone https://github.com/PabloMBarral/apps.git
cd apps
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Estructura

```
apps/
├── streamlit_app.py       # Home / landing
├── app_pages/             # Una página Streamlit por módulo (no se llama pages/: ver CLAUDE.md)
├── core/                  # Lógica de cálculo, sin dependencia de Streamlit
│   ├── fluids.py          # estado completo, saturación, región (CoolProp)
│   ├── state_report.py    # tablas, procedimiento didáctico y export del estado
│   ├── units_system.py    # sistemas SI / Técnico / Inglés
│   ├── latex.py           # cantidades y cadenas de igualdades en LaTeX
│   ├── export.py          # dict → CSV (campo, valor) para las descargas
│   ├── diagrams.py        # diagramas de propiedades (fluprodia)
│   ├── interpolation.py
│   ├── isentropic.py
│   ├── exergy.py
│   ├── combustion/
│   │   ├── stoichiometry.py
│   │   ├── heating_value.py
│   │   └── iso6976.py
│   ├── cycles/
│   └── plots.py
├── ui/                    # Helpers de UI (Streamlit): unidades, diagramas, créditos
├── tests/                 # pytest (+ páginas con streamlit.testing)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Cómo citar este software

Si lo usás en investigación o docencia, GitHub te ofrece un botón
"Cite this repository" en la columna derecha, generado a partir de
[`CITATION.cff`](CITATION.cff). Cita sugerida:

> Barral, P. M. (2026). *TA216 — Tecnología de Calor Avanzada — Apps* (Versión 0.9.0)
> [Software]. https://github.com/PabloMBarral/apps

---

## Referencias

Este proyecto se apoya en librerías y normas open source / públicas.
Si publicás resultados usando esta suite, te pedimos que cites también
las fuentes correspondientes:

**CoolProp** — motor de propiedades termofísicas:

```bibtex
@article{bell2014coolprop,
  author  = {Bell, Ian H. and Wronski, Jorrit and Quoilin, Sylvain and Lemort, Vincent},
  title   = {Pure and Pseudo-pure Fluid Thermophysical Property Evaluation
             and the Open-Source Thermophysical Property Library CoolProp},
  journal = {Industrial \& Engineering Chemistry Research},
  volume  = {53},
  number  = {6},
  pages   = {2498--2508},
  year    = {2014},
  doi     = {10.1021/ie4033999}
}
```

**TESPy** — simulación de ciclos termodinámicos:

```bibtex
@article{witte2020tespy,
  author  = {Witte, Francesco and Tuschy, Ilja},
  title   = {{TESPy}: Thermal Engineering Systems in Python},
  journal = {Journal of Open Source Software},
  volume  = {5},
  number  = {49},
  pages   = {2178},
  year    = {2020},
  doi     = {10.21105/joss.02178}
}
```

**IAPWS-95** — formulación de las propiedades del agua que usa CoolProp:

```bibtex
@article{wagner2002iapws95,
  author  = {Wagner, Wolfgang and Pru{\ss}, Andreas},
  title   = {The {IAPWS} Formulation 1995 for the Thermodynamic Properties of
             Ordinary Water Substance for General and Scientific Use},
  journal = {Journal of Physical and Chemical Reference Data},
  volume  = {31},
  number  = {2},
  pages   = {387--535},
  year    = {2002},
  doi     = {10.1063/1.1461829}
}
```

**fluprodia** — diagramas de propiedades de fluidos. Witte, F.
<https://github.com/fwitte/fluprodia>

**Streamlit** — framework de interfaz web. Streamlit Inc.
<https://streamlit.io>

**ISO 6976:2016** — *Natural gas — Calculation of calorific values, density,
relative density and Wobbe indices from composition*. International
Organization for Standardization, 2016.

---

## Notas de uso

- Las unidades por defecto son: presión en bar(a), temperatura en °C,
  entalpía en kJ/kg, entropía en kJ/(kg·K), título adimensional.
- El título x solo se informa dentro de la campana; fuera de ella la
  página muestra "—" (en la API de bajo nivel, `StatePoint.x = -1`).
- Las propiedades del agua salen de CoolProp con la formulación
  IAPWS-95; pueden diferir en el último decimal de las tablas impresas.
- La precisión de los resultados depende de las librerías subyacentes.
  Usar bajo propia responsabilidad. **Esta es una herramienta didáctica**,
  no apta para diseño de equipos sin verificación independiente.

---

## Licencia

MIT — ver [LICENSE](LICENSE). Las librerías citadas conservan sus
respectivas licencias (todas compatibles, mayormente MIT y Apache 2.0).

---

## Autoría y contacto

**Pablo M. Barral** — Profesor, TA216 — Tecnología de Calor Avanzada, FIUBA.
[ORCID 0000-0003-1125-4199](https://orcid.org/0000-0003-1125-4199)
· [Google Scholar](https://scholar.google.com/citations?user=nxvRCoUAAAAJ)
· pbarral@fi.uba.ar

Issues, pull requests y sugerencias didácticas son bienvenidos.

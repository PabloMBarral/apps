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
- ✅ **Propiedades del agua/vapor** a partir de cualquier par de variables
  independientes (T-p, p-h, h-s, p-x, T-x, p-s, T-s).
- ✅ **Interpolación lineal y doble entrada** sobre tablas, con procedimiento
  paso a paso (LaTeX + explicación en español) y comparación contra el valor
  exacto de CoolProp cuando la tabla es reconocida como saturación o vapor
  sobrecalentado. Tabla editable in-place (`st.data_editor`), comparación
  CoolProp como opt-in con selector de fluido (Water, R134a, R410A, R1234yf,
  NH₃, CO₂, Air), normalizador de unidades tolerante a notación variada
  (`kJ/(kg·K)`, `kJ/kg-K`, `kJ kg^-1 K^-1`, etc.). _Fase 1.2 cerrada._
- ✅ **Rendimientos isoentrópicos** de turbinas, compresores y bombas:
  modo directo (calcular el estado real dado η_s) e inverso (recuperar η_s
  a partir de los dos estados). Compresor multietapa con η_s por etapa,
  intercooler opcional entre etapas, y benchmark built-in contra una sola
  etapa equivalente. Validación de fase líquida del inlet de bomba vía
  `CoolProp.PhaseSI`. Procedimiento paso a paso con LaTeX y narrativa en
  español. Comparación opt-in en bomba contra el modelo de líquido
  incompresible `w_p ≈ v_1·Δp/η_s`. _Fase 1.3 cerrada._
- ✅ **Multifluido + selector global de unidades**: la página de
  Propiedades trabaja con cualquier fluido de `core.fluids.SUPPORTED_FLUIDS`
  (Water, R134a, R410A, R1234yf, NH₃, CO₂, Air). Selector global en
  sidebar entre **SI** (K, Pa, J/kg, J/(kg·K)), **Técnico** (°C, bar,
  kJ/kg, kJ/(kg·K)) — default — e **Inglés** (°F, psia, Btu/lb,
  Btu/(lb·°R)). El sistema se persiste vía `st.session_state` y aplica
  a Propiedades e Isoentrópicos. Interpolación respeta las unidades del
  CSV original; ISO 6976 usa las unidades de la norma. Los pasos
  didácticos isoentrópicos (LaTeX) quedan en Técnico hardcoded para
  consistencia con Cengel — su conversión al sistema activo entra en
  Fase 1.5. _Fase 1.4 cerrada._
- ✅ **ISO 6976:2016 — Poder calorífico de gases combustibles**:
  poder calorífico bruto y neto (molar, másico, volumétrico), densidad,
  densidad relativa al aire e índices de Wobbe G y N, con propagación
  de incertidumbre estándar (matriz identidad; la matriz de normalización
  queda deferida a fase futura por requerir ISO 14912:2003). Composición
  editable in-place con autocompletado de los 60 componentes de la norma.
  Ejemplos del Annex D (D.2, D.3, D.4) precargados y verificados contra
  los valores tabulados de la norma. _Fase 2.3 cerrada._

### En desarrollo (roadmap)
- **Multifluido**: refrigerantes (R134a, R410A, R1234yf, NH₃, CO₂), aire,
  combustibles puros y mezclas.
- **Diagramas de propiedades** (log p-h, T-s, h-s, p-v) con
  [fluprodia](https://github.com/fwitte/fluprodia), con superposición de
  puntos y procesos calculados.
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
├── pages/                 # Una página Streamlit por módulo
├── core/                  # Lógica de cálculo, sin dependencia de Streamlit
│   ├── fluids.py
│   ├── interpolation.py
│   ├── isentropic.py
│   ├── exergy.py
│   ├── combustion/
│   │   ├── stoichiometry.py
│   │   ├── heating_value.py
│   │   └── iso6976.py
│   ├── cycles/
│   └── plots.py
├── tests/                 # pytest
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

> Barral, P. M. (2026). *TA216 — Tecnología de Calor Avanzada — Apps* (Versión 0.2.0)
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
- `x = -1` en propiedades de agua/vapor indica que el estado no se
  encuentra dentro de la campana ni en su frontera.
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

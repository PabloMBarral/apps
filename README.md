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
  (Water, R134a, R410A, R1234yf, NH₃, CO₂, Air y, para el ORC, R-245fa,
  R-1233zd(E), isopentano y tolueno). Selector global en
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
  para los 11 fluidos del proyecto. La página de **Isoentrópicos** dibuja
  el proceso completo: la isoentrópica real 1→2s vía CoolProp, los
  segmentos 2s→2 y 1→2 como referencia visual (línea recta que **no**
  representa la trayectoria termodinámica real), y para el compresor
  multietapa cada etapa con su intercooler isobárico. Diagrama por
  ``(fluido, sistema de unidades)`` cacheado con `@st.cache_resource`
  para amortizar el cálculo de isolíneas, con isotermas e isobaras
  redondas en las unidades de cada sistema. _Fase 1.5a cerrada._
- ✅ **Ciclo de Rankine** (agua u ORC) resuelto con
  [TESPy](https://tespy.readthedocs.io) como una red de componentes
  (bomba, caldera, turbina, condensador): simple, **con
  recalentamiento** o **con regeneración** —hasta tres calentadores de
  agua de alimentación, abiertos o cerrados (con TTD, subenfriador de
  drenaje (DCA), desrecalentador y el drenaje en cascada o bombeado hacia
  adelante desde cualquier cerrado), combinables con el recalentamiento—,
  ideal o real (rendimientos isoentrópicos y **pérdidas de carga y de
  calor** del ciclo real de Cengel §10-3: caídas de presión en caldera,
  recalentador, condensador, calentadores y cañerías, subenfriamiento del
  condensado), con entrada a la turbina sobrecalentada o como vapor
  saturado seco, y caudal másico o potencia neta como dato. Con un fluido
  orgánico es un **ORC** (R-245fa, R-1233zd(E), isopentano, tolueno,
  R-134a, R-1234yf, amoníaco), con **recuperador** y la clasificación del
  fluido en seco, húmedo o casi isoentrópico. Numera los estados como
  Cengel (los ejemplos 10-5, 10-6 y 10-2 dan 1–7, 1–13 y 1–6). Muestra η
  térmico, las fracciones de extracción,
  trabajo neto, título a la salida de la turbina (aviso si baja de 0,88),
  relación de trabajo de retroceso, comparación con Carnot y con la
  temperatura media de aporte de calor, potencias, la tabla de estados
  (numeración de Cengel) y el ciclo sobre el diagrama T–s (u otro). El
  **procedimiento** lo resuelve estado por estado como con las tablas
  (bomba con v·Δp, regla de la palanca con h_fg y s_fg, balances de los
  calentadores con la fracción despejada, de mayor a menor presión —o, si
  un drenaje bombeado las acopla, el sistema con su verificación—, las
  cañerías y el recuperador, y trabajos y calores con (1 − y)), en LaTeX y
  en el sistema de unidades activo; calcula el **agua de enfriamiento** del
  condensador; un expansor
  muestra cómo lo arma TESPy (red y balances por componente) y otro, **cómo
  aumentar el rendimiento**: barridos de η y del título vs. presión de
  caldera, temperatura de entrada a la turbina y presión del condensador
  (Cengel §10-4), de la presión de cada extracción, que tiene un óptimo
  (§10-6), y de la efectividad del recuperador. Ejemplos precargados de
  Cengel (10-1, 10-2 completo con sus pérdidas, 10-3 a/b/c, 10-4, 10-5,
  10-6, uno basado en 10-2 sin pérdidas y un cerrado con drenaje al
  condensador), una planta con calentadores reales y tres ORC, verificados
  contra el libro y contra el cálculo a mano. Validación con mensajes en
  castellano (vapor que entraría líquido a la turbina, condensador por
  encima de la caldera, recalentamiento que enfriaría, calentador que no
  puede calentar o que necesitaría una extracción negativa, pérdidas que
  congelarían el condensado, desrecalentador sin sobrecalentamiento,
  recuperador con un fluido húmedo, agua de enfriamiento más caliente que
  el vapor…). Fórmulas del vademecum (§3.3, §9.2, §10.1, §10.4, §12, §13)
  y descarga CSV/JSON. _Fases 3.1a, 3.1b y 3.1c cerradas._
- ✅ **Refrigeración por compresión de vapor** resuelta con TESPy (como su
  tutorial de bomba de calor): ciclo **simple**, de **dos etapas con cámara
  de evaporación instantánea** («economizador») o **en cascada** con uno o
  dos refrigerantes (p. ej. CO₂ abajo y amoníaco arriba), ideal o real
  (rendimiento isoentrópico del compresor, sobrecalentamiento,
  subenfriamiento y caídas de presión), como **refrigerador o bomba de
  calor**. Refrigerantes: R-134a, R-1234yf, R-410A, R-32, amoníaco, CO₂
  (subcrítico), propano e isobutano. Los niveles se dan por presión o por
  temperatura de saturación (rocío en el evaporador, burbuja en el
  condensador) y el tamaño por caudal o por capacidad (con las toneladas de
  refrigeración). Muestra COP, Q̇_C, Ẇ, Q̇_H, el COP de Carnot, la
  temperatura de descarga, el caudal volumétrico aspirado, la comparación
  con el ciclo simple equivalente, la tabla de estados (numeración de
  Cengel) y el ciclo sobre el diagrama **log p–h** (una pestaña por
  refrigerante en la cascada; las válvulas sobre su línea de h constante).
  Con las temperaturas de las fuentes, el **segundo principio**: exergía
  destruida en cada componente (tabla, barras e interpretación física),
  trabajo mínimo y rendimiento exergético (Cengel §11-5). El
  **procedimiento** lo resuelve como con las tablas (A-11 a A-13 para el
  R-134a, con la aclaración de su estado de referencia), y los barridos
  muestran cómo cambia el COP con las temperaturas, el compresor, el
  subenfriamiento y el sobrecalentamiento, y el óptimo de la presión de la
  cámara y de la temperatura intermedia de la cascada. Ejemplos de Cengel
  (11-1 a 11-5, 8.ª ed.) verificados contra el libro, y propios (bomba de
  calor aire–agua, cascada CO₂/amoníaco, amoníaco con cámara, heladera con
  isobutano). Validación con mensajes en castellano (CO₂ transcrítico,
  punto triple, cámara sin vapor, cascada con el calor al revés, fuentes
  incompatibles con el refrigerante…). Fórmulas del vademecum (§3.3, §9.3,
  §10.4, §11, §12) y descarga CSV/JSON. _Fase 3.2 cerrada._
- ✅ **Caldera de recuperación (HRSG) de una, dos o tres presiones**:
  arma el **diagrama T–Q** a partir de los gases que entran (temperatura, caudal y
  composición en fracción molar o másica con N₂, O₂, CO₂, H₂O y Ar, o el
  escape de una turbina de gas a metano con exceso de aire λ), la presión
  de evaporación, la temperatura del agua de alimentación, el **pinch**, el
  **approach** y la temperatura del vapor sobrecalentado, o con **vapor
  saturado** (sin sobrecalentador). Balance de energía sección por sección
  (Kehlhofer et al., *Combined-Cycle Gas & Steam Turbine Power Plants*,
  2009): los gases como mezcla de gases ideales (vademecum §5; la entalpía
  de cada componente, del gas ideal de CoolProp) y el agua con IAPWS-95.
  Muestra el caudal de vapor, el calor de cada sección, las temperaturas
  de los gases entre secciones y de chimenea, el aprovechamiento (contra
  15 °C), el punto de rocío de los gases, las tablas de secciones y de
  estados, y la comparación entre vapor saturado y sobrecalentado. El
  diagrama marca el pinch y el approach (el escalón del domo); el
  **procedimiento** hace los balances a mano; los barridos muestran cómo
  cambian el vapor y la chimenea con el pinch, el approach, la presión, la
  temperatura del vapor, la del agua de alimentación y la de los gases.
  Una red de TESPy (`HeatExchanger` en serie, como su tutorial de turbina
  de gas) da lo mismo al 0,02 % en los tests. Validación con mensajes en
  castellano (cruce de temperaturas, chimenea bajo el punto de rocío,
  gases que no alcanzan para evaporar, presión supercrítica…). Fórmulas
  del vademecum (§3.3, §4.8, §5, §12, §13, §16) y descarga CSV/JSON.
  _Fase 3.3 cerrada._ El núcleo también la diseña por **temperatura de
  chimenea** (Cengel 10-9: el pinch pasa a ser un resultado), el modo que
  usa el ciclo combinado. Con **dos o tres niveles de presión** (Fase 3.5)
  los niveles van en **cascada** (Kehlhofer et al., cap. 5): los gases los
  recorren de alta a baja; el economizador de baja calienta toda el agua y
  cada domo manda el líquido que no evapora a la bomba del nivel siguiente.
  Cada nivel tiene su presión, su pinch, su approach y su vapor
  sobrecalentado o saturado. El T–Q muestra la curva del agua «en serrucho»
  (un color por nivel) y el pinch de cada uno; la página compara la misma
  caldera con 1, 2 y 3 presiones (chimenea, calor, vapor) y hace el
  **balance de exergía** (con T₀ = 15 °C): la que gana el agua, la
  **destruida en cada sección** (T₀·S_gen) y la que se va por la chimenea,
  con el rendimiento exergético; con más niveles, más cerca las curvas y
  menos exergía destruida. Procedimiento nivel por nivel, barridos (presión
  de baja y de alta, pinch, agua de alimentación) y validaciones que nombran
  el nivel. Con un nivel coincide exactamente con la caldera de una
  presión; una red de TESPy (un `DropletSeparator` y una `Pump` por domo)
  da lo mismo al 0,2 %. _Fase 3.5 cerrada._
- ✅ **Ciclo combinado gas–vapor de una, dos o tres presiones, con
  recalentamiento**: una **turbina de gas**
  (compresor, cámara de combustión y turbina) cuyo escape alimenta la HRSG
  de una presión, y el vapor mueve el **ciclo de Rankine** de la app, con
  desaireador opcional. La turbina de gas se calcula con calores
  específicos variables (mezclas de gases ideales con la función s°(T),
  como la tabla A-17 de Cengel, y entropías absolutas de NIST-JANAF),
  quemando **metano** (poder calorífico de ISO 6976:2016 a 25 °C; da la
  relación combustible/aire, el exceso de aire λ y la composición de los
  gases) o con el modelo de **aire estándar** (Cengel §9-3), con aire seco
  (el de las tablas de Cengel) o técnico (21 % O₂, vademecum §16.1); la
  misma turbina resuelta con **TESPy** (`Compressor`,
  `DiabaticCombustionChamber`, `Turbine`) queda al lado como control y
  coincide al 0,3 %. La HRSG se diseña por pinch o por temperatura de
  chimenea, con vapor sobrecalentado o saturado, y el agua de alimentación
  es la que entrega la bomba del ciclo de vapor. El tamaño sale del caudal
  de aire o de la potencia neta. Muestra los rendimientos de la turbina de
  gas, del ciclo de vapor y del **ciclo combinado** (con la relación de
  Kehlhofer η_CC = η_TG + η_HRSG·η_TV·(1 − η_TG) y el balance de energía,
  que cierran exactos), las potencias, el caudal de vapor, la chimenea y el
  heat rate; un **diagrama de Sankey** de a dónde va la energía del
  combustible; el **T–s** de la turbina de gas (con el enfriamiento de los
  gases en la HRSG), el **T–Q** de la HRSG y el diagrama del ciclo de
  vapor; las tablas de estados de las tres partes; el **procedimiento** por
  partes, y **barridos** que muestran que el óptimo de la relación de
  presiones del ciclo combinado está por debajo del de la turbina de gas
  sola (también la TIT, la presión del vapor y el pinch o la chimenea).
  Ejemplos: una planta típica de una presión, Cengel 10-9 (ṁ_v/ṁ_g = 0,131,
  η = 48,7 %; además 9-5 y 9-6 en los tests) y una turbina moderna (60,5 %).
  Validación con mensajes en castellano que nombran la parte (TIT por
  debajo de la salida del compresor, combustión con λ < 1, vapor más
  caliente que el escape, desaireador por encima de la caldera…). Fórmulas
  del vademecum (§3.3, §4.8, §5, §10.4, §16) y descarga CSV/JSON. _Fase 3.4
  cerrada._ Con **dos o tres presiones y recalentamiento** (Fase 3.6), la
  HRSG en cascada de la Fase 3.5 con el **recalentador en paralelo** con el
  sobrecalentador de alta (con tres niveles el vapor de media se suma al
  recalentamiento frío y los caudales de alta y media salen de un sistema
  lineal de 2×2) y una **turbina de vapor con admisiones**: el vapor de cada
  nivel entra mezclándose a la presión de su domo. Desaireador opcional y
  **regla de Baumann** (la humedad baja el rendimiento de la parte húmeda de
  la expansión). Muestra el título a la salida de la turbina, el T–Q con el
  recalentador, el T–s del ciclo de vapor con las admisiones, **«¿Cuánto
  ganás con más presiones y recalentamiento?»** (la misma turbina de gas con
  1, 2 y 3 presiones, con y sin recalentamiento, con η_T constante y con
  Baumann) y la **exergía del ciclo de fondo** por componente (HRSG,
  turbinas, mezclas, bombas, condensador y chimenea). Con una presión y sin
  recalentar coincide exactamente con la Fase 3.4; una red de TESPy de todo
  el lado agua–vapor da lo mismo al 0,2 %. _Fase 3.6 cerrada._
- ✅ **Turbina de gas: ciclo Brayton con interenfriamiento,
  recalentamiento y regenerador** (página propia, `/Brayton`): la turbina
  de gas del ciclo combinado con las tres mejoras de Cengel §9-9 y §9-10.
  **Compresión en etapas** con interenfriamiento (relaciones de presión
  iguales, el mínimo trabajo del vademecum §6.3, o presiones a elección),
  **expansión en etapas** con recalentamiento (con metano, una segunda
  cámara que quema en los gases: *combustión secuencial*; con aire
  estándar, un intercambiador) y **regenerador** con efectividad ε (la de
  Cengel §9-9; con combustión, la temperatura del aire se itera), con
  pérdidas de carga en cámaras, interenfriadores y regenerador. Estados
  numerados como Cengel (1–4, 1–6 con regenerador, 1–10 con dos etapas).
  Muestra η, el trabajo neto por kg de aire, la relación de trabajo de
  retroceso, el **T–s** con todas las etapas, las tablas de estados y de
  componentes, **«¿Cuánto ganás con cada mejora?»** (la misma turbina
  simple, con regenerador, con interenfriamiento, con recalentamiento y con
  todo: sin regenerador, interenfriar y recalentar suben el trabajo pero
  bajan el rendimiento) y la **exergía destruida en cada componente** (la
  del combustible ≈ PCI, vademecum §16.13). Procedimiento con las tablas de
  gas ideal, control con TESPy, export y barridos (r_p, TIT, temperatura
  ambiente, ε, la presión intermedia con su óptimo en √(p₁·p₂) y la
  cantidad de etapas, que con todo ideal tiende al ciclo de Ericsson).
  Reproduce los ejemplos 9-5 a 9-8 de Cengel (9-7: η = 36,9 %; 9-8:
  r_bw = 0,304 y η = 35,8 %, y 69,6 % con regenerador ideal); con una etapa
  y sin regenerador es idéntica a la turbina de la Fase 3.4, y TESPy da lo
  mismo dentro de 0,05 puntos de rendimiento. _Fase 3.7 cerrada._

Todas las páginas están pensadas para usarse **desde el celular**: las
ecuaciones del procedimiento se escriben en renglones cortos (una
igualdad por renglón) para que entren en el ancho de la pantalla.

### En desarrollo (roadmap)
- **Ciclo combinado (continuación)**: secciones intercaladas en la HRSG,
  quemadores suplementarios, pérdidas de carga, recirculación del
  precalentador, la turbina de gas con etapas (combustión secuencial) en el
  ciclo combinado y la exergía de toda la planta (con la química del
  combustible).
- **Ciclos termodinámicos**: cogeneración, refrigeración de álabes de la
  turbina de gas, turbinas de propulsión.
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
│   ├── ideal_gas.py       # mezclas de gases ideales: aire, gases de combustión, s°(T)
│   ├── interpolation.py
│   ├── isentropic.py
│   ├── exergy.py
│   ├── combustion/
│   │   ├── stoichiometry.py
│   │   ├── heating_value.py
│   │   └── iso6976.py
│   ├── cycles/
│   │   ├── tespy_utils.py # red de TESPy en SI y errores en castellano
│   │   ├── rankine.py     # Rankine simple, recalentamiento, regeneración, ciclo real y ORC (TESPy)
│   │   ├── rankine_layout.py     # topología y numeración de estados (sin TESPy)
│   │   ├── rankine_procedure.py  # procedimiento «como con las tablas»
│   │   ├── refrigeration.py      # refrigeración simple, con cámara y en cascada (TESPy)
│   │   ├── refrigeration_procedure.py  # su procedimiento, con la exergía destruida
│   │   ├── hrsg.py               # HRSG de una presión: balances, diagrama T–Q
│   │   ├── hrsg_procedure.py     # su procedimiento «a mano»
│   │   ├── hrsg_multi.py         # HRSG de dos y tres presiones (cascada) y su exergía
│   │   ├── hrsg_multi_procedure.py  # su procedimiento nivel por nivel
│   │   ├── brayton.py            # turbina de gas (gases ideales; TESPy de control)
│   │   ├── brayton_procedure.py  # su procedimiento con la función s°
│   │   ├── gas_turbine.py        # Brayton con interenfriamiento, recalentamiento y regenerador
│   │   ├── gas_turbine_procedure.py  # su procedimiento, con la exergía por componente
│   │   ├── combined.py           # ciclo combinado: turbina de gas + HRSG + Rankine
│   │   ├── combined_procedure.py # su procedimiento por partes
│   │   ├── combined_multi.py     # ciclo combinado de 2 y 3 presiones con recalentamiento
│   │   └── combined_multi_procedure.py  # su procedimiento, con las admisiones
│   └── plots.py
├── ui/                    # Helpers de UI (Streamlit): unidades, diagramas, gráficos
│                          # de los ciclos (T–Q, T–s, Sankey), créditos
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

> Barral, P. M. (2026). *TA216 — Tecnología de Calor Avanzada — Apps* (Versión 0.19.0)
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

**Ciclo de Rankine orgánico (ORC)** — clasificación de los fluidos de trabajo
(seco, húmedo, isoentrópico) y estado del arte:

```bibtex
@article{chen2010review,
  author  = {Chen, Huijuan and Goswami, D. Yogi and Stefanakos, Elias K.},
  title   = {A review of thermodynamic cycles and working fluids for the
             conversion of low-grade heat},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {14},
  number  = {9},
  pages   = {3059--3067},
  year    = {2010},
  doi     = {10.1016/j.rser.2010.07.006}
}

@article{quoilin2013orc,
  author  = {Quoilin, Sylvain and Van Den Broek, Martijn and Declaye, S{\'e}bastien
             and Dewallef, Pierre and Lemort, Vincent},
  title   = {Techno-economic survey of {Organic Rankine Cycle} ({ORC}) systems},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {22},
  pages   = {168--186},
  year    = {2013},
  doi     = {10.1016/j.rser.2013.01.028}
}
```

**Turbina de gas y ciclo combinado** — las entropías absolutas a 25 °C y 1 bar
de los gases (N₂, O₂, CO₂, H₂O, Ar) son las de las tablas NIST-JANAF; el
diseño de la HRSG y la relación entre los rendimientos, de Kehlhofer et al.:

```bibtex
@book{chase1998janaf,
  author    = {Chase, Jr., Malcolm W.},
  title     = {{NIST-JANAF} Thermochemical Tables},
  edition   = {4},
  series    = {Journal of Physical and Chemical Reference Data, Monograph 9},
  publisher = {American Chemical Society and American Institute of Physics},
  year      = {1998}
}

@book{kehlhofer2009combined,
  author    = {Kehlhofer, Rolf and Hannemann, Frank and Stirnimann, Franz
               and Rukes, Bert},
  title     = {Combined-Cycle Gas \& Steam Turbine Power Plants},
  edition   = {3},
  publisher = {PennWell},
  year      = {2009}
}
```

---

## Notas de uso

- Las unidades por defecto son: presión en bar(a), temperatura en °C,
  entalpía en kJ/kg, entropía en kJ/(kg·K), título adimensional.
- El título x solo se informa dentro de la campana; fuera de ella la
  página muestra "—" (en la API de bajo nivel, `StatePoint.x = -1`).
- Las propiedades del agua salen de CoolProp con la formulación
  IAPWS-95; pueden diferir en el último decimal de las tablas impresas.
- Las tablas del R-134a de Cengel (A-11 a A-13) toman h = s = 0 para el
  líquido saturado a −40 °C; CoolProp usa la referencia del IIR (h =
  200 kJ/kg y s = 1 kJ/(kg·K) a 0 °C). Las h, u y s de la app quedan
  148,14 kJ/kg y 0,7956 kJ/(kg·K) por encima de las del libro; las
  diferencias (calores, trabajos, COP) son iguales. Las páginas lo avisan.
  Las de los fluidos orgánicos (ORC), de las ecuaciones de estado de
  Helmholtz de CoolProp.
- Los ciclos usan la numeración de estados de Çengel & Boles,
  *Termodinámica* (cap. 10), y sus ejemplos precargados se verificaron
  contra los resultados del libro.
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

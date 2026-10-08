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
- ✅ **Psicrometría: aire húmedo, procesos de acondicionamiento y torres de
  enfriamiento** (página `/Psicrometria`): el modelo del vademecum §14 (aire
  seco y vapor como gases ideales con c_p constante; la presión de
  saturación de IAPWS, sobre agua líquida o sobre hielo). El **estado** sale
  de la presión (a nivel del mar, por altura con la atmósfera estándar de
  ASHRAE o dada) y de dos datos cualquiera entre T, φ, bulbo húmedo, punto
  de rocío, ω y h (trece pares): presiones parciales, ω, φ, μ, h, c_p, v, ρ,
  R, bulbo húmedo (saturación adiabática; bulbo de hielo bajo 0 °C), punto
  de rocío (o de escarcha), entropía y **exergía** (ψ_tm + ψ_qu), las masas
  en un recinto y la comparación con el modelo de gas real de CoolProp
  (ASHRAE RP-1485). Una **carta psicrométrica interactiva** a cualquier
  presión: al pasar el mouse muestra el estado de cada punto y al tocarlo lo
  carga como dato. Un **tren de procesos** (§14.12: calentamiento o
  enfriamiento sensible, calentamiento con humidificación, serpentín de
  enfriamiento y deshumidificación, humidificación adiabática y mezcla) con
  el recorrido en la carta, el calor, el agua y la **exergía destruida** en
  cada proceso, y la **torre de enfriamiento** (caudal de aire, reposición,
  rango, aproximación, efectividad). Procedimiento con las fórmulas del
  vademecum, export y barridos (altura, aire exterior, aproximación de la
  torre). Reproduce los ejemplos 14-1 a 14-9 de Cengel (14-6: 511 kJ/min y
  0,131 kg/min; 14-8: ω₃ = 0,0122, φ₃ = 89 %, T₃ = 19 °C; 14-9: 1,80 kg/s de
  reposición). _Fase 4 cerrada._
- ✅ **Combustión: estequiometría, humos, llama adiabática y calor** (página
  `/Combustion`): el modelo del vademecum §16 con los **polinomios NASA** de
  9 coeficientes de McBride, Zehe y Gordon (2002), la fuente de la tabla de
  §16.12 (32 especies de 200 a 6000 K, en `data/nasa9_thermo.csv`). El
  combustible puede ser una mezcla gaseosa (gas natural, GLP, hidrógeno,
  biogás o los componentes que quieras), un líquido puro (octano, metanol,
  etanol, querosén, propano líquido) o un sólido o líquido pesado por su
  análisis elemental, con el PCS como dato (carbón, bagazo, fueloil). El
  comburente es aire técnico, aire seco u oxígeno, con la humedad del
  ambiente y precalentado si hace falta. La cantidad de aire se da como λ,
  exceso, % de aire teórico, φ, AC o el O₂ medido en los humos secos.
  Resultados: el aire teórico y real, los **humos** en base húmeda, seca y
  másica (M_g, GC, volumen normal, CO₂ máximo), los **puntos de rocío** del
  agua y **ácido** (Verhoff y Banchero, 1974), el PCS y el PCI, la
  **temperatura adiabática de llama** con combustión completa y **con
  disociación** (equilibrio químico por minimización de la energía libre de
  Gibbs, como NASA CEA, verificado con las K_p), la combustión con CO o con
  defecto de aire, la combustión a volumen constante, el **calor** con los
  humos a una temperatura dada (con condensación: sobre el PCI el
  rendimiento puede pasar el 100 %), con el reparto del PCI, y el **segundo
  principio** (entropía generada y exergía destruida). En el modo **análisis
  de humos**, λ, el aire, el agua y la pérdida por CO salen del O₂ (y el CO)
  que mide un analizador o de un Orsat, con el **diagrama de combustión**.
  Procedimiento con los valores reemplazados en los tres sistemas, export y
  barridos (λ, precalentamiento del aire, temperatura de los humos).
  Reproduce los ejemplos 15-1 a 15-4, 15-6 a 15-8, 15-10 y 15-11 de Cengel
  (15-10: 1788 K, el libro 1789 K; 15-11: 871 700 kJ/kmol de calor y
  818 400 kJ/kmol de exergía destruida, el libro 871 400 y 818 000), y la
  llama de equilibrio coincide con Cantera dentro de 0,004 K. _Fase 5
  cerrada._
- ✅ **Poder calorífico por correlaciones** (página `/Poder_Calorifico`): el
  PCS de un carbón, una biomasa o un combustible líquido estimado con su
  análisis **elemental** (Dulong; Boie, 1953; Channiwala y Parikh, 2002) o
  **inmediato** (Parikh, Channiwala y Ghosal, 2005; Cordero et al., 2001),
  con los datos en cualquier base (tal cual, seca o seca y sin cenizas, con
  el O o el carbono fijo por diferencia). Resultados: el PCS de cada
  correlación en las tres bases, el PCI (el agua que forma el H y la
  humedad, con h_fg a 25 °C) y la humedad con la que el PCI se anula, la
  **comparación de las correlaciones** entre sí y contra el PCS medido o el
  exacto (las h_f de una sustancia pura), y avisos cuando una correlación
  se usa fuera de su rango o de su tipo de combustible. «¿Qué tan buenas
  son?» las prueba contra **536 biomasas** con el PCS medido (Ghugare et
  al., 2014) y **cinco carbones** del Argonne Premium Coal Sample Program
  (Vorres, 1990): en biomasa Channiwala y Parikh se equivoca 4,9 % en
  promedio, Boie 5,2 % y Dulong 11,5 % (subestima 10 % porque supone el O
  ya unido al H); con el inmediato, los carbones bituminosos quedan 10–23 %
  abajo. Procedimiento en los tres sistemas, export y barrido de la
  humedad. En `/Combustion`, el PCS de un análisis elemental se puede
  estimar con una de estas correlaciones. _Fase 6 cerrada._
- ✅ **Exergía: física, química y por componente** (página `/Exergia`, vademecum
  §11 y §16.13). **Física**: la exergía de flujo ψ y la de la masa φ de
  cualquier estado de un fluido (los pares de Propiedades, con CoolProp), con su
  parte térmica y mecánica (Kotas, 1985), la cinética y la potencial, por kg,
  para una masa o para un caudal, y el trabajo reversible de un proceso 1 → 2; la
  exergía de un calor a temperatura T (factor de Carnot) y la de un cuerpo que se
  enfría hasta el ambiente (fuente finita); el estado se ve en el h–s con la
  recta h₀ + T₀·(s − s₀), cuya distancia vertical es ψ. **Química**: la exergía
  química estándar de 42 sustancias en los dos ambientes de referencia de la
  tabla A-26 de Moran y Shapiro (Szargut, Morris y Steward, 1988, modelo II, y
  Ahrendts, 1980, modelo I) y el **método de Szargut** (Δg_f con los polinomios
  NASA + la exergía de los elementos), que reproduce la tabla dentro de 0,15 % en
  los hidrocarburos; la de una mezcla de gases (con el agua que condensa a 25 °C)
  y la de un combustible sólido o líquido por su análisis elemental, con las β
  de Szargut y Styrylska (1964) para carbones, madera y biomasa, y líquidos (con
  la humedad y el azufre como Kotas): con las biomasas de Ghugare de la Fase 6,
  e/PCS ≈ 1,05 en promedio. **Por componente**: un Rankine, una refrigeración,
  una turbina de gas o un ciclo combinado (un ejemplo o el último que calculaste
  en su página), con la exergía que gasta (Ẋ_F), produce (Ẋ_P), destruye
  (Ẋ_D = T₀·Ṡ_gen) y pierde (Ẋ_L) cada componente, su rendimiento ε y las
  razones y_D e y*_D (Bejan, Tsatsaronis y Moran, 1996), el **diagrama de
  Grassmann** y la exergía de cada corriente; el combustible entra con su
  exergía química (Szargut) o ≈ PCI (así η_II = η térmico). Todos los balances
  cierran a 10⁻⁹. Reproduce Cengel 10-8 (ψ = 1162 kJ/kg a la entrada de la
  turbina; 1110 kJ/kg destruidos en la caldera y 414 kJ/kg en el condensador)
  y los ejemplos del cap. 8 (tanque de aire comprimido, 281 MJ; compresor de
  R-134a, 38,0 kJ/kg; viento, 70,7 kW; hogar, 2195 Btu/s; bloque de hierro,
  8191 kJ). Procedimiento en los tres sistemas y export. _Fase 7 cerrada._

Todas las páginas están pensadas para usarse **desde el celular**: las
ecuaciones del procedimiento se escriben en renglones cortos (una
igualdad por renglón) para que entren en el ancho de la pantalla.

### En desarrollo (roadmap)
- **Ciclo combinado (continuación)**: secciones intercaladas en la HRSG,
  quemadores suplementarios, pérdidas de carga, recirculación del
  precalentador y la turbina de gas con etapas (combustión secuencial) en el
  ciclo combinado.
- **Ciclos termodinámicos**: cogeneración, refrigeración de álabes de la
  turbina de gas, turbinas de propulsión.
- **Psicrometría (continuación)**: cargas del local y factor de calor
  sensible, serpentín con ADP y factor de bypass, número de Merkel para
  dimensionar torres, niebla (mezclas sobresaturadas).
- **Combustión (continuación)**: hollín (carbono sólido en el equilibrio),
  NOx con cinética (mecanismo de Zeldovich), inquemados sólidos en las
  cenizas, los polinomios NASA en la turbina de gas y el ciclo combinado.
- **Poder calorífico (continuación)**: otras correlaciones (Mendeleev,
  IGT, Sheng y Azevedo; Goutal para carbones con el análisis inmediato),
  el C, el H y el O estimados desde el inmediato, y el PCS a volumen
  constante de la bomba contra el de presión constante (ISO 18125:2017).
- **Exergía (continuación)**: exergoeconomía (el costo de la exergía en cada
  componente, SPECO), el ambiente a elección para la exergía química
  (corrección a T₀ ≠ 25 °C), la tabla de Szargut (2007), y la HRSG, la
  psicrometría y la combustión en el diagrama de Grassmann.
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
│   ├── psychrometrics.py  # aire húmedo (vademecum §14): estado, exergía, carta
│   ├── hvac.py            # procesos de acondicionamiento y torre de enfriamiento
│   ├── psychrometrics_procedure.py  # su procedimiento «como en el pizarrón»
│   ├── interpolation.py
│   ├── isentropic.py
│   ├── exergy/
│   │   ├── physical.py    # exergía física: estado muerto, ψ, φ, calor, fuente finita
│   │   ├── chemical.py    # exergía química: Szargut y Ahrendts, mezclas, combustibles
│   │   ├── plant.py       # exergía por componente (F, P, D, L) y diagrama de Grassmann
│   │   └── exergy_procedure.py  # su procedimiento «como en el pizarrón»
│   ├── combustion/
│   │   ├── thermo.py      # polinomios NASA-9: c_p, h, s°, g° de cada especie
│   │   ├── fuels.py       # combustibles: mezclas, líquidos y análisis elemental
│   │   ├── stoichiometry.py  # aire, exceso, productos, rocíos, Orsat
│   │   ├── equilibrium.py # equilibrio químico (Gibbs, como NASA CEA)
│   │   ├── combustion.py  # llama, calor, segundo principio, análisis de humos
│   │   ├── combustion_procedure.py  # su procedimiento «como en el pizarrón»
│   │   ├── heating_value.py  # PCS por correlaciones (elemental e inmediato), bases, validación
│   │   ├── heating_value_procedure.py  # su procedimiento, con el aporte de cada componente
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
│                          # de los ciclos (T–Q, T–s, Sankey), carta psicrométrica,
│                          # gráficos de la combustión, del poder calorífico y de la
│                          # exergía (Grassmann), créditos
├── tests/                 # pytest (+ páginas con streamlit.testing)
├── data/                  # tablas: ISO 6976, polinomios NASA-9, biomasas (Ghugare),
│                          # carbones de Argonne con el PCS medido y exergías
│                          # químicas estándar (Szargut, Ahrendts)
├── scripts/               # extracción de datos (los polinomios de thermo.inp de NASA CEA)
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

> Barral, P. M. (2026). *TA216 — Tecnología de Calor Avanzada — Apps* (Versión 0.23.0)
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

**IAPWS (2011)** — presión de sublimación del hielo (aire húmedo bajo 0 °C):

```bibtex
@article{wagner2011sublimation,
  author  = {Wagner, Wolfgang and Riethmann, Thomas and Feistel, Rainer and
             Harvey, Allan H.},
  title   = {New Equations for the Sublimation Pressure and Melting Pressure
             of {H$_2$O} Ice {Ih}},
  journal = {Journal of Physical and Chemical Reference Data},
  volume  = {40},
  number  = {4},
  pages   = {043103},
  year    = {2011},
  doi     = {10.1063/1.3657937}
}
```

**Aire húmedo real (ASHRAE RP-1485)** — el modelo de CoolProp con el que se
compara el ideal:

```bibtex
@article{herrmann2009moistair,
  author  = {Herrmann, Sebastian and Kretzschmar, Hans-Joachim and Gatley, Donald P.},
  title   = {Thermodynamic Properties of Real Moist Air, Dry Air, Steam,
             Water, and Ice ({RP}-1485)},
  journal = {HVAC\&R Research},
  volume  = {15},
  number  = {5},
  pages   = {961--986},
  year    = {2009},
  doi     = {10.1080/10789669.2009.10390874}
}
```

**Exergía del aire húmedo y del agua** — Wepfer, W. J., Gaggioli, R. A. y
Obert, E. F. (1979). *Proper evaluation of available energy for HVAC*.
ASHRAE Transactions, 85(1), 214–230.

**ASHRAE Handbook—Fundamentals** (2017), cap. 1, *Psychrometrics*: atmósfera
estándar, entalpía del hielo y bulbo húmedo termodinámico. ASHRAE, Atlanta.

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

**Combustión** — los polinomios NASA de 9 coeficientes (de `thermo.inp` de
[NASA CEA](https://github.com/nasa/cea), licencia Apache 2.0), el equilibrio
químico por minimización de la energía libre de Gibbs, el punto de rocío
ácido y los combustibles de los ejemplos:

```bibtex
@techreport{mcbride2002nasa,
  author      = {McBride, Bonnie J. and Zehe, Michael J. and Gordon, Sanford},
  title       = {{NASA} {Glenn} Coefficients for Calculating Thermodynamic
                 Properties of Individual Species},
  institution = {NASA Glenn Research Center},
  number      = {NASA/TP-2002-211556},
  year        = {2002}
}

@techreport{gordon1994cea,
  author      = {Gordon, Sanford and McBride, Bonnie J.},
  title       = {Computer Program for Calculation of Complex Chemical
                 Equilibrium Compositions and Applications. {I}. Analysis},
  institution = {NASA Lewis Research Center},
  number      = {NASA RP-1311},
  year        = {1994}
}

@article{verhoff1974dew,
  author  = {Verhoff, Francis H. and Banchero, Julius T.},
  title   = {Predicting dew points of flue gases},
  journal = {Chemical Engineering Progress},
  volume  = {70},
  number  = {8},
  pages   = {71--72},
  year    = {1974}
}

@article{channiwala2002hhv,
  author  = {Channiwala, S. A. and Parikh, P. P.},
  title   = {A unified correlation for estimating {HHV} of solid, liquid and
             gaseous fuels},
  journal = {Fuel},
  volume  = {81},
  number  = {8},
  pages   = {1051--1063},
  year    = {2002},
  doi     = {10.1016/S0016-2361(01)00131-4}
}

@book{hugot1986cane,
  author    = {Hugot, E.},
  title     = {Handbook of Cane Sugar Engineering},
  edition   = {3},
  publisher = {Elsevier},
  year      = {1986}
}
```

**Poder calorífico por correlaciones** — las correlaciones (Dulong y Boie en
la forma que tabulan Channiwala y Parikh, 2002, citada arriba) y los datos
con que se validan: las biomasas de Ghugare et al. (2014), tal como las
distribuye el paquete de R `modeldata` (Posit, licencia MIT, ver
`data/LICENSE-modeldata.txt`), y los carbones de Argonne (Vorres, 1990):

```bibtex
@article{boie1953fuel,
  author  = {Boie, W.},
  title   = {Fuel technology calculations},
  journal = {Energietechnik},
  volume  = {3},
  pages   = {309--316},
  year    = {1953}
}

@article{parikh2005proximate,
  author  = {Parikh, Jigisha and Channiwala, S. A. and Ghosal, G. K.},
  title   = {A correlation for calculating {HHV} from proximate analysis of
             solid fuels},
  journal = {Fuel},
  volume  = {84},
  number  = {5},
  pages   = {487--494},
  year    = {2005}
}

@article{cordero2001proximate,
  author  = {Cordero, T. and Marquez, F. and Rodriguez-Mirasol, J. and
             Rodriguez, J. J.},
  title   = {Predicting heating values of lignocellulosics and carbonaceous
             materials from proximate analysis},
  journal = {Fuel},
  volume  = {80},
  number  = {11},
  pages   = {1567--1571},
  year    = {2001},
  doi     = {10.1016/S0016-2361(01)00034-5}
}

@article{ghugare2014biomass,
  author  = {Ghugare, Suhas B. and Tiwary, Shishir and Elangovan, Vinayagam
             and Tambe, Sanjeev S.},
  title   = {Prediction of Higher Heating Value of Solid Biomass Fuels Using
             Artificial Intelligence Formalisms},
  journal = {BioEnergy Research},
  volume  = {7},
  pages   = {681--692},
  year    = {2014}
}

@article{vorres1990argonne,
  author  = {Vorres, Karl S.},
  title   = {The {Argonne} Premium Coal Sample Program},
  journal = {Energy \& Fuels},
  volume  = {4},
  number  = {5},
  pages   = {420--426},
  year    = {1990},
  doi     = {10.1021/ef00023a001}
}
```

**Normas de las bases y los análisis**: ASTM D3180-25 (*Calculating Coal and
Coke Analyses from As-Determined to Different Bases*), ASTM D3176-24
(*Ultimate Analysis of Coal and Coke*) y ASTM D3172-13(2021)e1 (*Proximate
Analysis of Coal and Coke*), ASTM International.

**Exergía** — las exergías químicas estándar de los dos ambientes de referencia
(los valores digitales, de `tespy/data/ChemEx` de TESPy 0.7.9, licencia MIT,
contrastados con la tabla A-26 de Moran y Shapiro), las β de los combustibles
sólidos y líquidos, la parte térmica y mecánica de ψ y el análisis por
componente (combustible, producto, destrucción y pérdida):

```bibtex
@book{szargut1988exergy,
  author    = {Szargut, Jan and Morris, David R. and Steward, Frank R.},
  title     = {Exergy Analysis of Thermal, Chemical, and Metallurgical Processes},
  publisher = {Hemisphere},
  address   = {New York},
  year      = {1988}
}

@article{ahrendts1980reference,
  author  = {Ahrendts, Joachim},
  title   = {Reference states},
  journal = {Energy},
  volume  = {5},
  number  = {8--9},
  pages   = {666--677},
  year    = {1980},
  doi     = {10.1016/0360-5442(80)90087-0}
}

@article{szargut1964fuels,
  author  = {Szargut, Jan and Styrylska, Teresa},
  title   = {Angen{\"a}herte {B}estimmung der {E}xergie von {B}rennstoffen},
  journal = {Brennstoff-W{\"a}rme-Kraft},
  volume  = {16},
  number  = {12},
  pages   = {589--596},
  year    = {1964}
}

@book{kotas1985exergy,
  author    = {Kotas, Tadeusz J.},
  title     = {The Exergy Method of Thermal Plant Analysis},
  publisher = {Butterworths},
  address   = {London},
  year      = {1985}
}

@book{bejan1996thermal,
  author    = {Bejan, Adrian and Tsatsaronis, George and Moran, Michael},
  title     = {Thermal Design and Optimization},
  publisher = {Wiley},
  address   = {New York},
  year      = {1996}
}

@book{moran2014fundamentals,
  author    = {Moran, Michael J. and Shapiro, Howard N. and Boettner, Daisie D.
               and Bailey, Margaret B.},
  title     = {Fundamentals of Engineering Thermodynamics},
  edition   = {8},
  publisher = {Wiley},
  year      = {2014}
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

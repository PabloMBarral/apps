# CLAUDE.md

> Este archivo da contexto al agente Claude Code. Mantenelo actualizado:
> cualquier cambio de arquitectura, dependencia o convención debería
> reflejarse acá.

## Proyecto

Suite de herramientas didácticas de ingeniería térmica para la materia
**TA216 — Tecnología de Calor Avanzada** (FIUBA). Streamlit + CoolProp + TESPy + fluprodia.

Repo hermano de fórmulas teóricas:
[PabloMBarral/vademecum-termo](https://github.com/PabloMBarral/vademecum-termo).
Cada página de la app debe linkear la sección correspondiente del vademecum.

Autor: Pablo M. Barral (pbarral@fi.uba.ar, ORCID 0000-0003-1125-4199).
Licencia: MIT.

## Stack

- Python 3.11+
- **Streamlit** — UI (multipágina con `st.navigation`; páginas en `app_pages/`)
- **CoolProp** — propiedades termofísicas punto a punto
- **TESPy** — simulación de ciclos termodinámicos
- **fluprodia** — diagramas de propiedades de fluidos
- **NumPy, SciPy, pandas, matplotlib** — utilitarios numéricos y plots base
- **PyYAML** — lectura del `CITATION.cff` (página Acerca de)
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
├── app_pages/                 # Una página por módulo (numeradas). NO se llama
│                              # pages/: ver «Navegación» más abajo.
│   ├── 1_Propiedades.py       # ✅ Fase 1.6 — Estado completo del agua (y otros
│   │                          # fluidos): región, tablas, procedimiento,
│   │                          # diagrama, tabla de estados para ciclos.
│   ├── 2_Interpolacion.py     # ✅ Fase 1.1 (+ teoría y export, Fase 1.7)
│   ├── 3_Isoentropicos.py     # ✅ Fase 1.3 + 1.5a (diagrama del proceso) +
│   │                          # 1.5b y validaciones / defaults por fluido (1.7)
│   ├── 5_Rankine.py           # ✅ Fase 3.1a — Rankine simple / con
│   │                          # recalentamiento (TESPy), Carnot, barridos de η.
│   │                          # Fase 3.1b: regeneración (hasta 3 calentadores),
│   │                          # rótulos con la numeración del layout, agua de
│   │                          # enfriamiento, barrido de la presión de extracción.
│   │                          # Fase 3.1c: fluido de trabajo (ORC), pérdidas,
│   │                          # calentadores reales, recuperador, barrido de ε.
│   ├── 6_Refrigeracion.py     # ✅ Fase 3.2 — Compresión de vapor: simple, cámara
│   │                          # de evaporación instantánea y cascada (1 o 2
│   │                          # refrigerantes), refrigerador o bomba de calor,
│   │                          # niveles por p o por T_sat, ciclo real, segundo
│   │                          # principio (barras), pestañas por fluido, barridos.
│   ├── 7_Psicrometria.py      # ✅ Fase 4 — Aire húmedo (vademecum §14): estado con
│   │                          # 13 pares de datos, presión por altura, carta
│   │                          # psicrométrica interactiva (tocar carga el estado),
│   │                          # comparación con CoolProp, tren de procesos de
│   │                          # acondicionamiento y torre de enfriamiento.
│   ├── 8_Combustion.py        # ✅ Fase 5 — Combustión (vademecum §16): mezcla
│   │                          # gaseosa, líquido o análisis elemental; aire
│   │                          # técnico, seco u O₂, húmedo y precalentado; aire de
│   │                          # seis formas; humos, rocíos, PCS/PCI, llama completa
│   │                          # y con disociación (K_p), calor con condensación,
│   │                          # segundo principio, análisis de humos (O₂ u Orsat)
│   │                          # con el diagrama de combustión, barridos.
│   ├── 9_Poder_Calorifico.py  # ✅ Fase 6 — PCS por correlaciones (Dulong, Boie,
│   │                          # Channiwala–Parikh; Parikh, Cordero) en cualquier
│   │                          # base, PCI y W*, comparación contra el medido o el
│   │                          # exacto, validación con 536 biomasas y 5 carbones.
│   ├── 4_ISO6976.py           # ✅ Fase 2.3 (matriz identidad; teoría y export 1.7)
│   ├── 10_HRSG.py             # ✅ Fase 3.3 — HRSG de una presión: diagrama T–Q
│   │                          # (plotly) con vapor sobrecalentado o saturado,
│   │                          # composición con λ o cargada, secciones, estados,
│   │                          # comparación saturado/sobrecalentado, barridos.
│   │                          # Fase 3.5: selector de 1, 2 o 3 presiones (cascada),
│   │                          # T–Q en serrucho, exergía por sección, comparación
│   │                          # 1/2/3 presiones, procedimiento por nivel.
│   ├── 11_Exergia.py          # ✅ Fase 7 — Exergía (vademecum §11 y §16.13):
│   │                          # física de un fluido (ψ, φ, térmica y mecánica,
│   │                          # cinética y potencial; masa o caudal; proceso 1 → 2),
│   │                          # de un calor y de un cuerpo; química de una
│   │                          # sustancia (Szargut/Ahrendts y método de Szargut),
│   │                          # de una mezcla y de un combustible (Szargut y
│   │                          # Styrylska); una planta por componente (Rankine,
│   │                          # refrigeración, turbina de gas, ciclo combinado)
│   │                          # con el diagrama de Grassmann.
│   ├── 12_Ciclo_Combinado.py  # ✅ Fase 3.4 — Turbina de gas (metano o aire estándar)
│   │                          # + HRSG de una presión (por pinch o por chimenea) +
│   │                          # Rankine con desaireador: métricas, Sankey, T–s de
│   │                          # la TG, T–Q, ciclo de vapor, control con TESPy,
│   │                          # procedimiento por partes, barridos.
│   │                          # Fase 3.6: 1, 2 o 3 presiones y recalentamiento
│   │                          # (selector), Baumann, T–s con admisiones,
│   │                          # comparación de configuraciones, exergía del
│   │                          # ciclo de fondo, TESPy del lado agua–vapor.
│   ├── 13_Brayton.py          # ✅ Fase 3.7 — Turbina de gas con interenfriamiento,
│   │                          # recalentamiento (combustión secuencial) y
│   │                          # regenerador: T–s con las etapas, estados y
│   │                          # componentes, «¿Cuánto ganás con cada mejora?»,
│   │                          # exergía por componente, TESPy, barridos.
│   ├── 14_Transferencia_de_Calor.py # ✅ Fase 8.1 — Conducción (red de
│   │                          # resistencias de pared, caño o esfera con
│   │                          # contacto, partes en paralelo y radio crítico),
│   │                          # aletas (recta, aguja, anular; arreglos) y
│   │                          # convección (19 correlaciones de Nusselt:
│   │                          # externa, en un tubo y natural).
│   ├── 15_Radiacion.py        # ✅ Fase 8.2 — Cuerpo negro (Planck, Wien, S–B,
│   │                          # bandas; ε(λ) y α para el sol), factores de forma
│   │                          # (8 geometrías), dos superficies con pantallas,
│   │                          # recintos por radiosidades y radiación con
│   │                          # convección (y la termocupla).
│   ├── 16_Intercambiadores.py # ✅ Fase 8.2 — Verificación (ε-NTU),
│   │                          # dimensionamiento (ε-NTU y LMTD con F), ensayo con
│   │                          # las cuatro temperaturas y U global con
│   │                          # ensuciamiento; comparación de tipos y exergía.
│   ├── 17_Gases_Ideales.py    # ✅ Fase 9.1 — Un gas entre dos estados (c_p a
│   │                          # 25 °C, a la T media o NASA, y el error), mezclas
│   │                          # (composición, Dalton, Amagat, entropía) y mezcla
│   │                          # adiabática (tanque o cámara, S_gen por corriente),
│   │                          # transformaciones (los cinco caminos en el p–v y el
│   │                          # T–s, etapas y el n de dos estados).
│   └── 99_Acerca.py           # ✅ 0.25.1 — Créditos, cómo citar la app (APA y
│                              # BibTeX), las fuentes del CITATION.cff por tipo
│                              # (con búsqueda y la bibliografía .bib), las
│                              # licencias de los datos y las versiones.
├── core/                      # Lógica pura, sin dependencia de Streamlit
│   ├── __init__.py
│   ├── units.py               # Conversiones simples + normalizador de
│   │                          # headers (Fase 1.2). NO importa Streamlit.
│   ├── units_system.py        # ✅ Fase 1.4 — Sistema global SI/Técnico/Inglés.
│   │                          # Tabla (kind, system) → (factor, offset, label),
│   │                          # API format_quantity / convert_*_si / unit_label.
│   │                          # Fase 1.6: + ΔT, ρ, velocidad, μ, k, difusividad.
│   │                          # Fase 3.1a: + caudal másico y potencia.
│   │                          # Fase 3.2: + caudal volumétrico (volume_flow).
│   │                          # Fase 5: + entalpía y entropía molares (J/mol,
│   │                          # kJ/kmol, Btu/lbmol).
│   │                          # Fase 7: + masa (kg, lb), energía (J, kJ, Btu)
│   │                          # y longitud (m, ft).
│   │                          # Fase 8: + h, q″, R térmica, área, small_length
│   │                          # (mm, in), heat_rate y linear_heat_rate (en W,
│   │                          # no kW), heat_capacity_rate (ṁ·c_p = h·A),
│   │                          # inverse_length (m de una aleta),
│   │                          # expansion_coefficient (β), acceleration (g) y
│   │                          # pressure_drop (Pa | Pa | lbf/ft²).
│   │                          # Fase 8.2: + absolute_temperature (K | K | °R),
│   │                          # wavelength (μm), wavelength_temperature (λT),
│   │                          # spectral_emissive_power (E_bλ por μm),
│   │                          # fouling_resistance (R″_f) y entropy_rate (W/K).
│   │                          # Fase 9.1: + volume (m³ | m³ | ft³), entropy
│   │                          # (J/K | kJ/K | Btu/°R), entropy_flow, molar_mass
│   │                          # (kg/mol | kg/kmol | lb/lbmol) y amount (mol |
│   │                          # kmol | lbmol).
│   ├── fluids.py              # ✅ Fase 1.6 — Wrappers sobre CoolProp:
│   │                          # StatePoint/state_from_pair (cálculo) y
│   │                          # FluidState/fluid_state_from_pair (estado completo:
│   │                          # región, saturación, transporte, validación con
│   │                          # mensajes al alumno, suggested_inputs).
│   │                          # Fase 3.1c: + R-245fa, R-1233zd(E), isopentano, tolueno.
│   │                          # Fase 3.2: + R-32, propano, isobutano;
│   │                          # textbook_reference_offset (tablas del R-134a).
│   ├── state_report.py        # ✅ Fase 1.6 — Presentación pura de un FluidState:
│   │                          # tablas, notas, procedimiento LaTeX por sistema
│   │                          # de unidades, export JSON/CSV, tabla de estados.
│   │                          # Fase 3.2: textbook_reference_note (R-134a).
│   ├── latex.py               # ✅ Fase 1.7 — latex_number/unit/quantity,
│   │                          # latex_chain (una igualdad por renglón) y
│   │                          # latex_paren (negativos tras un signo).
│   ├── export.py              # ✅ Fase 1.7 — flatten / dict_to_csv: el dict de
│   │                          # un resultado → CSV (campo, valor) y JSON.
│   ├── citation.py            # ✅ 0.25.1 — CITATION.cff → Citation (PyYAML),
│   │                          # citation_apa / reference_apa (APA 7 en castellano,
│   │                          # texto o Markdown), bibliography_bibtex,
│   │                          # group_references, reference_matches,
│   │                          # DATA_SOURCES, LIBRARIES y package_info.
│   ├── interpolation.py       # ✅ Fase 1.1 — Interpolación lineal y doble entrada
│   │                          # (+ interpolation_to_dict, Fase 1.7).
│   ├── isentropic.py          # ✅ Fase 1.3 — Turbina / compresor / bomba; multietapa.
│   │                          # Fase 1.7: validaciones al alumno, DeviceDefaults /
│   │                          # suggested_device_inputs, pasos por sistema (1.5b),
│   │                          # tabla de estados y *_to_dict para exportar.
│   ├── diagrams.py            # ✅ Fase 1.5a — Wrappers tipados sobre fluprodia.
│   │                          # FLUPRODIA_UNITS, DEFAULT_RANGES, build_diagram,
│   │                          # isentropic/isobaric/isothermal_process, overlays,
│   │                          # axis_window_si, cycle_overlays (Fase 1.6).
│   │                          # Fase 3.1c: con fricción, casi isobárica.
│   │                          # Fase 3.2: válvulas a h constante (isenthalpic).
│   ├── ideal_gas.py           # ✅ Fase 3.4 — Mezclas de gases ideales (se mudó de
│   │                          # hrsg.py, que lo reexporta): FlueGas con h, cp, s°(T),
│   │                          # s(T, p) absoluta (NIST-JANAF), T_isentropic, T_from_h;
│   │                          # AIR_DRY / AIR_TECHNICAL, exhaust_composition(λ),
│   │                          # combustion_products(aire, átomos, λ).
│   │                          # Fase 3.7: bajo la T mínima de CoolProp (el agua
│   │                          # bajo 0,01 °C), gas ideal con c_p constante.
│   ├── psychrometrics.py      # ✅ Fase 4 — Aire húmedo (vademecum §14): p_vs de IAPWS
│   │                          # (líquido con CoolProp, hielo con IAPWS 2011),
│   │                          # MoistAirState, moist_air_state (13 pares), bulbo
│   │                          # húmedo y rocío, exergía ψ_tm + ψ_qu, DeadState,
│   │                          # water_exergy, altura (ASHRAE), comparación con
│   │                          # HAPropsSI, líneas y grilla de la carta, notas,
│   │                          # ejemplos, barrido de la altura y export.
│   ├── hvac.py                # ✅ Fase 4 — Procesos (§14.12): SensibleProcess,
│   │                          # HeatingHumidification, CoolingDehumidification,
│   │                          # AdiabaticHumidification, AdiabaticMixing; solve_hvac
│   │                          # (tren numerado, exergía por proceso), torre de
│   │                          # enfriamiento, notas, ejemplos, barridos y export.
│   ├── psychrometrics_procedure.py # ✅ Fase 4 — moist_air_steps, hvac_steps y
│   │                          # cooling_tower_steps.
│   ├── heat_transfer/         # ✅ Fase 8.1
│   │   ├── conduction.py      # Layer (capas, ParallelPart, contacto R''_c),
│   │   │                      # Boundary (fluido, superficie o calor dado),
│   │   │                      # solve_conduction (red en serie, U, r_cr),
│   │   │                      # temperature_profile, insulation_sweep, notas,
│   │   │                      # ejemplos de Cengel y Ghajar cap. 3 y export.
│   │   ├── fins.py            # FinInputs (recta, aguja, anular; cinco puntas),
│   │   │                      # solve_fin (tabla 3.4; Bessel escaladas en la
│   │   │                      # anular), fin_profile, tip_comparison,
│   │   │                      # efficiency_curve, arreglos (η_o) y export.
│   │   ├── convection.py      # Una función por correlación y CORRELATIONS;
│   │   │                      # fluid_properties (CoolProp, β), solve_external,
│   │   │                      # solve_natural, solve_internal (T_m iterada),
│   │   │                      # dimensionless_groups, nu_curve, tube_profile,
│   │   │                      # convection_fluids, notas, ejemplos y export.
│   │   ├── radiation.py       # ✅ Fase 8.2 — planck, blackbody_fraction (Chang y
│   │   │                      # Rhee) e inversa, bandas (ε y α), VIEW_FACTORS (8
│   │   │                      # geometrías) y view_factor_curve, solve_two_surface
│   │   │                      # (pantallas), solve_enclosure (radiosidades),
│   │   │                      # ENCLOSURE_LAYOUTS, enclosure_from_layout y
│   │   │                      # EnclosureCase, solve_surface_balance (h_rad, sol,
│   │   │                      # T de equilibrio) y su curva, la termocupla,
│   │   │                      # notas, ejemplos y export.
│   │   ├── exchangers.py      # ✅ Fase 8.2 — ε(NTU, C_r) y NTU(ε, C_r) de cada
│   │   │                      # tipo (flujo cruzado sin mezclar: serie de Mason),
│   │   │                      # correction_factor (F = NTU_cc/NTU), lmtd, Stream
│   │   │                      # (c_p dado o de CoolProp, cambio de fase),
│   │   │                      # solve_rating, solve_sizing,
│   │   │                      # solve_four_temperatures, ExergyBalance,
│   │   │                      # type_comparison, curvas ε–NTU y F–P, perfil del
│   │   │                      # doble tubo, overall_u (FOULING_FACTORS de TEMA),
│   │   │                      # notas, ejemplos y export.
│   │   ├── procedure_common.py # q, n, frac, sub, times, sum_rows, numbered.
│   │   ├── conduction_procedure.py # conduction_steps (símbolos de la red).
│   │   ├── fins_procedure.py  # fin_steps (con γ y Bessel en la anular).
│   │   ├── convection_procedure.py # convection_steps, internal_steps,
│   │   │                      # correlation_latex y big (adimensionales).
│   │   ├── radiation_procedure.py # ✅ Fase 8.2 — constants_latex y los pasos del
│   │   │                      # cuerpo negro, el factor de forma, las dos
│   │   │                      # superficies, el recinto, la superficie y la
│   │   │                      # termocupla.
│   │   └── exchangers_procedure.py # ✅ Fase 8.2 — exchanger_steps (los tres
│   │                          # problemas, con la exergía), overall_u_steps,
│   │                          # EPS_FORMULAS y NTU_FORMULAS (renglones).
│   ├── gases/                 # ✅ Fase 9.1 (la 9.2 suma los gases reales)
│   │   ├── ideal.py           # IdealGas (15 gases, NASA-9; el helio con 5/2·R):
│   │   │                      # cp, h, u, s°, p_r, v_r; model_cp/h/u/s (c_p a
│   │   │                      # 25 °C o variable); estados con dos de p, T, v;
│   │   │                      # ideal_gas_check (Z de CoolProp); solve_state_change
│   │   │                      # (tres modelos y su error), ejemplos y export.
│   │   ├── mixture.py         # solve_mixture (cuatro bases, Dalton, Amagat, s con
│   │   │                      # la mezcla, agua que condensaría) y solve_mixing
│   │   │                      # (tanque o cámara, c_p constante o variable,
│   │   │                      # S_gen_TP y S_gen_mix, X_dest), ejemplos y export.
│   │   ├── polytropic.py      # solve_process (cinco procesos, dato p₂, v₁/v₂ o
│   │   │                      # T₂, cerrado o abierto, la adiabática con c_p
│   │   │                      # variable), process_comparison (los cinco caminos),
│   │   │                      # process_curve, staged_compression y staged_curves,
│   │   │                      # exponent_from_states, ejemplos y export.
│   │   └── ideal_procedure.py # state_change_steps, mixture_steps, mixing_steps,
│   │                          # process_steps, staged_steps y exponent_steps.
│   ├── exergy/                # ✅ Fase 7 (reemplaza al placeholder exergy.py)
│   │   ├── physical.py        # Ambient (T₀, p₀), PhysicalExergy /
│   │   │                      # physical_exergy (ψ, φ, térmica y mecánica, V²/2,
│   │   │                      # g·z, por kg, masa o caudal), ProcessExergy,
│   │   │                      # heat_exergy, finite_source_exergy, notas,
│   │   │                      # ejemplos (Cengel cap. 8 y 10-8) y export.
│   │   ├── chemical.py        # Tabla de data/szargut_chemical_exergy.csv
│   │   │                      # (modelos II y I), szargut_method (Δg_f NASA +
│   │   │                      # elementos), species_exergy / species_available,
│   │   │                      # fuel_ratio_table, mixture_chemical_exergy (con el
│   │   │                      # agua que condensa), fuel_chemical_exergy (β de
│   │   │                      # Szargut y Styrylska), ejemplos y export.
│   │   ├── plant.py           # ComponentExergy (F, P, D, L), StreamExergy,
│   │   │                      # PlantExergy; rankine_exergy, refrigeration_exergy,
│   │   │                      # gas_turbine_plant_exergy, combined_plant_exergy;
│   │   │                      # grassmann_rows, notas y export.
│   │   └── exergy_procedure.py # physical_steps, heat_steps, finite_source_steps,
│   │                          # species_steps, mixture_steps, fuel_steps y
│   │                          # plant_steps (un paso por componente).
│   ├── combustion/
│   │   ├── __init__.py
│   │   ├── thermo.py          # ✅ Fase 5 — Polinomios NASA-9 (McBride et al., 2002)
│   │   │                      # desde data/nasa9_thermo.csv: Species (cp, h = h_f
│   │   │                      # + Δh, s° a 1 bar, g°), mezclas, h_fg del agua.
│   │   ├── fuels.py           # ✅ Fase 5 — Fuel: mezcla gaseosa, líquido puro
│   │   │                      # (GLP licuado con CoolProp) o UltimateAnalysis con
│   │   │                      # PCS dado; átomos, O₂ teórico, h, s, PCS/PCI; FUELS.
│   │   │                      # Fase 6: hhv_correlation (PCS estimado).
│   │   ├── stoichiometry.py   # ✅ Fase 5 — Oxidizer (técnico, seco, O₂; humedad a
│   │   │                      # T_humidity_K), AirSpec (6 formas), products (λ < 1
│   │   │                      # como Cengel 15-8 c), Stoichiometry (humos, GC, V_N,
│   │   │                      # rocíos, ácido de Verhoff y Banchero), Orsat.
│   │   ├── equilibrium.py     # ✅ Fase 5 — Equilibrio químico (Gibbs con
│   │   │                      # potenciales de elementos, como NASA CEA): TP, TV,
│   │   │                      # llama adiabática HP y UV.
│   │   ├── combustion.py      # ✅ Fase 5 — CombustionInputs / solve_combustion /
│   │   │                      # CombustionResult (llama completa y de equilibrio,
│   │   │                      # K_p, calor con condensación y reparto del PCI,
│   │   │                      # segundo principio), análisis de humos, notas,
│   │   │                      # ejemplos, barridos y export.
│   │   ├── combustion_procedure.py # ✅ Fase 5 — combustion_steps y flue_gas_steps.
│   │   ├── heating_value.py   # ✅ Fase 6 — Una función por correlación (cita y
│   │   │                      # rango en el docstring) y CORRELATIONS; DryUltimate,
│   │   │                      # DryProximate, HeatingValueInputs.from_basis (tal
│   │   │                      # cual, seca, sin cenizas), solve_heating_value, PCI y
│   │   │                      # W*, notas, ejemplos, ghugare_dataset/dataset_fit,
│   │   │                      # argonne_coals, estimate_hhv_as_fired (/Combustion),
│   │   │                      # barrido de la humedad y export.
│   │   ├── heating_value_procedure.py # ✅ Fase 6 — heating_value_steps y
│   │   │                      # correlation_latex (tabla de aportes).
│   │   └── iso6976.py         # ✅ Fase 2.3 — ISO 6976:2016 (matriz identidad;
│   │                          # iso6976_to_dict, Fase 1.7)
│   │                          # Normalization matrix: pendiente, requiere
│   │                          # ISO 14912:2003 Formula (69) — deferido.
│   ├── cycles/
│   │   ├── tespy_utils.py     # ✅ Fase 3.1a — new_network() en SI y
│   │   │                      # solve() (status de TESPy → ValueError en castellano).
│   │   ├── rankine.py         # ✅ Fase 3.1a — RankineInputs / solve_rankine /
│   │   │                      # RankineResult, validaciones, barridos y
│   │   │                      # rankine_to_dict. Fase 3.1b: red genérica desde el
│   │   │                      # layout (calentadores), componentes con fracciones,
│   │   │                      # agua de enfriamiento, barrido de extracción.
│   │   │                      # Fase 3.1c: Losses/PipeLoss (ciclo real), DCA,
│   │   │                      # desrecalentador, bombeados intermedios, fluid
│   │   │                      # (ORC), Recuperator, fluid_behavior,
│   │   │                      # suggested_rankine_inputs.
│   │   ├── rankine_layout.py  # ✅ Fase 3.1b — Topología y numeración tipo Cengel
│   │   │                      # (sin TESPy): FeedwaterHeater, plant_layout,
│   │   │                      # port_flows (caudales 1 − y… simbólicos).
│   │   │                      # Fase 3.1c: cañerías, recuperador, bombeados en
│   │   │                      # cualquier cerrado, "evaporador".
│   │   ├── rankine_procedure.py # ✅ Fase 3.1b — rankine_steps (se mudó; rankine.py
│   │   │                      # lo reexporta): simple, recalentamiento y regenerativo.
│   │   │                      # Fase 3.1c: ciclo real, recuperador, calentadores
│   │   │                      # reales, sistema acoplado, textos por fluido.
│   │   ├── refrigeration.py   # ✅ Fase 3.2 — RefrigerationInputs / solve_refrigeration /
│   │   │                      # RefrigerationResult: layout fijo por ciclo (numeración
│   │   │                      # de Cengel), red de TESPy, validaciones, ExergyAnalysis,
│   │   │                      # SingleStage, notas, ejemplos, barridos y export.
│   │   │                      # Fase 7: default_reservoirs y check_reservoirs
│   │   │                      # públicas (las usa core.exergy.plant).
│   │   ├── refrigeration_procedure.py # ✅ Fase 3.2 — refrigeration_steps (reexportado):
│   │   │                      # estados, cámara, mezcla, cascada, COP, Carnot,
│   │   │                      # potencias y exergía destruida por componente.
│   │   ├── hrsg.py            # ✅ Fase 3.3 — FlueGas (mezcla de gases ideales:
│   │   │                      # M, w, R, h_g(T), T(h), rocío), exhaust_composition(λ),
│   │   │                      # HRSGInputs / solve_hrsg / HRSGResult (balances por
│   │   │                      # sección), hrsg_tq_profile, notas, ejemplos,
│   │   │                      # other_steam_option, barridos y hrsg_to_dict. Sin TESPy.
│   │   │                      # Fase 3.4: el gas pasa a core/ideal_gas.py; diseño
│   │   │                      # por temperatura de chimenea (T_stack_K, pinch_K).
│   │   ├── hrsg_procedure.py  # ✅ Fase 3.3 — hrsg_steps: composición, entalpía de
│   │   │                      # los gases, agua, caudal de vapor, secciones, total
│   │   │                      # y aprovechamiento, punto de rocío. Fase 3.4: modo
│   │   │                      # chimenea, times_diff / factor_times_diff públicos.
│   │   ├── hrsg_multi.py      # ✅ Fase 3.5 — PressureLevel, MultiHRSGInputs /
│   │   │                      # solve_multi_hrsg / MultiHRSGResult (niveles en
│   │   │                      # cascada), from_single, multi_tq_profile,
│   │   │                      # hrsg_exergy (por sección), level_comparison,
│   │   │                      # notas, ejemplos, barridos y multi_hrsg_to_dict.
│   │   │                      # Fase 3.6: Reheater (en paralelo con el SH de
│   │   │                      # alta), reheat_system (2×2), eta_pump.
│   │   │                      # Fase 7: HRSGExergy.gained_W (lo que gana el
│   │   │                      # agua en cada sección, para el producto).
│   │   ├── hrsg_multi_procedure.py # ✅ Fase 3.5 — multi_hrsg_steps (por nivel)
│   │   │                      # y exergy_step (también para la de una presión).
│   │   ├── brayton.py         # ✅ Fase 3.4 — Fuel (ISO 6976), BraytonInputs /
│   │   │                      # solve_brayton / BraytonResult, validaciones, notas,
│   │   │                      # brayton_ts_lines y brayton_tespy (control).
│   │   ├── brayton_procedure.py # ✅ Fase 3.4 — brayton_steps: aire, compresor (s°),
│   │   │                      # cámara (PCI, aire teórico, f, λ) o aire estándar,
│   │   │                      # gases, turbina, rendimiento y potencias.
│   │   ├── combined.py        # ✅ Fase 3.4 — SteamCycle, CombinedInputs /
│   │   │                      # solve_combined / CombinedResult (Kehlhofer, balance),
│   │   │                      # notas, ejemplos, barridos y combined_to_dict.
│   │   ├── combined_procedure.py # ✅ Fase 3.4 — combined_sections: TG, HRSG,
│   │   │                      # ciclo de vapor y el acople con el rendimiento.
│   │   ├── combined_multi.py  # ✅ Fase 3.6 — MultiCombinedInputs (niveles,
│   │   │                      # ReheatSpec, baumann_alpha) / solve_combined_multi /
│   │   │                      # MultiCombinedResult: turbina con admisiones
│   │   │                      # (TurbineSection, Admission, MultiSteamCycle),
│   │   │                      # desaireador, Baumann, from_combined (= 3.4),
│   │   │                      # from_combined_with_pinch (Fase 7: también el
│   │   │                      # diseño por chimenea, con el pinch que resulta),
│   │   │                      # configuration_comparison, bottoming_exergy,
│   │   │                      # notas, ejemplos, barridos, export y
│   │   │                      # combined_multi_tespy (control).
│   │   ├── combined_multi_procedure.py # ✅ Fase 3.6 — combined_multi_sections:
│   │   │                      # TG, HRSG (RH y 2×2), ciclo de vapor y acople.
│   │   ├── gas_turbine.py     # ✅ Fase 3.7 — GasTurbineInputs (etapas, T_intercool_K,
│   │   │                      # T_reheat_K, regenerator, Δp, presiones intermedias,
│   │   │                      # W_net_W) / solve_gas_turbine / GasTurbineResult
│   │   │                      # (CycleState numerados como Cengel, Stage, Intercooler,
│   │   │                      # Combustor, Regenerator), from_brayton (= 3.4),
│   │   │                      # gas_turbine_exergy, improvement_comparison, notas,
│   │   │                      # ejemplos, barridos, gas_turbine_lines (T–s), export y
│   │   │                      # gas_turbine_tespy (control).
│   │   └── gas_turbine_procedure.py # ✅ Fase 3.7 — gas_turbine_steps: presiones,
│   │                          # etapas, cámaras, regenerador, rendimiento y exergía.
│   └── plots.py               # fluprodia + matplotlib helpers
├── ui/                        # Helpers de UI que sí importan Streamlit
│   ├── branding.py            # Bloque de créditos compartido (sidebar)
│   ├── units_ui.py            # ✅ Fase 1.4 — Selector global + number_input_si
│   │                          # (key real f"{key}@{sistema}": el valor físico
│   │                          # sobrevive al cambio de unidades, Fase 1.6).
│   ├── diagrams.py            # ✅ Fase 1.5a — Cache de FluidPropertyDiagram
│   │                          # (@st.cache_resource), render_diagram_plotly
│   │                          # con overlays de puntos / procesos.
│   └── cycle_charts.py        # ✅ Fase 3.4 — tq_figure (de /HRSG),
│                              # render_rankine_diagram (de /Rankine),
│                              # gas_turbine_ts_figure y energy_sankey_figure.
│                              # Fase 3.5: multi_tq_figure, exergy_split_figure
│                              # y exergy_sections_figure.
│                              # Fase 3.6: el RH en el T–Q,
│                              # render_steam_cycle_diagram,
│                              # bottoming_exergy_figure y configuration_figure.
│                              # Fase 3.7: gas_turbine_cycle_figure,
│                              # improvement_figure y gas_turbine_exergy_figure.
│   └── psychro_chart.py       # ✅ Fase 4 — psychrometric_chart_figure (carta a
│                              # cualquier presión, grilla para tocar),
│                              # grid_point_from_selection, hvac_chart_items y
│                              # hvac_exergy_figure.
│   └── combustion_charts.py   # ✅ Fase 5 — composition_figure,
│                              # combustion_diagram_figure, energy_split_figure
│                              # (cascada del PCI) y sweep_figure.
│   └── heating_value_charts.py # ✅ Fase 6 — comparison_figure (puntos),
│                              # parity_figure y error_vs_oxygen_figure.
│   └── exergy_charts.py       # ✅ Fase 7 — grassmann_figure (banda vertical),
│                              # mollier_exergy_figure (ψ en el h–s),
│                              # carnot_factor_figure, finite_source_figure,
│                              # fuel_ratio_figure y component_efficiency_figure.
│   └── heat_transfer_charts.py # ✅ Fase 8.1 — conduction_profile_figure (capas
│                              # sombreadas, fluidos y película),
│                              # insulation_figure, fin_profile_figure,
│                              # fin_efficiency_figure, nu_curve_figure,
│                              # correlation_figure y tube_figure.
│   └── radiation_charts.py    # ✅ Fase 8.2 — planck_figure, spectral_match_figure,
│                              # view_factor_figure, network_figure,
│                              # enclosure_figure y surface_curve_figure.
│   └── exchanger_charts.py    # ✅ Fase 8.2 — effectiveness_figure (ε–NTU),
│                              # f_factor_figure (F–P), profile_figure,
│                              # type_comparison_figure y u_resistance_figure.
│   └── gas_charts.py          # ✅ Fase 9.1 — cp_figure, composition_figure,
│                              # mixing_entropy_figure, pv_figure, ts_figure (con
│                              # far_paths ocultos), work_figure y staged_figure.
├── tests/                     # pytest: tests/test_<modulo>.py; páginas con
│                              # streamlit.testing (tests/test_page_<pagina>.py)
├── data/                      # Tablas, propiedades por componente, etc.
│   ├── iso6976_components.csv # Valores tabulados por componente puro
│   ├── nasa9_thermo.csv       # ✅ Fase 5 — 32 especies (60 tramos) de NASA CEA
│   ├── ghugare2014_biomass.csv # ✅ Fase 6 — 536 biomasas (de modeldata, MIT;
│   │                          # LICENSE-modeldata.txt al lado)
│   ├── argonne_premium_coals.csv # ✅ Fase 6 — 5 carbones (Vorres, 1990)
│   └── szargut_chemical_exergy.csv # ✅ Fase 7 — 42 sustancias, modelos II
│                              # (Szargut et al., 1988) y I (Ahrendts, 1980)
├── scripts/
│   └── extract_nasa9.py       # ✅ Fase 5 — thermo.inp de NASA CEA → el CSV
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
- Cache de CoolProp con `@st.cache_data` en los wrappers de Streamlit
  (`app_pages/`) que envuelven funciones de `core.fluids`. `core/` no
  importa Streamlit.
- Idioma de la UI: **español rioplatense**. Los identificadores de
  código en inglés, comentarios y docstrings en español.
- Cada función académicamente relevante incluye en su docstring una
  cita corta a la fuente (libro de texto, paper, norma).

### Navegación

- `streamlit_app.py` arma el menú con `st.navigation` + `st.Page` y es
  el único entry point. Las páginas viven en `app_pages/` y la URL de
  cada una sale del nombre de archivo sin el número (`1_Propiedades.py`
  → `/Propiedades`).
- **No crear una carpeta `pages/`** en la raíz: Streamlit la detecta y
  arranca en el modo multipágina viejo (bandera global del proceso)
  hasta que corre `st.navigation`; los links directos a una página
  después de un reinicio mostraban el menú con nombres de archivo. Un
  test (`tests/test_navigation.py`) lo vigila.

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

Notas de implementación (aprendidas en la Fase 1.6):

- Guardar el resultado en `st.session_state` y renderizarlo en cada
  corrida: si se muestra solo dentro de `if submit:`, desaparece al tocar
  cualquier otro widget (p. ej. el selector de diagrama).
- Los links al vademecum salen de las constantes `VADEMECUM_*` de
  `ui/branding.py`; citar la sección (§N) además del link.
- Pensar en el celular: tablas con símbolo, valor y unidad primero; la
  unidad de `st.metric` en el rótulo; selectores en el cuerpo de la
  página (el sidebar queda oculto).
- Testear la página con `streamlit.testing.v1.AppTest` (ver
  `tests/test_page_propiedades.py`) y validar el LaTeX nuevo con KaTeX,
  que es el motor de `st.latex`.

Notas de la Fase 1.7:

- **Ancho de las ecuaciones**: KaTeX no parte una ecuación en renglones
  y en un celular de 390 px hay ~324 px útiles (dentro de un expansor).
  Las sustituciones con números se escriben con `core.latex.latex_chain`
  (una igualdad por renglón, `aligned`); si un renglón sigue largo, se
  corta antes de un `+`/`−` con `\\ &\quad +`. Dos ecuaciones en la
  misma línea (`\qquad`) solo si son cortas; si no, van en `st.latex`
  separados o en un `aligned`. Los negativos después de un signo, con
  `latex_paren`. Medir el ancho real renderizando con KaTeX en un
  navegador (p. ej. Chromium con Playwright) en los tres sistemas de
  unidades. En SI los números llevan ×10ⁿ (J/kg, Pa): una resta con un
  factor delante (η, v, la fracción y, x de la palanca) pasa el segundo
  número a otro renglón (`_factor_diff` del Rankine, `times_diff` de la
  HRSG, `latex_is_wide` en Propiedades e Isoentrópicos, que también corta
  con negativos). Desde la 0.17.0 todas las páginas entran en 324 px en
  los tres sistemas (un test lo vigila en el Rankine).
- **Isolíneas de fluprodia**: `set_isolines` interpreta los valores en
  las unidades activas del diagrama (`set_unit_system`). Generarlas en
  las unidades de cada sistema (`core.diagrams._isoline_grid(fluid,
  system)`); hasta la 0.9.0 iban siempre en °C/bar y en SI quedaban
  isobaras de 0,01–1000 Pa.
- Un diagrama que falla no debe tumbar la página: envolverlo en
  `try/except` y mostrar `st.warning` (como Propiedades e Isoentrópicos).
- Los valores por defecto de cada página tienen que ser calculables para
  todos los fluidos (`suggested_inputs`, `suggested_device_inputs`) y
  hay tests que los recorren.
- En el estado de referencia de cada fluido (agua: u = s = 0 en el punto
  triple; aire: h = s = 0 líquido saturado a 1 atm) CoolProp devuelve
  ruido (−6.6×10⁻⁸ J/kg); `core.fluids` lo pasa a 0 (`_zero_if_noise`)
  en `StatePoint`, `FluidState` y la saturación. Si se lee CoolProp
  directo, aplicar lo mismo.
- **Pseudo-puros** (aire, R410A; `FluidLimits.is_pure` False): tienen
  deslizamiento de temperatura en la campana. Con p y (h, s, v o u) el
  estado se calcula con (p, x), con x de la regla de la palanca (el
  flash de CoolProp falla cerca de la línea de burbuja); T-x con
  0 < x < 1 no está definido. En los procedimientos, T_f ≤ T ≤ T_g en vez
  de T = T_sat.

Notas de la Fase 3.1a (TESPy 0.11):

- Armar la red como el tutorial oficial (`tutorial/basics/rankine.py`:
  `CycleCloser` + `SimpleHeatExchanger` + `Turbine` + `Pump`) con
  `core.cycles.tespy_utils.new_network()`, que fija unidades SI (incluida
  `pressure_difference`: si falta, TESPy 0.11 avisa con un FutureWarning).
- Resolver con `tespy_utils.solve(network, what=...)`: exige
  `network.status == 0` (1 = convergió con parámetros fuera de rango, p. ej.
  una turbina que comprime; 2 = no convergió; 3 = singular; 99 = error) y
  explica el problema en castellano.
- TESPy acepta datos imposibles sin quejarse (entrada a la turbina por
  debajo de la saturación, un "recalentamiento" que enfría): validar en
  `core` antes de resolver y revisar el resultado después.
- De TESPy leer solo p y h (`.val_SI`) y reconstruir cada estado con
  `core.fluids.fluid_state_from_pair` (p, h): región, título y saturación
  salen con la convención del proyecto (no usar el x de TESPy).
- Cachear el resultado (dataclasses picklables) con `st.cache_data`, nunca
  la `Network`. `import tespy` tarda ~3,6 s en frío: queda dentro de
  `core/cycles/`. Un Rankine se resuelve en ~60 ms; un barrido, una red por
  punto.
- Ecuaciones: el factor de unidades de v·Δp y los términos con ×10ⁿ (SI)
  van en un renglón de continuación (`\\ &\quad`); la regla de la palanca
  usa h_fg y s_fg, como Cengel.

Notas de la Fase 3.1b (regeneración):

- Calentador cerrado = `Condenser` (`ttd_u` se mide contra T_sat a la
  presión de la extracción; `ttd_u = 0` es el ideal de Cengel y la salida
  caliente queda en líquido saturado); abierto = `Merge` con x = 0 a la
  salida; extracción = `Splitter`; drenaje en cascada = `Valve` + `Merge`
  (antes del condensador o de la entrada caliente del cerrado de menor
  presión); drenaje hacia adelante = bomba + `Merge`. Todo como los ejemplos
  oficiales (tutorial de optimización de una central, modelo SEGS de sus
  tests, CCPP).
- La presión se fija una sola vez: entrada de la turbina y salida de cada
  tramo; bombas, válvulas y `Merge` la toman de la red.
- Topologías: hasta 3 calentadores; solo el cerrado de mayor presión
  bombea el drenaje hacia adelante. Así cada fracción se despeja en orden
  (de mayor a menor presión) en el procedimiento.
- η_T se mide desde la entrada de cada turbina (soluciones de Cengel): los
  tramos llevan el `eta_s` local que reproduce esa línea de expansión.
- TESPy no avisa una extracción negativa hacia un abierto (un `Merge` no
  tiene límites): `solve_rankine` revisa cada fracción después de resolver.
  Antes de resolver, cada calentador tiene que poder calentar (incluido el
  calentamiento en la bomba).
- La numeración sale de `core.cycles.rankine_layout` antes de resolver: la
  página la usa para los rótulos (lee de `session_state` los widgets que
  están más abajo). Las opciones de los radios no llevan números: cambiar
  las opciones reinicia el widget.
- Con regeneración η < 1 − T_C/T̄_H (los calentadores generan entropía).
- Numeración del cap. 10 de Cengel (ediciones 7.ª a 9.ª): §10-2 Rankine
  ideal, §10-3 desviaciones del ciclo real (pérdidas de carga y de calor,
  ejemplo 10-2), §10-4 cómo aumentar el rendimiento, §10-5 recalentamiento,
  §10-6 regeneración, §10-9 ciclos combinados. El autor confirmó §10-3 y
  §10-5: hasta la 0.15.0 el ciclo real citaba §10-5 por error, y un test
  (`test_procedure_cites_the_textbook_sections`) lo vigila.

Notas de la Fase 3.1c (ciclo real, calentadores reales y ORC):

- Ciclo real (`Losses`, `PipeLoss`): los datos siguen siendo los de la
  turbina (p y T de entrada, presión de escape, presión de recalentamiento
  a la salida de la de alta) y la bomba compensa todas las caídas: cada
  componente lleva su `dp` y TESPy resuelve la presión de la bomba.
  Cañerías = `Pipe` con `dp` y la T de salida (la de alimentación, con
  `Ref(entrada, 1, −ΔT)`); sin ΔT, `Q = 0`. Subenfriamiento con
  `td_bubble` a la salida del condensador. Primer principio:
  w_neto = q_H − q_C − q_pérd. `T_low_K` es la de condensación (T_sat a
  la presión de escape), no la del condensado subenfriado.
- Calentador cerrado real: subenfriador de drenaje = `Condenser` con
  `subcooling=True` y `ttd_l` = DCA (el ejemplo de la doc de TESPy);
  desrecalentador = `Desuperheater` en serie (la extracción sale como vapor
  saturado) y el TTD se fija como la T del agua a la salida: el `ttd_u` de
  TESPy tiene mínimo 0 y un TTD < 0 en un `Condenser` da status 1.
- Un drenaje bombeado desde un cerrado que no es el de mayor presión
  acopla las fracciones (la mezcla cambia la h que entra al calentador
  siguiente): el procedimiento muestra el sistema, la solución y la
  verificación de cada balance en vez de despejar y de a una.
- Antes de resolver se calculan la línea de expansión (`_expansion_line`)
  y la de agua (`_feedwater_line`, `_line_pressures`) para validar:
  extracción sobrecalentada para el desrecalentador, escape sobrecalentado
  para el recuperador, calentador que no puede calentar, presión de la
  bomba.
- ORC: `RankineInputs.fluid` (`RANKINE_FLUIDS`). TESPy recibe `water` para
  el agua (como el tutorial) y el nombre de CoolProp para el resto. En la
  página las keys de los widgets llevan el fluido, salvo el agua (las de la
  0.12.0 no cambian); los valores por defecto de cada fluido salen de
  `suggested_rankine_inputs` (un test los recorre). Recuperador =
  `HeatExchanger` con `eff_max` = ε (q/q_máx), solo sin calentadores.
  Seco, húmedo o casi isoentrópico (`fluid_behavior`): T·(ds_g/dT)/s_fg a
  0,8·T_c, con umbral ±0,25 (Chen, Goswami y Stefanakos, 2010).
- Textos según el fluido: tablas de Cengel para el agua (A-4 a A-7) y el
  R-134a (A-11 a A-13); si no, "ecuación de estado". En el LaTeX, los
  subíndices con tilde van en `\text{}`: KaTeX estricto avisa con
  `\mathrm{pérd}`.
- Diagramas: un intercambiador o una cañería con caída de presión
  (presiones a menos de 1/4 una de otra) se dibuja con p lineal en h, no
  como recta.
- Streamlit no recarga los módulos de `core/` al editarlos: reiniciar el
  servidor antes del smoke test (si no, el síntoma es un diagrama viejo).
- Regresión: un snapshot de la 0.12.0 (24 casos: estados, red de TESPy con
  todas sus especificaciones, pasos en los tres sistemas, export y
  barridos) queda idéntico.

Notas de la Fase 3.2 (refrigeración por compresión de vapor):

- Red como el tutorial de bomba de calor de TESPy 0.11
  (`tutorial/basics/heat_pump.py`): `CycleCloser` + `SimpleHeatExchanger`
  + `Compressor` + `Valve`; sobrecalentamiento con `td_dew` y
  subenfriamiento con `td_bubble` (`tutorial/advanced/stepwise.py`). Cámara
  de evaporación instantánea = `DropletSeparator` (out1 líquido, out2
  vapor) + `Merge`; cascada = `HeatExchanger` entre dos lazos, cada uno con
  su `CycleCloser` y su fluido. Con ΔT = 0 en el intercambiador (ejemplo
  11-4) TESPy da `kA = nan` sin avisar: no importa (no se usa).
- Base de cálculo: 1 kg/s por el condensador (la de Cengel en los cinco
  ejemplos) y después se escala al caudal o a la capacidad.
- `p_evap_Pa` y `p_cond_Pa` son la aspiración y la descarga del compresor;
  la T de evaporación es la de rocío ahí y la de condensación la de
  burbuja (el R-410A tiene deslizamiento). En la página, los niveles por
  temperatura se convierten con esas mismas definiciones.
- En una cascada con dos fluidos, la presión del evaporador puede superar
  a la del condensador (CO₂ abajo, R-134a arriba): las presiones se
  comparan dentro de cada ciclo, nunca entre ciclos.
- Notación del vademecum (§9.3): Q_C sale de la fuente fría (Cengel: Q_L).
  Segundo principio: T₀ es el ambiente (la fuente caliente del
  refrigerador, la fría de la bomba de calor); X_dest = T₀·S_gen por
  componente y Ẇ = Ẇ_mín + ΣX_dest cierra (test). La cámara no destruye
  exergía (separar fases a la misma T es reversible).
- Cengel 8.ª ed.: §11-5 (segundo principio) es nuevo, así que los
  ejemplos de cascada y cámara son 11-4 y 11-5 (11-3 y 11-4 en la 7.ª).
- Tablas del R-134a (A-11 a A-13): h = s = 0 en el líquido saturado a
  −40 °C; CoolProp usa la del IIR. No se cambia la referencia de CoolProp
  (es global del proceso y afecta a TESPy): `textbook_reference_note` lo
  explica con las constantes (148,14 kJ/kg y 0,7956 kJ/(kg·K)) donde se
  citan esas tablas (Propiedades, Rankine con R-134a y Refrigeración).
- Con un fluido «seco» (isobutano) la compresión isoentrópica desde vapor
  saturado termina dentro de la campana: es una nota, no un error.
- La comparación con el ciclo simple usa el refrigerante del condensador a
  la misma temperatura de evaporación; en la cascada CO₂/NH₃ el COP casi
  no cambia, pero la relación de presiones y la descarga sí (la nota lo
  dice).
- Diagramas: una válvula (h igual, p distinta) se dibuja rayada sobre su
  línea de h constante («estrangulamiento (h constante)»), también en los
  drenajes del Rankine y en una cañería sin pérdida de calor.
- Smoke test: la primera carga de la página puede seguir dibujando
  después de que aparece la segunda fila de métricas; esperar ~3 s antes
  de leer el texto.
- Regresión: el snapshot de la 0.13.0 (29 casos) queda idéntico salvo la
  nota del R-134a en el ORC.

Notas de la Fase 3.3 (HRSG de una presión):

- Cálculo directo en `core/cycles/hrsg.py` (los balances que se hacen a
  mano, sin TESPy: rápido y transparente). TESPy es el control cruzado en
  los tests: `HeatExchanger` en serie con los gases como mezcla (como su
  tutorial de turbina de gas); coincide al 0,02 % (TESPy evalúa cada
  componente a su presión parcial). Sirve de base para el ciclo combinado.
- Gases = mezcla de gases ideales: la h de cada componente sale de CoolProp
  a 1 Pa (`AbstractState`, ~16 µs por punto; `PropsSI` es 8 veces más
  lento), todos a la misma presión, con h_g = 0 a 25 °C. Con la presión
  parcial el H₂O cae en la campana a temperaturas bajas. `T_from_h` busca
  entre el punto triple del agua y 1720 °C.
- Diseño (Kehlhofer et al., 2009): los gases salen del evaporador a
  T_sat + pinch y el agua del economizador a T_sat − approach. El caudal
  de vapor sale del tramo entre la entrada de los gases y el pinch; el
  economizador fija la chimenea (por eso T_alim no cambia ṁ_v).
- Diagrama T–Q con la convención del domo: el agua sube vertical de
  T_sat − approach a T_sat al entrar al evaporador, y los tubos del
  evaporador ven agua a T_sat (su ΔT frío es el pinch). Así el mínimo ΔT
  del perfil es exactamente el pinch (test).
- Errores al alumno: cruce de temperaturas en el economizador o dentro de
  la caldera, chimenea bajo el punto de rocío (el modelo no condensa),
  gases que no alcanzan T_sat + pinch, vapor más caliente que los gases,
  presión supercrítica. Notas: agua de alimentación bajo el punto de rocío,
  ΔT de un extremo menor que el pinch, approach 0.
- El barrido de la presión no es monótono: cerca de la crítica h_fg se
  achica y el caudal vuelve a subir. Se limita a 1–160 bar.
- La página recalcula sola en cada cambio (~0,2 s con la comparación,
  `st.cache_data`); los barridos van con botón. En el diagrama, el pinch y
  el approach (unos pocos kelvin) se señalan con flechas que vienen de
  zonas libres del evaporador, y los nombres de las secciones van arriba;
  medido en 1280 y 390 px.
- Los textos usan °C y bar (como el Rankine); el LaTeX sigue el sistema.
  En SI las restas con ×10ⁿ se parten (`_times_diff`) y el calor total va
  un sumando por renglón: máximo 304 px.

Notas de la Fase 3.4 (turbina de gas y ciclo combinado de una presión):

- Gases ideales en `core/ideal_gas.py` (`hrsg.py` lo reexporta: la API no
  cambió). h y s°(T) desde 25 °C; la entropía absoluta suma la S° de
  NIST-JANAF a 25 °C y 1 bar, −R·ln(p/p°) y el término de mezcla, así el
  aire y los gases quedan en la misma escala del T–s. La isoentrópica sale
  de s°(T₂s) = s°(T₁) + R·ln(p₂/p₁) (método de la A-17 de Cengel). Cada
  especie tiene la T mínima de CoolProp (el CO₂ del aire seco limita a
  216,6 K; con agua, 273,17 K).
- Aire seco (N₂ 0,7808, O₂ 0,2095, Ar 0,0093, CO₂ 0,0004): reproduce las
  tablas de Cengel (Δh̄ de A-18 a A-23 al 0,1 %; ejemplos 9-5, 9-6 y
  10-9). El técnico (21/79) es el del vademecum §16.1.
- Cámara, por kg de aire y con h desde 25 °C (la referencia del PCI):
  h_a(T₂) + f·PCI = (1 + f)·h_g(T₃). La composición depende de f: `brentq`
  entre ~0 y el estequiométrico (si no alcanza, λ < 1 y error). PCI de
  ISO 6976:2016 a 25 °C: PCS_m − (b/2)·L₀ (metano: 50,027 MJ/kg). Aire
  estándar = `fuel=None`. TIT ≤ T₂ se valida antes (sin eso `brentq` falla
  sin explicación).
- TESPy de control (`brayton_tespy`, como su tutorial de turbina de gas):
  `Compressor` + `DiabaticCombustionChamber` (combustible a 25 °C con
  `p=Ref(c2, 1.05, 0)`) + `Turbine`; aire estándar con
  `SimpleHeatExchanger`. Sin `T0`/`m0` del cálculo directo, Newton pasa por
  63 K y termina en status 99. Coincide al 0,3 % (gas real a r_p alta).
- HRSG por chimenea (Cengel 10-9): ṁ_v del balance de toda la caldera; el
  pinch es un resultado (≤ 0: cruce de temperaturas, con mensaje).
- Acople: el Rankine se resuelve por kg (bomba → agua de alimentación de
  la HRSG) y se escala con el ṁ_v de la HRSG (`dataclasses.replace`).
  η_HRSG = Q̇_HRSG / (Q̇_comb − Ẇ_TG): con esa definición la relación de
  Kehlhofer y el balance Q̇_comb = Ẇ_TG + Ẇ_TV + Q̇_cond + Q̇_chim (la
  chimenea contra el aire a T₁) cierran exactos (tests). Los errores llevan
  el nombre de la parte («Turbina de gas: », «Caldera de recuperación: »,
  «Ciclo de vapor: »).
- Barridos: en los de r_p y TIT el vapor se sobrecalienta como mucho hasta
  T₄ − 25 K (`SH_HOT_END_K`); los puntos sin sentido físico se omiten. En
  el ejemplo típico el óptimo del ciclo combinado está en r_p ≈ 20 y el de
  la turbina de gas sola, en el extremo del barrido (40).
- Página: botón «Calcular» como el Rankine; el TESPy de control se
  cachea (devuelve el resultado o el texto del error). El diseño por
  chimenea arranca con la del diseño por pinch del ejemplo redondeada hacia
  arriba a 5 °C (con 150 °C el típico cruza). Los rótulos del Sankey van
  en varios renglones (`<br>`): en uno solo se pisaban a 390 px. En el T–s,
  el tramo gris horizontal en 2 es el cambio de composición de la cámara.
- Procedimiento con la notación de Cengel (s°₁, s°₂s; s°_g,3 para los
  gases). En SI la turbina calcula primero Δh_s y w_T usa
  `factor_times_diff`: máximo 306 px en los tres sistemas (2592
  ecuaciones). Hasta la 0.16.0 los pasos del Rankine en SI se
  deslizaban (hasta 393 px); desde la 0.17.0 también entran.
- `ui/cycle_charts.py` junta los gráficos de los ciclos: /HRSG y /Rankine
  importan de ahí el T–Q y el diagrama del Rankine, sin cambios visibles.

Notas de la Fase 3.5 (HRSG de dos y tres presiones):

- Arreglo en cascada (`core/cycles/hrsg_multi.py`): los gases recorren los
  niveles de alta a baja (SH, EV y ECO de cada uno); el ECO de baja calienta
  toda el agua y cada domo manda el líquido saturado que no evapora a la
  bomba (isoentrópica, fuera de la caldera) del nivel siguiente. Así cada
  caudal sale en orden, de alta a baja, de un balance hasta el pinch del
  nivel, con el calor del domo del agua que sube (ṁ_sube·(h₃ − h₂)). Las
  calderas reales intercalan secciones (economizadores partidos o en
  paralelo): queda como mejora.
- Con un nivel (`from_single`) reproduce bit a bit `solve_hrsg` (test). El
  evaporador de cada nivel incluye el domo: en el T–Q el agua sube vertical
  de T_sat − approach a T_sat y los tubos ven T_sat (el mínimo ΔT de cada
  nivel es su pinch).
- Límite de la cascada: el SH de media o de baja ve los gases que ya pasaron
  por el ECO de alta (con 100/20/4 bar, 257 °C): media a 300 °C no se puede
  (error que nombra el nivel) y la nota del extremo caliente lo explica.
  Las presiones demasiado cercanas (T_sat de abajo ≥ T_sat − approach de
  arriba) también son error.
- Exergía (`hrsg_exergy`, T₀ = T_ref = 15 °C): física de los gases a su
  presión (h y s° desde T₀); X_gases = X_agua + Σ X_dest (T₀·S_gen de cada
  sección) + X_chimenea, cierra a 10⁻⁹ (test). η_II = X_agua / X_gases (la
  chimenea cuenta como pérdida, para comparar 1, 2 y 3 niveles). La de una
  presión se calcula con el mismo modelo (`from_single`) y su procedimiento
  suma `exergy_step` antes del punto de rocío.
- TESPy de control (tests): `HeatExchanger` en serie para los gases; cada
  domo de abajo es un `DropletSeparator` (out1 líquido a la `Pump`
  isoentrópica, out2 vapor al SH) y el de alta evapora todo (x = 1). Con
  `m0` del cálculo directo converge en 0,1–0,25 s; coincide al 0,12 % en
  caudales y 0,04 K en la chimenea (presión parcial).
- Los puntos de los gases van a, b, c… sin la «g» (h_g(T_g) se confundía con
  la entalpía de los gases); los estados del agua se numeran 1–5 por nivel,
  como la caldera de una presión, con el nivel de subíndice (h_{4,A}).
- Página: radio «Niveles de presión» (`hr_levels`); con 1 todo queda como
  en la 0.16.0 (mismas keys) más la exergía; con 2 o 3, ejemplos filtrados
  por cantidad de niveles y keys `hm{n}_{ejemplo}_{A|M|B}_…`. En el T–Q los
  nombres de las secciones van en dos alturas alternadas y sin las de menos
  del 5 % del calor, y se ocultan los rótulos de puntos de gas pegados; en
  las barras de exergía, `uniformtext` oculta los % que no entran. Medido a
  390 px.
- LaTeX: 1212 expresiones distintas, máx. 313 px; en SI las búsquedas en
  tabla con ×10ⁿ (h = h(p, T) = …) van en dos renglones (`_lookup`).

Notas de la Fase 3.6 (ciclo combinado de dos y tres presiones con recalentamiento):

- Recalentador en `hrsg_multi` (`MultiHRSGInputs.reheat`, `None` = 0.17.0
  idéntico): en paralelo con el SH de alta (los dos bancos ven los gases de
  entrada y salen a una T común; en `_flows` cada banco lleva su parte de los
  gases, `gas_share`, para el T–Q y la exergía). En serie no alcanza: el banco
  que va segundo ve gases más fríos. Con tres niveles el vapor de media se suma
  al recalentamiento frío (p_RH = p_media) y alta y media salen del 2×2
  (`reheat_system`, Cramer); la mezcla queda en el ciclo de vapor. Bombas
  entre niveles con `eta_pump` (en el ciclo combinado, el η_B del ciclo).
- `combined_multi`: cálculo directo (el agua de alimentación y el
  recalentamiento frío no dependen de los caudales) → HRSG → turbina con
  admisiones. Cada tramo entre admisiones (o el RH) es una turbina con η_T
  desde su entrada; la extracción del desaireador va sobre esa línea. Estados
  numerados en el sentido del flujo: coincide con Cengel en el Rankine simple
  (1–4), con recalentamiento (1–6) y con desaireador (1–7). Con un nivel y
  sin RH reproduce `solve_combined` a 1e-9 (test).
- Regla de Baumann (`baumann_alpha`): la expansión se parte donde la línea de
  expansión cruza x = 1 (`brentq`); la parte húmeda va con η_T·(1 − α·ȳ),
  iterando la humedad de salida. Sin ella, el RH casi no suma rendimiento (con
  una sola presión lo baja: la chimenea se calienta) y solo seca el vapor;
  con ella, 3P → 3PRH suma ~0,8 puntos. La página la trae apagada (como
  Cengel); la comparación muestra las dos columnas.
- TESPy de control (`combined_multi_tespy`, tests y página): banco paralelo
  con `Splitter` + `Merge` de gases, `Ref(otro, 1, 0)` en la T de salida y sin
  `pr1` en el RH (si no, la presión queda sobredeterminada: «circular
  dependency»); cada tramo de turbina con el η local que reproduce la línea
  (también con Baumann). Coincide al 0,2 % en caudales, 0,02 % en Ẇ_TV y
  0,1 K en la chimenea; 0,1–0,8 s.
- Exergía del ciclo de fondo (`bottoming_exergy`): Ẋ_gases = Ẇ_TV + Σ Ẋ_dest
  (HRSG por sección, turbinas, mezclas, desaireador, bombas) + Ẋ_cond +
  Ẋ_chim, cierra a ~1e-7 W (test).
- Página: radio `cc_levels` y casilla `cc_reheat`; con 1 nivel y sin RH,
  la 0.17.0 sin cambios. Si no, ejemplos de esa combinación (hay uno o dos
  por combinación) y keys `cm{n}{r}_{ejemplo}_…`. Gráficos con la paleta de
  referencia validada para daltonismo (azul, naranja, aguamarina: también en
  las barras de exergía de /HRSG; el verde y el naranja de antes no se
  distinguían con protanopía). La comparación es un gráfico de puntos: las
  diferencias son de décimas y unas barras desde cero las esconderían.
- LaTeX: 4007 expresiones distintas, KaTeX estricto, máx. 320 px. Con RH el
  calor del tramo de alta es `\dot{Q}_{\mathrm{tramo},A}` (`SH+RH+EV` no
  entraba).

Notas de la Fase 3.7 (turbina de gas con interenfriamiento, recalentamiento y regenerador):

- `core/cycles/gas_turbine.py` generaliza `brayton.py`, que sigue siendo la
  turbina del ciclo combinado: `from_brayton` + una etapa y sin regenerador
  dan idéntico (test a 1e-12). Con aire estándar los estados de la turbina se
  rotulan «aire» (la 3.4 decía «gases»).
- Etapas: sin presiones dadas, la misma relación en cada etapa, compensando
  las Δp de interenfriadores y cámaras (`_compressor_pressures`,
  `_turbine_pressures`); es el mínimo trabajo del vademecum §6.3
  (p_x = √(p₁p₂)). `r_p` es la del compresor completo; la turbina descarga a
  p₁/(1 − Δp_reg) con regenerador.
- Recalentamiento con combustible = segunda cámara que quema en los gases
  (combustión secuencial): (1 + F_ant)·h_ent + f·PCI = (1 + F)·h_g(T); la
  composición sale de `combustion_products` con el F acumulado (λ = f_t/F,
  con λ ≥ 1; si no alcanza el O₂, error que nombra la cámara).
- Regenerador: ε de Cengel §9-9 con las entalpías del aire
  (h₅ = h₂ + ε·(h_a(T₄) − h₂)); con combustión T₄ depende de f y f de T₅: punto
  fijo acelerado con la secante (con aire estándar converge en una vuelta).
  T₄ ≤ T₂ es error («el regenerador no sirve»). En TESPy, `HeatExchanger` con
  `eff_cold` = ε; con ε = 1 TESPy termina en status 1, así que el control lo
  explica (el núcleo sí acepta ε = 1, el 9-8 b de Cengel).
- Numeración de Cengel: 1–4; 1–6 con regenerador y una etapa (5 y 6 son sus
  salidas, figura 9-38); con etapas, en el sentido del flujo (1–10 en el 9-8,
  figura 9-43). El 9-8 a) del libro numera como si hubiera regenerador; acá,
  sin regenerador, va de 1 a 8.
- Exergía (`gas_turbine_exergy`): T₀ = T ambiente; la del combustible ≈ PCI
  (vademecum §16.13), así que η_II = η; con aire estándar entra Q·(1 − T₀/T)
  con T la salida de cada calentador. ψ de cada estado contra la misma mezcla
  a T₀ y p₀. Compresores, turbinas y regenerador: T₀·S_gen; cada cámara, el
  resto de su balance; interenfriadores y escape, «perdida». Cierra a 1e-9.
- `ideal_gas`: bajo la T mínima de CoolProp (el agua bajo 0,01 °C) cada
  componente sigue como gas ideal con c_p constante. Hace falta para el estado
  muerto con ambiente bajo cero; arregló también la exergía del ciclo
  combinado, que hasta la 0.18.0 fallaba con un error crudo de CoolProp.
- TESPy de control (`gas_turbine_tespy`): `Compressor` por etapa,
  `SimpleHeatExchanger` en los interenfriadores (T de salida en la entrada de
  la etapa siguiente), `HeatExchanger` regenerador, `DiabaticCombustionChamber`
  en cada cámara (p de cada recalentamiento fijada en su entrada) y `Turbine`
  por etapa. Hacen falta `T0`/`p0` en la entrada de cada cámara: sin ellos,
  con aire técnico y tres etapas, Newton termina en status 99. Coincide dentro
  de 0,05 puntos de η, 0,3 % en f y 2 K en el escape; 0,1–0,7 s.
- Página: botón «Calcular» (keys `bt_{ejemplo}_…`; las presiones a elección,
  `…_pic{n}_{k}` y `…_prh{n}_{k}`); la comparación de mejoras usa ε = 0,80 si
  los datos no tienen regenerador y dos etapas si tienen una. Gráficos con la
  paleta validada (cámaras naranja, interenfriamiento azul, regenerador
  aguamarina); la comparación es un gráfico de puntos en dos paneles (η y
  w_neto: nunca un eje doble); el T–s lleva una leyenda corta y sin las
  isobaras (a 390 px ocupaba cinco renglones).
- LaTeX: 3299 expresiones distintas (procedimiento y teoría), KaTeX
  estricto, máx. 316 px. Corta con números negativos (`_wrap_wide`, s° bajo
  25 °C en Inglés); el regenerador como Cengel (q_reg,máx, q_reg =
  ε·q_reg,máx, h₅ = h₂ + q_reg); el recalentamiento calcula Δh antes de f (un
  `\\` dentro de `\frac` no es válido); la exergía de cada cámara, un término
  por renglón; los subíndices por componente (x_{d,C1}, x_{d,\text{cám}},
  x_{d,\text{RH}}).

Notas de la Fase 4 (psicrometría):

- Modelo del vademecum §14 (gas ideal, c_p constantes: 1,005 y 1,864 kJ/(kg·K),
  r₀ = 2501 kJ/kg, 0,622 y 1,608) para que el alumno reproduzca los números a
  mano. R_v = 1,608·R_a = 461,5 J/(kg·K) (Cengel A-1: 0,4615; el vademecum
  redondea a 0,462): así s (§14.10) y ψ (§14.11) son coherentes y el balance de
  exergía con ψ da T₀·S_gen a 1e-10 en los procesos sin agua (con agua difiere
  hasta 3e-4 por el redondeo 0,622 vs 1/1,608).
- p_vs(T): sobre líquido desde el punto triple con un `AbstractState` de
  CoolProp **por hilo** (`threading.local`: Streamlit corre cada sesión en un
  hilo; ~100 veces más rápido que `PropsSI`), y sobre hielo con la ecuación de
  sublimación de IAPWS R14-08(2011) (230 K → 8,94735 Pa, test). Bajo 0 °C el
  rocío es de escarcha y el bulbo húmedo, de hielo (h = −333,4 + 2,1·t, ASHRAE).
  ω < 1e-12 se trata como aire seco (sin rocío): con una ω de 1e-17 el `brentq`
  del rocío fallaba con un error crudo.
- Bulbo húmedo = saturación adiabática, con `brentq` entre el rocío y T. Solo
  dos pares (h–φ, T_bh–φ) buscan T con `brentq`; los demás son cerrados.
  ω y T_pr son la misma información, y h y T_bh son casi paralelas: esos dos
  pares no se aceptan (quedan 13).
- h_w del agua que entra o sale (condensado, humidificador, torre) de tablas
  (IAPWS), como Cengel y el vademecum. Exergía del agua (Wepfer et al., 1979)
  contra el vapor del ambiente con el vapor del **modelo** a T₀ y p_v0: queda
  coherente con la ψ del aire.
- `coolprop_comparison`: los mismos datos a `HAPropsSI` (RP-1485). A 1 atm ω
  da 0,4 % más (factor de mejora) y T_bh/T_pr difieren < 0,02 K; a 7 bar, 2 %.
  Con ω = 0, HAPropsSI devuelve un rocío de 149 K: se muestra «—».
- Procesos: el tren numera en el sentido del flujo (la corriente que se mezcla
  va antes de la mezcla, como 14-8). La exergía del calor usa T_b: la fuente
  (por defecto 60 °C o 10 K sobre la salida) o la superficie del serpentín (=
  el condensado). Un enfriamiento sensible con el serpentín bajo el rocío es
  error. En 14-5 el libro no hace el balance del humidificador: con vapor
  saturado a 100 °C hace falta además ~1,6 kW (nota).
- Carta (`ui/psychro_chart.py`): ω a la derecha, familias en gris (la paleta
  validada queda para los procesos: calentar naranja, enfriar azul,
  humidificar aguamarina, mezcla gris), T_bh apagada en la leyenda (casi
  paralela a h). La **primera traza es una grilla invisible**: hover con el
  estado completo y `on_select` (callback) que carga T y φ en los widgets; la
  key del gráfico cambia después de cada toque para que no quede la selección.
  Rótulos de φ sobre una diagonal y solo 10, 20, 30, 40, 60 y 80 % (en 390 px
  se pisaban); sin rótulos de h que nacen arriba del 85 % de ω_máx.
- Página: recalcula sola (cada estado tarda ms; la carta se cachea con
  `st.cache_data`); barridos con botón. Keys `ps_{ejemplo}_…`, `hv_…`, `ct_…`;
  al cambiar de par, los datos nuevos salen del último estado
  (`{key}_last`). Los valores en °C usan `number_input_si` (key `…@sistema`).
- LaTeX: 3466 expresiones distintas, KaTeX estricto, máx. 316 px. Lo que hizo
  falta: h en tres renglones (h_a, h_v, h = h_a + ω·h_v); ω por bulbo húmedo
  con h_fg* y h_g − h_w* (Cengel ec. 14-14); lados izquierdos cortos (en un
  `aligned` el ancho del lado izquierdo se suma al del renglón más ancho:
  h*, Δh_a, Ẋ_Q); búsquedas en tabla en dos renglones con ×10ⁿ o negativos
  (`_is_wide` detecta también los negativos entre paréntesis); ω con 4 cifras.

Notas de la Fase 5 (combustión):

- Datos: los polinomios NASA de 9 coeficientes de McBride, Zehe y Gordon
  (2002), la fuente de la tabla del vademecum §16.12, extraídos de `thermo.inp`
  de NASA CEA (github.com/nasa/cea, Apache 2.0) con `scripts/extract_nasa9.py`
  (32 especies; los tramos de más de 6000 K se descartan). h = h_f +
  [poly(T) − poly(298,15 K)], así h(25 °C) = h_f exacto (el polinomio solo
  dejaba ~2 J/mol de diferencia). s° a 1 bar; en una mezcla, s° − R·ln(y·p/p°).
  Contra el vademecum: h_f a menos de 0,05 %, s° de los líquidos hasta 0,9
  kJ/(kmol·K); el H₂O de NASA se aparta de la A-23 de Cengel hasta 0,5 % a
  2000 K. h_fg del agua a 25 °C = 44 004 kJ/kmol (el vademecum dice 44 011).
- Combustibles: con especies van por mol (como Cengel); un análisis elemental,
  por kg, con h_f = PCS + Σ ν_p·h_f,p (agua líquida) y PCI = PCS −
  n_agua·h_fg. Propano y butano líquidos: el gas de NASA menos el h_fg de
  CoolProp a T. El PCS del carbón del ejemplo es 30,5 MJ/kg: con 31,1 la h_f
  salía positiva.
- Estequiometría como el vademecum (aire técnico 21/79, M = 28,85) o con el
  aire seco de Cengel u O₂ puro. Con λ < 1, la regla de Cengel 15-8 c (H → H₂O
  y S → SO₂ primero, el C reparte el resto entre CO₂ y CO) y error de hollín
  bajo el mínimo. Orsat: balances de C (con el SO₂) y N₂ (con el Ar) en un 2×2;
  el de H da el agua y el de O queda de control (residuo relativo).
- Humedad: φ se da a la temperatura del ambiente (`Oxidizer.T_humidity_K`) y
  precalentar no cambia el vapor; con φ a la T del comburente, 60 % a 200 °C no
  es aire húmedo y el barrido de precalentamiento se caía arriba de 80 °C. En
  la página, T₀ es el aire ambiente y el estado muerto (los ejemplos tienen
  T₀ = T del aire salvo los precalentados).
- Equilibrio (`equilibrium.py`): potenciales de elementos con la
  amortiguación de NASA CEA (Gordon y McBride, 1994, RP-1311, ec. 2.24–2.26);
  converge con Σ n_j·|Δln n_j| / Σ n_j < 0,5·10⁻⁵; la llama HP/UV con `brentq`
  en T (arranque en caliente). Control con Cantera 3.2 (solo en el scratchpad,
  no es dependencia) con los mismos polinomios y `reference-pressure: 1 bar`
  (con su 1 atm por defecto daba 0,5 K de diferencia): 0,004 K.
- Llama completa con `brentq` en [250, 6000] K; si pasa 6000 K (carbón con O₂
  puro) es error. A volumen constante, U = H − R_u·T·n_gas. Calor con
  condensación: n_v = p_sat/(p − p_sat)·n_seco; el reparto Q = PCI + q_reac −
  q_inq − q_humos + q_lat es una identidad (test).
- 15-8 c: el libro da 2236 K, pero con sus mismos productos el balance cierra
  en 2284 K («revisá en tu edición»; los tests van contra el balance).
- Segundo principio solo con especies (el análisis elemental no tiene s°):
  S_gen = S_p − S_r (+ Q/T_b), X_dest = T₀·S_gen y X_comb ≈ PCI (§16.13).
- Página: recalcula sola (~7 ms un caso con equilibrio; `st.cache_data`) y los
  barridos van con botón (el de λ con equilibrio, 0,3 s). Keys `cb_{ejemplo}_…`
  y `fg_{ejemplo}_…`; al cambiar la forma de dar el aire, el valor nuevo sale
  del último resultado (`{key}_last`). La composición y el análisis van en
  filas de dos columnas (un `st.columns` por par): con un solo
  `st.columns(2)`, en el celular se apilaban columna por columna y
  desordenaban los componentes. Las tablas muestran todas las filas
  (`height = 35·(n + 1) + 3`).
- Gráficos: la paleta validada (el violeta de «otros gases» se validó junto al
  naranja y al gris de las barras apiladas). En los barridos, el título arriba
  y la leyenda en su propio renglón a la izquierda: anclada a x = 0,35 no
  entraba en 390 px y plotly achicaba el gráfico para hacerle lugar; la línea
  punteada de los datos se explica en el texto (un rótulo pisaba la leyenda).
- Streamlit 1.65: `number_input`, `radio` y `checkbox` con key se identifican
  solo por la key (`key_as_main_identity`): cambiar el rótulo o el valor por
  defecto no los reinicia; `selectbox` y `multiselect` suman pocos parámetros.
- LaTeX: 14 745 expresiones distintas (procedimiento y teoría), KaTeX estricto,
  máx. 321 px. Lo que hizo falta: entalpías molares enteras con espacio fino
  (−393\,510), la reacción con el aire entre paréntesis en renglones propios,
  la cadena de K_p con el factor de presión en otro renglón si hay ×10ⁿ y los
  negativos del reparto con `latex_paren`. AppTest devuelve el LaTeX de
  `st.latex` envuelto en `$$`: sacarlo antes de validar.
- Smoke test de las ecuaciones: Streamlit ajusta la caja de cada `st.latex` a
  su contenido y KaTeX sobresale ~2 px por los glifos (scrollWidth =
  clientWidth + 2 sin barra visible, también en /Psicrometria). Lo que cuenta
  es si la ecuación se pasa del borde de su expansor (izquierda + scrollWidth
  contra el borde derecho del `stExpanderDetails` menos su padding).

Notas de la Fase 6 (poder calorífico por correlaciones):

- Cada correlación es una función de `core/combustion/heating_value.py` con la
  cita, el tipo de combustible y el rango de ajuste en el docstring (la regla
  de abajo); los coeficientes quedan en su forma publicada (MJ/kg por % en
  masa, **base seca**) y `CORRELATIONS` los junta con el LaTeX y los rangos.
  Dulong y Boie, en la forma que tabulan Channiwala y Parikh (2002): Dulong no
  tiene un trabajo original y Boie (1953) es ≈ su forma en kcal/kg × 4,1868
  (test). Cordero: 0,1708 (hay tablas secundarias con 0,17008).
- Bases (ASTM D3180-25): el H y el O del análisis son los de la materia seca
  (la convención de `UltimateAnalysis`); `from_basis` acepta tal cual, seca o
  seca y sin cenizas (con las cenizas en base seca), el O o el CF por
  diferencia, tolera 0,1 puntos en la suma y renormaliza. PCI = PCS −
  h_fg·(8,937·H + W): M_H₂O/(2·M_H) y h_fg = 2442,6 kJ/kg de NASA (el
  vademecum redondea a 2442); para una sustancia pura coincide con el PCI de
  la Fase 5 a 1e-9 (test).
- Validación con datos reales: las 536 biomasas de Ghugare et al. (2014) del
  paquete de R `modeldata` (MIT: la licencia va en `data/`; un nombre trae un
  espacio al final, por eso se compara con `strip`) y los carbones de Argonne
  (Vorres, 1990; O por diferencia, con el Cl). La red bloqueó varios hosts
  académicos: los carbones salen de reproducciones del *Users Handbook* que
  coinciden entre sí, con las cenizas y el S del inmediato pasados a seco
  iguales a los del elemental.
- Lo didáctico: Dulong subestima la biomasa (−10 %, −18 % con O > 45 %)
  porque supone el O unido al H; las del inmediato subestiman los carbones
  bituminosos 10–23 % aunque estén dentro de los rangos de cada variable
  (estar en rango no garantiza que la correlación sirva); en sustancias puras
  Dulong se pasa +11 % (metano: ignora la h_f) y acierta en el H₂, donde las
  de ajuste no extrapolan. Una nota que explica una correlación reemplaza sus
  avisos genéricos de rango y tipo (`explained`), que igual quedan en la tabla.
- Página: keys `hv_{ejemplo}_…`; los campos del análisis llevan la base
  (`{key}_{base}_{campo}`) y arrancan del último resultado pasado a esa base
  (`{key}_last`, con `to_basis`); la humedad (`{key}_W`) es una sola; las
  cenizas, `{key}_Aar` o `{key}_Ad` (seca y sin cenizas también usa la seca);
  el PCS medido, `{key}_refv_{base}`. El PCS exacto de una sustancia pura se
  deja de comparar si cambia la composición (tolerancia 1e-4: los widgets
  redondean). El selector de la correlación principal lleva el modo de
  análisis en la key (cambian sus opciones). El barrido de la humedad es
  instantáneo: va sin botón.
- /Combustion: selector «PCS» (dato o estimado con Channiwala y Parikh, Boie
  o Dulong); `Fuel.hhv_correlation` (`None` = dato, la 0.21.0 sin cambios)
  hace que el procedimiento muestre la correlación y el export la nombre.
- LaTeX: la sustitución de una correlación es una **tabla de aportes** (un
  renglón por componente, `C:\quad 0.3491 \cdot 48.64 &= 16.98`, y la suma
  sin unidad, que dice el renglón siguiente): con dos términos por renglón
  llegaba a 322–412 px. Las fórmulas, de a dos términos. 2078 expresiones,
  KaTeX estricto, máx. 311 px.
- Gráficos: la comparación es un gráfico de puntos (las diferencias son de
  pocos %); en la paridad y el error contra el O la leyenda va debajo del eje
  (a 390 px ocupa tres renglones y arriba pisaba el título), y el eje del
  error va de −50 a +40 % (fuera quedan menos de 10 muestras dudosas, que la
  página cuenta).
- Smoke test: `pkill -f "streamlit run"` también mata al shell que lo corre
  (su línea de comando contiene el patrón): usar `pkill -f "[s]treamlit run"`,
  y nunca en el mismo comando que lo vuelve a arrancar (esa línea contiene
  «streamlit run» y el shell se mata a sí mismo).

Notas de la Fase 7 (exergía física, química y por componente):

- `core/exergy/` es un paquete (el `exergy.py` de antes era un placeholder). Física: el
  estado muerto a (T₀, p₀) con CoolProp; ψ = (h − h₀) − T₀(s − s₀), φ con
  p₀(v − v₀); la parte térmica y la mecánica pasan por el estado a T₀ y p
  (Kotas, 1985); si (T₀, p) cae en la campana (p = p_sat(T₀)), ese estado es el
  líquido saturado (PX con x = 0). Las diferencias de CoolProp en el estado
  muerto dan ruido (Δs = −1,3e-14): `_noise` las pasa a 0 si son ≤ 1e-11
  relativas. `Ambient` valida −50 a 60 °C y 0,4 a 10 bar. Reproduce Cengel 10-8
  (ψ = 1162,1 y 449,0 kJ/kg) y los ejemplos del cap. 8 (tanque 281 MJ, R-134a
  38,0 kJ/kg, viento 70,7 kW, hogar 2195 Btu/s, bloque de hierro 8191 kJ).
- Química: `data/szargut_chemical_exergy.csv` (42 sustancias) con los dos
  modelos de la tabla A-26 de Moran y Shapiro: II = Szargut, Morris y Steward
  (1988), p₀ = 1 atm; I = Ahrendts (1980), p₀ = 1,019 atm. Valores digitales de
  `tespy/data/ChemEx` (TESPy 0.7.9, MIT); grafito y azufre, de la A-26. Lo que
  la tabla no trae sale del **método de Szargut**, ē = Δḡ_f + Σν·ē_elemento, con
  los polinomios NASA y s° del grafito (5,74) y del azufre rómbico (32,054), que
  no están en el CSV de NASA: reproduce la tabla dentro de 0,15 % en los
  hidrocarburos; el NO da 1,2 % (h_f de NASA actualizada) y el agua líquida 5 %
  relativo pero 0,05 kJ/mol (la página muestra los dos). El modelo I no incluye
  He, Ne, Kr ni Xe, ni los hidrocarburos sin datos de NASA (`species_available`;
  el error dice «elegí» el otro modelo). Gases del aire: x⁰⁰ = exp(−ē/R̄T₀).
- Mezclas: Σx·ē + R̄T₀·Σx·ln x; si el vapor supera p_sat(25 °C), la parte que
  condensa entra como líquido con su ē (como el «ideal-cond» de TESPy).
- Combustibles sólidos y líquidos (Szargut y Styrylska, 1964; Kotas, 1985): β
  en masa y base seca, con tres formas (carbón o/c ≤ 0,667; madera y biomasa
  hasta 2,67, que con o/c → 0 da la de carbón; líquidos);
  e = β·(PCI + W·h_fg) + (e_S − PCI_S)·S + e_w·W. La de líquidos contra
  sustancias puras: octano +0,6 %, etanol +1,2 %, metanol +2,7 % (aviso con
  mucho O); con las biomasas de Ghugare, e/PCS ≈ 1,054. Con humedad, e/PCI tal
  cual sube (el PCI descuenta W·h_fg y la exergía no): la página lo explica.
- Por componente (Bejan, Tsatsaronis y Moran, 1996): Ẋ_F = Ẋ_P + Ẋ_D + Ẋ_L,
  ε = Ẋ_P/Ẋ_F, y_D y y*_D. Convención: el calor que va al ambiente a T₀
  (condensador, interenfriadores) no lleva exergía, así que lo que pierde el
  fluido ahí se **destruye** (Cengel 10-8: caldera 1110 y condensador 414 kJ/kg);
  si el sumidero está más caliente que T₀, la exergía del calor es una
  **pérdida**; el escape y la chimenea (física + química) también. Válvulas y
  condensador son disipativos (sin producto). El Rankine usa por defecto el
  ambiente de 10-8 (290 K, 100 kPa) y la fuente a 1600 K con agua (T_máx + 50 K
  en un ORC). En la refrigeración, las mezclas toman el caudal de cada entrada y
  la cámara el de cada salida (con el de la mezcla, el balance quedaba 0,6 %
  abierto); la D coincide con la de la Fase 3.2. En la turbina de gas, la
  numeración de los caudales es explícita (`_mass_factors`: con regenerador el
  estado 5 es aire); con e_comb ≈ PCI, η_II = η térmico. Ciclo combinado: la HRSG
  por sección (`HRSGExergy.gained_W`), turbinas, mezclas, desaireador, bombas,
  condensador y chimenea. Todos los balances cierran a ≤ 7,5e-10 (tests).
- Grassmann (`ui/exergy_charts.py`): una banda vertical dibujada con `shapes`
  de plotly (el Sankey, en bucle, desenrollado o en cascada, no se leía a
  390 px). Azul la exergía que entra, gris lo destruido y violeta lo perdido
  (cada rama con su rótulo en su renglón), naranja el producto abajo; alto
  120 + 62·n px; las ramas de menos de 0,5 % se juntan en «Otros (n)» y como
  mucho van 10 (`grassmann_rows`). Los rótulos arrancan en x = 0,55 y se parten
  a 26 caracteres sin separar un número de su unidad (`_wrap_label`, espacio
  duro); el título y el producto pasan a dos renglones con más de 52 caracteres.
  A 390 px se cortaba «Escape (gases a 581,4 °C) (pérdida)». El gráfico de ε
  por componente parte los nombres igual: con «Mezcla del vapor de media
  (recalentamiento frío)» en un renglón, el eje se comía el gráfico. En el ciclo
  combinado la turbina de gas se llama así, junto a «Turbina de vapor de alta».
- Página: tres modos (`ex_mode`) con keys `xf_…` (fluido), `xh_…` (calor),
  `xs_…` (cuerpo), `xq_…` (química) y `xp_…` (planta). «El último que
  calculaste» lee `rk_inputs`, `rf_inputs`, `bt_inputs` y `cc_inputs` (lo que
  guarda cada página de ciclo) y lleva una huella de los datos en la key
  (`xp_{c}_last_{sha1[:8]}`): otro ciclo arranca con sus valores por defecto
  (T_H, T_L). Un `CombinedInputs` de la 3.4 se pasa al modelo de varias
  presiones adentro de la función cacheada con `from_combined_with_pinch`
  (Cengel 10-9 se diseña por chimenea: mismo ciclo a 1e-9). Con aire estándar
  no hay modelo de combustible para elegir (el calor entra con Q̇·(1 − T₀/T)).
- LaTeX: 6338 expresiones del procedimiento y 505 de la teoría y del ciclo
  combinado de una presión, KaTeX estricto, máx. 310 px. Lo que hizo falta: las
  búsquedas en tabla siempre en dos renglones, p₀·Δv con `_wrap`, las β de a un
  término por renglón, PCI* para PCI + W·h_fg, T₀ y p₀ en renglones propios, y
  en la teoría φ y e^{ch} de los combustibles en dos renglones (362 y 410 px).

Notas de la Fase 8.1 (transferencia de calor: conducción, aletas y convección):

- La Fase 8 se partió: 8.1 = conducción, aletas y convección (esta entrega);
  8.2 = radiación e intercambiadores (LMTD, ε-NTU). El vademecum todavía no
  tiene un capítulo de transferencia de calor: la teoría cita a Çengel y
  Ghajar (2015) e Incropera et al. (2007) y lo dice.
- Citas **por sección**, no por número de ecuación (cambian entre ediciones):
  Cengel y Ghajar §3-1 pared plana y U, §3-2 contacto, §3-3 redes
  generalizadas (partes en paralelo), §3-4 cilindros y esferas, §3-5 radio
  crítico, §3-6 aletas; Incropera §3.6.2–§3.6.5 (aletas), §7.2 placa, §7.4
  cilindro, §7.5 esfera, §8.4–§8.5 tubos, §9.6.1–§9.6.4 convección natural
  externa. Hasta este PR la conducción citaba §3-5 (paralelo) y §3-7 (radio
  crítico): se corrigió con la tabla de contenidos.
- Unidades de transferencia de calor en W (no kW) en los tres sistemas, como
  los libros; las magnitudes nuevas hacen que cada sustitución cierre sin
  factores (h·A y ṁ·c_p en W/K o Btu/(h·°F); en el inglés ṁ·c_p sale en
  Btu/(s·°F) y el procedimiento lo dice: ×3600; en el técnico c_p está en
  kJ/(kg·K): ×1000). Los espesores y diámetros se muestran en mm o in
  (`small_length`) pero en las fórmulas van en m o ft (`length`). Un test de
  coherencia (`test_heat_transfer_procedure_units_are_coherent`) lo vigila.
- Conducción: Q̇ > 0 va del lado 1 al 2; un borde «calor dado» del lado 2 da
  Q̇ = −Q̇_dato. Las temperaturas de la red se numeran T_∞,1, T₁, T₂… (dos en
  un contacto), T_∞,2. El radio crítico solo se comenta con una capa exterior
  aislante (k < 1 W/(m·K)): con el acero de un tanque es cierto pero no viene
  al caso. Reproduce Cengel y Ghajar exacto (630 W, 266,2 W, 69,25 W,
  261,9 W, 120,8 W/m, 105,0 °C).
- Aletas: la tabla 3.4 de Incropera en forma que no desborda (exp de mL) y la
  anular con las funciones de Bessel escaladas (`i0e`, `k0e`…) en un 2×2. Se
  validan contra `solve_bvp` adimensional (1e-6; 1e-5 la anular). En el
  procedimiento la anular se escribe con γ = C₁/C₂ y M = k·A_c·m·θ_b; las
  funciones de Bessel van a 5 cifras (la resta K₁ − γ·I₁ pierde precisión:
  con 4 el q salía 0,08 % corrido; test).
- Convección: las propiedades salen de CoolProp a T_f (Whitaker a T∞; el tubo
  a la T media iterada a 1e-9 K). El aire de CoolProp tiene k ~2,7 % mayor y
  Pr ~3 % menor que la tabla A-15: h queda 1,5–2 % arriba del libro; los tests
  piden ±3 % contra el libro y exacto contra el cálculo a mano con las
  propiedades del libro (Incropera 7.4: Hilpert 37,3 y Churchill y Bernstein
  40,6; Incropera 9.2: 147; Cengel y Ghajar: 124, 17,40 y 69,4). Los rangos
  se verifican con 1 % de tolerancia (el Pr del aire a 20 °C es 0,708 y
  Whitaker pide 0,71). La mixta de la placa da Nu < 0 con Re chico: no se
  muestra como alternativa, y su curva Nu(Re) arranca en Re_cr (el margen
  alrededor del punto la llevaba abajo y caía a Nu < 0). Las notas comparan las correlaciones que valen
  (≤ 25 %: incertidumbre típica; más: se usa la más general) y en la placa
  explican la transición (no son correlaciones rivales).
- Tubo: error si el fluido llega a la saturación dentro del tubo (con el
  T_sal de cada vuelta: «subí la presión» o «bajá la presión»), si q″ lo
  llevaría bajo el cero absoluto, y aviso de ebullición o condensación en la
  pared. El ejemplo de las resistencias de Cengel y Ghajar va a 2 bar (a 1 atm
  la pared, a 115 °C, pasaría la saturación). R1233zd(E) no tiene viscosidad
  en CoolProp: `convection_fluids()` lo saca de la lista.
- LaTeX: los adimensionales entre 10³ y 10⁷ van enteros, con espacio fino
  desde 10⁴ (`big`); las correlaciones largas se parten en factores con
  nombre (a, b, c, ψ, φ) y Ra^{1/6} se calcula antes; con ×10ⁿ o un Q̇
  negativo, Q̇·R va en otro renglón; las sumas de resistencias con ×10ⁿ, un
  término por renglón (al lado de R_total no entran dos). 1865 expresiones
  del procedimiento (ejemplos y variantes) y 37 de la teoría, KaTeX estricto,
  máx. 308 y 311 px. Python 3.11: nada de `\` dentro de las llaves de una
  f-string (armar los pedazos antes).
- Página: tres modos (`ht_mode`) con keys `qc_{ejemplo}_…` (conducción; las
  capas llevan la geometría: `qc_{e}_{geo}_{j}_…`), `qa_{e}_…` (aletas) y
  `qv_{e|i|n}{e}_…` (convección); recalcula sola (`st.cache_data` con los
  dataclasses). Gráficos con la paleta validada: el sólido azul con las capas
  sombreadas y numeradas (los nombres no entraban a 390 px), los fluidos
  naranjas con la película punteada; la comparación de correlaciones es un
  gráfico de puntos (hueco si está fuera de rango).

Notas de la Fase 8.2 (radiación e intercambiadores):

- Dos páginas nuevas en vez de más modos en /Transferencia_de_Calor (habría
  pasado de 2500 renglones): /Radiacion (☀️, keys `rd_mode`, `rb_…`, `rv_…`,
  `rt_…`, `re_…`, `rc_…` y `rk_…`) e /Intercambiadores (🔄, `hx_mode`, `xv_…`,
  `xd_…`, `xe_…` y `xu_…`). Las dos recalculan solas (`st.cache_data` con los
  dataclasses). El vademecum no tiene radiación ni intercambiadores, pero la
  exergía del intercambiador sigue a §11.10 (η_ex) y §13.2 (Δs = c·ln(T₂/T₁)).
- Unidades: la radiación usa la temperatura absoluta (`absolute_temperature`:
  K, K, °R) y λ en μm en los tres sistemas; E_bλ va por μm y σ en el Inglés es
  0,1712·10⁻⁸ Btu/(h·ft²·°R⁴). Los modos de cuerpo negro, factor de forma, dos
  superficies y recinto piden T absoluta; la superficie con convección y la
  termocupla piden la T del sistema y el procedimiento suma 273,15 o 459,67.
  En los intercambiadores, Q̇ = ṁ·h_fg y ṁ = Q̇/(c_p·ΔT) llevan el factor de
  unidades a la vista (1000 en el Técnico, 3600 en el Inglés).
- ε-NTU: el flujo cruzado con los dos fluidos sin mezclar va con la serie
  exacta (Mason, 1955; `gammainc`): la fórmula aproximada de las tablas se
  aparta hasta 3,8 % en ε y mueve mucho a F (11-6: 0,933 contra 0,970, el libro
  lee 0,97). El procedimiento muestra las dos. F = NTU_cc/NTU para cualquier
  tipo (igual a la fórmula cerrada de Bowman con un casco); P y R como Cengel y
  Ghajar con t el fluido de los tubos (el radio `tubes` solo cambia los
  rótulos: F(P, R) = F(P·R, 1/R)). Con C_r = 0 todos los tipos dan lo mismo.
- 11-5 con un solo paso de casco no se puede: P = 0,667 con R = 0,75 es justo
  el ε máximo (área infinita); es un error que sugiere más pasos de casco, y el
  ejemplo usa dos con caudales que reproducen las temperaturas del libro.
- c_p de CoolProp a la temperatura media de cada corriente (se itera, porque la
  salida depende de c_p), con la corriente en una sola fase («Subí la presión»
  o «Bajá la presión, o marcá que cambia de fase»). Una corriente que cambia de
  fase entra a T_sat (C → ∞) y su caudal sale de Q̇/h_fg; los pseudo-puros se
  rechazan (deslizamiento). En el ensayo, una corriente que cambia de fase pide
  su caudal solo si es el caudal conocido.
- Radiación: f(λT) con la serie de Chang y Rhee (contra `quad`, 5·10⁻⁹), los
  factores de forma contra Monte Carlo (3D), las cuerdas cruzadas (2D) y una
  caja cerrada (perpendiculares). Los recintos se arman desde la geometría
  (`ENCLOSURE_LAYOUTS`, `layout_geometry`: F de la regla de la suma y la
  reciprocidad) y cada ejemplo guarda su configuración
  (`RadiationExample.case`): la página cambia la condición de cada superficie
  (T dada, Q̇ dado o rerradiante). La boca de una cavidad y los costados entre
  dos placas son una superficie negra a la T de los alrededores (se valida). El
  ducto se calcula por metro de largo (las áreas son el ancho por 1 m).
- Gráficos: ε–NTU y F–P como los de los libros, con la familia en una rampa
  azul ordinal (validada) y la curva de los datos en naranja rayada; F–P solo
  en casco y tubos y flujo cruzado. Los espectros de la superficie y de la
  fuente van divididos por su máximo junto a ε(λ): todo adimensional en un
  solo eje (no es un eje doble). En los ejes logarítmicos plotly rotula 0,2
  como «2»: van marcas 1-2-5 explícitas (`_log_ticks`).
- LaTeX: 3094 expresiones distintas (procedimientos con los ejemplos y sus
  variantes, y la teoría), KaTeX estricto, máx. 321 px. Lo que hizo falta: un
  lado izquierdo largo sin alinear (`_flush_chain`, o solo en el primer
  renglón: las ecuaciones de las radiosidades, un término por renglón); el
  calor de cada superficie con J_i − J_j ya restado si las J llevan ×10ⁿ y el
  área en su propio renglón; las fórmulas con una variable auxiliar en dos
  renglones (`EPS_FORMULAS` y `NTU_FORMULAS` son tuplas de renglones); P y R
  por separado; el ln de adentro calculado antes (NTU del flujo cruzado
  mezclado), y en las fórmulas de los factores de forma, los términos con
  nombre (t₁…t₄; u, v, a, b, c).
- CITATION.cff: un número de artículo con cero adelante (025010) va entre
  comillas: sin ellas YAML lo lee en octal (10760).

Notas de la Fase 9.1 (gases ideales):

- La Fase 9 se partió como la 8: 9.1 = gases ideales (esta entrega, página
  /Gases_Ideales, 🎈, keys `ig_mode`, `gi_…`, `gm_kind`, `gm_c…`, `gm_a…`,
  `gp_kind`, `gp_p…`, `gp_s_…` y `gp_n_…`); 9.2 = gases reales (/Gases_Reales).
  La página recalcula sola (`st.cache_data` con los dataclasses).
- **Secciones del vademecum, verificadas contra `vademecum.tex`**: §4.2 p·v = R·T
  y R = R_u/M, §4.3 h = u + R·T, §4.5 Δu y Δh, §4.6 Mayer, §4.7 la tabla de
  gases, §4.8 los polinomios NASA; §5.1 a §5.7 las mezclas; §6.1 la ley general,
  §6.2 y §6.3 los trabajos, §6.3.1 las etapas, §6.4.1 c = c_v·(n − k)/(n − 1),
  §6.4.2 k, §6.4.5 el resumen; §10.5.1 a §10.5.3 la entropía (p_r, v_r); §11.8
  T₀·S_gen. Los primeros borradores citaban §4.5 (Mayer), §4.6 (tabla), §4.7
  (NASA) y §6.5 (resumen), que no existen o son otra cosa.
- c_p variable con los polinomios NASA-9 de McBride et al. (2002) que ya usaba la
  combustión (el vademecum §4.8 trae la forma de cinco términos): a 25 °C
  coinciden con la tabla del vademecum al 0,05 %. h, u y s° desde 25 °C; el helio
  es monoatómico (c_p = 5/2·R exacto). El tercer modelo, c_p a la temperatura
  media, es el de Çengel §7-9.
- Mezclas con la notación del vademecum: x es la fracción másica e y la molar.
  La entropía de mezcla se muestra como −R_M·Σ y·ln y (= −Σ x·R·ln y, porque
  x_i·R_i = y_i·R_M): con el producto x·R·ln y no entraba en 324 px. Un gas con
  fracción 0 se acepta (sin entropía, «—» en la tabla).
- Ruido de redondeo: la adiabática reversible tiene Δs = 0 exacto y una corriente
  que ya está a la T y la p final, Δs_TP = 0 (si no, el procedimiento mostraba
  7·10⁻¹⁵ y 6,8·10⁻¹⁸).
- Los cinco caminos (`process_comparison`): la politrópica usa el n del dato o
  1,3. La isócora hasta la p₂ de un compresor llega a 9·T₁ y aplastaba el T–s y
  el gráfico del trabajo: `far_paths` (T₂ − T₁ más de 5 veces la del dato, o
  50 K) la oculta en el T–s (`visible="legendonly"`, se ve tocándola en la
  leyenda) y la saca del gráfico del trabajo; la página lo dice y la tabla tiene
  todo. En el p–v queda (es vertical). Los rótulos de los estados van con
  `cliponaxis=False` (el «2» en el borde se cortaba).
- Gráficos con la paleta validada en su orden fijo (azul, naranja, aguamarina,
  amarillo, magenta, verde; validada para barras y líneas): cada proceso tiene
  su color y los gases de una mezcla toman los colores en el orden en que se
  cargan. Los tres claros no llegan a 3:1: las barras llevan sus porcentajes y la
  página trae las tablas.
- LaTeX: 3486 expresiones distintas del procedimiento (los ejemplos y variantes
  de cada proceso, dato, modelo y sistema) y 26 de la teoría, KaTeX estricto,
  máx. 315 y 293 px. Lo que hizo falta: las sumas de una mezcla de a un término
  por renglón (el lado izquierdo de un `aligned` se suma al ancho); en la mezcla
  adiabática, Δs_j de cada corriente antes de S_gen; el w_f de la isócora con la
  resta en otro renglón si hay factor de p·v o ×10ⁿ; la teoría sin `\qquad` de
  tres fórmulas (394 px). p·v lleva el factor a la vista: 100 en el Técnico
  (bar·m³/kg = 100 kJ/kg) y 0,18505 en el Inglés.
- Smoke test a 390 y 1280 px: 16 casos (los tres modos y sus submodos, en los
  tres sistemas) sin errores ni desbordes.

### Citas y licencias

- **Toda librería externa de cálculo** (no UI) que se sume al proyecto
  exige actualizar `CITATION.cff` con su referencia formal y `README.md`
  con su BibTeX.
- **Normas técnicas** (ISO, ASHRAE, IRAM): citar siempre versión y año.
- La página `99_Acerca.py` lee el `CITATION.cff` con `core/citation.py`: la
  cita de la app, todas las referencias (APA y BibTeX, con la bibliografía
  para descargar) y las licencias. Una referencia nueva en el CFF aparece
  sola; un archivo nuevo en `data/`, una dependencia nueva o una versión
  nueva hacen fallar un test hasta actualizar `DATA_SOURCES`, `LIBRARIES`,
  la cita del README y la versión de la home. No es un módulo de cálculo:
  no lleva teoría ni procedimiento (descarga el CFF y la bibliografía). Un
  número con cero adelante en el CFF va entre comillas (YAML lo lee en
  octal): un test lo vigila.

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

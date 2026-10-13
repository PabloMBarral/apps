"""Transferencia de calor (Fases 8 y 11): conducción, aletas, convección, radiación,
intercambiadores, conducción transitoria, ebullición y condensación.

- :mod:`core.heat_transfer.conduction`: red de resistencias térmicas en una
  pared plana, un cilindro o una esfera de varias capas; radio crítico.
- :mod:`core.heat_transfer.fins`: aletas rectas, de aguja y anulares; arreglos.
- :mod:`core.heat_transfer.convection`: correlaciones de convección forzada
  (externa e interna) y natural, con las propiedades de CoolProp.
- :mod:`core.heat_transfer.radiation`: cuerpo negro, bandas, factores de forma,
  dos superficies con pantallas, recintos y la radiación con convección.
- :mod:`core.heat_transfer.exchangers`: ε-NTU, LMTD con F, el ensayo con las
  cuatro temperaturas, el U global y la exergía.
- :mod:`core.heat_transfer.transient`: conducción transitoria (capacidad
  concentrada, la serie de la pared, el cilindro y la esfera, el sólido
  semi-infinito, el contacto de dos sólidos y el producto de soluciones).
- :mod:`core.heat_transfer.phase_change`: los fluidos y las propiedades de
  saturación (y de una fase) para el cambio de fase.
- :mod:`core.heat_transfer.boiling`: ebullición en pileta (Rohsenow, el flujo
  crítico de Zuber con Lienhard y Dhir, el mínimo, la película de Bromley) y la
  curva de ebullición.
- :mod:`core.heat_transfer.condensation`: condensación en película (Nusselt,
  Kutateladze, Labuntsov; placa, tubos, banco, esfera y Chato por dentro).

Todo en SI y sin Streamlit. El vademecum de la cátedra no tiene (todavía) un
capítulo de transferencia de calor: las fórmulas siguen a Cengel y Ghajar
(*Transferencia de calor y masa*) e Incropera et al. (*Fundamentals of Heat and
Mass Transfer*).
"""

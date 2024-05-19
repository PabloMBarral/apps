import streamlit as st
import CoolProp.CoolProp as cp

# Notas
# CoolProp usa pascales absolutos y kelvin, y devuelve en J / kg.

# Título de la aplicación
st.title("Calculador de entalpías")

# Entradas para la temperatura y presión como números decimales
t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
p = st.number_input("Ingrese la presión [bar(a)]", value=0.0, step=0.01, format="%.2f")


# Botón para calcular la entalpía
if st.button("Calcular"):
    h = cp.PropsSI('H', 'P', p * 1e5, 'T', t + 273.15, 'Water') / 1000  # Entalpía en [kJ/kg]
    st.write(f"La entalpía a {t:.2f} °C y {p:.2f} bar(a) es {h:.2f} kJ/kg.")

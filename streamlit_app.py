import streamlit as st
from fluids.units import *
from thermo.units import Stream


# Título de la aplicación
st.title("Calculador de entalpías")

# Entradas para los dos números
T = st.number_input("Ingrese la temperatura [°C]", value=0)
P = st.number_input("Ingrese la presión [bar(a)]", value=0)

#T = T * u.degC
#P = P * u.bar

# Botón para realizar la suma
if st.button("Sumar"):
    steam = Stream('water', T = T, P = P, m = 40000) #* u.kg / u.hr)
    st.write(f"La entalpía a {T} °C y {P} bar(a) es {steam.rho} kJ/kg.")

import streamlit as st
import CoolProp.CoolProp as cp


# Título de la aplicación
st.title("Calculador de entalpías")

# Entradas para los dos números
t = st.number_input("Ingrese la temperatura [°C]", value=0)
p = st.number_input("Ingrese la presión [bar(a)]", value=0)


# Botón para realizar la suma
if st.button("Sumar"):
    h = cp.PropsSI('H', 'P', p * 1e5, 'T', t + 273.15, 'Water')  / 1000 # Entalpía en [kJ/kg]
    st.write(f"La entalpía a {t} °C y {p} bar(a) es {h} kJ/kg.")

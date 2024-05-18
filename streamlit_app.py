import streamlit as st

# Título de la aplicación
st.title("Suma de dos números")

# Entradas para los dos números
num1 = st.number_input("Ingrese el primer número", value=0)
num2 = st.number_input("Ingrese el segundo número", value=0)

# Botón para realizar la suma
if st.button("Sumar"):
    suma = num1 + num2
    st.write(f"La suma de {num1} y {num2} es {suma}")
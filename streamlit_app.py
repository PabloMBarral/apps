import streamlit as st
import CoolProp.CoolProp as cp

# Título de la aplicación
st.title("Calculador de propiedades del agua")

# Sección para el ingreso de temperatura y presión
st.header("Calcular desde Temperatura y Presión")
t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
p_t = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")

# Botón para calcular las propiedades desde temperatura y presión
if st.button("Calcular desde Temperatura y Presión"):
    # Convertir las unidades
    t_kelvin = t + 273.15
    p_pascal_t = p_t * 1e5
    
    # Calcular la entalpía
    h_tp = cp.PropsSI('H', 'P', p_pascal_t, 'T', t_kelvin, 'Water') / 1000  # Entalpía en [kJ/kg]
    st.write(f"La entalpía a {t:.2f} °C y {p_t:.2f} bar(a) es {h_tp:.2f} kJ/kg.")
    
    # Calcular la entropía
    s_tp = cp.PropsSI('S', 'P', p_pascal_t, 'T', t_kelvin, 'Water') / 1000  # Entropía en [kJ/(kg·K)]
    st.write(f"La entropía a {t:.2f} °C y {p_t:.2f} bar(a) es {s_tp:.4f} kJ/(kg·K).")
    
    # Calcular el título (calidad del vapor)
    try:
        x_tp = cp.PropsSI('Q', 'P', p_pascal_t, 'T', t_kelvin, 'Water')  # Calidad del vapor
        if 0 <= x_tp <= 1:
            st.write(f"El título (calidad del vapor) a {t:.2f} °C y {p_t:.2f} bar(a) es {x_tp:.2f}.")
        else:
            st.write(f"La mezcla a {t:.2f} °C y {p_t:.2f} bar(a) no es una mezcla de líquido y vapor (título fuera del rango 0-1).")
    except ValueError:
        st.write(f"No se puede calcular el título (calidad del vapor) para las condiciones dadas.")

# Separador
st.markdown("---")

# Sección para el ingreso de presión y entalpía
st.header("Calcular desde Presión y Entalpía")
h = st.number_input("Ingrese la entalpía [kJ/kg]", value=0.0, step=0.01, format="%.2f")
p_h = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")

# Botón para calcular las propiedades desde presión y entalpía
if st.button("Calcular desde Presión y Entalpía"):
    # Convertir las unidades
    h_joules = h * 1000
    p_pascal_h = p_h * 1e5
    
    # Calcular la temperatura
    t_hk = cp.PropsSI('T', 'P', p_pascal_h, 'H', h_joules, 'Water')
    t_h = t_hk - 273.15  # Convertir de Kelvin a Celsius
    st.write(f"La temperatura a {h:.2f} kJ/kg y {p_h:.2f} bar(a) es {t_h:.2f} °C.")
    
    # Calcular la entropía
    s_h = cp.PropsSI('S', 'P', p_pascal_h, 'H', h_joules, 'Water') / 1000  # Entropía en [kJ/(kg·K)]
    st.write(f"La entropía a {h:.2f} kJ/kg y {p_h:.2f} bar(a) es {s_h:.4f} kJ/(kg·K).")
    
    # Calcular el título (calidad del vapor)
    try:
        x_h = cp.PropsSI('Q', 'P', p_pascal_h, 'H', h_joules, 'Water')  # Calidad del vapor
        if 0 <= x_h <= 1:
            st.write(f"El título (calidad del vapor) a {h:.2f} kJ/kg y {p_h:.2f} bar(a) es {x_h:.2f}.")
        else:
            st.write(f"La mezcla a {h:.2f} kJ/kg y {p_h:.2f} bar(a) no es una mezcla de líquido y vapor (título fuera del rango 0-1).")
    except ValueError:
        st.write(f"No se puede calcular el título (calidad del vapor) para las condiciones dadas.")

import streamlit as st
import CoolProp.CoolProp as cp

# Título de la aplicación
st.title("Calculador de propiedades del agua")

# Definir funciones para cálculos específicos
def calcular_propiedades(desde, **kwargs):
    t = None
    p = None
    h = None
    s = None
    x = None
    
    try:
        if desde == 'TP':
            t_kelvin = kwargs['t'] + 273.15
            p_pascal = kwargs['p'] * 1e5
            h = cp.PropsSI('H', 'P', p_pascal, 'T', t_kelvin, 'Water') / 1000
            s = cp.PropsSI('S', 'P', p_pascal, 'T', t_kelvin, 'Water') / 1000
            x = cp.PropsSI('Q', 'P', p_pascal, 'T', t_kelvin, 'Water')
        elif desde == 'PH':
            h_joules = kwargs['h'] * 1000
            p_pascal = kwargs['p'] * 1e5
            t_kelvin = cp.PropsSI('T', 'P', p_pascal, 'H', h_joules, 'Water')
            s = cp.PropsSI('S', 'P', p_pascal, 'H', h_joules, 'Water') / 1000
            x = cp.PropsSI('Q', 'P', p_pascal, 'H', h_joules, 'Water')
            t = t_kelvin - 273.15
        elif desde == 'HS':
            h_joules = kwargs['h'] * 1000
            s_joules = kwargs['s'] * 1000
            t_kelvin = cp.PropsSI('T', 'H', h_joules, 'S', s_joules, 'Water')
            p_pascal = cp.PropsSI('P', 'H', h_joules, 'S', s_joules, 'Water')
            x = cp.PropsSI('Q', 'H', h_joules, 'S', s_joules, 'Water')
            t = t_kelvin - 273.15
            p = p_pascal / 1e5
        elif desde == 'HX':
            h_joules = kwargs['h'] * 1000
            x = kwargs['x']
            t_kelvin = cp.PropsSI('T', 'H', h_joules, 'Q', x, 'Water')
            p_pascal = cp.PropsSI('P', 'H', h_joules, 'Q', x, 'Water')
            s = cp.PropsSI('S', 'H', h_joules, 'Q', x, 'Water') / 1000
            t = t_kelvin - 273.15
            p = p_pascal / 1e5
        elif desde == 'PX':
            p_pascal = kwargs['p'] * 1e5
            x = kwargs['x']
            t_kelvin = cp.PropsSI('T', 'P', p_pascal, 'Q', x, 'Water')
            h = cp.PropsSI('H', 'P', p_pascal, 'Q', x, 'Water') / 1000
            s = cp.PropsSI('S', 'P', p_pascal, 'Q', x, 'Water') / 1000
            t = t_kelvin - 273.15
        elif desde == 'TX':
            t_kelvin = kwargs['t'] + 273.15
            x = kwargs['x']
            p_pascal = cp.PropsSI('P', 'T', t_kelvin, 'Q', x, 'Water')
            h = cp.PropsSI('H', 'T', t_kelvin, 'Q', x, 'Water') / 1000
            s = cp.PropsSI('S', 'T', t_kelvin, 'Q', x, 'Water') / 1000
            p = p_pascal / 1e5
        elif desde == 'SX':
            s_joules = kwargs['s'] * 1000
            x = kwargs['x']
            t_kelvin = cp.PropsSI('T', 'S', s_joules, 'Q', x, 'Water')
            p_pascal = cp.PropsSI('P', 'S', s_joules, 'Q', x, 'Water')
            h = cp.PropsSI('H', 'S', s_joules, 'Q', x, 'Water') / 1000
            t = t_kelvin - 273.15
            p = p_pascal / 1e5
        elif desde == 'PS':
            p_pascal = kwargs['p'] * 1e5
            s_joules = kwargs['s'] * 1000
            t_kelvin = cp.PropsSI('T', 'P', p_pascal, 'S', s_joules, 'Water')
            h = cp.PropsSI('H', 'P', p_pascal, 'S', s_joules, 'Water') / 1000
            x = cp.PropsSI('Q', 'P', p_pascal, 'S', s_joules, 'Water')
            t = t_kelvin - 273.15
        elif desde == 'TH':
            t_kelvin = kwargs['t'] + 273.15
            h_joules = kwargs['h'] * 1000
            p_pascal = cp.PropsSI('P', 'T', t_kelvin, 'H', h_joules, 'Water')
            s = cp.PropsSI('S', 'T', t_kelvin, 'H', h_joules, 'Water') / 1000
            x = cp.PropsSI('Q', 'T', t_kelvin, 'H', h_joules, 'Water')
            p = p_pascal / 1e5
        elif desde == 'TS':
            t_kelvin = kwargs['t'] + 273.15
            s_joules = kwargs['s'] * 1000
            p_pascal = cp.PropsSI('P', 'T', t_kelvin, 'S', s_joules, 'Water')
            h = cp.PropsSI('H', 'T', t_kelvin, 'S', s_joules, 'Water') / 1000
            x = cp.PropsSI('Q', 'T', t_kelvin, 'S', s_joules, 'Water')
            p = p_pascal / 1e5

        # Devolver todas las propiedades calculadas
        return t, p, h, s, x

    except Exception as e:
        st.error(f"Error en el cálculo: {e}")
        return None, None, None, None, None

# Sección para diferentes combinaciones de propiedades
st.header("Calcular propiedades del agua")

# Formulario para Temperatura y Presión
with st.form(key='tp_form'):
    t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
    p_t = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")
    tp_submit_button = st.form_submit_button(label='Calcular desde Temperatura y Presión')

if tp_submit_button:
    t, p, h, s, x = calcular_propiedades('TP', t=t, p=p_t)
    if t is not None:
        st.write(f"Resultados a {t:.2f} °C y {p:.2f} bar(a):")
        st.write(f"Entalpía: {h:.2f} kJ/kg")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Formulario para Presión y Entalpía
with st.form(key='ph_form'):
    h = st.number_input("Ingrese la entalpía [kJ/kg]", value=0.0, step=0.01, format="%.2f")
    p_h = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")
    ph_submit_button = st.form_submit_button(label='Calcular desde Presión y Entalpía')

if ph_submit_button:
    t, p, h, s, x = calcular_propiedades('PH', h=h, p=p_h)
    if t is not None:
        st.write(f"Resultados a {h:.2f} kJ/kg y {p:.2f} bar(a):")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Formulario para Entalpía y Entropía
with st.form(key='hs_form'):
    h = st.number_input("Ingrese la entalpía [kJ/kg]", value=0.0, step=0.01, format="%.2f")
    s = st.number_input("Ingrese la entropía [kJ/(kg·K)]", value=0.0, step=0.01, format="%.4f")
    hs_submit_button = st.form_submit_button(label='Calcular desde Entalpía y Entropía')

if hs_submit_button:
    t, p, h, s, x = calcular_propiedades('HS', h=h, s=s)
    if t is not None:
        st.write(f"Resultados a {h:.2f} kJ/kg y {s:.4f} kJ/(kg·K):")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Formulario para Entalpía y Título
with st.form(key='hx_form'):
    h = st.number_input("Ingrese la entalpía [kJ/kg]", value=0.0, step=0.01, format="%.2f")
    x = st.number_input("Ingrese el título (calidad del vapor) [0-1]", value=0.0, step=0.01, format="%.2f")
    hx_submit_button = st.form_submit_button(label='Calcular desde Entalpía y Título')

if hx_submit_button:
    t, p, h, s, x = calcular_propiedades('HX', h=h, x=x)
    if t is not None:
        st.write(f"Resultados a {h:.2f} kJ/kg y {x:.2f}:")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")

# Separador
st.markdown("---")

# Formulario para Presión y Título
with st.form(key='px_form'):
    p = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")
    x = st.number_input("Ingrese el título (calidad del vapor) [0-1]", value=0.0, step=0.01, format="%.2f")
    px_submit_button = st.form_submit_button(label='Calcular desde Presión y Título')

if px_submit_button:
    t, p, h, s, x = calcular_propiedades('PX', p=p, x=x)
    if t is not None:
        st.write(f"Resultados a {p:.2f} bar(a) y {x:.2f}:")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Entalpía: {h:.2f} kJ/kg")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")

# Separador
st.markdown("---")

# Formulario para Temperatura y Título
with st.form(key='tx_form'):
    t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
    x = st.number_input("Ingrese el título (calidad del vapor) [0-1]", value=0.0, step=0.01, format="%.2f")
    tx_submit_button = st.form_submit_button(label='Calcular desde Temperatura y Título')

if tx_submit_button:
    t, p, h, s, x = calcular_propiedades('TX', t=t, x=x)
    if t is not None:
        st.write(f"Resultados a {t:.2f} °C y {x:.2f}:")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Entalpía: {h:.2f} kJ/kg")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")

# Separador
st.markdown("---")

# Formulario para Entropía y Título
with st.form(key='sx_form'):
    s = st.number_input("Ingrese la entropía [kJ/(kg·K)]", value=0.0, step=0.01, format="%.4f")
    x = st.number_input("Ingrese el título (calidad del vapor) [0-1]", value=0.0, step=0.01, format="%.2f")
    sx_submit_button = st.form_submit_button(label='Calcular desde Entropía y Título')

if sx_submit_button:
    t, p, h, s, x = calcular_propiedades('SX', s=s, x=x)
    if t is not None:
        st.write(f"Resultados a {s:.4f} kJ/(kg·K) y {x:.2f}:")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Entalpía: {h:.2f} kJ/kg")

# Separador
st.markdown("---")

# Formulario para Presión y Entropía
with st.form(key='ps_form'):
    p = st.number_input("Ingrese la presión [bar(a)]", value=1.0, step=0.01, format="%.2f")
    s = st.number_input("Ingrese la entropía [kJ/(kg·K)]", value=0.0, step=0.01, format="%.4f")
    ps_submit_button = st.form_submit_button(label='Calcular desde Presión y Entropía')

if ps_submit_button:
    t, p, h, s, x = calcular_propiedades('PS', p=p, s=s)
    if t is not None:
        st.write(f"Resultados a {p:.2f} bar(a) y {s:.4f} kJ/(kg·K):")
        st.write(f"Temperatura: {t:.2f} °C")
        st.write(f"Entalpía: {h:.2f} kJ/kg")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Formulario para Temperatura y Entalpía
with st.form(key='th_form'):
    t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
    h = st.number_input("Ingrese la entalpía [kJ/kg]", value=0.0, step=0.01, format="%.2f")
    th_submit_button = st.form_submit_button(label='Calcular desde Temperatura y Entalpía')

if th_submit_button:
    t, p, h, s, x = calcular_propiedades('TH', t=t, h=h)
    if t is not None:
        st.write(f"Resultados a {t:.2f} °C y {h:.2f} kJ/kg:")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Entropía: {s:.4f} kJ/(kg·K)")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Formulario para Temperatura y Entropía
with st.form(key='ts_form'):
    t = st.number_input("Ingrese la temperatura [°C]", value=0.0, step=0.01, format="%.2f")
    s = st.number_input("Ingrese la entropía [kJ/(kg·K)]", value=0.0, step=0.01, format="%.4f")
    ts_submit_button = st.form_submit_button(label='Calcular desde Temperatura y Entropía')

if ts_submit_button:
    t, p, h, s, x = calcular_propiedades('TS', t=t, s=s)
    if t is not None:
        st.write(f"Resultados a {t:.2f} °C y {s:.4f} kJ/(kg·K):")
        st.write(f"Presión: {p:.2f} bar(a)")
        st.write(f"Entalpía: {h:.2f} kJ/kg")
        st.write(f"Título: {x:.2f}")

# Separador
st.markdown("---")

# Texto adicional
st.write("Desarrollado por Pablo M. Barral. Contacto: pbarral@fi.uba.ar. Powered by CoolProp.")

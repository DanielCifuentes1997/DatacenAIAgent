import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# Importamos las funciones que creamos en el archivo analysis.py
import analysis

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="Centinela de Operaciones 1.0",
    page_icon="⚡",
    layout="wide"
)

st.title("Centinela de Operaciones 1.0 ⚡")
st.markdown("Carga los datos de sensores de un activo para detectar anomalías operativas mediante IA.")

# --- 2. Carga de Archivos ---
uploaded_file = st.file_uploader(
    "Selecciona un archivo .csv con los datos de los sensores",
    type="csv"
)

# --- 3. Lógica Principal de la Aplicación ---
if uploaded_file is not None:
    with st.spinner('Analizando datos... Este proceso puede tardar unos segundos.'):
        df_original = pd.read_csv(uploaded_file)
        
        # --- PASO A: LIMPIEZA Y PREPARACIÓN ---
        # La función devuelve un dataframe con todas las etapas intermedias
        df_limpio = analysis.pipeline_limpieza_completa(df_original.copy())

        # --- PASO B: INGENIERÍA DE CARACTERÍSTICAS ---
        df_features = analysis.crear_caracteristicas_ingenieria(df_limpio.copy())
        
        # --- PASO C: MODELO DE DETECCIÓN DE ANOMALÍAS ---
        features_para_modelo = [
            'voltaje_diff_media', 'frecuencia_diff_media',
            'temperatura_diff_media', 'vibracion_diff_media',
            'voltaje_std_movil', 'vibracion_std_movil'
        ]
        # Corregido: 'contamination' debe ser un número (float), no un texto.
        modelo = IsolationForest(contamination=0.005, random_state=42)
        df_features['anomalia'] = modelo.fit_predict(df_features[features_para_modelo])
        df_features['es_anomalia'] = df_features['anomalia'].apply(lambda x: 'Anomalía' if x == -1 else 'Normal')
        
    st.success("¡Análisis completado!")
    
    anomalias = df_features[df_features['es_anomalia'] == 'Anomalía']

    # --- PASO D: VISUALIZACIÓN DE RESULTADOS ---
    col1, col2 = st.columns(2)
    col1.metric("Registros Analizados", f"{len(df_features):,}")
    col2.metric("Anomalías Detectadas", f"{len(anomalias):,}", delta_color="inverse")

    st.subheader("Visualización de Series Temporales con Anomalías")
    
    sensor_a_visualizar = st.selectbox(
        'Selecciona una métrica de sensor para visualizar:',
        options=['voltaje', 'temperatura', 'frecuencia', 'vibracion']
    )
    columna_visual = f'{sensor_a_visualizar}_visualizacion'
    
    fig = px.line(
        df_features, x='timestamp', y=columna_visual,
        title=f'Comportamiento del sensor: {sensor_a_visualizar.capitalize()}'
    )
    fig.add_trace(px.scatter(anomalias, x='timestamp', y=columna_visual).data[0])
    fig.data[1].marker.color = 'red'
    fig.data[1].marker.size = 8
    fig.data[1].name = 'Anomalía Detectada'
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla de Anomalías Detectadas")
    columnas_tabla = {
        'timestamp': 'Fecha y Hora', 'voltaje_visualizacion': 'Voltaje',
        'frecuencia_visualizacion': 'Frecuencia', 'temperatura_visualizacion': 'Temperatura',
        'vibracion_visualizacion': 'Vibración'
    }
    st.dataframe(
        anomalias[columnas_tabla.keys()].rename(columns=columnas_tabla).reset_index(drop=True)
    )

    # --- SECCIÓN DE VERIFICACIÓN (Restaurada y completa) ---
    with st.expander("Verificar el Proceso de Limpieza de Datos"):
        st.markdown("Aquí puedes comparar los datos antes y después de cada etapa del pipeline de limpieza.")
        
        sensor_a_verificar = st.selectbox(
            'Selecciona un sensor para verificar la limpieza:',
            options=['voltaje', 'temperatura', 'frecuencia', 'vibracion'],
            key='verificacion_sensor'
        )

        # Gráfico para verificar la Reducción de Ruido
        st.subheader(f"1. Verificación de Reducción de Ruido para '{sensor_a_verificar.capitalize()}'")
        fig_ruido = go.Figure()
        fig_ruido.add_trace(go.Scatter(x=df_limpio['timestamp'], y=df_limpio[f'{sensor_a_verificar}_visualizacion'], mode='lines', name='Señal Original'))
        fig_ruido.add_trace(go.Scatter(x=df_limpio['timestamp'], y=df_limpio[f'{sensor_a_verificar}_suavizado'], mode='lines', name='Señal Suavizada', line=dict(color='red')))
        fig_ruido.update_layout(title="Comparación: Señal Original vs. Señal Suavizada")
        st.plotly_chart(fig_ruido, use_container_width=True)

        # Gráfico para verificar el Manejo de Outliers
        st.subheader(f"2. Verificación de Manejo de Outliers para '{sensor_a_verificar.capitalize()}'")
        st.write("Las líneas punteadas muestran los límites calculados con el método IQR. Los valores fuera de estas líneas son considerados 'outliers' y son recortados.")
        
        col_suavizada = f'{sensor_a_verificar}_suavizado'
        Q1 = df_limpio[col_suavizada].quantile(0.25)
        Q3 = df_limpio[col_suavizada].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Scatter(x=df_limpio['timestamp'], y=df_limpio[col_suavizada], mode='lines', name='Señal Suavizada'))
        fig_outliers.add_hline(y=limite_superior, line_dash="dash", line_color="red", annotation_text="Límite Superior IQR")
        fig_outliers.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior IQR")
        fig_outliers.update_layout(title="Señal Suavizada y Límites para Detección de Outliers")
        st.plotly_chart(fig_outliers, use_container_width=True)

# --- 4. Guía para el Usuario ---
else:
    st.info("Por favor, carga un archivo .csv para comenzar el análisis.")
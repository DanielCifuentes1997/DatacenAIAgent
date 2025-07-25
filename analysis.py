import pandas as pd
from sklearn.preprocessing import StandardScaler

def pipeline_limpieza_completa(df):
    """
    Ejecuta un pipeline de preprocesamiento completo sobre los datos de los sensores.
    """
    print("Iniciando pipeline de limpieza completo...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    columnas_sensores = ['voltaje', 'frecuencia', 'temperatura', 'vibracion']

    # Etapa 1: Manejo de Valores Faltantes
    df[columnas_sensores] = df[columnas_sensores].fillna(method='ffill').fillna(method='bfill')
    print("Etapa 1: Valores faltantes rellenados.")

    # Etapa 2: Reducción de Ruido
    for col in columnas_sensores:
        df[f'{col}_suavizado'] = df[col].rolling(window=5, center=True, min_periods=1).mean()
    print("Etapa 2: Ruido reducido mediante suavizado.")

    # Etapa 3: Manejo de Outliers
    for col in columnas_sensores:
        Q1 = df[f'{col}_suavizado'].quantile(0.25)
        Q3 = df[f'{col}_suavizado'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df[f'{col}_limpio'] = df[f'{col}_suavizado'].clip(lower=limite_inferior, upper=limite_superior)
    print("Etapa 3: Outliers manejados.")

    # Etapa 4: Normalización
    scaler = StandardScaler()
    columnas_limpias = [f'{col}_limpio' for col in columnas_sensores]
    # Guardamos las columnas originales para visualización antes de normalizar
    for col in columnas_sensores:
        df[f'{col}_visualizacion'] = df[col] # Creamos copia para los gráficos
    df[columnas_limpias] = scaler.fit_transform(df[columnas_limpias])
    print("Etapa 4: Datos normalizados.")

    # Usamos las columnas limpias para el modelo, renombrándolas a su nombre original
    for col in columnas_sensores:
        df[col] = df[f'{col}_limpio']
    
    print("Pipeline de limpieza completado.")
    return df

def crear_caracteristicas_ingenieria(df, ventana_minutos=5):
    """
    Crea características de ingeniería basadas en ventanas de tiempo móviles.
    ESTA ES LA FUNCIÓN QUE FALTABA.
    """
    print(f"Iniciando ingeniería de características con ventana de {ventana_minutos} minutos...")
    # Usamos una copia para evitar el SettingWithCopyWarning
    df_features = df.copy()
    df_temp = df_features.set_index('timestamp')
    
    tamano_ventana = ventana_minutos * 60
    columnas_sensores = ['voltaje', 'frecuencia', 'temperatura', 'vibracion']

    for col in columnas_sensores:
        # Media móvil sobre los datos ya limpios y normalizados
        df_features[f'{col}_media_movil'] = df_temp[col].rolling(window=tamano_ventana, min_periods=1).mean().values
        # Desviación estándar móvil
        df_features[f'{col}_std_movil'] = df_temp[col].rolling(window=tamano_ventana, min_periods=1).std().values
        # Diferencia con la media móvil
        df_features[f'{col}_diff_media'] = df_features[col] - df_features[f'{col}_media_movil']

    # Rellena los posibles NaNs generados al principio
    df_features = df_features.fillna(0)

    print("Ingeniería de características completada.")
    return df_features
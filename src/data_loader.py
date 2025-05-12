import pandas as pd
from sklearn.preprocessing import LabelEncoder

def cargar_datos(ruta_csv="C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/data/Data_completa.csv"):
    df = pd.read_csv(ruta_csv)

    # Convertir a datetime
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.dayofweek

    # Mapeo de gravedad a clase numérica
    gravedad_map = {
        'Solo danos': 0,
        'Con heridos': 1,
        'Con muertos': 2
    }
    df['gravedad_cat'] = df['gravedad'].map(gravedad_map)

    # IMPUTACIÓN DE NULOS
    
    df['choque'] = df['choque'].fillna(df['choque'].mode()[0])
    df['clasevehiculo'] = df['clasevehiculo'].fillna(df['clasevehiculo'].mode()[0])
    df['servicio'] = df['servicio'].fillna("Desconocido")

    # Columnas categóricas para codificar
    cat_cols = ['periodo_dia', 'clase','localidad', 'disenolugar', 'clasevehiculo', 'causa', 'servicio']

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        label_encoders[col] = le

    features = ['anio', 'mes', 'dia_semana'] + [col + '_enc' for col in cat_cols]
    target = 'gravedad_cat'

    X = df[features]
    y = df[target]

    return X, y, df, label_encoders

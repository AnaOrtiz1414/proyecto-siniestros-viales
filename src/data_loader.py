import pandas as pd
from sklearn.preprocessing import LabelEncoder
import kagglehub

def cargar_datos():
    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "williamrrubio/data-siniestros-bogot-2015-2021",
        "Data_completa.csv"
    )

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
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import kagglehub

def cargar_datos():
    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "williamrrubio/data-siniestros-bogot-2015-2021",
        "Data_completa.csv"
    )

    # Procesar fecha
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.dayofweek

    # Codificar gravedad
    gravedad_map = {'Solo danos': 0, 'Con heridos': 1, 'Con muertos': 2}
    df['gravedad_cat'] = df['gravedad'].map(gravedad_map)

    # Imputación
    df['clasevehiculo'] = df['clasevehiculo'].fillna(df['clasevehiculo'].mode()[0])
    df['servicio'] = df['servicio'].fillna("Desconocido")

    # Columnas categóricas (todas)
    cat_cols = ['periodo_dia', 'clase', 'localidad', 'disenolugar',
                'clasevehiculo', 'causa', 'servicio', 'anio', 'mes', 'dia_semana']

    # Codificación ordinal
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    X = df[cat_cols]
    y = df['gravedad_cat']

    return X, y

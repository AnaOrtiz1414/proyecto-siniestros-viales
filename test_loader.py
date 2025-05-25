
from src.data_loader import cargar_datos_kaggle

# Ejecutar carga de datos
X, y, df, encoder = cargar_datos_kaggle()

# Mostrar resultados básicos
print(f"✅ DataFrame original: {df.shape}")
print(f"✅ Features (X): {X.shape}")
print(f"✅ Target (y): {y.shape}")
print(f"✅ Clases únicas en y: {y.unique()}")

# Verificar si hay valores nulos
print("\n🔍 Valores nulos en X:")
print(X.isnull().sum())

# Mostrar primeras filas
print("\n📌 Primeras filas de X:")
print(X.head())

print("\n📌 Primeras filas de y:")
print(y.head())

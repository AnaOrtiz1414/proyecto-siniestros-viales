# 📄 src/analisis_causas.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar la base de datos
df = pd.read_csv("C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/data/Data_completa.csv")

# 2. Validar existencia de la columna
if 'causa' not in df.columns:
    raise ValueError("La columna 'causa' no existe en el archivo.")

# 3. Contar las causas más frecuentes
causas = df['causa'].value_counts().head(10)

# 4. Mostrar resultados en consola
print("\n🔍 Principales causas de accidentes:")
print(causas)

# 5. Visualización
plt.figure(figsize=(10, 6))
sns.barplot(x=causas.values, y=causas.index, palette="Reds_r")
plt.title("Top 10 causas más frecuentes de accidentes")
plt.xlabel("Número de casos")
plt.ylabel("Causa")
plt.tight_layout()
plt.show()

# 5. (Opcional) Exportar a CSV para usar en reporte o Power BI
# causas.to_csv("resultados/top_causas_accidentes.csv")

# print("\n✅ Análisis exportado a: resultados/top_causas_accidentes.csv")

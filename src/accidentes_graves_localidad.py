# 📄 src/accidentes_graves_localidad.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo de gráficos
sns.set(style="whitegrid")

# 1. Cargar los datos
df = pd.read_csv("C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/data/Data_completa.csv", encoding='latin1')

# 2. Filtrar siniestros graves
df_graves = df[df['gravedad'] == 'Con heridos']

# 3. Conteo por localidad
conteo_localidades = df_graves['localidad'].value_counts()

# 4. Localidad con más siniestros graves
top_localidad = conteo_localidades.idxmax()
total_localidad = conteo_localidades.max()

# 5. Gráfico de todas las localidades
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=conteo_localidades.values, y=conteo_localidades.index, palette="Blues_r")
plt.title("Número de siniestros graves por localidad")
plt.xlabel("Número de casos")
plt.ylabel("Localidad")

# Resaltar la barra de Kennedy si existe
for i, localidad in enumerate(conteo_localidades.index):
    if localidad.lower() == "kennedy":
        ax.patches[i].set_facecolor("red")
        ax.patches[i].set_edgecolor("black")

plt.tight_layout()
plt.show()

# 6. En Kennedy, ¿cuándo ocurren más?
graves_en_kennedy = df_graves[df_graves['localidad'].str.lower() == "kennedy"]
periodo_top = graves_en_kennedy['periodo_dia'].value_counts().idxmax()
total_periodo = graves_en_kennedy['periodo_dia'].value_counts().max()

# 7. Gráfico del periodo del día en Kennedy
plt.figure(figsize=(6, 4))
sns.countplot(data=graves_en_kennedy, x='periodo_dia', order=graves_en_kennedy['periodo_dia'].value_counts().index, palette="Reds_r")
plt.title("Distribución de siniestros graves por periodo del día en Kennedy")
plt.xlabel("Periodo del día")
plt.ylabel("Número de casos")
plt.tight_layout()
plt.show()

# 8. Imprimir resultados
print(f"📍 La localidad con más siniestros graves es **{top_localidad}** con {total_localidad} casos.")
print(f"🕒 En Kennedy, el periodo del día con más siniestros graves es **{periodo_top}** con {total_periodo} registros.")

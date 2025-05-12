# 04_comparacion_modelos.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar resultados desde la carpeta resultados/
carpeta = "C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/resultados"
archivos = [f for f in os.listdir(carpeta) if f.endswith(".json")]

registros = []
for archivo in archivos:
    with open(os.path.join(carpeta, archivo), "r") as f:
        data = json.load(f)
        registros.append({
            "Modelo": data["modelo"].replace("_model.h5", "").upper(),
            "Accuracy": round(data["accuracy"], 3),
            "Macro_F1": round(data["macro_f1"], 3),
            "AUC_Leve": round(data["auc_leve"], 2),
            "AUC_Grave": round(data["auc_grave"], 2),
            "AUC_Fatal": round(data["auc_fatal"], 2),
            "AUC_Macro": round(data["auc_macro"], 2)
        })

resultados = pd.DataFrame(registros).sort_values(by="Macro_F1", ascending=False)

# 2. Mostrar tabla comparativa
print("\n🔍 Comparación automática de modelos:\n")
print(resultados)

# 3. Gráfica F1 macro
plt.figure(figsize=(6, 4))
sns.barplot(data=resultados, x="Modelo", y="Macro_F1", palette="Greens")
plt.title("F1-Score Macro por Modelo")
plt.ylim(0, 1)
plt.ylabel("F1-Score Macro")
plt.tight_layout()
plt.show()

# 4. AUC por clase
auc_melted = resultados.melt(id_vars="Modelo", value_vars=["AUC_Leve", "AUC_Grave", "AUC_Fatal"],
                             var_name="Clase", value_name="AUC")
auc_melted["Clase"] = auc_melted["Clase"].str.replace("AUC_", "")

plt.figure(figsize=(8, 5))
sns.barplot(data=auc_melted, x="Clase", y="AUC", hue="Modelo")
plt.title("AUC por Clase y Modelo")
plt.ylim(0.6, 1.0)
plt.ylabel("AUC")
plt.tight_layout()
plt.show()

# 5. Conclusión automática
print("\n✅ Comparación completada.")

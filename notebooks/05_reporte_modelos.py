import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# === Cargar reportes
model_paths = {
    "MLP Clásico": "C:/ProyectoSiniestros_Limpio/resultados/mlp_report_smote.json",
    "MLP Embeddings": "C:/ProyectoSiniestros_Limpio/resultados/mlp_embeddings_report_smote.json",
    "LSTM": "C:/ProyectoSiniestros_Limpio/resultados/lstm_report_smote.json"
}

reportes = {}
for modelo, path in model_paths.items():
    with open(path, "r") as f:
        reportes[modelo] = json.load(f)

# === Comparar métricas por clase
clases = ["0", "1", "2"]
metricas = ["precision", "recall", "f1-score"]

for metrica in metricas:
    plt.figure(figsize=(8, 5))
    for modelo, datos in reportes.items():
        valores = [datos["classification_report"][cl][metrica] for cl in clases]
        plt.plot(clases, valores, marker='o', label=modelo)
    plt.title(f"{metrica.capitalize()} por Clase")
    plt.xlabel("Clase (0=Leve, 1=Herido, 2=Fatal)")
    plt.ylabel(metrica.capitalize())
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_{metrica}.png")
    plt.close()

# === Comparar Accuracy
plt.figure(figsize=(6, 4))
modelos = []
accuracies = []
for modelo, datos in reportes.items():
    modelos.append(modelo)
    accuracies.append(datos["classification_report"]["accuracy"])

plt.bar(modelos, accuracies, color=["#4B9CD3", "#63C5DA", "#9C89B8"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparación de Accuracy Global")
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_accuracy.png")
plt.close()

# === Comparar Matrices de Confusión
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (modelo, datos) in enumerate(reportes.items()):
    matriz = np.array(datos["confusion_matrix"])
    sns.heatmap(matriz, annot=True, fmt='d', cmap="Blues", ax=axes[idx],
                xticklabels=["Leve", "Herido", "Fatal"],
                yticklabels=["Leve", "Herido", "Fatal"])
    axes[idx].set_title(modelo)
    axes[idx].set_xlabel("Predicción")
    axes[idx].set_ylabel("Real")

plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_matrices_confusion.png")
plt.close()

print("Comparación gráfica completada. Archivos en 'resultados/graficos/'.")

import json
import matplotlib.pyplot as plt
import numpy as np

# === 1. Cargar archivos .json
paths = {
    "MLP Clásico": "C:/ProyectoSiniestros_Limpio/resultados/mlp_report_smote.json",
    "MLP Embeddings": "C:/ProyectoSiniestros_Limpio/resultados/mlp_embeddings_report_smote.json",
    "LSTM": "C:/ProyectoSiniestros_Limpio/resultados/lstm_report_smote.json"
}

reports = {}
for modelo, path in paths.items():
    with open(path, "r") as f:
        reports[modelo] = json.load(f)

# === 2. Métricas por clase
clases = ["0", "1", "2"]
metricas = ["precision", "recall", "f1-score"]

for metrica in metricas:
    plt.figure(figsize=(8, 5))
    for modelo, data in reports.items():
        valores = [data["classification_report"][cl][metrica] for cl in clases]
        plt.plot(clases, valores, marker='o', label=modelo)
    plt.title(f"{metrica.capitalize()} por clase")
    plt.xlabel("Clase (0=Leve, 1=Herido, 2=Fatal)")
    plt.ylabel(metrica.capitalize())
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_{metrica}.png")
    plt.close()

# === 3. Accuracy global
plt.figure(figsize=(6, 4))
modelos = []
accuracies = []
for modelo, data in reports.items():
    modelos.append(modelo)
    acc = data["classification_report"]["accuracy"]
    accuracies.append(acc)

plt.bar(modelos, accuracies, color=["#4B9CD3", "#63C5DA", "#9C89B8"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparación de Accuracy Global")
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_accuracy.png")
plt.close()

print("Comparación completa. Imágenes en resultados/graficos/")


import seaborn as sns

# === 4. Comparar matrices de confusión lado a lado
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_names = ["MLP Clásico", "MLP Embeddings", "LSTM"]

for i, modelo in enumerate(model_names):
    cm = np.array(reports[modelo]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=axes[i],
                xticklabels=["Leve", "Herido", "Fatal"],
                yticklabels=["Leve", "Herido", "Fatal"])
    axes[i].set_title(modelo)
    axes[i].set_xlabel("Predicción")
    axes[i].set_ylabel("Real")

plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/comparacion_confusion_matrices.png")
plt.close()

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# === 1. Cargar modelo y datos de validación
model = load_model("C:/ProyectoSiniestros_Limpio/models/mlp_model_embeddings_smote.keras")
data = np.load("C:/ProyectoSiniestros_Limpio/models/mlp_embeddings_val_data.npz", allow_pickle=True)

X_val_inputs = data["X_val"]  # list de arrays, uno por feature
y_val_cat = data["y_val"]
y_true = np.argmax(y_val_cat, axis=1)

# === 2. Predicción
y_pred = model.predict(X_val_inputs)
y_pred_classes = np.argmax(y_pred, axis=1)

# === 3. Métricas
report = classification_report(y_true, y_pred_classes, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# === 4. Guardar métricas
os.makedirs("resultados", exist_ok=True)
with open("C:/ProyectoSiniestros_Limpio/resultados/mlp_embeddings_report_smote.json", "w") as f:
    json.dump({
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }, f, indent=4)

# === 5. Matriz de confusión
os.makedirs("C:/ProyectoSiniestros_Limpio/resultados/graficos", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=["Leve", "Herido", "Fatal"],
            yticklabels=["Leve", "Herido", "Fatal"])
plt.title("Matriz de confusión - MLP Embeddings")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/mlp_embeddings_confusion_matrix.png")
plt.close()

# === 6. ROC y AUC por clase
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curvas ROC - MLP Embeddings' )
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/mlp_embeddings_roc_auc.png")
plt.close()

print("Evaluación completa. Resultados en carpeta resultados/")

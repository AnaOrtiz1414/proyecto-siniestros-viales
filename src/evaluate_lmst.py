import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# === 1. Cargar modelo y datos
model = load_model("C:/ProyectoSiniestros_Limpio/models/lstm_model_smote.h5")
data = np.load("models/lstm_val_data.npz", allow_pickle=True)
X_val = data["X_val"]
y_val_cat = data["y_val"]
y_true = np.argmax(y_val_cat, axis=1)

# === 2. Predicción
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# === 3. Reporte
report = classification_report(y_true, y_pred_classes, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# === 4. Guardar JSON
os.makedirs("resultados", exist_ok=True)
with open("C:/ProyectoSiniestros_Limpio/resultados/lstm_report_smote.json", "w") as f:
    json.dump({
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }, f, indent=4)

# === 5. Matriz de confusión
os.makedirs("resultados/graficos", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Leve", "Herido", "Fatal"],
            yticklabels=["Leve", "Herido", "Fatal"])
plt.title("Matriz de Confusión - LSTM")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/lstm_confusion_matrix.png")
plt.close()

# === 6. Curvas ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curvas ROC - LSTM')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/lstm_roc_auc.png")
plt.close()

print("Evaluación LSTM completa. Resultados en carpeta resultados/graficos/")

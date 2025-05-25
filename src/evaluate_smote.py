import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os

# === Cargar modelo y validación
model = load_model("C:/ProyectoSiniestros_Limpio/models/mlp_model_smote.h5")
data = np.load("C:/ProyectoSiniestros_Limpio/models/mlp_val_data.npz", allow_pickle=True)
X_val = data["X_val"]
y_val_cat = data["y_val"]
y_true = np.argmax(y_val_cat, axis=1)

# === Predicción
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# === Reporte y matriz
report = classification_report(y_true, y_pred_classes, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# === Crear carpetas si no existen
os.makedirs("resultados", exist_ok=True)
os.makedirs("C:/ProyectoSiniestros_Limpio/resultados/graficos", exist_ok=True)

# === Guardar JSON
with open("C:/ProyectoSiniestros_Limpio/resultados/mlp_report_smote.json", "w") as f:
    json.dump({
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }, f, indent=4)

# === Graficar matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Leve", "Herido", "Fatal"], yticklabels=["Leve", "Herido", "Fatal"])
plt.title("Matriz de confusión - MLP")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

#plt.tight_layout()
#plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/mlp_confusion_matrix.png")
#plt.close()

# === ROC y AUC por clase
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# === Plot ROC
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curvas ROC - MLP')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
#plt.tight_layout()
#plt.savefig("C:/ProyectoSiniestros_Limpio/resultados/graficos/mlp_roc_auc.png")
#plt.close()

#print("Evaluación guardada con gráficos en carpeta resultados/graficos/")

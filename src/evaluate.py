import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from data_loader import cargar_datos
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

def evaluar_modelo(nombre_modelo):
    print(f"🔍 Evaluando modelo: {nombre_modelo}")

    # Cargar datos
    X, y, df, encoders = cargar_datos()
    cat_cols = [col for col in X.columns if col.endswith('_enc')]
    num_cols = ['anio', 'mes', 'dia_semana']
    X_cat = X[cat_cols].astype('int32')
    X_num = X[num_cols].astype('float32')

    # Dividir conjunto de prueba
    _, X_cat_test, _, X_num_test, _, y_test = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_inputs = [X_cat_test[col].values for col in cat_cols] + [X_num_test.values]

    # Cargar modelo
    model = load_model(f"../models/{nombre_modelo}")

    # Predicciones
    y_pred_probs = model.predict(X_test_inputs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Reporte de clasificación
    etiquetas = ['Leve', 'Grave', 'Fatal']
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    # Matriz de Confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

    # ROC y AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Gráfico ROC
    plt.figure(figsize=(8, 6))
    colores = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(3), colores):
        plt.plot(fpr[i], tpr[i], lw=2, color=color, label=f'Clase {etiquetas[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], linestyle=':', color='navy', lw=3, label=f'Macro AUC = {roc_auc["macro"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC por Clase')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Guardar resultados
    reporte = classification_report(y_test, y_pred, output_dict=True)
    resumen = {
        "modelo": nombre_modelo,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(reporte["accuracy"]),
        "macro_f1": float(reporte["macro avg"]["f1-score"]),
        "auc_leve": float(roc_auc[0]),
        "auc_grave": float(roc_auc[1]),
        "auc_fatal": float(roc_auc[2]),
        "auc_macro": float(roc_auc["macro"])
    }

    os.makedirs("C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/resultados", exist_ok=True)
    ruta = f"C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/resultados/{nombre_modelo.replace('.h5', '')}.json"
    with open(ruta, "w") as f:
        json.dump(resumen, f, indent=4)
    print(f"\n💾 Resultados guardados en: {ruta}")

# Uso por consola
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, required=True, help="Nombre del archivo .h5 en la carpeta models/")
    args = parser.parse_args()
    evaluar_modelo(args.modelo)

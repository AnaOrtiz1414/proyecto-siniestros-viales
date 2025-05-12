# 📊 Proyecto Deep Learning: Clasificación de Gravedad de Siniestros Viales en Bogotá

Este proyecto utiliza modelos de redes neuronales (MLP y LSTM) para predecir la gravedad de siniestros viales a partir de un conjunto de datos con múltiples 
características categóricas y numéricas.

## 📁 Estructura del Proyecto

```bash
Proyecto_siniestros/
├── data/                    # Archivos CSV (Data_completa.csv)
├── models/                 # Modelos entrenados (mlp_model.h5, lstm_model.h5)
├── notebooks/              # Notebook y scripts de análisis
│   └── 04_comparacion_modelos.py
├── resultados/             # Reportes .json con métricas de evaluación
├── src/                    # Código fuente
│   ├── data_loader.py      # Carga, limpieza e imputación de datos
│   ├── train.py            # Entrenamiento modelo MLP
│   ├── train_lstm.py       # Entrenamiento modelo LSTM
│   ├── evaluate.py         # Evaluación de modelos y generación de curvas ROC
│   ├── model_mlp_embeddings.py  # Versión alternativa del modelo MLP
│   └── analisis_causas.py       # Análisis de causas más frecuentes de accidentes
├── requirements.txt        # Aquí se encuentran todos los paquetes que son requeridos 
├── test_loader.py          # Código para validar que la data este bien cargada
├── .gitignore
└── README.md               # Este archivo
```

---

## ⚙️ Preparación de Datos

El archivo `src/data_loader.py`:
- Convierte fechas y extrae `anio`, `mes` y `día de la semana`.
- Imputa valores nulos en columnas categóricas como `clasevehiculo`, `servicio`, y `choque` (inicialmente incluida pero posteriormente descartada por bajo aporte).
- Codifica las variables categóricas mediante `LabelEncoder`.
- Mapea la variable objetivo `gravedad` a valores numéricos:  
  `"Solo danos" → 0`, `"Con heridos" → 1`, `"Con muertos" → 2`

---

## 🧠 Modelos Implementados

### 🔹 MLP (Multilayer Perceptron)
- Utiliza embeddings para variables categóricas.
- Entrada numérica concatenada con embeddings.
- Capas densas con Dropout para evitar overfitting.
- Entrenamiento con `EarlyStopping`.

Script: `src/train.py`

### 🔸 LSTM (Long Short-Term Memory)
- Embeddings concatenados como secuencia.
- Procesamiento secuencial con capa LSTM.
- Salida combinada con entradas numéricas.

Script: `src/train_lstm.py`

---

## 📊 Evaluación de Modelos

Script: `src/evaluate.py`
- Carga el modelo guardado (`.h5`).
- Evalúa sobre el conjunto de prueba (`accuracy`, `precision`, `recall`, `f1-score`).
- Genera curva ROC por clase.
- Guarda reporte `.json` con métricas clave para posterior comparación.

---

## 🔍 Comparación de Modelos

Script: `notebooks/04_comparacion_modelos.py`
- Lee archivos `.json` desde la carpeta `resultados/`.
- Construye tabla comparativa.
- Muestra gráficos de:
  - F1 macro por modelo
  - AUC por clase

---

## 🧪 Otros Scripts Complementarios

- `src/model_mlp_embeddings.py`: modelo MLP con arquitectura alternativa basado en embeddings.
- `src/analisis_causas.py`: genera visualización con las causas más frecuentes de siniestros según los registros históricos.

---

## 📝 Resultados

| Modelo | Accuracy | Macro F1 | AUC Macro | Clase Grave AUC |
|--------|----------|----------|-----------|-----------------|
| MLP    | 0.807    | 0.578    | 0.83      | 0.82            |
| LSTM   | 0.794    | 0.590    | 0.84      | 0.81            |

- **LSTM mejora ligeramente la capacidad de detectar casos graves y fatales.**
- Ambos modelos logran buen desempeño general y mejores resultados sin la variable `choque`.

---

## 🚀 Ejecución

```bash
# Entrenar MLP
python src/train.py

# Entrenar LSTM
python src/train_lstm.py

# Evaluar un modelo
python src/evaluate.py --modelo mlp_model.h5

# Comparar resultados
python notebooks/04_comparacion_modelos.py

# Ver causas frecuentes
python src/analisis_causas.py
```

---

## ✅ Recomendaciones

- Continuar explorando nuevas variables como `estado de la vía`, `tipo de vía`, o `condiciones climáticas` si están disponibles.
- Probar técnicas como `SMOTE` o `undersampling` para abordar el desbalance.
- Ajustar arquitectura y optimizadores para mejorar precisión en clase **Fatal**.

---

## 📌 Conclusiones

## 🧾 Conclusiones

- La limpieza de datos y la correcta imputación de variables categóricas fueron clave para lograr mejoras significativas en la precisión del modelo.
- Eliminar la variable `choque`, que inicialmente se asumía relevante, mejoró el rendimiento general del modelo MLP y permitió una mejor detección de las clases `Grave` y `Fatal`.
- El modelo LSTM mostró un mejor desempeño en métricas globales (F1 Macro y AUC Macro), lo que indica una mayor capacidad para generalizar sobre secuencias de entradas categóricas complejas.
- Ambos modelos alcanzaron una buena capacidad predictiva sobre la clase `Leve`, pero el LSTM destacó en su habilidad para identificar siniestros con mayor gravedad.
- Las curvas ROC por clase evidencian una separación clara de las predicciones para las tres categorías, con un AUC superior a 0.80 en todos los casos.
- El flujo modular de scripts (`data_loader`, `train`, `evaluate`, `comparacion_modelos`) permite escalar el proyecto fácilmente hacia otras regiones, segmentos o nuevas variables.

---

## 📌 Autores

Ana María Ortiz
Karen Velasquez
William Rubio
Alejandro Gomez
Guillermo Gomez


**Siniestros Viales Bogotá – Deep Learning aplicado**


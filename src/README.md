# 🧐 Explicación del Proyecto de Modelos de Deep Learning (MLP y LSTM)

Este documento explica de forma clara y sencilla cómo se desarrollaron y evaluaron los dos modelos de Deep Learning del proyecto: **MLP** y **LSTM**, con el fin de predecir la gravedad de un siniestro vial (leve, grave o fatal) en la ciudad de Bogotá.

---

## 🔄 1. Flujo general del proyecto

1. Se parte de dos bases de datos diferentes:
   - `Data_SinNull.csv`: versión completa, sin valores nulos.
   - `Data.csv`: versión más balanceada, con mejor distribución entre clases. (Contiene nulos)

2. Se elige `Data.csv` para el entrenamiento final porque mejora el rendimiento en clases minoritarias. Lo que se realiza es una imputación de datos, en donde: 
    - Servicio se dejó los faltante como "Desconocido".
    - Clase de vehiculo se imputó con la categoría más frecuente (moda)
3. Se procesan las variables categóricas usando codificación con **embeddings** mediante `LabelEncoder`.
4. Se dividen los datos en conjuntos de entrenamiento (70%), validación (20%) y prueba (10%).
5. Se entrenan dos modelos distintos:
   - Un modelo **MLP** (red neuronal multicapa)
   - Un modelo **LSTM** (red secuencial)
6. Se evalúa cada modelo con el conjunto de prueba y se comparan los resultados.

---

## 🔢 2. Variables utilizadas

- Variables **categóricas** codificadas con embeddings:
  - CLASE (tipo de accidente - se determinó Choque)
  - TIPO_SERVICIO
  - CLASE_VEHICULO
  - CAUSA_PROBABLE

- Variables **numéricas**:
  - AÑO
  - MES
  - DIA_SEMANA

La variable objetivo (`GRAVEDAD`) fue convertida a valores numéricos: 0 = Leve, 1 = Grave, 2 = Fatal.

---

## 🔧 3. Modelo MLP (Multilayer Perceptron) – Paso a paso

1. Se definen las dimensiones de embeddings para cada variable categórica.
2. Se codifican las variables categóricas con `LabelEncoder`.
3. Se construye el modelo:
   - Entradas de cada variable categórica → capa `Embedding` → `Flatten`
   - Se concatenan todas las salidas de embeddings con las variables numéricas
   - Se pasa por capas densas (`Dense`) con activación ReLU y `Dropout`
   - Salida con `softmax` para 3 clases
4. Se entrena el modelo con `sparse_categorical_crossentropy`, `Adam`, y pesos por clase.

**Ventajas:** Simple, rápido de entrenar.

**Limitación:** Menor sensibilidad a clases desbalanceadas como "Fatal".

---

## 🔧 4. Modelo LSTM (Long Short-Term Memory) – Paso a paso

1. Se usan los mismos embeddings por variable categórica.
2. En lugar de concatenar y aplanar directamente, se crea una "secuencia artificial":
   - Cada vector embedding se mantiene como (1, emb_dim)
   - Se concatena como una secuencia de pasos temporales (aunque no lo son en realidad)
3. Esta secuencia se pasa por una capa `LSTM` para aprender combinaciones complejas.
4. La salida del LSTM se concatena con las variables numéricas y pasa por capas densas.
5. Se entrena con los mismos parámetros y estructura de validación.

**Ventajas:** Mayor capacidad de generalización y sensibilidad a clases poco frecuentes.

**Limitación:** Requiere más tiempo de entrenamiento.

---

## 📊 5. Evaluación de modelos

Cada modelo se evaluó con:
- Accuracy (precisión global)
- F1-Score por clase y macro promedio
- AUC por clase (curvas ROC)

Los resultados fueron almacenados en `resultados/*.json` y comparados automáticamente con el script `04_comparacion_modelos.py`.

---

## 📅 6. Conclusión

- Ambos modelos pueden predecir la gravedad de un siniestro, pero **LSTM ofrece mejor equilibrio** entre clases.
- El MLP puede ser útil para sistemas en tiempo real por su velocidad.
- La combinación embeddings + Deep Learning fue efectiva para tratar datos categóricos.
- El uso de una versión más balanceada de la base (`Data.csv`) permitió mejorar el rendimiento global.




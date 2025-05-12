from data_loader import cargar_datos
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Cargar y preparar datos
X, y, df, encoders = cargar_datos()

# Variables categóricas y numéricas
cat_cols = [col for col in X.columns if col.endswith('_enc')]
num_cols = ['anio', 'mes', 'dia_semana']

X_cat = X[cat_cols].astype('int32')
X_num = X[num_cols].astype('float32')

# Separar en train / val / test
X_cat_temp, X_cat_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y
)
X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_cat_temp, X_num_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Inputs para Keras
X_train_inputs = [X_cat_train[col].values for col in cat_cols] + [X_num_train.values]
X_val_inputs = [X_cat_val[col].values for col in cat_cols] + [X_num_val.values]
X_test_inputs = [X_cat_test[col].values for col in cat_cols] + [X_num_test.values]

# Embeddings
embedding_sizes = []
for col in cat_cols:
    n_cat = df[col].nunique()
    emb_dim = min(50, (n_cat + 1) // 2)
    embedding_sizes.append((n_cat, emb_dim))

# Pesos por clase
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("📌 Pesos por clase aplicados:", class_weight_dict)

# Modelo LSTM corregido
def construir_modelo_lstm(embedding_sizes, num_numeric, num_classes):
    inputs = []
    embeddings = []

    for i, (n_cat, emb_dim) in enumerate(embedding_sizes):
        input_i = Input(shape=(1,), name=f'cat_input_{i}')
        emb_i = Embedding(input_dim=n_cat + 1, output_dim=emb_dim, name=f'emb_{i}')(input_i)
        emb_i = Flatten()(emb_i)
        inputs.append(input_i)
        embeddings.append(emb_i)

    # Concatenar todos los embeddings
    x_emb = Concatenate()(embeddings)
    x_emb = Reshape((1, -1))(x_emb)  # ahora es una secuencia de 1 paso

    x_lstm = LSTM(64, return_sequences=False)(x_emb)

    # Entrada numérica
    input_num = Input(shape=(num_numeric,), name='num_input')
    inputs.append(input_num)

    # Concatenar salida LSTM + numéricas
    x = Concatenate()([x_lstm, input_num])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Construir y entrenar el modelo
model = construir_modelo_lstm(
    embedding_sizes=embedding_sizes,
    num_numeric=X_num.shape[1],
    num_classes=3
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train_inputs, y_train,
    validation_data=(X_val_inputs, y_val),
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

model.save("C:/Users/ANA/Documents/Maestria/Clase Deeplearning/Proyecto_siniestros/models/lstm_model.h5")
print("✅ Modelo LSTM entrenado y guardado en models/lstm_model.h5")

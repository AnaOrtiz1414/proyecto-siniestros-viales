from data_loader_smote import cargar_datos
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# === 1. Cargar y dividir datos
X, y = cargar_datos()

# Convertir a enteros y luego a secuencia para LSTM
X_seq = X.astype(int).values  # shape: (n_samples, n_features)

# Ajustar largo a secuencia (padding por filas)
X_seq_padded = pad_sequences(X_seq, maxlen=X_seq.shape[1], padding='post')

X_train, X_val, y_train, y_val = train_test_split(
    X_seq_padded, y, test_size=0.2, stratify=y, random_state=42
)

# === 2. SMOTE solo a clase 2
smote = SMOTE(sampling_strategy={2: int(sum(y_train == 2) * 1.5)}, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === 3. One-hot encoding
y_train_cat = to_categorical(y_train_bal, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)

# === 4. Modelo LSTM
vocab_size = int(X_seq.max()) + 2  # tamaño del vocabulario total

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=X_train.shape[1]))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 5. Entrenamiento
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_train_bal, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# === 6. Guardar modelo y datos de validación
os.makedirs("models", exist_ok=True)
model.save("C:/ProyectoSiniestros_Limpio/models/lstm_model_smote.h5")
np.savez("C:/ProyectoSiniestros_Limpio/models/lstm_val_data.npz", X_val=X_val, y_val=y_val_cat)

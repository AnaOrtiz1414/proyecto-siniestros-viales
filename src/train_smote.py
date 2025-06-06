from data_loader_smote import cargar_datos
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import os

# === 1. Cargar y preparar datos
X, y = cargar_datos()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 2. Balancear solo clase 2 con SMOTE
smote = SMOTE(sampling_strategy={2: int(sum(y_train == 2) * 1.5)}, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === 3. One-hot encoding
y_train_cat = to_categorical(y_train_bal, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)

# === 4. MLP
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 5. Entrenar
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_train_bal, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=35,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# === 6. Guardar modelo y validación
os.makedirs("models", exist_ok=True)
model.save("C:/ProyectoSiniestros_Limpio/models/mlp_model_smote.h5")
np.savez("C:/ProyectoSiniestros_Limpio/models/mlp_val_data.npz", X_val=X_val, y_val=y_val_cat)

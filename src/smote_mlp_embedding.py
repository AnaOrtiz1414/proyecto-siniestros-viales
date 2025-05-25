import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from data_loader_smote import cargar_datos
import os

# === 1. Cargar datos
X, y = cargar_datos()
X = X.astype(int)  # asegurar enteros

# === 2. División y SMOTE (solo clase 2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(sampling_strategy={2: int(sum(y_train == 2) * 1.5)}, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

y_train_cat = to_categorical(y_train_bal, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)

# === 3. Calcular pesos por clase
weights = compute_class_weight('balanced', classes=np.unique(y_train_bal), y=y_train_bal)
class_weights = dict(enumerate(weights))

# === 4. Arquitectura con embeddings
inputs = []
embeddings = []

for i in range(X.shape[1]):
    input_i = Input(shape=(1,))
    vocab_size = int(X.iloc[:, i].max()) + 2
    embed_i = Embedding(input_dim=vocab_size, output_dim=8)(input_i)
    flat_i = Flatten()(embed_i)
    inputs.append(input_i)
    embeddings.append(flat_i)

x = Concatenate()(embeddings)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 5. Preparar inputs separados

X_train_inputs = [X_train_bal.iloc[:, i].values for i in range(X.shape[1])]
X_val_inputs = [X_val.iloc[:, i].values for i in range(X.shape[1])]


# === 6. Entrenamiento
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_train_inputs, y_train_cat,
    validation_data=(X_val_inputs, y_val_cat),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === 7. Guardado
os.makedirs("models", exist_ok=True)
model.save("C:/ProyectoSiniestros_Limpio/models/mlp_model_embeddings_smote.keras")
np.savez("C:/ProyectoSiniestros_Limpio/models/mlp_embeddings_val_data.npz", X_val=X_val_inputs, y_val=y_val_cat)

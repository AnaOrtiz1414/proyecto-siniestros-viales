import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

def construir_modelo_mlp(embedding_sizes, num_numeric, num_classes=3):
    """
    Crea un modelo MLP con embeddings para variables categóricas.

    embedding_sizes: lista de tuplas (num_categorias, dimension_embedding)
    num_numeric: número de variables numéricas
    num_classes: número de clases a predecir (default=3)

    return: modelo compilado
    """

    # Entradas categóricas
    inputs_cat = []
    embeddings = []
    for i, (n_cat, emb_dim) in enumerate(embedding_sizes):
        input_i = Input(shape=(1,), name=f'cat_input_{i}')
        embed_i = Embedding(input_dim=n_cat + 1, output_dim=emb_dim, name=f'embed_{i}')(input_i)
        embed_i = Flatten()(embed_i)
        inputs_cat.append(input_i)
        embeddings.append(embed_i)

    # Entrada numérica
    input_num = Input(shape=(num_numeric,), name='num_input')

    # Concatenar todo
    x = Concatenate()(embeddings + [input_num])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs_cat + [input_num], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
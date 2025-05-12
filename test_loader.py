from src.data_loader import cargar_datos

X, y, df, encoders = cargar_datos("data/Data_completa.csv")

print(X.shape)
print(y.value_counts())
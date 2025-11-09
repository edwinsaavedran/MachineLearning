# -----------------------------------------------------------------
# ARCHIVO: workflow_completo.py 
# -----------------------------------------------------------------
# 
# Demuestra el flujo de trabajo profesional de la Semana 13:
# 1. Cargar datos con Pandas
# 2. Entrenar un modelo
# 3. Guardar (Persistir) el modelo con Joblib
# 4. Cargar el modelo guardado para hacer predicciones
#
import pandas as pd
import joblib 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("--- Flujo de Trabajo Profesional (Pandas + Joblib) ---")

# --- PASO 1 y 2: Simular Carga de Datos desde un Archivo ---

print(f"\n[Paso 1] Cargando datos de Iris (X, y) como DataFrames...")
# Pylance sigue confundido con el tipo de 'y', así que usaremos un workaround.
X, y = load_iris(return_X_y=True, as_frame=True)

# --- GUARDAR LOS DATOS EN UN ARCHIVO CSV ---
# 1. Unimos X e y. Por defecto, 'y' se llama 'target'
data_to_save = X.join(y)

# 2. Renombramos la columna 'target' a 'species' usando .rename()
# Esto evita el error de Pylance en 'y.name ='
data_to_save = data_to_save.rename(columns={'target': 'species'})

# 3. Guardamos el DataFrame en un archivo CSV
data_to_save.to_csv('iris_dataset.csv', index=False)
print(f"   -> Datos guardados en 'iris_dataset.csv'")


# --- PASO 3: Cargar y Preparar Datos con Pandas ---
# Esta parte no cambia.
print(f"[Paso 2] Cargando datos desde 'iris_dataset.csv'...")
data = pd.read_csv('iris_dataset.csv')

# Separamos las variables (features) 'X' de la variable objetivo 'y'
X = data.drop('species', axis=1) 
y = data['species']


# --- PASO 4: Entrenar el Modelo ---
# Dividimos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"[Paso 3] Entrenando modelo de Regresión Logística...")
model = LogisticRegression(max_iter=200) 
model.fit(X_train, y_train)

# Evaluamos
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"   -> Precisión del modelo (Accuracy): {acc:.4f}")


# --- PASO 5: Guardar (Persistir) el Modelo ---
MODEL_FILENAME = 'modelo_iris_logistico.joblib'
print(f"[Paso 4] Guardando modelo entrenado en '{MODEL_FILENAME}'...")
joblib.dump(model, MODEL_FILENAME)


# --- PASO 6: Cargar y Usar el Modelo ---
print(f"[Paso 5] Cargando modelo desde '{MODEL_FILENAME}'...")
loaded_model = joblib.load(MODEL_FILENAME)

# Creemos una flor Iris nueva para clasificar:
nueva_flor_1_data = {
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)': 0.2
}
nueva_flor_2_data = {
    'sepal length (cm)': 6.7,
    'sepal width (cm)': 3.0,
    'petal length (cm)': 5.2,
    'petal width (cm)': 2.3
}
nueva_flor_1 = pd.DataFrame([nueva_flor_1_data])
nueva_flor_2 = pd.DataFrame([nueva_flor_2_data])


# Usamos el modelo CARGADO
prediccion_1 = loaded_model.predict(nueva_flor_1)
prediccion_2 = loaded_model.predict(nueva_flor_2)

# Mapeamos la predicción (0, 1, 2) al nombre de la especie
species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

print(f"\n--- Predicción en Producción ---")
print(f"   Datos de flor 1: {nueva_flor_1_data}")
print(f"   Predicción: Especie {prediccion_1[0]} ({species_names[prediccion_1[0]]})")
print(f"   Datos de flor 2: {nueva_flor_2_data}")
print(f"   Predicción: Especie {prediccion_2[0]} ({species_names[prediccion_2[0]]})")
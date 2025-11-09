# -----------------------------------------------------------------
# ARCHIVO: pipeline_workflow.py
# -----------------------------------------------------------------
# 
# Nivel 2: Flujo de Trabajo con Pipelines de Preprocesamiento
#
# 1. Carga datos de un CSV.
# 2. Define un Pipeline que incluye Escalado (StandardScaler) y un Modelo.
# 3. Entrena el Pipeline (automáticamente escala y entrena).
# 4. Guarda (Persiste) el Pipeline COMPLETO con Joblib.
# 5. Carga el Pipeline y lo usa para predecir sobre datos *crudos*.
#
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # ¡Nuevo! Para escalar datos
from sklearn.pipeline import Pipeline            # ¡Nuevo! Para encadenar pasos

print("--- Nivel 2: Flujo de Trabajo con Pipelines (Scaler + Modelo) ---")

# --- PASO 1: Cargar y Preparar Datos con Pandas ---
print(f"[Paso 1] Cargando datos desde 'iris_dataset.csv'...")
data = pd.read_csv('iris_dataset.csv')

X = data.drop('species', axis=1) 
y = data['species']

# Dividimos los datos ANTES de cualquier preprocesamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"   -> Datos divididos: {len(X_train)} para entrenar, {len(X_test)} para probar.")


# --- PASO 2: Definir el Pipeline ---
# Aquí está la magia. Creamos una lista de pasos (tuplas).
# ('nombre_del_paso', objeto_del_paso)
pipeline_steps = [
    ('scaler', StandardScaler()), # Paso 1: Escalar los datos
    ('model', LogisticRegression(max_iter=200)) # Paso 2: Entrenar el modelo
]

# Creamos el objeto Pipeline
pipeline = Pipeline(steps=pipeline_steps)
print("[Paso 2] Pipeline creado: StandardScaler -> LogisticRegression")


# --- PASO 3: Entrenar el Pipeline ---
# Solo llamamos .fit() UNA VEZ en el pipeline.
# Él se encarga de:
# 1. Aplicar .fit_transform() del scaler a X_train
# 2. Aplicar .fit() del modelo a los datos ya escalados
print("[Paso 3] Entrenando el Pipeline completo...")
pipeline.fit(X_train, y_train)


# --- PASO 4: Evaluar el Pipeline ---
# Al llamar a .predict() en el pipeline:
# 1. Aplica .transform() del scaler a X_test (¡sin re-entrenar!)
# 2. Aplica .predict() del modelo a los datos de prueba ya escalados
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"   -> Precisión del Pipeline (Accuracy): {acc:.4f}")


# --- PASO 5: Guardar (Persistir) el Pipeline COMPLETO ---
PIPELINE_FILENAME = 'pipeline_completo_iris.joblib'
print(f"[Paso 4] Guardando el Pipeline completo en '{PIPELINE_FILENAME}'...")
joblib.dump(pipeline, PIPELINE_FILENAME)


# --- PASO 6: Cargar y Usar el Pipeline ---
print(f"[Paso 5] Cargando Pipeline desde '{PIPELINE_FILENAME}'...")
loaded_pipeline = joblib.load(PIPELINE_FILENAME)

# Creemos las mismas flores de antes
# Nota: Son datos "crudos", sin escalar.
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

# El pipeline cargado se encarga de TODO:
# 1. Escala los datos de nueva_flor_1
# 2. Predice sobre esos datos escalados
prediccion_1 = loaded_pipeline.predict(nueva_flor_1)
prediccion_2 = loaded_pipeline.predict(nueva_flor_2)

# Mapeamos la predicción (0, 1, 2) al nombre de la especie
species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

print(f"\n--- Predicción en Producción (con Pipeline) ---")
print(f"   Datos crudos flor 1: {nueva_flor_1_data}")
print(f"   Predicción: Especie {prediccion_1[0]} ({species_names[prediccion_1[0]]})")
print(f"   Datos crudos flor 2: {nueva_flor_2_data}")
print(f"   Predicción: Especie {prediccion_2[0]} ({species_names[prediccion_2[0]]})")

print("\n¡Flujo 'Nivel 2' completado! El pipeline maneja el preprocesamiento automáticamente.")
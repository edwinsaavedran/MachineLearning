# -----------------------------------------------------------------
# ARCHIVO: Semana14_ML_Clasificacion/benchmark_clasificadores.py
# -----------------------------------------------------------------
#
# Implementa y compara los clasificadores de la Semana 14:
# - K-Vecinos Cercanos (KNN)
# - Árboles de Decisión (DT)
# - Máquinas de Soporte Vectorial (SVM)
#
# Utiliza un Pipeline completo con ColumnTransformer para el preprocesamiento
# basado en el EDA.
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC # Support Vector Classifier

print("--- Iniciando Benchmark de Clasificadores (Semana 14) ---")

# --- 1. Cargar y Limpiar Datos (basado en EDA) ---
DATA_PATH = '../data/telco_churn.csv'
df = pd.read_csv(DATA_PATH)

# Convertir TotalCharges a numérico (como en el EDA)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convertir Churn a 0/1
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Separar X (features) de y (target)
X = df.drop(['customerID', 'Churn'], axis=1) # customerID no es un feature
y = df['Churn']

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 2. Definir Columnas para Preprocesamiento ---
# Identificar columnas numéricas y categóricas (¡clave!)

# Numéricas: aquellas que necesitan escalado e imputación
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Categóricas: aquellas que necesitan codificación (One-Hot Encoding)
# Tomamos todas las que son 'object'
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Features numéricos: {list(numeric_features)}")
print(f"Features categóricos: {list(categorical_features)}")


# --- 3. Crear el Pipeline de Preprocesamiento ---

# Pipeline para datos numéricos:
# 1. Imputar: Rellenar NaN (que encontramos en TotalCharges) con la media.
# 2. Escalar: Estandarizar los datos.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline para datos categóricos:
# 1. Imputar: Rellenar NaN (si los hubiera) con la categoría 'missing'.
# 2. Codificar: Aplicar One-Hot Encoding (ignora categorías desconocidas)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Usar ColumnTransformer para aplicar los pipelines a las columnas correctas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# --- 4. Definir Modelos y Benchmark ---
print("\n--- ¡Iniciando Entrenamiento! ---")

# Diccionario de modelos que queremos probar
models = {
    "K-Vecinos Cercanos (KNN)": KNeighborsClassifier(n_neighbors=7),
    "Árbol de Decisión (DT)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Máquina de Soporte Vectorial (SVM)": SVC(kernel='linear', C=1, random_state=42)
}

# Iterar y entrenar cada modelo
for name, model in models.items():
    
    print(f"\nEntrenando: {name}")
    
    # Crear el Pipeline final: [PASO 1: Preprocesar] -> [PASO 2: Modelo]
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Entrenar el pipeline completo
    clf_pipeline.fit(X_train, y_train)
    
    # Predecir en los datos de prueba
    y_pred = clf_pipeline.predict(X_test)
    
    # Mostrar reporte de clasificación
    print(f"--- Reporte para: {name} ---")
    # 'target_names' nos da etiquetas 'No' y 'Yes' en el reporte
    print(classification_report(y_test, y_pred, target_names=['No (Churn=0)', 'Yes (Churn=1)']))
    print("-" * 50)

print("--- Benchmark Completado ---")
# -----------------------------------------------------------------
# ARCHIVO: Semana14_ML_Clasificacion/boosting_model.py
# -----------------------------------------------------------------
#
# Implementa un modelo de Gradient Boosting (XGBoost).
# Este tipo de modelo (Boosting) suele ser el de mayor rendimiento
# para datos tabulares como el dataset de Churn.
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Preprocesamiento (reutilizamos la lógica)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modelo
from xgboost import XGBClassifier # ¡Nuevo! El modelo de Boosting

print("--- Iniciando Modelo de Boosting (Semana 14) ---")

# --- 1. Cargar y Limpiar Datos (basado en EDA) ---
DATA_PATH = '../data/Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 2. Definir Columnas para Preprocesamiento ---
# (Idéntico al script de benchmark)
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Features numéricos: {list(numeric_features)}")
print(f"Features categóricos: {list(categorical_features)}")


# --- 3. Crear el Pipeline de Preprocesamiento ---
# (Idéntico al script de benchmark)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# --- 4. Definir y Entrenar el Pipeline de Boosting ---
print("\n--- ¡Iniciando Entrenamiento de XGBoost! ---")

# Parámetros comunes para XGBoost:
# n_estimators: Número de árboles a construir.
# learning_rate: Qué tanto corrige cada árbol al anterior (un valor bajo es mejor).
# max_depth: Profundidad de cada árbol (para evitar sobreajuste).
# use_label_encoder=False y eval_metric='logloss': Para evitar warnings comunes.
model_xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Crear el Pipeline final: [PASO 1: Preprocesar] -> [PASO 2: XGBoost]
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_xgb)
])

# Entrenar el pipeline
xgb_pipeline.fit(X_train, y_train)

# --- 5. Evaluar el Modelo ---
y_pred = xgb_pipeline.predict(X_test)

print(f"--- Reporte para: XGBoost ---")
print(classification_report(y_test, y_pred, target_names=['No (Churn=0)', 'Yes (Churn=1)']))
print("-" * 50)

# --- 6. (Opcional) Importancia de Features ---
# Una gran ventaja de los modelos de árbol es que nos dicen qué features
# fueron más importantes para tomar la decisión.

print("\n--- Importancia de Features (Top 10) ---")

# 1. Obtener el modelo entrenado (XGBoost) desde el pipeline
trained_model = xgb_pipeline.named_steps['classifier']

# 2. Obtener los nombres de todos los features (numéricos + categóricos)
# (Esto es un poco avanzado, pero muy útil)
cat_features_out = xgb_pipeline.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_features)
    
all_feature_names = list(numeric_features) + list(cat_features_out)

# 3. Crear un DataFrame de importancias
importances = pd.DataFrame(
    data=trained_model.feature_importances_,
    index=all_feature_names,
    columns=['Importancia']
).sort_values(by='Importancia', ascending=False)

print(importances.head(10))
print("-" * 50)
print("--- Script de Boosting Completado ---")
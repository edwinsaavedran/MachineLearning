# -----------------------------------------------------------------
# ARCHIVO: visual_pipeline_workflow.py
# -----------------------------------------------------------------
# 
# Nivel 3: Visualización del Proceso del Pipeline
#
# 1. Genera un gráfico "Antes y Después" del StandardScaler.
# 2. Genera un gráfico de las "Fronteras de Decisión" del modelo.
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

print("--- Nivel 3: Visualización del Pipeline ---")

# --- PASO 1: Cargar y Preparar Datos (Simplificado a 2 Features) ---
print("[Paso 1] Cargando datos (solo 2 features) para visualización...")
X, y = load_iris(return_X_y=True, as_frame=True)

# ¡Simplificación! Usaremos solo 'petal length' y 'petal width'
X = X[['petal length (cm)', 'petal width (cm)']]
y.name = 'species'

# Renombramos columnas para que sean más cortas en los gráficos
X.columns = ['Petal Length', 'Petal Width']

# Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- PASO 2: Visual 1 - El Efecto del StandardScaler ---
print("[Paso 2] Generando 'visual_1_scaling.png'...")

# Creamos un scaler y lo entrenamos (fit) solo con los datos de train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Creamos una figura con 2 subplots (lado a lado)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: ANTES de Escalar
ax1.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
ax1.set_title(f'Antes de Escalar\nMedia (L: {X_train.iloc[:, 0].mean():.2f}, A: {X_train.iloc[:, 1].mean():.2f})')
ax1.set_xlabel(X_train.columns[0])
ax1.set_ylabel(X_train.columns[1])
ax1.grid(True)

# Plot 2: DESPUÉS de Escalar
ax2.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
ax2.set_title(f'Después de Escalar (StandardScaler)\nMedia (L: {X_train_scaled[:, 0].mean():.2f}, A: {X_train_scaled[:, 1].mean():.2f})')
ax2.set_xlabel('Petal Length (Estandarizado)')
ax2.set_ylabel('Petal Width (Estandarizado)')
ax2.grid(True)
ax2.axhline(0, color='grey', linestyle='--') # Línea en media 0
ax2.axvline(0, color='grey', linestyle='--') # Línea en media 0

fig.suptitle('Visual 1: El Proceso de Preprocesamiento (StandardScaler)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('visual_1_scaling.png')
print("   -> 'visual_1_scaling.png' guardado.")
plt.clf() # Limpiar la figura

# --- PASO 3: Definir y Entrenar el Pipeline ---
print("[Paso 3] Entrenando el Pipeline completo (Scaler + Modelo)...")
# Usaremos los datos crudos (sin escalar) X_train
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# --- PASO 4: Visual 2 - La Frontera de Decisión ---
print("[Paso 4] Generando 'visual_2_decision_boundary.png'...")

# Primero, escalamos nuestros datos de TEST para el plot
# Usamos el scaler que está DENTRO del pipeline entrenado
X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

# Crear un "meshgrid": una grilla fina de puntos para colorear el fondo
# Usamos los datos escalados para definir los límites del gráfico
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Usamos el MODELO (paso 'model') para predecir cada punto de la grilla
# El modelo espera datos escalados, por eso le pasamos la grilla (xx, yy)
Z = pipeline.named_steps['model'].predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Crear el plot
plt.figure(figsize=(10, 7))
cmap_light = ListedColormap(['#FFDDDD', '#DDFFDD', '#DDDDFF']) # Colores de fondo
cmap_bold = 'viridis' # Colores de los puntos

# Dibujar el fondo coloreado (la decisión del modelo)
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Dibujar los puntos de prueba (escalados) encima
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=cmap_bold,
            edgecolor='k', s=60, label='Datos de Prueba (Test)')

plt.title('Visual 2: Frontera de Decisión del Pipeline (LogisticRegression)')
plt.xlabel('Petal Length (Estandarizado)')
plt.ylabel('Petal Width (Estandarizado)')
plt.legend()
plt.grid(True)
plt.savefig('visual_2_decision_boundary.png')
print("   -> 'visual_2_decision_boundary.png' guardado.")

print("\n--- ¡Visualización completada! ---")
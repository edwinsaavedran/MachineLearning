# -----------------------------------------------------------------
# ARCHIVO: regresion_logistica.py
# -----------------------------------------------------------------
# 
# Implementa una Regresión Logística para CLASIFICACIÓN binaria.
# 
# Temario: Regresión Logística, Clasificación vs Regresión
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns # Para una mejor visualización de la matriz

print("\n--- 2. Modelo de Regresión Logística (Clasificar Aprobados) ---")

# 1. Preparar los Datos (Sintéticos)
# X = Horas de Estudio
# y = Aprobó (1) o Reprobó (0)
X = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0,   0, 0,   0, 1,   0, 1,   1, 1,   1, 1, 1, 1, 1])

# 2. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear y Entrenar el Modelo
# Instanciamos el modelo
modelo_logistico = LogisticRegression()

# Entrenamos
print(f"Entrenando modelo logístico con {len(X_train)} muestras...")
modelo_logistico.fit(X_train, y_train)

# 4. Evaluar el Modelo
y_pred = modelo_logistico.predict(X_test)

# La Métrica Clave: Matriz de Confusión
# [[Verdaderos Negativos (TN), Falsos Positivos (FP)]
#  [Falsos Negativos (FN),  Verdaderos Positivos (TP)]]
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Evaluación del Modelo ---")
print(f"Datos de Prueba (X_test): \n{X_test.T[0]}")
print(f"Predicciones (y_pred): \n{y_pred}")
print(f"Valores Reales (y_test): \n{y_test}")

print(f"\nAccuracy (Precisión Global): {acc:.4f}")
print("\nMatriz de Confusión:")
print(cm)

# 5. Usar el modelo (Predicción)
# ¿Alguien que estudió 2.8 horas aprueba?
# ¿Alguien que estudió 6.5 horas aprueba?
horas_1 = 2.8
horas_2 = 6.5
pred_1 = modelo_logistico.predict(np.array([[horas_1]]))
pred_2 = modelo_logistico.predict(np.array([[horas_2]]))

print(f"\nPredicción para {horas_1} horas: {'Aprueba' if pred_1[0] == 1 else 'Reprueba'}")
print(f"Predicción para {horas_2} horas: {'Aprueba' if pred_2[0] == 1 else 'Reprueba'}")

# 6. Visualización de la curva logística
plt.figure(figsize=(10, 6))
# Gráfica de la probabilidad (la curva 'S')
X_visual = np.linspace(0, 11, 100).reshape(-1, 1)
y_visual_prob = modelo_logistico.predict_proba(X_visual)[:, 1] # Probabilidad de ser '1'

plt.scatter(X, y, color='blue', label='Datos Reales (0=Reprueba, 1=Aprueba)')
plt.plot(X_visual, y_visual_prob, color='red', linewidth=2, label='Curva Logística (Probabilidad)')
plt.axhline(y=0.5, color='green', linestyle='--', label='Límite de Decisión (0.5)')
plt.title('Regresión Logística: Horas de Estudio vs. Aprobación')
plt.xlabel('Horas de Estudio')
plt.ylabel('Probabilidad de Aprobar')
plt.legend()
plt.grid(True)
plt.savefig('regresion_logistica.png')
print("\n* Gráfico 'regresion_logistica.png' generado.")
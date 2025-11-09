# -----------------------------------------------------------------
# ARCHIVO: regresion_lineal.py
# -----------------------------------------------------------------
# 
# Implementa una Regresión Lineal Simple para predecir un valor numérico.
# 
# Temario: Regresión lineal
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- 1. Modelo de Regresión Lineal (Predecir Salario) ---")

# 1. Preparar los Datos (Sintéticos)
# X = Años de Experiencia (variable independiente)
# y = Salario (variable dependiente, lo que queremos predecir)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([30000, 35000, 45000, 50000, 65000, 70000, 80000, 85000, 95000, 110000])

# 2. Dividir los datos: Entrenamiento y Prueba (¡Práctica fundamental!)
# Usamos 80% para entrenar el modelo y 20% para probar qué tan bueno es.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear y Entrenar el Modelo
# Instanciamos el modelo
modelo_lineal = LinearRegression()

# Entrenamos el modelo con los datos de entrenamiento
print(f"Entrenando modelo lineal con {len(X_train)} muestras...")
modelo_lineal.fit(X_train, y_train)

# 4. Evaluar el Modelo
# Usamos el modelo para predecir sobre los datos de "prueba" (que nunca vio)
y_pred = modelo_lineal.predict(X_test)

# Comparamos las predicciones (y_pred) con los valores reales (y_test)
# R^2 (Coeficiente de Determinación):
#   - 1.0 = El modelo es perfecto.
#   - 0.0 = El modelo no explica nada.
#   - Cerca de 1.0 es bueno.
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n--- Evaluación del Modelo ---")
print(f"Datos de Prueba (X_test): \n{X_test}")
print(f"Predicciones (y_pred): \n{y_pred.round(0)}")
print(f"Valores Reales (y_test): \n{y_test}")
print(f"\nCoeficiente R-cuadrado (R^2): {r2:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

# 5. Usar el modelo (Predicción)
# ¿Cuánto ganaría alguien con 12 años de experiencia?
anios_exp = 12
prediccion_salario = modelo_lineal.predict(np.array([[anios_exp]]))
print(f"\nPredicción para {anios_exp} años de exp: ${prediccion_salario[0]:.2f}")

# 6. Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos Reales')
plt.plot(X, modelo_lineal.predict(X), color='red', linewidth=2, label='Línea de Regresión')
plt.title('Regresión Lineal: Años de Experiencia vs. Salario')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.legend()
plt.grid(True)
plt.savefig('regresion_lineal.png')
print("\n* Gráfico 'regresion_lineal.png' generado.")
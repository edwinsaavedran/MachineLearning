# -----------------------------------------------------------------
# ARCHIVO: regresion_no_lineal.py
# -----------------------------------------------------------------
# 
# Implementa una Regresión Polinomial (un tipo de Regresión No Lineal).
# 
# Temario: Regresión no lineal
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline # ¡Muy útil para esto!
from sklearn.metrics import r2_score

print("\n--- 3. Modelo de Regresión No Lineal (Polinomial) ---")

# 1. Preparar los Datos (Sintéticos, con una curva)
# Datos que siguen una función cuadrática (y = 0.5x^2 + x + 2 + ruido)
np.random.seed(42)
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 5, X.shape[0]).reshape(-1, 1)

# 2. Crear y Entrenar el Modelo (Lineal vs. Polinomial)

# Modelo Lineal simple (para comparar)
modelo_lineal = LinearRegression()
modelo_lineal.fit(X, y)
y_pred_lineal = modelo_lineal.predict(X)
r2_lineal = r2_score(y, y_pred_lineal)
print(f"R^2 (Lineal Simple): {r2_lineal:.4f} (¡Muy bajo!)")

# Modelo Polinomial (Grado 2, es decir, cuadrático)
# Usamos un 'Pipeline' para automatizar:
# 1. Transformar X a (X, X^2) -> PolynomialFeatures
# 2. Aplicar Regresión Lineal a esos nuevos features

grado_polinomio = 2
modelo_polinomial = Pipeline([
    ('poly', PolynomialFeatures(degree=grado_polinomio)),
    ('linear', LinearRegression())
])

modelo_polinomial.fit(X, y)
y_pred_polinomial = modelo_polinomial.predict(X)
r2_polinomial = r2_score(y, y_pred_polinomial)
print(f"R^2 (Polinomial Grado {grado_polinomio}): {r2_polinomial:.4f} (¡Mucho mejor!)")


# 3. Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos Reales (Curvos)', s=10)
plt.plot(X, y_pred_lineal, color='red', linewidth=2, label=f'Regresión Lineal (R2={r2_lineal:.2f})')
plt.plot(X, y_pred_polinomial, color='green', linewidth=2, label=f'Regresión Polinomial (R2={r2_polinomial:.2f})')
plt.title('Regresión Lineal vs. Polinomial')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.legend()
plt.grid(True)
plt.savefig('regresion_no_lineal.png')
print("\n* Gráfico 'regresion_no_lineal.png' generado.")
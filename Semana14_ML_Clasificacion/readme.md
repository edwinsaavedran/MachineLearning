# Semana 14: Clasificadores Avanzados

[cite_start]Esta semana implementa y compara los modelos de clasificación clásicos  (KNN, Árboles de Decisión y SVM) para resolver un problema de negocio real: la **predicción de abandono de clientes**.

## Archivos

### 1. `benchmark_clasificadores.py`
Este es el script principal. En lugar de crear un archivo por modelo, este script realiza un **benchmark** (comparativa) de los tres clasificadores usando un flujo de trabajo profesional.

**Flujo de Trabajo Implementado:**
1.  **Carga de Datos:** Carga el dataset `telco_churn.csv` de Kaggle.
2.  **Limpieza:** Aplica las correcciones identificadas en el EDA (ej. `TotalCharges` a numérico).
3.  **Definición de Pipelines:**
    * Crea un `Pipeline` para **features numéricos** (`SimpleImputer` + `StandardScaler`).
    * Crea un `Pipeline` para **features categóricos** (`SimpleImputer` + `OneHotEncoder`).
4.  **ColumnTransformer:** Utiliza `ColumnTransformer` para aplicar los pipelines correctos a las columnas correctas.
5.  **Benchmark:**
    * [cite_start]Define un diccionario de modelos (KNN, DT, SVM).
    * Itera sobre cada modelo, lo "enchufa" al final del `ColumnTransformer` y lo entrena.
    * Imprime el `classification_report` (Precision, Recall, F1-Score) para cada uno, permitiendo una comparación directa.

### 2. (Próximamente) `boosting.py`
[cite_start]El temario  también menciona **Boosting**. Este se implementará en un script separado, ya que los modelos de Boosting (como XGBoost) suelen ser el siguiente nivel de rendimiento.
# Semana 14: Clasificadores Avanzados

Esta carpeta contiene las implementaciones prácticas de los modelos de clasificación avanzados cubiertos en la **Semana 14** del sílabo:
* Máquinas de Soporte Vectorial (SVM)
* Vecinos más Cercanos (KNN)
* Árboles de Decisión (DT)
* Boosting

El objetivo es aplicar estos modelos a un problema de negocio real (predicción de abandono de clientes) y comparar su rendimiento.

## Archivos en este Directorio

### 1. `benchmark_clasificadores.py`
Este script realiza una **comparativa (benchmark)** de los tres clasificadores clásicos (KNN, SVM, DT).

**Flujo de Trabajo Implementado:**
1.  **Carga de Datos:** Carga el dataset `telco_churn.csv` desde la carpeta `/data`.
2.  **Limpieza:** Aplica las correcciones identificadas en el EDA (ej. `TotalCharges` a numérico, `Churn` a 0/1).
3.  **Pipeline de Preprocesamiento:**
    * Define un `Pipeline` para **features numéricos** (`SimpleImputer` + `StandardScaler`).
    * Define un `Pipeline` para **features categóricos** (`SimpleImputer` + `OneHotEncoder`).
    * Utiliza un `ColumnTransformer` para aplicar los pipelines correctos a las columnas correctas.
4.  **Benchmark:**
    * Define un diccionario de modelos (KNN, DecisionTree, SVC).
    * Itera sobre cada modelo, lo "enchufa" al final del `ColumnTransformer` en un `Pipeline` final.
    * Entrena y evalúa cada modelo, imprimiendo el `classification_report` (Precision, Recall, F1-Score) para una comparación directa.

### 2. `boosting_model.py`
Este script implementa el modelo de **Boosting** (específicamente, `XGBClassifier`), que representa el siguiente nivel de rendimiento.

**Flujo de Trabajo Implementado:**
1.  **Reutiliza el Pipeline:** Utiliza exactamente el mismo `ColumnTransformer` de preprocesamiento que el script de benchmark. Esto es **crucial** para una comparación justa.
2.  **Entrena XGBoost:** Define y entrena un `Pipeline` que combina el preprocesador con `XGBClassifier`.
3.  **Reporte Avanzado:**
    * Imprime el `classification_report` para comparar su rendimiento con los modelos clásicos (se espera que sea superior).
    * Extrae y muestra la **Importancia de Features** (`feature_importances_`) del modelo. Esto nos da un *insight* de negocio invaluable, mostrándonos *qué* variables (ej. "Contrato mensual") son las que más influyen en la predicción del abandono.

##  Conclusión de la Semana
Al ejecutar ambos scripts, no solo implementamos los algoritmos del sílabo, sino que creamos un sistema de evaluación que nos permite determinar objetivamente qué modelo es el mejor para nuestro problema de negocio.

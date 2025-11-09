# Proyecto: Implementación de IA y Machine Learning (Curso 100000TD09)

Este repositorio documenta el desarrollo práctico y técnico de los componentes de Inteligencia Artificial y Machine Learning del curso **Innovación y Transformación Digital (100000TD09)**.

[cite_start]El proyecto sigue una progresión lógica, comenzando con los algoritmos fundamentales de la IA Clásica y evolucionando hacia *pipelines* de Machine Learning modernos y listos para la producción, aplicando las tecnologías clave para la transformación digital[cite: 32].

## Objetivo del Proyecto

El objetivo es traducir la teoría del sílabo en implementaciones prácticas y robustas. Buscamos construir un portafolio de soluciones de datos que demuestren la capacidad de:
1.  **Resolver problemas** de optimización (IA Clásica).
2.  **Predecir resultados** (Aprendizaje Supervisado).
3.  **Descubrir patrones** ocultos (Aprendizaje No Supervisado).
4.  **Construir flujos de trabajo** profesionales (Pipelines, Preprocesamiento).

---

## Estructura del Repositorio

El proyecto está organizado en carpetas semanales que corresponden a los temas del sílabo, construyendo incrementalmente sobre el conocimiento anterior.

* `/Semana11_12_IA_Clasica/`: Fundamentos de IA y algoritmos de búsqueda.
* `/Semana13_ML_Fundamentos/`: El salto a Machine Learning (Regresión y Workflows).
* `/Semana14_ML_Clasificacion/`: Modelos avanzados de clasificación.
* `/Semana15_17_ML_Clustering/`: Aprendizaje No Supervisado y Segmentación.

---

## El Viaje: De la IA Clásica al ML Avanzado

Abordamos los temas del curso como un GDataScience, construyendo una solución integral.

### 1. Fundamentos de IA Clásica (Semanas 11-12)
* [cite_start]**Temas:** Espacios de problemas, Razonamiento, Búsqueda[cite: 49].
* [cite_start]**Implementación:** Se implementan algoritmos de búsqueda no informada (BFS, DFS) e informada (Hill Climbing, A\*) para resolver problemas de optimización de rutas[cite: 50]. Aquí, le *decimos* a la máquina cómo "razonar".

### 2. Aprendizaje Supervisado: Fundamentos (Semana 13)
* [cite_start]**Temas:** Introducción a ML, Regresión Lineal, Regresión Logística[cite: 51].
* **Implementación:** Damos el salto al ML. Aquí, la máquina *aprende* de los datos. Construimos modelos para predecir valores numéricos y categorías.
* **Workflow Profesional:** Introducimos el flujo de trabajo estándar con **Pandas** (para ingesta de datos), **StandardScaler** (para preprocesamiento), **Pipelines** (para automatizar) y **Joblib** (para persistir modelos).

### 3. Aprendizaje Supervisado: Clasificadores Avanzados (Semana 14)
* [cite_start]**Temas:** Máquinas de Soporte Vectorial (SVM), K-Vecinos Cercanos (KNN), Árboles de Decisión (DT) y Boosting[cite: 53].
* **Implementación:** Usando un dataset real (ver sección de Dataset), comparamos el rendimiento de estos modelos avanzados para resolver un problema de negocio (predecir el abandono de clientes).

### 4. Aprendizaje No Supervisado: Clustering (Semanas 15-17)
* [cite_start]**Temas:** Introducción al Clustering, K-Means, Mean-Shift, DBSCAN, GMM, etc.[cite: 54, 55, 56].
* **Implementación:** Cambiamos de objetivo. Usando el mismo dataset, aplicamos técnicas de clustering para realizar **análisis de comportamiento** y encontrar segmentos de clientes ocultos (ej. "Clientes leales", "Clientes en riesgo").

---

## Dataset Principal: Kaggle - Telco Customer Churn

Para hacer este proyecto lo más realista posible, a partir de la Semana 14 utilizaremos un dataset público de Kaggle sobre abandono de clientes de una empresa de telecomunicaciones.

Este dataset es ideal porque nos permite:
1.  **Para Clasificación (Semana 14):** Predecir la columna `Churn` (Sí/No).
2.  **Para Clustering (Semana 15+):** Usar `tenure`, `MonthlyCharges` y otros features para segmentar clientes.

---

## Stack Tecnológico

Este proyecto utiliza el ecosistema PyData estándar para la ciencia de datos.

* **Lenguaje:** Python 3
* **Análisis de Datos:** Pandas, Numpy
* **Machine Learning:** Scikit-learn (sklearn)
* **Visualización:** Matplotlib, Seaborn
* **Persistencia:** Joblib

---

## Cómo Empezar

Para ejecutar este proyecto, se recomienda un entorno virtual.

1.  Clona este repositorio:
    ```bash
    git clone [URL_DE_TU_REPO_AQUI]
    cd MachineLearning
    ```

2.  Crea y activa un entorno virtual:
    ```bash
    python3 -m venv innovacion
    source innovacion/bin/activate
    ```

3.  Instala todas las dependencias:
    *(Asegúrate de mantener un archivo `requirements.txt` en esta carpeta raíz)*
    ```bash
    pip install -r requirements.txt
    ```

4.  Navega a la carpeta de la semana que deseas explorar:
    ```bash
    cd Semana13_ML_Fundamentos/
    python pipeline_workflow.py
    ```
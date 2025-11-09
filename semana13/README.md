# Proyecto Semana 13: Fundamentos de Machine Learning (Regresión)

Este directorio contiene las implementaciones prácticas correspondientes a la **Semana 13** del curso "Innovación y Transformación Digital" (100000TD09).

El objetivo de esta semana es realizar la transición de la IA Clásica (basada en reglas y búsqueda) al **Aprendizaje Automático (Machine Learning)**, donde los modelos "aprenden" patrones a partir de los datos.

## Contenido del Proyecto

Cubrimos los dos tipos principales de problemas de **Aprendizaje Supervisado**:

1.  **Regresión (Predecir un número):**
    * `regresion_lineal.py`: Implementa un modelo de Regresión Lineal simple para predecir un salario basado en años de experiencia.
    * `regresion_no_lineal.py`: Implementa una Regresión Polinomial para ajustar un modelo a datos que siguen un patrón curvo (no lineal).

2.  **Clasificación (Predecir una categoría):**
    * `regresion_logistica.py`: A pesar de su nombre, implementa un modelo de Regresión Logística para **clasificación** binaria (ej. predecir si un estudiante "Aprueba" o "Reprueba" basado en horas de estudio).

---

## Recomendaciones
# 1. Crear el Entorno Virtual
Ejecuta este comando. Usará el módulo venv de Python 3 para crear una nueva carpeta llamada innovacion que contendrá todos los archivos del entorno.

python3 -m venv innovacion

# 2. Activar el Entorno Virtual
source innovacion/bin/activate

Sabrás que funcionó porque tu prompt (el texto de la terminal) cambiará

Ese (innovacion) al inicio te confirma que el entorno virtual está activo. Todo lo que instales ahora se quedará dentro de esa carpeta.

# 3. Instalar las Dependencias (Dentro del Entorno)
Ahora que estás "dentro" de la burbuja, vuelve a ejecutar el comando de instalación. Esta vez, pip instalará los paquetes en la carpeta innovacion, no en tu sistema global.

pip install -r requirements.txt

## Pasos para la ejecución
# 1. Para ejecutar la Regresión Lineal:
python3 regresion_lineal.py

# 2. Para ejecutar la Regresión Logística (Clasificación):
python3 regresion_logistica.py

# 3. Para ejecutar la Regresión No Lineal (Polinomial):
python3 regresion_no_lineal.py

## Cómo Desactivar (Cuando Termines)
Cuando termines de trabajar en tu proyecto y quieras salir de la "burbuja" en tu terminal, simplemente escribe:
deactivate

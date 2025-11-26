# Semana 15-17: Aprendizaje No Supervisado (Clustering)

Este módulo contiene la suite completa de algoritmos de segmentación exigidos por el sílabo.

## Scripts Disponibles

### 1. `clustering_kmeans.py` (Semana 15)
* **Algoritmo:** K-Means.
* **Uso:** Segmentación estándar.
* **Clave:** Usa el "Método del Codo" para decidir cuántos grupos existen.
* **Salida:** Perfilado de clientes (ej. "Nuevos de Bajo Valor").

### 2. `clustering_dbscan.py` (Semana 16)
* **Algoritmo:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
* **Uso:** Detección de anomalías y grupos con formas irregulares.
* **Clave:** No requiere definir el número de clusters. Identifica **Ruido** (Cluster -1), útil para detectar fraude o datos erróneos.

### 3. `clustering_hierarchical.py` (Semana 17)
* **Algoritmo:** Clustering Jerárquico Aglomerativo.
* **Uso:** Análisis visual de la estructura de los datos.
* **Clave:** Genera un **Dendrograma** para visualizar cómo se agrupan los clientes desde el nivel individual hasta formar grandes grupos.

---

## Ejecución

Puedes ejecutar cada script secuencialmente para comparar los resultados:

```bash
# 1. Encontrar segmentos generales
python clustering_kmeans.py

# 2. Detectar anomalías/ruido
python clustering_dbscan.py

# 3. Ver la jerarquía visual
python clustering_hierarchical.py
# Semana 15-17: Aprendizaje No Supervisado (Clustering)

En este módulo cambiamos el enfoque de predecir etiquetas (Supervisado) a descubrir patrones ocultos en los datos (No Supervisado)

El objetivo es realizar una **Segmentación de Clientes** utilizando el dataset de Telco Churn, agrupando a los usuarios según su comportamiento de pago y antigüedad.

## Archivos

### 1. `clustering_kmeans.py` (Semana 15) 
Implementa el algoritmo **K-Means**, la técnica de clustering más popular.

**Flujo de Trabajo:**
1.  **Selección de Features:** Utilizamos `tenure`, `MonthlyCharges` y `TotalCharges`.
2.  **Escalado:** Aplicamos `StandardScaler` (obligatorio para K-Means debido a la distancia Euclideana).
3.  **Método del Codo:** Genera el gráfico `kmeans_elbow_plot.png` probando K=1 a 10, ayudándonos a decidir el número óptimo de grupos.
4.  **Modelado:** Aplica K-Means (con K=4 por defecto).
5.  **Interpretación:** Calcula el promedio de cada cluster para darle un "nombre" de negocio (ej. "Cliente Nuevo de Bajo Valor").
6.  **Visualización:** Genera `kmeans_clusters_visual.png` para ver la separación de grupos en 2D.

---

## Resultados Esperados

Al ejecutar el script, descubrirás 4 segmentos claros de clientes que suelen alinearse con:
* **Cluster A:** Nuevos / Bajo Gasto.
* **Cluster B:** Nuevos / Alto Gasto (Riesgo alto de churn).
* **Cluster C:** Antiguos / Bajo Gasto (Leales económicos).
* **Cluster D:** Antiguos / Alto Gasto (VIPs).
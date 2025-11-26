# -----------------------------------------------------------------
# ARCHIVO: Semana15_17_ML_Clustering/clustering_dbscan.py
# -----------------------------------------------------------------
#
# Implementación de DBSCAN (Density-Based Spatial Clustering).
# Semana 16 del Sílabo.
#
# Ventaja: Detecta 'Ruido' (Outliers) y no necesita que le digas K.
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

print("--- Semana 16: Clustering Basado en Densidad (DBSCAN) ---")

# --- 1. Carga y Preprocesamiento ---
DATA_PATH = '../data/Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)

# Limpieza
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Features (igual que K-Means para comparar)
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]

# Escalado (OBLIGATORIO para DBSCAN, es muy sensible a distancias)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Aplicar DBSCAN ---
# eps=0.5: Radio de vecindad (ajustado tras pruebas para este dataset estandarizado)
# min_samples=5: Necesito 5 vecinos para ser un "núcleo" de cluster
print("Ejecutando DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# DBSCAN devuelve -1 para el RUIDO (Outliers)
df['Cluster_DBSCAN'] = clusters

# Contamos cuántos grupos encontró y cuánto ruido hay
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"\nResultados de DBSCAN:")
print(f" -> Clusters encontrados: {n_clusters}")
print(f" -> Puntos de Ruido (Outliers): {n_noise} (Clientes atípicos)")


# --- 3. Visualización ---
plt.figure(figsize=(10, 7))
# Usamos una paleta donde el ruido (-1) destaque (ej. gris o rojo)
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', 
                hue='Cluster_DBSCAN', palette='deep', s=50)
plt.title(f'Segmentación DBSCAN (Ruido = -1)')
plt.xlabel('Antigüedad (Meses)')
plt.ylabel('Cargos Mensuales ($)')
plt.grid(True)
plt.savefig('dbscan_visual.png')
print("-> Gráfico 'dbscan_visual.png' generado.")

# --- 4. Análisis de Outliers ---
# Veamos quiénes son esos "ruidosos" (Cluster -1)
print("\n--- Análisis de Anomalías (Cluster -1) ---")
outliers = df[df['Cluster_DBSCAN'] == -1]
print(outliers[features].describe())
print("\nNota: Los outliers en DBSCAN suelen ser clientes con comportamientos extremos")
print("(ej. muy nuevos pagando mucho, o muy antiguos pagando muy poco).")
# -----------------------------------------------------------------
# ARCHIVO: Semana15_17_ML_Clustering/clustering_hierarchical.py
# -----------------------------------------------------------------
#
# Implementación de Clustering Jerárquico y Dendrogramas.
# Semana 17 del Sílabo.
#
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch # Librería para dibujar el dendrograma

print("--- Semana 17: Clustering Jerárquico y Dendrogramas ---")

# --- 1. Carga y Preprocesamiento ---
DATA_PATH = '../data/Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Escalado
X_scaled = StandardScaler().fit_transform(X)

# --- 2. Muestreo para Dendrograma ---
# Dibujar 7000 ramas es ilegible. Tomamos 100 clientes al azar para visualizar.
import numpy as np
np.random.seed(42)
indices = np.random.choice(X_scaled.shape[0], 100, replace=False)
X_sample = X_scaled[indices]

# --- 3. Visualizar Dendrograma ---
print("Generando Dendrograma (Muestra de 100 clientes)...")
plt.figure(figsize=(12, 6))
plt.title('Dendrograma de Clientes (Jerarquía)')
plt.xlabel('Clientes (Índices)')
plt.ylabel('Distancia Euclidiana')

# 'ward' es el método de enlace más común (minimiza varianza)
dendrogram = sch.dendrogram(sch.linkage(X_sample, method='ward'))
plt.axhline(y=5, color='r', linestyle='--') # Línea de corte sugerida
plt.savefig('hierarchical_dendrogram.png')
print("-> Gráfico 'hierarchical_dendrogram.png' generado.")


# --- 4. Aplicar Clustering Jerárquico (Dataset Completo) ---
# Basado en el dendrograma (o en nuestro K-Means anterior), elegimos 4 clusters
print("\nAplicando Agglomerative Clustering al dataset completo...")
hc = AgglomerativeClustering(n_clusters=4, linkage='ward') # 'affinity' fue renombrado o eliminado en versiones nuevas de sklearn, linkage='ward' usa euclideana por defecto
y_hc = hc.fit_predict(X_scaled)

df['Cluster_HC'] = y_hc

print("Asignación completada. Primeros 5 clientes:")
print(df[['tenure', 'MonthlyCharges', 'Cluster_HC']].head())
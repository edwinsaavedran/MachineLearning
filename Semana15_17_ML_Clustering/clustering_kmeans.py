# -----------------------------------------------------------------
# ARCHIVO: Semana15_17_ML_Clustering/clustering_kmeans.py
# -----------------------------------------------------------------
#
# Implementación de K-Means para Segmentación de Clientes.
# Incluye:
# 1. Método del Codo para encontrar K óptimo.
# 2. Visualización de Clusters.
# 3. Interpretación de Negocio de los segmentos.
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("--- Semana 15: Segmentación de Clientes con K-Means ---")

# --- 1. Carga y Preprocesamiento ---
DATA_PATH = '../data/Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)

# Limpieza básica (igual que semanas anteriores)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Selección de Features para Clustering
# Usaremos solo variables numéricas de comportamiento
features_to_cluster = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features_to_cluster]

print(f"Datos seleccionados para clustering: {X.shape}")

# --- 2. Escalado de Datos (CRUCIAL para K-Means) ---
# K-Means usa distancia Euclideana. Si una variable es 2000 y otra es 1,
# el modelo solo le hará caso a la de 2000. Debemos estandarizar.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 3. Método del Codo (Elbow Method) ---
# Buscamos el K óptimo probando del 1 al 10
print("\nCalculando inercia para el Método del Codo (esto puede tardar un poco)...")
inertia = []
K_range = range(1, 11)

for k in K_range:
    # n_init=10 es el estándar, corre el algoritmo 10 veces con diferentes semillas
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el Codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.title('Método del Codo para determinar K óptimo')
plt.grid(True)
plt.savefig('kmeans_elbow_plot.png')
print("-> Gráfico 'kmeans_elbow_plot.png' generado.")


# --- 4. Aplicar K-Means con K elegido ---
# Basado en la experiencia con este dataset, el codo suele estar en K=3 o K=4.
# Usaremos K=4 para buscar perfiles detallados.
optimal_k = 4
print(f"\nAplicando K-Means con K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Agregamos la etiqueta del cluster al DataFrame original (¡Importante!)
# para poder interpretar los resultados con los valores reales ($ y meses), no los escalados.
df['Cluster'] = clusters


# --- 5. Visualización de Clusters (2D) ---
# Graficamos Tenure vs MonthlyCharges coloreado por Cluster
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis', s=50)
plt.title(f'Segmentación de Clientes (K={optimal_k})')
plt.xlabel('Antigüedad (Meses)')
plt.ylabel('Cargos Mensuales ($)')
plt.legend(title='Cluster ID')
plt.grid(True)
plt.savefig('kmeans_clusters_visual.png')
print("-> Gráfico 'kmeans_clusters_visual.png' generado.")


# --- 6. Interpretación de Negocio (Perfilado) ---
# Agrupamos por cluster y calculamos la media de las variables clave
print("\n--- Perfil de los Segmentos (Promedios) ---")
cluster_summary = df.groupby('Cluster')[features_to_cluster].mean().reset_index()
print(cluster_summary)

# Análisis automático simple
print("\n--- Interpretación Rápida ---")
for index, row in cluster_summary.iterrows():
    cluster_id = int(row['Cluster'])
    tenure = row['tenure']
    spending = row['MonthlyCharges']
    
    desc = f"Cluster {cluster_id}: "
    if tenure < 20:
        desc += "Clientes NUEVOS "
    elif tenure > 50:
        desc += "Clientes VETERANOS "
    else:
        desc += "Clientes REGULARES "
        
    if spending < 40:
        desc += "de BAJO consumo."
    elif spending > 80:
        desc += "de ALTO consumo (Premium)."
    else:
        desc += "de consumo PROMEDIO."
        
    print(desc)
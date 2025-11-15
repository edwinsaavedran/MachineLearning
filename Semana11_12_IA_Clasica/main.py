# -----------------------------------------------------------------
# ARCHIVO: main.py
# -----------------------------------------------------------------
#
# Orquestador principal para ejecutar y comparar los algoritmos de búsqueda.
# Para ejecutar este proyecto, corre en tu terminal:
# python main.py
#

# 1. Importar el problema y los algoritmos
from graph import MAPA_RUTAS, HEURISTICA
from busqueda_no_informada import bfs, dfs
from busqueda_informada import hill_climbing, a_star

# 2. Definir el problema
START_NODE = "A"
GOAL_NODE = "G"


# Función auxiliar para calcular el costo de un camino
def calcular_costo(path, graph):
    cost = 0
    # Esta función SÓLO debe llamarse si path NO es None
    for i in range(len(path) - 1):
        nodo_actual = path[i]
        siguiente_nodo = path[i + 1]
        # Encontrar el costo en el grafo
        for vecino, costo_tramo in graph[nodo_actual]:
            if vecino == siguiente_nodo:
                cost += costo_tramo
                break
    return cost


# 3. Ejecutar y mostrar resultados
print("=" * 40)
print(f"Buscando ruta de {START_NODE} a {GOAL_NODE}...")
print("=" * 40)

# --- Búsquedas No Informadas (Semana 11) ---
print("\n--- 1. BÚSQUEDAS NO INFORMADAS ---")

# BFS
print(f"\nAlgoritmo: Búsqueda por Anchura (BFS)")
path_bfs = bfs(MAPA_RUTAS, START_NODE, GOAL_NODE)

# ¡Corrección! Comprobar si se encontró un camino
if path_bfs:
    cost_bfs = calcular_costo(path_bfs, MAPA_RUTAS)
    print(f"   Camino: {path_bfs}")
    print(f"   Costo:  {cost_bfs} KM")
    print(f"   Paradas: {len(path_bfs) - 1}")
    print("   *Análisis: Encuentra el camino con MENOS PARADAS.")
else:
    print("   Camino: No se encontró ruta.")

# DFS
print(f"\nAlgoritmo: Búsqueda por Profundidad (DFS)")
path_dfs = dfs(MAPA_RUTAS, START_NODE, GOAL_NODE)

# ¡Corrección! Comprobar si se encontró un camino
if path_dfs:
    cost_dfs = calcular_costo(path_dfs, MAPA_RUTAS)
    print(f"   Camino: {path_dfs}")
    print(f"   Costo:  {cost_dfs} KM")
    print(f"   Paradas: {len(path_dfs) - 1}")
    print("   *Análisis: Encuentra un camino rápido, pero no es el óptimo.")
else:
    print("   Camino: No se encontró ruta.")


# --- Búsquedas Informadas (Semana 12) ---
print("\n--- 2. BÚSQUEDAS INFORMADAS (usan heurística) ---")

# Hill Climbing
print(f"\nAlgoritmo: Hill Climbing (Voraz)")
path_hc = hill_climbing(MAPA_RUTAS, START_NODE, GOAL_NODE, HEURISTICA)

# ¡Corrección! Comprobar si se encontró un camino
if path_hc:
    cost_hc = calcular_costo(path_hc, MAPA_RUTAS)
    print(f"   Camino: {path_hc}")
    print(f"   Costo:  {cost_hc} KM")
    print(f"   Paradas: {len(path_hc) - 1}")
    print("   *Análisis: Es 'voraz'. Sigue el camino que 'parece' más cercano.")
else:
    print("   Camino: No se encontró ruta.")


# A* (A-Estrella)
print(f"\nAlgoritmo: A* (A-Estrella)")
# A* devuelve (path, cost) o (None, 0)
path_a_star, cost_a_star = a_star(MAPA_RUTAS, START_NODE, GOAL_NODE, HEURISTICA)

# ¡Corrección! Comprobar si se encontró un camino
if path_a_star:
    print(f"   Camino: {path_a_star}")
    print(f"   Costo:  {cost_a_star} KM")
    print(f"   Paradas: {len(path_a_star) - 1}")
    print("   *Análisis: Encuentra el camino ÓPTIMO (menor costo total).")
else:
    print("   Camino: No se encontró ruta.")

print("\n" + "=" * 40)
print("Conclusión de la Semana 11-12:")
print("BFS y DFS exploran 'a ciegas'.")
print("A* es 'inteligente', equilibra el costo real (g) y la estimación (h)")
print("para GARANTIZAR la ruta más corta (óptima).")
print("=" * 40)

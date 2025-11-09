# Semana 11 y 12.
Objetivo: aquí es entender cómo una IA "piensa" para encontrar una solución. No se trata de aprender (como en Machine Learning), sino de razonar y explorar sistemáticamente un mapa de posibilidades para encontrar la mejor ruta.

# Caso Ejemplo
Vamos a modelar un problema clásico de optimización de rutas. Olvídate de un laberinto, usemos un grafo de ciudades que es más flexible.
El Escenario: Imagina que eres una empresa de logística y necesitas encontrar la ruta más corta para un envío desde la ciudad 'A' hasta la ciudad 'G'.
El Modelo (Nuestro Espacio de Problemas):
Estados: Las ciudades (A, B, C, D, E, F, G).
Estado Inicial: 'A'
Estado Objetivo: 'G'
Acciones: Moverse de una ciudad a otra conectada.
Costos: Las distancias (en KM) entre ellas.


# Analisis de Resultados
# ========================================
Buscando ruta de A a G...
# ========================================

--- 1. BÚSQUEDAS NO INFORMADAS ---

Algoritmo: Búsqueda por Anchura (BFS)
   Camino: ['A', 'C', 'F', 'G']
   Costo:  13 KM
   Paradas: 3
   *Análisis: Encuentra el camino con MENOS PARADAS (A->C->F->G).

Algoritmo: Búsqueda por Profundidad (DFS)
   Camino: ['A', 'C', 'E', 'G']
   Costo:  25 KM
   *Análisis: Encuentra un camino rápido (A->C->E->G), pero no es el óptimo.

--- 2. BÚSQUEDAS INFORMADAS (usan heurística) ---

Algoritmo: Hill Climbing (Voraz)
   Camino: ['A', 'C', 'F', 'G']
   Costo:  13 KM
   *Análisis: Es 'voraz'. Elige A->C (h=4) sobre A->B (h=10).
              Sigue el camino que 'parece' más cercano (A->C->F->G).

Algoritmo: A* (A-Estrella)
   Camino: ['A', 'B', 'E', 'G']
   Costo:  22 KM
   *Análisis: Encuentra el camino ÓPTIMO (A->B->E->G) con 22 KM.
              Aunque A->C->F->G tiene menos paradas (13 KM), A*
              explora y encuentra la ruta de menor costo real.

# ========================================
Conclusión de la Semana 11-12:
BFS y DFS exploran 'a ciegas'.
A* es 'inteligente', equilibra el costo real (g) y la estimación (h)
para GARANTIZAR la ruta más corta (óptima).
# ========================================

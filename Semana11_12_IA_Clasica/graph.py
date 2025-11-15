# -----------------------------------------------------------------
# ARCHIVO: graph.py
# -----------------------------------------------------------------
# Define el espacio de problemas: el mapa de rutas y la heurística.

# El mapa (grafo) representado como un diccionario de adyacencia.
# Clave: Nodo (Ciudad)
# Valor: Lista de tuplas (Vecino, Costo/Distancia)
MAPA_RUTAS = {
    "A": [("B", 5), ("C", 6)],
    "B": [("A", 5), ("D", 8), ("E", 7)],
    "C": [("A", 6), ("E", 9), ("F", 4)],
    "D": [("B", 8)],
    "E": [("B", 7), ("C", 9), ("G", 10)],
    "F": [("C", 4), ("G", 3)],
    "G": [("E", 10), ("F", 3)],  # Nodo objetivo
}

# Definición de la Heurística (h(n))
# Es nuestra "estimación" de la distancia en línea recta desde cada nodo
# hasta el objetivo 'G'.
#
# Nota clave: La heurística para el nodo objetivo 'G' siempre debe ser 0.
# Estos valores son "inventados" pero consistentes (admisibles).
HEURISTICA = {
    "A": 13,
    "B": 10,
    "C": 4,
    "D": 15,
    "E": 3,
    "F": 2,
    "G": 0,  # El costo para llegar a G desde G es 0
}

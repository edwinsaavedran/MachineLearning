# -----------------------------------------------------------------
# ARCHIVO: busqueda_no_informada.py
# -----------------------------------------------------------------
# 
# Implementa algoritmos de búsqueda "a ciegas" (Semana 11).
# Estos algoritmos no usan heurística, solo el mapa.
#
from collections import deque

def bfs(graph, start, goal):
    """
    Búsqueda por Anchura (Breadth-First Search)
    Usa una Fila (Queue) para explorar por niveles.
    """
    # Fila (Queue): Almacena tuplas de (nodo_actual, camino_hasta_aqui)
    queue = deque([(start, [start])]) 
    
    # Set de nodos ya visitados para evitar bucles
    visited = set()

    while queue:
        # Saca el primer elemento de la fila
        (node, path) = queue.popleft() 

        if node not in visited:
            visited.add(node)

            if node == goal:
                # ¡Meta encontrada!
                return path
            
            # Explorar vecinos
            for neighbor, cost in graph[node]:
                if neighbor not in visited:
                    # Añade el vecino al final de la fila
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append((neighbor, new_path))
    
    # No se encontró un camino
    return None

def dfs(graph, start, goal):
    """
    Búsqueda por Profundidad (Depth-First Search)
    Usa una Pila (Stack) para explorar a fondo una rama primero.
    """
    # Pila (Stack): Almacena tuplas de (nodo_actual, camino_hasta_aqui)
    stack = [(start, [start])] 
    
    # Set de nodos ya visitados para evitar bucles
    visited = set()

    while stack:
        # Saca el último elemento de la pila (LIFO)
        (node, path) = stack.pop() 

        if node not in visited:
            visited.add(node)

            if node == goal:
                # ¡Meta encontrada!
                return path
            
            # Explorar vecinos
            for neighbor, cost in graph[node]:
                if neighbor not in visited:
                    # Añade el vecino a la pila
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append((neighbor, new_path))
    
    # No se encontró un camino
    return None
# -----------------------------------------------------------------
# ARCHIVO: busqueda_informada.py
# -----------------------------------------------------------------
# 
# Implementa algoritmos de búsqueda "informada" (Semana 12).
# Estos algoritmos usan la HEURISTICA para tomar decisiones.
#
import heapq # Para implementar la Fila de Prioridad (Priority Queue)

def hill_climbing(graph, start, goal, heuristic):
    """
    Algoritmo Hill Climbing (Búsqueda Voraz / Greedy Best-First)
    Prioridad = h(n)
    Siempre explora el nodo que *parece* estar más cerca de la meta.
    """
    # Fila de Prioridad: (prioridad, nodo_actual, camino_hasta_aqui)
    # La prioridad es h(n)
    p_queue = [(heuristic[start], start, [start])] 
    visited = set()

    while p_queue:
        # Saca el nodo con la MENOR prioridad (menor h(n))
        (h_cost, node, path) = heapq.heappop(p_queue)

        if node not in visited:
            visited.add(node)

            if node == goal:
                # ¡Meta encontrada!
                return path

            for neighbor, cost in graph[node]:
                if neighbor not in visited:
                    # La prioridad es solo la heurística del vecino
                    new_priority = heuristic[neighbor]
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(p_queue, (new_priority, neighbor, new_path))
    return None

def a_star(graph, start, goal, heuristic):
    """
    Algoritmo A* (A-Estrella)
    Prioridad = g(n) + h(n)
    Equilibra el costo real recorrido (g(n)) con la heurística (h(n)).
    Garantiza encontrar la ruta de menor costo.
    """
    # Fila de Prioridad: (prioridad, costo_g, nodo_actual, camino)
    # prioridad = f(n) = g(n) + h(n)
    # g(n) = costo real desde el inicio hasta el nodo actual
    
    # Empezamos con g(n) = 0 y f(n) = h(start)
    p_queue = [(0 + heuristic[start], 0, start, [start])] 
    visited = set()

    while p_queue:
        # Saca el nodo con la MENOR prioridad (menor f(n))
        (f_cost, g_cost, node, path) = heapq.heappop(p_queue)

        if node not in visited:
            visited.add(node)

            if node == goal:
                # ¡Meta encontrada! Retorna el camino y su costo total
                return path, g_cost

            for neighbor, cost in graph[node]:
                if neighbor not in visited:
                    # Calculamos los nuevos costos
                    new_g = g_cost + cost            # Costo real acumulado
                    new_h = heuristic[neighbor]      # Heurística del vecino
                    new_f = new_g + new_h            # Prioridad A*
                    
                    new_path = list(path)
                    new_path.append(neighbor)
                    
                    heapq.heappush(p_queue, (new_f, new_g, neighbor, new_path))
    
    # No se encontró camino
    return None, 0
"""
Implementación del Algoritmo Genético Clásico (GA) para el TSP.
"""

import random
import numpy as np


def calcular_distancia_tour(tour, matriz_distancias):
    """
    Calcula la distancia total de un tour (ruta).
    
    Args:
        tour: Lista de índices de ciudades que representan el orden del tour
        matriz_distancias: Matriz de distancias entre ciudades
    
    Returns:
        Distancia total del tour
    """
    distancia = 0
    for i in range(len(tour)):
        ciudad_actual = tour[i]
        ciudad_siguiente = tour[(i + 1) % len(tour)]  # Vuelve al inicio
        distancia += matriz_distancias[ciudad_actual][ciudad_siguiente]
    return distancia


def crear_individuo_aleatorio(num_ciudades):
    """
    Crea un individuo aleatorio (permutación de ciudades).
    
    Args:
        num_ciudades: Número de ciudades en el problema
    
    Returns:
        Lista con una permutación aleatoria de índices de ciudades
    """
    individuo = list(range(num_ciudades))
    random.shuffle(individuo)
    return individuo


def crossover_order(padre1, padre2):
    """
    Operador de crossover Order Crossover (OX) para TSP.
    Toma un segmento del padre1 y llena el resto con genes del padre2 en orden.
    
    Args:
        padre1: Primer padre (lista de índices)
        padre2: Segundo padre (lista de índices)
    
    Returns:
        Hijo resultante del crossover
    """
    # Seleccionar dos puntos de corte aleatorios
    a, b = sorted(random.sample(range(len(padre1)), 2))
    
    # Crear hijo vacío
    hijo = [None] * len(padre1)
    
    # Copiar segmento del padre1
    hijo[a:b] = padre1[a:b]
    
    # Llenar con genes del padre2 que no están en el hijo
    genes_faltantes = [ciudad for ciudad in padre2 if ciudad not in hijo]
    
    idx = 0
    for i in range(len(hijo)):
        if hijo[i] is None:
            hijo[i] = genes_faltantes[idx]
            idx += 1
    
    return hijo


def mutar_swap(individuo):
    """
    Operador de mutación por intercambio (swap).
    Intercambia dos ciudades aleatorias en el tour.
    
    Args:
        individuo: Lista de índices de ciudades (se modifica in-place)
    """
    i, j = random.sample(range(len(individuo)), 2)
    individuo[i], individuo[j] = individuo[j], individuo[i]


def seleccionar_torneo(poblacion, matriz_distancias, tam_torneo=3):
    """
    Selección por torneo: elige el mejor individuo de un subconjunto aleatorio.
    
    Args:
        poblacion: Lista de individuos
        matriz_distancias: Matriz de distancias entre ciudades
        tam_torneo: Tamaño del torneo
    
    Returns:
        Individuo ganador del torneo
    """
    torneo = random.sample(poblacion, tam_torneo)
    return min(torneo, key=lambda ind: calcular_distancia_tour(ind, matriz_distancias))


def algoritmo_genetico_clasico(matriz_distancias, params):
    """
    Implementa el Algoritmo Genético Clásico para resolver el TSP.
    
    Args:
        matriz_distancias: Matriz de distancias entre ciudades
        params: Instancia de ParametrosGA con la configuración
    
    Returns:
        Tupla (mejor_ruta, mejor_distancia, historial_fitness)
    """
    num_ciudades = len(matriz_distancias)
    
    # Inicializar población
    poblacion = [crear_individuo_aleatorio(num_ciudades) for _ in range(params.poblacion)]
    
    # Historial para tracking
    historial_fitness = []
    mejor_global = None
    mejor_distancia_global = float('inf')
    
    # Evolución
    for generacion in range(params.generaciones):
        # Evaluar y ordenar población
        poblacion.sort(key=lambda ind: calcular_distancia_tour(ind, matriz_distancias))
        
        # Actualizar mejor solución global
        distancia_actual = calcular_distancia_tour(poblacion[0], matriz_distancias)
        if distancia_actual < mejor_distancia_global:
            mejor_distancia_global = distancia_actual
            mejor_global = poblacion[0][:]
        
        # Guardar mejor fitness de esta generación
        historial_fitness.append(mejor_distancia_global)
        
        # Crear nueva generación
        nueva_generacion = []
        
        # Elitismo: mantener los mejores individuos
        nueva_generacion.extend([ind[:] for ind in poblacion[:params.elitismo]])
        
        # Generar resto de la población
        while len(nueva_generacion) < params.poblacion:
            # Selección de padres
            padre1 = seleccionar_torneo(poblacion, matriz_distancias)
            padre2 = seleccionar_torneo(poblacion, matriz_distancias)
            
            # Crossover
            if random.random() < params.tasa_crossover:
                hijo = crossover_order(padre1, padre2)
            else:
                hijo = padre1[:]
            
            # Mutación
            if random.random() < params.tasa_mutacion:
                mutar_swap(hijo)
            
            nueva_generacion.append(hijo)
        
        # Reemplazar población
        poblacion = nueva_generacion
        
        # Imprimir progreso cada 50 generaciones
        if (generacion + 1) % 50 == 0:
            print(f"Generación {generacion + 1}/{params.generaciones} - "
                  f"Mejor distancia: {mejor_distancia_global:.2f}")
    
    return mejor_global, mejor_distancia_global, historial_fitness

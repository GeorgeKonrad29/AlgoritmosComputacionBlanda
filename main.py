"""
Comparación de tres algoritmos para resolver el TSP (Traveling Salesman Problem):
1. Algoritmo Genético Clásico (GA)
2. Optimización de Colonia de Hormigas (ACO)
3. Algoritmo Genético Celular de Bhu-Beasley (CBGA)
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from algoritmo_genetico import algoritmo_genetico_clasico
from algoritmo_colonia_hormigas import algoritmo_colonia_hormigas
from algoritmo_chu_beasly import chu_beasley


def leer_archivo_tsp(nombre_archivo):
    """
    Lee un archivo TSP de TSPLIB y retorna las coordenadas de las ciudades.
    
    Args:
        nombre_archivo: Nombre del archivo .tsp a leer
    
    Returns:
        numpy array con las coordenadas (x, y) de cada ciudad
    """
    coordenadas = []
    lectura_coordenadas = False
    
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            if "NODE_COORD_SECTION" in linea:
                lectura_coordenadas = True
                continue
            if "EOF" in linea:
                break
            if lectura_coordenadas:
                partes = linea.strip().split()
                if len(partes) >= 3:
                    # partes[0] es el índice, partes[1] y partes[2] son las coordenadas
                    coordenadas.append([float(partes[1]), float(partes[2])])
    
    return np.array(coordenadas)


def calcular_distancia_euclidiana(coord1, coord2):
    """Calcula la distancia euclidiana entre dos coordenadas."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def calcular_matriz_distancias(coordenadas):
    """
    Calcula la matriz de distancias entre todas las ciudades.
    
    Args:
        coordenadas: Array de numpy con coordenadas de ciudades
    
    Returns:
        Matriz de distancias simétrica
    """
    n = len(coordenadas)
    matriz_distancias = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calcular_distancia_euclidiana(coordenadas[i], coordenadas[j])
            matriz_distancias[i][j] = dist
            matriz_distancias[j][i] = dist
    
    return matriz_distancias


# ============================================================================
# CLASES DE PARÁMETROS PARA CADA ALGORITMO
# ============================================================================

class ParametrosGA:

    def __init__(self):
        # Tamaño de la población: Número de individuos (rutas) en cada generación
        self.poblacion = 100
        
        # Número de generaciones: Cantidad de iteraciones del algoritmo
        self.generaciones = 500
        
        # Tasa de mutación: Probabilidad de que un gen mute (0.0 a 1.0)
        self.tasa_mutacion = 0.2
        
        # Tasa de crossover: Probabilidad de cruce entre dos padres (0.0 a 1.0)
        self.tasa_crossover = 0.8
        
        # Elitismo: Número de mejores individuos que pasan directamente a la siguiente generación
        self.elitismo = 2


class ParametrosACO:

    def __init__(self):
        # Número de hormigas: Cantidad de hormigas que construyen soluciones en cada iteración
        self.num_hormigas = 50
        
        # Número de iteraciones: Cantidad de ciclos completos del algoritmo
        self.iteraciones = 500
        
        # Alpha: Parámetro que controla la importancia de la feromona (valores típicos: 1.0-2.0)
        self.alpha = 1.0
        
        # Beta: Parámetro que controla la importancia de la información heurística (valores típicos: 2.0-5.0)
        self.beta = 5.0
        
        # Rho: Tasa de evaporación de feromonas (0.0 a 1.0, valores típicos: 0.1-0.5)
        self.rho = 0.5
        
        # Q: Constante de deposición de feromonas (influye en la cantidad depositada)
        self.q = 100
        
        # Feromona inicial: Cantidad inicial de feromona en todas las aristas
        self.feromona_inicial = 1.0


class ParametrosCBGA:

    def __init__(self):
        # Tamaño de la población: Número total de individuos en la cuadrícula celular
        self.poblacion = 100
        
        # Dimensiones de la cuadrícula: Se organiza como una grilla (ej: 10x10 para 100 individuos)
        self.filas = 10
        self.columnas = 10
        
        # Número de generaciones: Cantidad de iteraciones del algoritmo
        self.generaciones = 500
        
        # Tasa de mutación: Probabilidad de que un gen mute (0.0 a 1.0)
        self.tasa_mutacion = 0.2
        
        # Tasa de crossover: Probabilidad de cruce entre vecinos (0.0 a 1.0)
        self.tasa_crossover = 0.8
        
        # Radio de vecindad: Define la estructura de vecindad en la cuadrícula (1: von Neumann, 2: Moore extendido)
        self.radio_vecindad = 1


# ============================================================================
# FUNCIONES DE EJECUCIÓN DE ALGORITMOS
# ============================================================================

def ejecutar_ga_clasico(coordenadas, matriz_distancias, params):
    """
    
    Args:
        coordenadas: Coordenadas de las ciudades
        matriz_distancias: Matriz de distancias entre ciudades
        params: Instancia de ParametrosGA con la configuración del algoritmo
    
    """
    print("\n" + "="*70)
    print("EJECUTANDO ALGORITMO GENÉTICO CLÁSICO (GA)")
    print("="*70)
    print(f"Población: {params.poblacion}, Generaciones: {params.generaciones}")
    print(f"Tasa de mutación: {params.tasa_mutacion}, Tasa de crossover: {params.tasa_crossover}")
    print(f"Elitismo: {params.elitismo}")
    
    tiempo_inicio = time()
    
    # Ejecutar algoritmo genético
    mejor_ruta, mejor_distancia, historial_fitness = algoritmo_genetico_clasico(
        matriz_distancias, params
    )
    
    tiempo_ejecucion = time() - tiempo_inicio
    
    print(f"\nMejor distancia encontrada: {mejor_distancia:.2f}")
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    
    return mejor_ruta, mejor_distancia, historial_fitness, tiempo_ejecucion


def ejecutar_aco(coordenadas, matriz_distancias, params):
    """
    
    Args:
        coordenadas: Coordenadas de las ciudades
        matriz_distancias: Matriz de distancias entre ciudades
        params: Instancia de ParametrosACO con la configuración del algoritmo
    

    """
    print("\n" + "="*70)
    print("EJECUTANDO OPTIMIZACIÓN DE COLONIA DE HORMIGAS (ACO)")
    print("="*70)
    print(f"Hormigas: {params.num_hormigas}, Iteraciones: {params.iteraciones}")
    print(f"Alpha: {params.alpha}, Beta: {params.beta}")
    print(f"Rho: {params.rho}, Q: {params.q}")
    
    tiempo_inicio = time()
    
    # Ejecutar algoritmo de colonia de hormigas
    mejor_ruta, mejor_distancia, historial_fitness = algoritmo_colonia_hormigas(
        matriz_distancias, params
    )
    
    tiempo_ejecucion = time() - tiempo_inicio
    
    print(f"\nMejor distancia encontrada: {mejor_distancia:.2f}")
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    
    return mejor_ruta, mejor_distancia, historial_fitness, tiempo_ejecucion
    return None, float('inf'), [], tiempo_ejecucion


def ejecutar_cbga(coordenadas, matriz_distancias, params):
    """

    
    Args:
        coordenadas: Coordenadas de las ciudades
        matriz_distancias: Matriz de distancias entre ciudades
        params: Instancia de ParametrosCBGA con la configuración del algoritmo
    

    """
    print("\n" + "="*70)
    print("EJECUTANDO ALGORITMO GENÉTICO CELULAR BHU-BEASLEY (CBGA)")
    print("="*70)
    
    tiempo_inicio = time()
    
    # TODO: Implementar CBGA
    # La implementación se agregará en el siguiente 
    
    tiempo_ejecucion = time() - tiempo_inicio
    
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    
    # Valores de retorno temporales
    return None, float('inf'), [], tiempo_ejecucion


def ejecutar_chu_beasley(coordenadas, matriz_distancias, restarts=20, seed=None):
    """
    Ejecuta el algoritmo Chu-Beasley implementado en `algoritmo_chu_beasly.py`.

    `coordenadas` se pasa como numpy array; la función convierte a lista de tuplas.
    """
    print("\n" + "="*70)
    print("EJECUTANDO ALGORITMO CHU-BEASLEY (heurística NN + 2-opt)")
    print("="*70)
    coords_list = [tuple(x) for x in coordenadas]
    from time import time
    t0 = time()
    tour, distancia = chu_beasley(coords_list, restarts=restarts, seed=seed)
    tiempo = time() - t0
    print(f"Mejor distancia (Chu-Beasley): {distancia:.2f} | Tiempo: {tiempo:.2f}s")
    return tour, distancia, [], tiempo


def comparar_resultados(resultados):
    """
    Compara y muestra los resultados de los tres algoritmos.
    
    Args:
        resultados: Diccionario con los resultados de cada algoritmo para cada archivo TSP
    """
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    
    for archivo, datos in resultados.items():
        print(f"\n{archivo}:")
        print("-" * 70)
        
        for algoritmo, (ruta, distancia, historial, tiempo) in datos.items():
            print(f"{algoritmo:10s} | Distancia: {distancia:10.2f} | Tiempo: {tiempo:6.2f}s")
    
    print("\n" + "="*70)


def main():
    """Función principal que ejecuta la comparación de los tres algoritmos."""
    
    print("="*70)
    print("COMPARACIÓN DE ALGORITMOS PARA EL TSP")
    print("="*70)
    
    # Inicializar
    params_ga = ParametrosGA()
    params_aco = ParametrosACO()
    params_cbga = ParametrosCBGA()
    
    archivos_tsp = [
        'berlin52.tsp',
        'eil51.tsp',
        'st70.tsp'
    ]
    
    resultados = {}
    
    # Procesar cada archivo TSP
    for archivo in archivos_tsp:
        print(f"\n{'='*70}")
        print(f"PROCESANDO: {archivo}")
        print(f"{'='*70}")
        

        coordenadas = leer_archivo_tsp(archivo)
        print(f"Número de ciudades: {len(coordenadas)}")
        
        # Calcular matriz de distancias
        matriz_distancias = calcular_matriz_distancias(coordenadas)
        
        # Ejecutar cada algoritmo
        resultados[archivo] = {}
        
        resultados[archivo]['GA'] = ejecutar_ga_clasico(coordenadas, matriz_distancias, params_ga)
        resultados[archivo]['ACO'] = ejecutar_aco(coordenadas, matriz_distancias, params_aco)
        resultados[archivo]['ChuBeasley'] = ejecutar_chu_beasley(coordenadas, matriz_distancias, restarts=20, seed=42)
        resultados[archivo]['CBGA'] = ejecutar_cbga(coordenadas, matriz_distancias, params_cbga)
    
    # Mostrar comparación final
    comparar_resultados(resultados)


if __name__ == "__main__":
    main()

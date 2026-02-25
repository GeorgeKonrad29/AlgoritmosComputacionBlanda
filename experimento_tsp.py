import time
import random
import numpy as np
import matplotlib.pyplot as plt

# === IMPORTACIÓN DE ALGORITMOS ===
from algoritmo_genetico import algoritmo_genetico_clasico
from algoritmo_colonia_hormigas import algoritmo_colonia_hormigas
from algoritmo_chu_beasly import cbga

import math

def parse_tsp(path):
    coords = []
    with open(path, "r") as f:
        in_section = False
        for line in f:
            if line.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if in_section:
                if line.strip() == "EOF":
                    break
                _, x, y = line.split()
                coords.append((float(x), float(y)))
    return coords


def matriz_distancias(coords):
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1]
            )
    return dist

coords = parse_tsp("C:\\Users\\jonat\\Downloads\\AlgoritmosComputacionBlanda-main\\AlgoritmosComputacionBlanda-main\\berlin52.tsp")
dist_matrix = matriz_distancias(coords)


class ParamsGA:
    poblacion = 100
    generaciones = 100
    elitismo = 2
    tasa_crossover = 0.9
    tasa_mutacion = 0.2


class ParamsACO:
    num_hormigas = 50
    iteraciones = 100
    alpha = 1.0
    beta = 5.0
    rho = 0.5
    q = 100
    feromona_inicial = 1.0

params_ga = ParamsGA()
params_aco = ParamsACO()
params_cbga = {
    "pop_size": 100,
    "generations": 100,
    "min_diversity": 0.2,
    "p_mut": 0.2,
    "p_ls_child": 0.1,
    "seed": None,
    "apply_2opt_on_new": False,
    "tournament_k": 3
}

results = {
    "GA": {"best": [], "time": [], "history": []},
    "ACO": {"best": [], "time": [], "history": []},
    "CBGA": {"best": [], "time": [], "history": []}
}


NUM_RUNS = 30

for seed in range(NUM_RUNS):
    random.seed(seed)
    np.random.seed(seed)

    # ---- GA ----
    start = time.perf_counter()
    _, dist_ga, hist_ga = algoritmo_genetico_clasico(dist_matrix, params_ga)
    results["GA"]["time"].append(time.perf_counter() - start)
    results["GA"]["best"].append(dist_ga)
    results["GA"]["history"].append(hist_ga)

    # ---- ACO ----
    start = time.perf_counter()
    _, dist_aco, hist_aco = algoritmo_colonia_hormigas(dist_matrix, params_aco)
    results["ACO"]["time"].append(time.perf_counter() - start)
    results["ACO"]["best"].append(dist_aco)
    results["ACO"]["history"].append(hist_aco)

    # ---- CBGA ----
    start = time.perf_counter()
    _, dist_cbga = cbga(coords, **params_cbga)
    hist_cbga = []  
    results["CBGA"]["time"].append(time.perf_counter() - start)
    results["CBGA"]["best"].append(dist_cbga)
    results["CBGA"]["history"].append(hist_cbga)



def avg_curve(histories):
    histories = [h for h in histories if len(h) > 0]
    if len(histories) == 0:
        return None
    min_len = min(len(h) for h in histories)
    return np.mean([h[:min_len] for h in histories], axis=0)


plt.plot(avg_curve(results["GA"]["history"]), label="GA")
plt.plot(avg_curve(results["ACO"]["history"]), label="ACO")
curve_cbga = avg_curve(results["CBGA"]["history"])
if curve_cbga is not None:
    plt.plot(curve_cbga, label="CBGA")

plt.xlabel("Iteración / Generación")
plt.ylabel("Mejor distancia promedio")
plt.title("Convergencia promedio – berlin52")
plt.legend()
plt.grid(True)
plt.show()



plt.boxplot(
    [
        results["GA"]["best"],
        results["ACO"]["best"],
        results["CBGA"]["best"]
    ],
    labels=["GA", "ACO", "CBGA"],
    showmeans=True
)

plt.ylabel("Mejor distancia")
plt.title("Estabilidad de soluciones")
plt.grid(True)
plt.show()



plt.boxplot(
    [
        results["GA"]["time"],
        results["ACO"]["time"],
        results["CBGA"]["time"]
    ],
    labels=["GA", "ACO", "CBGA"],
    showmeans=True
)

plt.ylabel("Tiempo (s)")
plt.title("Costo computacional")
plt.grid(True)
plt.show()

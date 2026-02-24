import math
import random
import sys
import argparse
from typing import List, Tuple


def parse_tsp(path: str) -> List[Tuple[float, float]]:
	coords = []
	with open(path, "r", encoding="utf-8") as f:
		in_node_section = False
		for line in f:
			line = line.strip()
			if line.upper().startswith("NODE_COORD_SECTION"):
				in_node_section = True
				continue
			if not in_node_section:
				continue
			if line == "EOF" or line == "":
				break
			parts = line.split()
			if len(parts) >= 3:
				try:
					x = float(parts[1])
					y = float(parts[2])
				except ValueError:
					continue
				coords.append((x, y))
	return coords


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


def total_distance(tour: List[int], coords: List[Tuple[float, float]]) -> float:
	n = len(tour)
	if n == 0:
		return 0.0
	dist = 0.0
	for i in range(n):
		a = coords[tour[i]]
		b = coords[tour[(i + 1) % n]]
		dist += euclidean(a, b)
	return dist


def nearest_neighbor(coords: List[Tuple[float, float]], start: int = 0) -> List[int]:
	n = len(coords)
	unvisited = set(range(n))
	tour = [start]
	unvisited.remove(start)
	while unvisited:
		last = tour[-1]
		next_node = min(unvisited, key=lambda j: euclidean(coords[last], coords[j]))
		tour.append(next_node)
		unvisited.remove(next_node)
	return tour


def two_opt(tour: List[int], coords: List[Tuple[float, float]]) -> List[int]:
	improved = True
	n = len(tour)
	best = tour
	best_dist = total_distance(best, coords)
	while improved:
		improved = False
		for i in range(1, n - 1):
			for k in range(i + 1, n):
				new_tour = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
				new_dist = total_distance(new_tour, coords)
				if new_dist + 1e-12 < best_dist:
					best = new_tour
					best_dist = new_dist
					improved = True
					break
			if improved:
				break
	return best


def chu_beasley(coords: List[Tuple[float, float]], restarts: int = 10, seed: int | None = None) -> Tuple[List[int], float]:
	if seed is not None:
		random.seed(seed)
	n = len(coords)
	if n == 0:
		return [], 0.0
	best_tour = None
	best_dist = float("inf")
	for r in range(restarts):
		start = random.randrange(n)
		tour = nearest_neighbor(coords, start=start)
		tour = two_opt(tour, coords)
		dist = total_distance(tour, coords)
		if dist < best_dist:
			best_dist = dist
			best_tour = tour
	return best_tour, best_dist


def parse_args():
	p = argparse.ArgumentParser(description="Algoritmo Chu-Beasley (heurística TSP)")
	p.add_argument("instance", nargs="?", default="berlin52.tsp", help="Archivo .tsp (EUC_2D)")
	p.add_argument("--restarts", type=int, default=20, help="Número de reinicios aleatorios")
	p.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
	return p.parse_args()


def main():
	args = parse_args()
	coords = parse_tsp(args.instance)
	if not coords:
		print("No se encontraron coordenadas en la instancia.")
		sys.exit(1)
	tour, dist = chu_beasley(coords, restarts=args.restarts, seed=args.seed)
	print(f"Instancia: {args.instance}")
	print(f"Nodos: {len(coords)}")
	print(f"Distancia mejor tour: {dist:.4f}")
	print("Tour (0-based indices):")
	print(tour)


if __name__ == "__main__":
	main()


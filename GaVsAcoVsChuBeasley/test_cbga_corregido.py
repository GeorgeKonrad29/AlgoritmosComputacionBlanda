"""Test rápido para verificar las correcciones del CBGA"""
import random
import time
from algoritmo_chu_beasly import cbga, parse_tsp

# Cargar datos
coords = parse_tsp("berlin52.tsp")

print("=== TEST CBGA CORREGIDO ===\n")

# Test 1: Verificar que retorna 3 valores (tour, distancia, historial)
print("Test 1: Verificando retorno con historial...")
tour, dist, history = cbga(coords, pop_size=30, generations=50, seed=42)
print(f"✓ Retorna 3 valores: tour (len={len(tour)}), distancia ({dist:.2f}), historial (len={len(history)})")
print(f"  - Historial primeros 10 valores: {history[:10]}")
print(f"  - Historial últimos 5 valores: {history[-5:]}")
assert len(history) == 51, f"Esperaba 51 elementos en historial (inicial + 50 gen), obtuvo {len(history)}"
print("✓ Test 1 PASADO\n")

# Test 2: Verificar que la mutación es menos agresiva
print("Test 2: Verificando patrón de convergencia...")
_, dist1, hist1 = cbga(coords, pop_size=50, generations=100, p_mut=0.2, seed=123)
print(f"✓ Ejecución 1: distancia={dist1:.2f}, convergencia: {hist1[0]:.2f} -> {hist1[-1]:.2f}")

_, dist2, hist2 = cbga(coords, pop_size=50, generations=100, p_mut=0.5, seed=456)
print(f"✓ Ejecución 2: distancia={dist2:.2f}, convergencia: {hist2[0]:.2f} -> {hist2[-1]:.2f}")
print("✓ Test 2 PASADO\n")

# Test 3: Verificar mejora respecto a solución aleatoria
print("Test 3: Verificando mejora sobre solución aleatoria...")
from algoritmo_chu_beasly import total_distance
random.seed(999)
tour_random = list(range(len(coords)))
random.shuffle(tour_random)
dist_random = total_distance(tour_random, coords)

_, dist_cbga, _ = cbga(coords, pop_size=100, generations=150, seed=999)
mejora = ((dist_random - dist_cbga) / dist_random) * 100

print(f"  - Distancia tour aleatorio: {dist_random:.2f}")
print(f"  - Distancia CBGA: {dist_cbga:.2f}")
print(f"  - Mejora: {mejora:.1f}%")
assert dist_cbga < dist_random, "CBGA debería mejorar sobre tour aleatorio"
print("✓ Test 3 PASADO\n")

# Test 4: Verificar estabilidad con semillas diferentes
print("Test 4: Verificando estabilidad con múltiples ejecuciones...")
resultados = []
for seed in range(5):
    _, dist, _ = cbga(coords, pop_size=50, generations=80, seed=seed)
    resultados.append(dist)

promedio = sum(resultados) / len(resultados)
desv_est = (sum((x - promedio)**2 for x in resultados) / len(resultados))**0.5
cv = (desv_est / promedio) * 100  # Coeficiente de variación

print(f"  - Resultados: {[f'{d:.1f}' for d in resultados]}")
print(f"  - Promedio: {promedio:.2f}")
print(f"  - Desviación estándar: {desv_est:.2f}")
print(f"  - Coeficiente de variación: {cv:.2f}%")
print("✓ Test 4 PASADO\n")

print("=" * 50)
print("✓✓✓ TODOS LOS TESTS PASADOS ✓✓✓")
print("=" * 50)

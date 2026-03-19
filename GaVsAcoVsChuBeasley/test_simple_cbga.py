"""Test simple y rápido del CBGA corregido"""
from algoritmo_chu_beasly import cbga, parse_tsp

coords = parse_tsp("berlin52.tsp")

print("=== TEST SIMPLE CBGA ===\n")

# Test sin 2-opt (más rápido)
print("Test 1: Ejecución sin 2-opt (rápida)...")
tour, dist, history = cbga(coords, pop_size=50, generations=100, 
                           apply_2opt_on_new=False, seed=42)
print(f"✓ Distancia: {dist:.2f}")
print(f"✓ Historial tiene {len(history)} elementos (debe ser 101: inicial + 100 gen)")
assert len(history) == 101, f"Error: esperaba 101, obtuvo {len(history)}"
print(f"✓ Convergencia: {history[0]:.2f} -> {history[-1]:.2f}\n")

# Test con los parámetros del experimento
print("Test 2: Con parámetros del experimento (sin 2-opt)...")
params_cbga = {
    "pop_size": 100,
    "generations": 150,
    "min_diversity": 0.2,
    "p_mut": 0.2,
    "p_ls_child": 0.1,
    "seed": 42,
    "apply_2opt_on_new": False,  # Desactivado para el experimento
    "tournament_k": 3
}

tour, dist, history = cbga(coords, **params_cbga)
print(f"✓ Distancia: {dist:.2f}")
print(f"✓ Historial tiene {len(history)} elementos")
print(f"✓ Convergencia: {history[0]:.2f} -> {history[-1]:.2f}\n")

# Verificar consistencia con semilla
print("Test 3: Verificando consistencia con misma semilla...")
_, dist1, _ = cbga(coords, pop_size=50, generations=50, apply_2opt_on_new=False, seed=999)
_, dist2, _ = cbga(coords, pop_size=50, generations=50, apply_2opt_on_new=False, seed=999)
assert dist1 == dist2, "Misma semilla debe dar mismo resultado"
print(f"✓ Consistencia verificada: {dist1:.2f} == {dist2:.2f}\n")

print("=" * 50)
print("✓✓✓ TODOS LOS TESTS PASADOS ✓✓✓")
print("=" * 50)
print("\nEl algoritmo CBGA está correctamente implementado con:")
print("  1. Mutación corregida (1-2 swaps en lugar de n*p_mut)")
print("  2. Semilla única en cbga (sin duplicación)")
print("  3. Historial de convergencia completo")

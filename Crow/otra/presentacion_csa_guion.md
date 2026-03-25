# Guion de presentación (10 min) — Taller CSA + MLP

## 1) Problema (1 min)
- Queremos aproximar $y=\sin(x)$ con MLP.
- Restricción: no usar backpropagation.
- Enfoque: optimización directa de pesos con CSA.

## 2) Arquitectura y representación (1 min)
- MLP: `1 → 2 → 2 → 1`.
- Activaciones: sigmoide en ocultas, identidad en salida.
- Cada solución: vector de 13 parámetros.
- Fitness: MSE sobre datos de entrenamiento.

## 3) CSA explicado simple (2 min)
- Cada cuervo = una solución candidata.
- Memoria = mejor solución individual encontrada.
- Movimiento:
  - Si objetivo no detecta: seguir su memoria.
  - Si detecta: ir a posición aleatoria.
- Se actualiza memoria y mejor global por iteración.

## 4) Implementación y métricas (1 min)
- Dataset: puntos de `sin(x)` en $[-\pi,\pi]$.
- Métricas: MSE train/test, convergencia y residuos.
- Historial guardado por iteración para análisis.

## 5) Variación obligatoria: ACSA (2 min)
- Cambio propuesto: `AP_t` y `FL_t` decrecientes.
- Idea clave:
  - Inicio: explorar más y saltar más lejos.
  - Final: explotar y afinar solución.
- Mostrar gráfica de schedules + comparación promedio por semillas.

## 6) Resultados (2 min)
- Mostrar:
  1. Curva de convergencia.
  2. Real vs predicción.
  3. Tabla/plot de CSA vs PSO vs GA.
  4. CSA base vs ACSA (media ± std).
- Mensaje final: qué algoritmo ganó y por qué (calidad vs estabilidad).

## 7) Cierre y defensa (1 min)
- Decisiones tomadas:
  - Rango de búsqueda.
  - Tamaño de población e iteraciones.
  - Diseño de variación adaptativa.
- Lecciones:
  - Metaheurísticas dependen de hiperparámetros.
  - Comparar por múltiples semillas es clave.

---

## Preguntas típicas de defensa (preparación rápida)
1. ¿Por qué no backpropagation?
2. ¿Qué hace exactamente `AP` en CSA?
3. ¿Por qué 13 parámetros?
4. ¿Cómo garantizas que no es azar de una sola corrida?
5. ¿Qué harías para mejorar aún más ACSA?

## Respuestas cortas sugeridas
- Se evitó backprop para cumplir la restricción y estudiar optimización bio-inspirada.
- `AP` controla exploración aleatoria vs seguimiento de memoria.
- Los 13 parámetros salen de la arquitectura fija `1-2-2-1` con pesos+bias.
- Se usaron varias semillas y se reportó media ± desviación estándar.
- Se podría añadir elitismo, reinicios adaptativos o búsqueda local al final.

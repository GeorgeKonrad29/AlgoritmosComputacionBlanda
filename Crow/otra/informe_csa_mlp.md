# Informe — Entrenamiento de MLP con Crow Search Algorithm (CSA)

## 1. Introducción
Este trabajo aborda la aproximación de la función $y=\sin(x)$ mediante una red neuronal MLP entrenada sin backpropagation, formulando el problema como optimización continua de parámetros.

Objetivo de optimización:

$$
\min_{\theta}\; MSE(\theta)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

## 2. Arquitectura de la red
Se implementó una MLP desde cero con la arquitectura obligatoria:
- Entrada: 1 neurona
- Capa oculta 1: 2 neuronas (sigmoide)
- Capa oculta 2: 2 neuronas (sigmoide)
- Salida: 1 neurona (identidad)

Total de parámetros optimizados: 13.

## 3. Representación y fitness
Cada individuo/cuervo representa un vector $\theta\in\mathbb{R}^{13}$ con todos los pesos y sesgos de la MLP.

El fitness usado fue el error cuadrático medio (MSE) sobre el conjunto de entrenamiento.

## 4. Algoritmo CSA implementado
El Crow Search Algorithm se implementó con los pasos clásicos:
1. Inicialización aleatoria de posiciones y memorias.
2. Evaluación de fitness de cada cuervo.
3. Actualización de posición por seguimiento o reubicación aleatoria según `AP`.
4. Actualización de memorias personales y mejor global.
5. Registro de historial y criterio de parada.

Parámetros base usados en el experimento principal:
- `n_crows = 40`
- `max_iter = 1000`
- `fl = 2.0`
- `ap = 0.1`
- `lb, ub = [-5, 5]`

## 5. Variación propuesta (mejora obligatoria)
Se propuso un **CSA adaptativo (ACSA)** con:
- `AP_t` decreciente en el tiempo.
- `FL_t` decreciente en el tiempo.

Ecuaciones:

$$
AP_t = AP_{max} - (AP_{max}-AP_{min})\frac{t}{T}
$$

$$
FL_t = FL_{max} - (FL_{max}-FL_{min})\frac{t}{T}
$$

Intuición: mayor exploración al inicio y refinamiento al final.

## 6. Experimentos realizados
1. Convergencia (MSE vs iteraciones).
2. Predicción vs función real y análisis de residuos.
3. Sensibilidad a parámetros (`fl`, `AP`).
4. Comparación de desempeño con PSO y GA.
5. Comparación adicional CSA base vs ACSA en múltiples semillas.

## 7. Resultados (completar con tus valores)
- MSE entrenamiento (CSA): `_____`
- MSE prueba (CSA): `_____`
- MSE final PSO: `_____`
- MSE final GA: `_____`
- MSE promedio CSA base (5 semillas): `_____ ± _____`
- MSE promedio ACSA (5 semillas): `_____ ± _____`

Interpretación breve sugerida:
- Si ACSA < CSA base en promedio: la variación mejora robustez/convergencia.
- Si no mejora siempre: justificar sensibilidad a hiperparámetros y estocasticidad.

## 8. Problemas encontrados
- Alta variabilidad entre corridas por naturaleza estocástica.
- Sensibilidad a límites de búsqueda y configuración de `fl/AP`.
- Mayor costo computacional al comparar múltiples algoritmos y semillas.

## 9. Conclusiones
- Se logró entrenar una MLP sin backpropagation mediante CSA.
- El enfoque bio-inspirado es viable para optimización de pesos en baja dimensión.
- La variación ACSA aporta una estrategia explícita de exploración-explotación temporal.
- La comparación con PSO/GA permite contextualizar fortalezas y debilidades del método.

## 10. Trabajo futuro
- Ajuste automático de hiperparámetros.
- Hibridación CSA + búsqueda local.
- Validación en funciones objetivo más complejas.

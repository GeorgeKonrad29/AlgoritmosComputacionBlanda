# 🧪 Taller: Entrenamiento de una MLP con Algoritmos Bio-Inspirados

## 🎯 Objetivo
Diseñar e implementar un algoritmo bio-inspirado para entrenar una red neuronal MLP que aproxime:

y = sin(x)

---

## 🧠 Contexto

- No usar backpropagation
- Resolver como problema de optimización:
  min MSE(θ)

---

## 🧬 Algoritmos permitidos

Los algoritmos seleccionados por cada grupo de estudiantes, debe ser usado para entrenar la MLP. Estos son:

1. Eagle Strategy (ES)
2. Grey Wolf Optimizer (GWO)
3. Artificial Fish Swarm Algorithm (AFSA)
4. Lion Optimization Algorithm (LOA)
5. Shark Smell Optimization (SSO)
6. Cuckoo Search
7. Cat Swarm Optimization (CSO)
8. Salp Swarm Algorithm (SSA)
9. Firefly Algorithm (FA)
10. Water Cycle Algorithm (WCA)
11. Whale Optimization Algorithm (WOA)  
12. Gravitational Search Algorithm (GSA)
13. Crow Search Algorithm (CSA)
14. Harmony Search (HS) 
15. Wolf Pack Algorithm
16. Bat Algorithm (BA)
17. Crow Search Algorithm (CSA)

---

## 🏗️ Arquitectura obligatoria

- 1 entrada
- 2 capas ocultas (2 neuronas cada una)
- Activación: sigmoide
- Salida: identidad
- Total parámetros: 13

---

## ⚙️ Requisitos técnicos

### Representación
Cada individuo = vector de 13 pesos

### Forward
Implementación desde cero (sin librerías ML)

### Fitness
MSE entre predicción y sin(x)

### Algoritmo
Debe incluir:
- Inicialización
- Evaluación
- Actualización
- Criterio de parada

---

## 📊 Experimentos

1. Convergencia (error vs iteraciones)
2. Sensibilidad a parámetros
3. Gráfica real vs predicción
4. Comparación con PSO o GA

---

## 📦 Entregables

### Notebook (.ipynb)
- Código completo
- Resultados
- Gráficas

### Informe (máx 5 páginas)
- Explicación del algoritmo
- Adaptación a la red
- Problemas
- Resultados

### Presentación (10 min)
- Explicación
- Resultados
- Justificación

---

## 🚨 Estrategia ANTI-IA

### Defensa oral
- Explicar algoritmo
- Explicar decisiones

### Prueba individual
- Preguntas conceptuales
- Dibujos del algoritmo

### Código en vivo
- Modificaciones en clase

### Variación obligatoria
- Mejora propia del algoritmo

### Registro de evolución
- Historial del fitness

---

## 🏆 Rúbrica

- Implementación: 25%
- Uso del algoritmo: 20%
- Análisis: 20%
- Presentación: 15%
- Defensa oral: 20%

---

## 🔥 Bonus

- Comparación de algoritmos
- Algoritmos híbridos
- Análisis profundo
import math
from scipy.stats import norm

# =========================================
# EJERCICIO 1 — E-COMMERCE (BINOMIAL → NORMAL)
# =========================================

n = 500
p = 0.025

mu = n * p
sigma = math.sqrt(n * p * (1 - p))

# Corrección de continuidad: P(X >= 15) → P(X >= 14.5)
z = (14.5 - mu) / sigma

prob_1 = 1 - norm.cdf(z)

print("Ejercicio 1 (E-commerce):", prob_1)


# =========================================
# EJERCICIO 2 — CALL CENTER (POISSON → NORMAL)
# =========================================

lam = 20

mu = lam
sigma = math.sqrt(lam)

# Corrección: P(X > 25) → P(X > 25.5)
z = (25.5 - mu) / sigma

prob_2 = 1 - norm.cdf(z)

print("Ejercicio 2 (Call Center):", prob_2)


# =========================================
# EJERCICIO 3 — TIEMPOS DE RESPUESTA (NORMAL)
# =========================================

mu = 200
sigma = 30

z = (250 - mu) / sigma

prob_3 = 1 - norm.cdf(z)

print("Ejercicio 3 (Tiempos > 250ms):", prob_3)


# =========================================
# EJERCICIO 4 — FIABILIDAD (EXPONENCIAL)
# =========================================

lam = 1 / 30  # tasa

prob_4 = math.exp(-lam * 60)

print("Ejercicio 4 (Durar > 60 días):", prob_4)
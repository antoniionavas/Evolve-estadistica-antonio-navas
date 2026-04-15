"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 1
Análisis Estadístico Descriptivo
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Crear la carpeta de salida por si no existe
if not os.path.exists('output'):
    os.makedirs('output')
    
# Cargar dataset
df = pd.read_csv("data/diamonds.csv", index_col=0)

# Mostrar primeras filas para verificar que se ha cargado el dataset correctamente
print(df.head())

"""
 A) Resumen estructural
    ----------------------
    • Número de filas, columnas y tamaño en memoria.
    • Tipos de dato de cada columna (dtypes).
    • Porcentaje de valores nulos por columna y decisión de tratamiento.
"""

# Número de filas y columnas
n_filas, n_columnas = df.shape

print(f"\nNúmero de filas: {n_filas}")
print(f"\nNúmero de columnas: {n_columnas}")

# Tamaño en memoria
memoria = df.memory_usage(deep=True).sum() / (1024**2)
print(f"\nTamaño en memoria: {memoria:.2f} MB")

# Tipos de datos
print("\nTipos de datos por columna:", df.dtypes)

# Valores nulos
nulos = df.isnull().mean() * 100
print("\nPorcentaje de nulos:")
print(nulos)


"""
B) Estadísticos descriptivos de variables numéricas
    -----------------------------------------------
    • Media, mediana, moda, desviación típica, varianza, mínimo, máximo y cuartiles.
    • Rango intercuartílico (IQR) de la variable objetivo.
    • Coeficiente de asimetría (skewness) y curtosis para al menos la variable objetivo.
"""

# Usando la funcion .describe muestro todos los datos pedidos excepto la varianza, la mediana y la moda que lo añado aparte indicando que es solo para numeros. 
stats_dates = df.describe()
stats_dates.loc['var'] = df.var(numeric_only=True)    # Varianza
stats_dates.loc['median'] = df.median(numeric_only=True) # Mediana 
stats_dates.loc['mode'] = df.mode(numeric_only=True).iloc[0] # Moda
stats_dates.to_csv("output/ej1_descriptivo.csv")

print(stats_dates)

# Rango intercuartílico de la variable objetivo 

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
print(f"\n IQR de price: {IQR}")

# Asimetría 
asimetria = df['price'].skew()
print(f"\n Asimetría de price: {asimetria}")

# Curtosis
curtosis = df['price'].kurt()
print(f"\n Curtosis de price: {curtosis}")


"""
C) Distribuciones
    --------------
    • Histogramas de todas las variables numéricas
    • Boxplots de la variable objetivo, segmentados por cada variable categórica.
    • Detección y tratamiento de outliers (método IQR o Z-score; justifica cuál usas).
"""
# Identifico las variables numéricas para posteriormente hacer los histogramas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(numeric_cols)

# Crear el histograma
df[numeric_cols].hist(bins=30, figsize=(15, 10), color='grey', edgecolor='black')

# Añado un título, configuro el nombre de la imagen y la guardo
plt.suptitle('Distribución de Variables Numéricas', fontsize=16) 
plt.tight_layout()
plt.savefig("output/ej1_histogramas.png")
plt.close()

#Boxplots de la variable objetivo, segmentado por cada variable
categoric_vars = df.select_dtypes(include=['object', 'category', 'string']).columns
fig, axes = plt.subplots(1, len(categoric_vars), figsize=(15, 5))

for i, col in enumerate(categoric_vars):
    sns.boxplot(x=col, y='price', data=df, ax=axes[i])
plt.tight_layout()
plt.savefig("output/ej1_boxplots.png")
plt.close()

# Outliers método IQR debido a que es más fiable ya que los diamantes tienen precios variados, aparte puedo reutilizar la variable creada anteriormente IQR y directamente empezar a definir los limites
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# detectar cuantos outliers hay
outliers = len(df[(df['price'] < lower_bound) | (df['price'] > upper_bound)])
#tratamiento de outliers mediante cliping
df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
print(f"Outliers detectados en price: {outliers}")


"""
D) Variables categóricas
    --------------
    • Frecuencia absoluta y relativa de cada categoría.
    • Gráfico de barras o de sectores para cada variable categórica.
    • Análisis de si alguna categoría domina el dataset (desbalance).
"""
# Uso variables categóricas ya en la variable categoric_vars
fig, axes = plt.subplots(1, len(categoric_vars), figsize=(10, 6))
for i, col in enumerate(categoric_vars):
    print(f"\nFrecuencias para {col}:")
    abs_freq = df[col].value_counts()
    rel_freq = df[col].value_counts(normalize=True) * 100
    # Gráfico de barras
    sns.countplot(x=col, data=df, ax=axes[i], palette='viridis', order=abs_freq.index)
    axes[i].set_title(f'Distribución de {col}')

plt.tight_layout()
plt.savefig("output/ej1_categoricas.png")
plt.close()


"""
E) Correlaciones
    --------------
    • Mapa de calor (heatmap) de la matriz de correlaciones de Pearson de las variables numéricas.
    • Identificación de las tres variables con mayor correlación (en valor absoluto) con la variable objetivo.
    • Detección de posible multicolinealidad entre predictoras (pares con |r| > 0,9).
"""

# Mapa de calor de la matriz con variables numéricas
matriz_corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor: Correlaciones de Pearson')
plt.savefig("output/ej1_heatmap_correlacion.png") 
plt.close()

# Identificacion de las tres variables con mayor correlacion

ranking_corr = matriz_corr['price'].abs().sort_values(ascending=False) # Se extrae la columna price de la matriz
top_3_vars = ranking_corr.drop('price').head(3) # se extren las tres variables con mayor correlacion
print(f"\n Las 3 variables con mayor correlacion con el precio son: {top_3_vars} ")


# Detección de la posible multicolinealidad entre predictores r > 0.9

corr_predictors = matriz_corr.drop(index='price', columns='price') # Se quita la variable price para comparar las demas
for i in range(len(corr_predictors.columns)): # bucle for para comparar cada columna
    for j in range(i):
        coeficiente = corr_predictors.iloc[i, j]
        if abs(coeficiente) > 0.9:
            var_1 = corr_predictors.columns[i]
            var_2 = corr_predictors.columns[j]
            print(f"Alerta de Multicolinealidad: {var_1} y {var_2} (r = {coeficiente:.4f})")
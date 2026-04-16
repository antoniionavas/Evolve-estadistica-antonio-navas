"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 2
Inferencia con Scikit-Learn
=============================================================================
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Crear la carpeta de salida por si no existe
if not os.path.exists('output'):
    os.makedirs('output')
    
# Cargar dataset
df = pd.read_csv("data/diamonds.csv", index_col=0)

# Mostrar primeras filas para verificar que se ha cargado el dataset correctamente
print(df.head())


"""
 2.1) Preprocesamiento
    ----------------------
    • Aplica las transformaciones necesarias: codificación de variables categóricas
    (LabelEncoder, OneHotEncoder o get_dummies), escalado si procede (StandardScaler o
    MinMaxScaler) y eliminación de columnas que no aporten información.
    • Divide los datos en Train (80 %) y Test (20 %) usando train_test_split(...,
    random_state=42).
"""

# Seleccion de variables y limpieza para que la regresion lineal no se confunda 
X = df.drop(columns=['price', 'x', 'y', 'z']) 
y = df['price']
print("\nVariables:", X.columns.tolist())

# Convertir variables categóricas a columnas de 0 y 1
categoric_vars = df.select_dtypes(include=['object', 'category', 'string']).columns

X = pd.get_dummies(X, columns=categoric_vars, drop_first=True)
# Poner todos los numeros a la misma escala
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("\nDatos procesados:")
print(X.head())

#Dividimos los datos en Train al 80 % y en Test al 20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nMuestras para entrenar: {len(X_train)}")
print(f"\nMuestras para evaluar: {len(X_test)}")


"""
 2.2) Modelo A - Regresión Lineal
    -------------------------------
    • Entrena el modelo con los datos de entrenamiento.
    • Evalúa sobre el test set calculando: MAE, RMSE y R2.
    • Genera el gráfico de residuos (valores predichos en X, residuos en Y).
    • Comenta los resultados en Respuestas.md: ¿el modelo es bueno?, ¿hay overfitting o
    underfitting?, ¿qué variables son más influyentes?
"""

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModelo entrenado con éxito")

# Predicciones
y_pred = model.predict(X_test)

# Calcular métricas 
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

print(f"\nMAE (Error medio): {MAE:.2f}$")
print(f"\nRMSE (Desviación del error): {RMSE:.2f}$")
print(f"\nR2 Score (Precisión): {R2:.4f}")

# 4. Generación del archivo de texto con las métricas
with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write("RESUMEN: REGRESIÓN LINEAL\n")
    f.write("======================================\n\n")
    f.write(f"MAE:  {MAE:.2f}$\n")
    f.write(f"RMSE: {RMSE:.2f}$\n")
    f.write(f"R2:   {R2:.4f}\n\n")
    f.write("ANÁLISIS:\n")
    f.write(f"- El modelo explica el {R2*100:.2f}% de la varianza del precio.\n")
    f.write(f"- Error promedio por diamante es de {MAE:.2f} $.\n")

# Gráfico de residuos
residuos = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, alpha=0.3, color='teal') # Puntos de error
plt.axhline(y=0, color='red', linestyle='--')          # Línea de error cero
plt.title('Gráfico de Residuos: Regresión Lineal')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')

plt.savefig("output/ej2_residuos.png")
plt.close()
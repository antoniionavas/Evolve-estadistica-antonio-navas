import numpy as np
import pandas as pd

def media_evolve(lista_datos: list):
     return round(sum(lista_datos) / len(lista_datos),2)

def mediana_evolve(lista_datos: list):
    sorted_list = sorted(lista_datos)
    n = len(sorted_list)
    
    if n % 2 == 0:
        return (sorted_list[n//2 - 1] + sorted_list[n//2]) / 2
    else:
        return round(sorted_list[n//2],2)

def percentil_evolve(lista_datos: list, percentil: int):
    sorted_list = sorted(lista_datos)
    n = len(sorted_list)
    
    k = (n - 1) * (percentil / 100)
    f = int(k)
    c = f + 1

    if c >= n:
        return sorted_list[f]

    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return round(d0 + d1,2)

def varianza_evolve(lista_datos: list):
    media = sum(lista_datos) / len(lista_datos)
    n = len(lista_datos)
    
    suma = 0
    for x in lista_datos:
        suma += (x - media) ** 2
    
    return round(suma / (n - 1),2)

def desviacion_evolve(lista_datos: list):
    return round(varianza_evolve(lista_datos) ** 0.5, 2)

def IQR_evolve(lista_datos: list):
    qsup = percentil_evolve(lista_datos, 75)
    qinf = percentil_evolve(lista_datos, 25)
    return round(qsup - qinf,2)

def numero_outlier(lista_datos) -> int:
    q1 = percentil_evolve(lista_datos, 25)
    q3 = percentil_evolve(lista_datos, 75)

    iqr = q3 - q1

    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr

    count = 0
    for x in lista_datos:
        if x < limite_inf or x > limite_sup:
            count += 1

    return count


def skewness_evolve(lista_datos) -> float:
    n = len(lista_datos)
    media = media_evolve(lista_datos)

    desviacion = varianza_evolve(lista_datos) ** 0.5

    suma_cubos = 0
    for x in lista_datos:
        suma_cubos += (x - media) ** 3

    skewness = (suma_cubos / n) / (desviacion ** 3)

    return round(skewness, 4)

def kurtosis_evolve(lista_datos) -> float:
    n = len(lista_datos)
    media = media_evolve(lista_datos)

    desviacion = varianza_evolve(lista_datos) ** 0.5

    suma_cuartos = 0
    for x in lista_datos:
        suma_cuartos += (x - media) ** 4

    kurtosis = (suma_cuartos / n) / (desviacion ** 4)

    return round(kurtosis - 3, 4)


if __name__ == "__main__":

    np.random.seed(42)
    
    edad = list(np.random.randint(20, 60, 100))
    salario = list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))

    np.random.seed(42)
    
    df = pd.DataFrame({
        'edad': np.random.randint(20, 60, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })

    print("Resultado pandas")
    print("------------------")
    print(df.describe())

    print("\nResultado edad")
    print("------------------")
    print("Media:", media_evolve(edad))
    print("Mediana:", mediana_evolve(edad))
    print("Percentil 50:", percentil_evolve(edad, 50))
    print("Varianza:", varianza_evolve(edad))
    print("Desviación típica:", desviacion_evolve(edad))
    print("IQR:", IQR_evolve(edad))
    print("Outliers:", numero_outlier(edad))
    print("Skewness:", skewness_evolve(edad))
    print("Kurtosis:", kurtosis_evolve(edad))


    print("\nResultado salario")
    print("------------------")
    print("Media:", media_evolve(salario))
    print("Mediana:", mediana_evolve(salario))
    print("Percentil 50:", percentil_evolve(salario, 50))
    print("Varianza:", varianza_evolve(salario))
    print("Desviación típica:", desviacion_evolve(salario))
    print("IQR:", IQR_evolve(salario))
    print("Outliers:", numero_outlier(salario))
    print("Skewness:", skewness_evolve(salario))
    print("Kurtosis:", kurtosis_evolve(salario))


    print("\nResultado experiencia")
    print("------------------")
    print("Media:", media_evolve(experiencia))
    print("Mediana:", mediana_evolve(experiencia))
    print("Percentil 50:", percentil_evolve(experiencia, 50))
    print("Varianza:", varianza_evolve(experiencia))
    print("Desviación típica:", desviacion_evolve(experiencia))
    print("IQR:", IQR_evolve(experiencia))
    print("Outliers:", numero_outlier(experiencia))
    print("Skewness:", skewness_evolve(experiencia))
    print("Kurtosis:", kurtosis_evolve(experiencia))
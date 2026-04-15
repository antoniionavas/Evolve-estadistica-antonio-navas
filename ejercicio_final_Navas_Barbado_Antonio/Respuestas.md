# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:
Mi dataset llamado Diamonds tiene un total de 10 columnas, contando con 53940 filas y con un tamaño en memoria de 10.98 MB. Tiene diferentes tipos de datos como son: float, int y str. 
Tras el análisis descriptivo, se puede observar que el diamante tiene un peso de unos 0.8 quilates y un precio cercano a 3.900 $, pero la alta desviación estandar muestra que los valores varían mucho entre sí. 
Las variables price y carat tiene una distribución sesgada a la derecha, con muchos diamantes pequeños y pocos diamantes muy grandes o caros. Con los outliers se han detectado con el método IQR ya que es más adecuado para los datos no normales, estos han sido tratados con el método cliping para limitar los valores extremos sin eliminarlos. 

El análisis de las variables categóricas muestra un desbalance, sobre todo en el corte, donde predomina la categoría Ideal. También se ve que los diamantes de menor calidad son poco frecuentes, lo que indica que hay más productos de calidad media-alta en el dataset. 

El precio está muy influido por el peso del diamante (carat) y sus dimensiones. De hecho, estas variables están relacionadas entre sí, lo que puede causar problemas en modelos de predicción. Por eso, más adelante sería adecuado simplificarlas o quedarse solo con una para evitar errores. 
---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset se llama Diamonds y proviene de la web Kaggle. La variable objetivo es price. Tiene sentido hacer el análisis de regresión sobre esta variable porque es una variable numérica continua, por lo tanto, price (el precio) puede tomar cualquier valor. El objetivo de negocio está entorno al valor de mercadona de una piedra preciosa como son los diamantes basándose en sus diferentes características físicas (carat, dimensiones) y de calidad (corte, claridad, color).

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las principales variables numéricas son price y carat. Ambas variables tienen una distribución con asimetría positiva, esto indica que la mayoría de diamantes son pequeños y económicos. La minoría de estos tienen precios y pesos altos. Mediante el método de Boxplots se han encontrado outliers en las variables price, carat y en las dimensiones x,y,z. He decidido aplicar el método del Rango Intercuartílico (IQR) para detectar los límites ya que las variables numéricas, sobretodo el precio no sigue una distribución normal y están bastantes sesgadas. En el análisis descriptivo inicial se mantienen para observar la realidad del mercado, pero para el modelado futuro se debería hacer un clippeo o eliminar los registros donde x,y o z sean 0, ya que son errores de medición puesto que un diamante no puede tener una dimensión cero. 

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Analizando la matriz de correlación de Pearson, las tres variables con mayor impacto en el precio son: carat con un coeficiente de 0.92, x con un coeficiente de 0.88 y por último tanto y o z ya que ambas tienen de coeficiente 0.86. Teniendo en cuenta que estos coeficientes mencionados son aproximados.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Una vez comprobado si existen valores nulos, he detectado que en mi dataset Diamonds hay un 0 % de valores nulos en sus columnas. Por lo tanto, no ha sido necesario realizar ninguna técnica ni siquiera eliminar filas en valores nulos. Lo que sí se ha detectado son valores que se 

--- 

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> _Escribe aquí tu respuesta_


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> _Escribe aquí tu respuesta_

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

> _Escribe aquí tu respuesta_

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> _Escribe aquí tu respuesta_

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*

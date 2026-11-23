# Investigación y Desarrollo sobre Algoritmos de Clustering

**Módulo:** Data Analytics & Machine Learning  
**Tema:** Algoritmos no supervisados de Clustering  
**Tipo de entrega:** Individual

---

# Índice

1. [Aprendizaje no supervisado y clustering](#1-qué-es-el-aprendizaje-no-supervisado-y-qué-es-un-algoritmo-de-clustering)
2. [Clustering basado en centroides y clustering jerárquico](#2-diferencia-entre-clustering-basado-en-centroides-y-clustering-jerárquico)
3. [Clustering basado en densidad: DBSCAN](#3-en-qué-consiste-el-clustering-basado-en-densidad-como-dbscan)
4. [Elección del número de clusters y evaluación](#4-por-qué-es-importante-elegir-correctamente-el-número-de-clusters-o-los-hiperparámetros)
5. [Principales algoritmos de clustering](#5-explicación-de-los-principales-algoritmos-de-clustering)
6. [Aplicaciones en el mundo real](#6-aplicaciones-del-clustering-en-el-mundo-real)
7. [Conclusiones](#7-conclusiones)
8. [Referencias bibliográficas](#8-referencias-bibliográficas)

---

# 1. ¿Qué es el aprendizaje no supervisado y qué es un algoritmo de clustering?

## Aprendizaje no supervisado (Unsupervised Learning)

El aprendizaje no supervisado es una rama del Machine Learning en la que un algoritmo aprende a partir de datos que no tienen etiquetas ni respuestas previamente conocidas.

A diferencia del aprendizaje supervisado, no se proporciona al modelo una respuesta correcta para cada ejemplo. Su objetivo es analizar los datos, encontrar patrones, identificar relaciones y descubrir estructuras que no son evidentes a simple vista.

Por ejemplo, imaginemos una tienda online que dispone de información de miles de clientes, como:

- Edad.
- Frecuencia de compra.
- Importe medio de los pedidos.
- Productos que suelen comprar.
- Antigüedad como clientes.

La empresa no sabe de antemano qué tipos de clientes existen. Mediante aprendizaje no supervisado puede descubrir grupos de clientes con comportamientos similares.

## ¿Qué es un algoritmo de clustering?

El clustering, también conocido como agrupamiento, es una técnica de aprendizaje no supervisado que consiste en dividir un conjunto de datos en grupos llamados *clusters*.

Los elementos que pertenecen a un mismo grupo deben ser lo más similares posible entre sí, mientras que los elementos de grupos diferentes deben presentar características distintas.

### Objetivos principales del clustering

- Descubrir patrones ocultos en los datos.
- Agrupar elementos con características similares.
- Identificar comportamientos y perfiles.
- Detectar valores atípicos (*outliers*).
- Facilitar la toma de decisiones basada en datos.

## Diferencias entre aprendizaje supervisado y no supervisado

| Característica | Supervisado | No supervisado |
|---|---|---|
| Datos | Etiquetados | Sin etiquetar |
| Objetivo | Predecir resultados | Descubrir patrones |
| Respuestas correctas | Sí | No |
| Tareas habituales | Clasificación y regresión | Clustering y reducción de dimensionalidad |
| Ejemplo | Predecir si un cliente abandonará | Agrupar clientes por comportamiento |

**Ejemplo práctico:**

- **Aprendizaje supervisado:** entrenamos un modelo con correos identificados como spam o no spam para que aprenda a clasificar nuevos correos.
- **Aprendizaje no supervisado:** analizamos miles de correos sin etiquetas para descubrir grupos con características similares.

**Conclusión:** el aprendizaje supervisado aprende a partir de ejemplos con respuestas conocidas, mientras que el no supervisado descubre estructuras y relaciones sin disponer de respuestas previas.

---

# 2. Diferencia entre clustering basado en centroides y clustering jerárquico

Existen diferentes formas de agrupar datos. Dos de los enfoques más conocidos son el clustering basado en centroides y el clustering jerárquico.

## A. Clustering basado en centroides (particional)

Este método divide los datos en un número determinado de grupos. Cada grupo está representado por un punto central llamado *centroide*.

El algoritmo asigna cada observación al centroide más cercano, intentando que los elementos de cada grupo estén lo más próximos posible a su centro.

El algoritmo más representativo es **K-Means**.

### ¿Cómo funciona?

1. Se establece el número de clusters (K).
2. Se inicializan K centroides.
3. Cada punto se asigna al centroide más cercano.
4. Se recalcula la posición de cada centroide mediante la media de los puntos asignados.
5. Se repiten las asignaciones y actualizaciones hasta que los grupos se estabilizan o se alcanza el criterio de parada.

### Ventajas

- Es sencillo de implementar.
- Tiene un buen rendimiento con conjuntos de datos grandes.
- Es relativamente rápido.
- Produce grupos fáciles de interpretar cuando los datos tienen una estructura adecuada.

### Desventajas

- Hay que indicar previamente el número de clusters.
- Es sensible a los valores atípicos.
- Depende de la inicialización de los centroides.
- Funciona peor con grupos de formas irregulares o densidades muy diferentes.

## B. Clustering jerárquico (Aglomerativo)

El clustering jerárquico construye una estructura de agrupaciones organizadas en diferentes niveles, formando una especie de árbol llamado *dendrograma*.

A diferencia de K-Means, no necesita establecer obligatoriamente el número de clusters al principio.

En su versión aglomerativa, comienza considerando cada observación como un grupo independiente y va fusionando progresivamente los grupos más similares.

### ¿Cómo funciona?

1. Cada observación comienza como un cluster individual.
2. Se calcula la distancia entre los grupos.
3. Se fusionan los dos grupos más cercanos.
4. Se actualizan las distancias entre los grupos restantes.
5. El proceso continúa hasta que todos los datos forman un único grupo o se alcanza un criterio de parada.

La distancia entre grupos depende del método de enlace (*linkage*) utilizado:

- **Single linkage:** distancia mínima entre dos puntos de diferentes grupos.
- **Complete linkage:** distancia máxima entre dos puntos de diferentes grupos.
- **Average linkage:** distancia media entre los puntos de ambos grupos.
- **Ward linkage:** fusiona grupos intentando minimizar el incremento de la variabilidad interna.

### Ventajas

- Permite visualizar las relaciones entre los grupos.
- No requiere establecer K obligatoriamente desde el principio.
- Permite estudiar diferentes niveles de agrupamiento.
- Puede ser útil para descubrir estructuras jerárquicas.

### Desventajas

- Puede ser costoso computacionalmente.
- No es especialmente eficiente con grandes volúmenes de datos.
- Las fusiones realizadas normalmente no se pueden deshacer.
- Los resultados dependen de la métrica y del método de enlace seleccionados.

## Comparación conceptual

| Característica | Centroides | Jerárquico |
|---|---|---|
| Representante | Centroide | Estructura de árbol |
| Número de grupos | Se establece K | Se puede decidir posteriormente |
| Funcionamiento | Asignación y actualización | Fusión o división |
| Visualización | Grupos | Dendrograma |
| Algoritmo | K-Means | Agglomerative Clustering |

**Diferencia fundamental:** K-Means busca dividir los datos en K grupos alrededor de unos centros, mientras que el clustering jerárquico construye una estructura de relaciones entre observaciones y grupos.

---

# 3. ¿En qué consiste el clustering basado en densidad (como DBSCAN)?

El clustering basado en densidad es un enfoque que identifica grupos buscando regiones donde los datos están concentrados.

Su algoritmo más representativo es **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**.

A diferencia de K-Means, DBSCAN no busca centros geométricos. Busca zonas con una concentración suficiente de puntos y considera que los puntos aislados pueden ser ruido o valores atípicos.

## ¿Cómo funciona DBSCAN?

DBSCAN utiliza principalmente dos hiperparámetros:

- **Epsilon (eps):** establece el radio de distancia que define el vecindario de un punto.
- **Min_samples:** establece el número mínimo de puntos que debe haber dentro del vecindario para considerar que existe suficiente densidad.

El algoritmo clasifica los puntos en tres categorías:

1. **Puntos núcleo (Core Points):** tienen al menos el número mínimo de vecinos establecido dentro del radio Epsilon.
2. **Puntos frontera (Border Points):** no tienen suficientes vecinos para ser puntos núcleo, pero están dentro del vecindario de un punto núcleo.
3. **Puntos de ruido (Noise Points):** no pertenecen a ningún grupo porque no están suficientemente cerca de una región densa.

## Ejemplo práctico

Imaginemos que tenemos un mapa con la ubicación de miles de personas.

DBSCAN podría identificar:

- Una zona con muchas personas concentradas como un primer cluster.
- Otra zona de concentración como un segundo cluster.
- Personas aisladas como ruido.

## Ventajas

- No necesita especificar previamente el número de clusters.
- Puede detectar grupos de formas irregulares.
- Identifica automáticamente puntos de ruido.
- No depende de la posición de centroides.

## Desventajas

- Es sensible a la elección de Epsilon y Min_samples.
- Puede tener dificultades cuando existen grupos con densidades muy diferentes.
- La elección de Epsilon puede ser complicada en espacios de muchas dimensiones.
- Su rendimiento puede disminuir en conjuntos de datos de alta dimensionalidad.

## Diferencias entre los tres enfoques

| Característica | K-Means | Jerárquico | DBSCAN |
|---|---|---|---|
| Enfoque | Centroides | Distancias entre grupos | Densidad |
| Necesita K | Sí | No obligatoriamente | No |
| Detecta ruido | No directamente | No directamente | Sí |
| Grupos irregulares | Limitado | Depende del enlace | Sí |
| Parámetros principales | K | Distancia y linkage | Eps y Min_samples |

**Conclusión:** DBSCAN agrupa los datos según su densidad, no según la distancia a un centro. Esto permite detectar grupos irregulares y reconocer observaciones aisladas.

---

# 4. ¿Por qué es importante elegir correctamente el número de clusters o los hiperparámetros?

Elegir correctamente el número de clusters y los hiperparámetros es fundamental porque determina cómo se organizan los datos.

Una mala configuración puede producir agrupaciones que no representan correctamente la estructura real del conjunto de datos.

Por ejemplo:

- Si elegimos muy pocos clusters, podemos agrupar elementos que tienen comportamientos muy diferentes.
- Si elegimos demasiados clusters, podemos dividir artificialmente grupos que deberían permanecer juntos.
- Si seleccionamos un Epsilon inadecuado en DBSCAN, podemos unir grupos diferentes o considerar demasiados puntos como ruido.

Por eso necesitamos métodos que nos ayuden a evaluar la calidad de los agrupamientos.

## Método 1. Método del codo (Elbow Method)

El método del codo se utiliza principalmente para seleccionar el número adecuado de clusters en K-Means.

Consiste en ejecutar el algoritmo con diferentes valores de K y medir la variabilidad interna de los grupos mediante la inercia (*Inertia* o WCSS).

La inercia mide la suma de las distancias cuadráticas entre cada punto y el centroide de su cluster.

**Fórmula:**

\[
WCSS = \sum_{k=1}^{K}\sum_{x_i \in C_k}
\|x_i-\mu_k\|^2
\]

Donde:

- K: número de clusters.
- Ck: conjunto de puntos del cluster k.
- xi: observación.
- μk: centroide del cluster.

### ¿Cómo se interpreta?

1. Ejecutamos K-Means con diferentes valores de K.
2. Calculamos la inercia para cada valor.
3. Representamos los resultados en una gráfica.
4. Buscamos el punto donde la reducción de la inercia empieza a ser mucho menor.

Ese punto se conoce como el codo y puede indicar un número adecuado de clusters.

**Ventaja:** es fácil de aplicar y permite observar cómo mejora la compactación de los grupos al aumentar K.

**Limitación:** no siempre aparece un codo claramente definido y no garantiza que el número elegido sea el mejor desde el punto de vista de la utilidad real de los grupos.

## Método 2. Coeficiente de silueta (Silhouette Score)

El coeficiente de silueta mide lo bien que está asignado cada punto a su propio cluster en comparación con los clusters vecinos.

Tiene en cuenta dos aspectos:

- **Cohesión:** qué tan cerca está un punto de los demás elementos de su propio cluster.
- **Separación:** qué tan lejos está de los puntos de otros clusters.

Su valor está comprendido entre -1 y 1.

### Interpretación

| Valor | Interpretación |
|---|---|
| Cercano a 1 | Buena asignación y separación |
| Cercano a 0 | Punto situado entre dos clusters |
| Cercano a -1 | Posible asignación incorrecta |

**Fórmula:**

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

Donde:

- a(i): distancia media entre el punto y los elementos de su propio cluster.
- b(i): menor distancia media entre el punto y los elementos de otro cluster.

### Ventajas

- Permite evaluar la cohesión y separación simultáneamente.
- Facilita comparar diferentes agrupamientos.
- Su interpretación es relativamente sencilla.

### Desventajas

- Puede ser costoso en conjuntos de datos grandes.
- Puede favorecer estructuras de clusters compactas.
- Su interpretación depende de la métrica de distancia utilizada.

## Comparación de los métodos

| Método | Qué evalúa | Principal utilidad |
|---|---|---|
| Método del codo | Inercia o variabilidad interna | Seleccionar K en K-Means |
| Silhouette Score | Cohesión y separación | Evaluar y comparar agrupamientos |

**Conclusión:** no debemos elegir el número de clusters únicamente porque el algoritmo lo permita. Debemos evaluar si los grupos tienen sentido matemático y si son útiles para resolver el problema real.

---

# 5. Explicación de los principales algoritmos de clustering

## 5.1. K-Means

### ¿Qué es?

K-Means es un algoritmo de clustering basado en centroides que divide un conjunto de datos en K grupos.

Su objetivo es minimizar la distancia entre los puntos y el centroide de su grupo.

### Mecanismo de funcionamiento

1. Se establece el número de clusters K.
2. Se inicializan los centroides.
3. Se asigna cada punto al centroide más cercano.
4. Se recalculan los centroides.
5. Se repite el proceso hasta alcanzar la convergencia.

### Ventajas

- Fácil

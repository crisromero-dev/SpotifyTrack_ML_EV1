# Predicción de Popularidad de Canciones — Spotify Tracks

- **Asignatura:** MLY1101 — Machine Learning
- **Evaluación:** Parcial N°1 — Presentación y defensa técnica del proyecto
- **Caso:** C — Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)
- **Equipo:** 05 Cristian Romero y Rolando Paredes
- **Notebook:** `SpotifyTrack_ML_EV1.ipynb`

---

## Índice

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [KPIs del proyecto y criterio de éxito](#3-kpis-del-proyecto-y-criterio-de-éxito)
4. [Metodología: CRISP-DM](#4-metodología-crisp-dm)
5. [Fuente de datos y herramientas](#5-fuente-de-datos-y-herramientas)
6. [Análisis exploratorio de los datos (EDA)](#6-análisis-exploratorio-de-los-datos-eda)
7. [Preparación de los datos](#7-preparación-de-los-datos)
8. [Análisis de sesgos](#8-análisis-de-sesgos)
9. [Consideraciones de ética y privacidad](#9-consideraciones-de-ética-y-privacidad)
10. [Reproducibilidad y ejecución del notebook](#10-reproducibilidad-y-ejecución-del-notebook)
11. [Próximos pasos](#11-próximos-pasos)
12. [Referencias](#12-referencias)

---

## 1. Descripción del problema de negocio

Un sello discográfico, un distribuidor digital o un equipo de curaduría de playlists debe decidir
continuamente en qué canciones invertir recursos de promoción. Hoy esa decisión depende en buena
medida de la intuición de los equipos de A&R (*Artists and Repertoire*), lo que trae dos problemas
prácticos: es difícil de escalar a catálogos de miles de lanzamientos, y el criterio detrás de cada
decisión rara vez queda documentado.

El proyecto aborda la siguiente pregunta: **¿es posible estimar la popularidad que alcanzará una
canción en Spotify a partir de sus atributos musicales y sus metadatos?** Una estimación de este
tipo permitiría priorizar canciones en campañas de promoción, detectar canciones con potencial no
aprovechado, y entender qué atributos se asocian con el éxito dentro de cada género.

Se trata de un problema de **aprendizaje supervisado de regresión**: la variable objetivo
`popularity` es un entero de 0 a 100 calculado por Spotify a partir de las reproducciones
recientes, y se dispone de canciones históricas ya etiquetadas con ese valor. Se descartó formular
el problema como clasificación (por ejemplo, "popular" versus "no popular") porque eso obligaría a
fijar un umbral arbitrario y perdería la información que aporta la escala continua.

---

## 2. Objetivos del proyecto

### Objetivo general

Preparar un conjunto de datos confiable, sin fuga de información y documentado, que permita
entrenar un modelo de regresión para estimar la popularidad de una canción en Spotify a partir de
sus características musicales y metadatos.

### Objetivos específicos

1. Caracterizar el dataset e identificar sus problemas de calidad: nulos, duplicados, valores
   inválidos y outliers.
2. Identificar mediante EDA las variables con mayor relación con la popularidad y los patrones
   relevantes por género.
3. Construir un pipeline de preparación reproducible — limpieza, ingeniería de características,
   codificación, transformación y escalado — que evite la fuga de datos entre entrenamiento y
   prueba.
4. Evaluar los sesgos, las limitaciones y los riesgos éticos del uso de estos datos antes de pasar
   a la etapa de modelado.

### Alcance

Esta entrega cubre las fases de comprensión del negocio, comprensión de los datos, preparación de
los datos y evaluación de sesgos de CRISP-DM (fases 1, 2, 3 y 5). El entrenamiento y la
optimización de modelos —fase 4— quedan para la siguiente etapa del proyecto.

---

## 3. KPIs del proyecto y criterio de éxito

Los KPIs no se fijaron antes de conocer los datos: se calibraron con la evidencia que entregaron el
análisis exploratorio y la preparación de los datos, y por eso se presentan después de esas
secciones tanto en el notebook como en la presentación del grupo. Varios umbrales se expresan **en
relación con una línea base** — un modelo trivial que siempre predice la popularidad media del
conjunto de entrenamiento —, cuyo valor se calculó sobre el conjunto de prueba ya preparado.

| KPI | Qué mide | Umbral de aceptación |
|---|---|---|
| R² (test) | Proporción de la variabilidad de la popularidad explicada por el modelo | ≥ 0,40 |
| RMSE (test) | Error que penaliza con más fuerza los errores grandes | ≤ 77% del RMSE de la línea base |
| MAE (test) | Error absoluto medio, en puntos de popularidad | ≤ 80% del MAE de la línea base |
| Brecha de R² train–test | Control de sobreajuste | ≤ 0,05 |
| MAE en canciones de alta popularidad (≥ 70) | Calidad de la estimación en el segmento de mayor interés comercial | ≤ 20 puntos |
| Brecha de MAE entre cuartiles de género | Equidad: que el error no se concentre en los géneros menos populares | ≤ 5 puntos |

**Valores de la línea base** (referencia para los umbrales relativos, calculados sobre el 20% de
prueba): MAE 16,83, RMSE 20,15, y un error de 41,2 puntos en las canciones con popularidad ≥ 70. La
brecha de error entre el cuartil de géneros más populares y el menos popular es de 6,1 puntos.

### Qué mide cada métrica

- **MAE (*Mean Absolute Error*, error absoluto medio).** Promedia cuánto se equivoca el modelo en
  puntos de popularidad, sin importar si se equivoca por exceso o por defecto. Es la métrica más
  intuitiva de leer.
- **RMSE (*Root Mean Squared Error*, raíz del error cuadrático medio).** Eleva cada error al
  cuadrado antes de promediar y luego saca la raíz. Al elevar al cuadrado, un error grande pesa
  desproporcionadamente más que varios errores pequeños, así que el RMSE es más sensible a fallos
  puntuales severos que el MAE.
- **R² (coeficiente de determinación).** Indica qué proporción de la variación total de la
  popularidad logra explicar el modelo, en una escala de 0 a 1 (puede ser negativo si el modelo es
  peor que predecir siempre la media).
- **Equidad.** No es una métrica estándar de un curso de Machine Learning: mide si el error del
  modelo se reparte de forma pareja entre géneros populares y géneros de nicho, o si se concentra
  en perjudicar a unos más que a otros. Se incluye porque, como muestra la sección de sesgos, esa
  desigualdad ya existe en los datos antes de entrenar cualquier modelo.

### Fundamento de cada umbral

- **R² ≥ 0,40 y RMSE ≤ 77% de la línea base son la misma exigencia expresada de dos formas**
  (un R² de 0,40 equivale, en términos de error, a un RMSE de aproximadamente 77% de la línea
  base). El umbral se fija porque el género por sí solo ya explica el 32,6% de la varianza de la
  popularidad (sección de EDA): el modelo debe aportar información adicional a esa sola variable,
  sin exigir un nivel de ajuste irreal para un fenómeno que depende de factores que no están en los
  datos, como la fama previa del artista, el marketing o las playlists editoriales.
- **MAE ≤ 80% de la línea base** asegura que la mejora del modelo no dependa solo de acertar mejor
  en unos pocos casos extremos.
- **El KPI de alta popularidad** existe porque la línea base ya comete un error de más de 40 puntos
  en ese segmento: un error promedio aceptable puede esconder que el modelo falla justo donde el
  negocio toma decisiones de inversión.
- **El KPI de equidad** responde a que la línea base ya reparte su error de forma desigual entre
  grupos de géneros. Un modelo que no reduzca esa brecha heredaría y reforzaría el sesgo.

---

## 4. Metodología: CRISP-DM

| Fase | Aplicación en este proyecto |
|---|---|
| 1. Comprensión del negocio | Problema, objetivos y KPIs (secciones 1 a 3 de este informe) |
| 2. Comprensión de los datos | Glosario de variables, calidad de datos, EDA univariado y bivariado |
| 3. Preparación de los datos | Limpieza, deduplicación, ingeniería de características, división, codificación y escalado |
| 4. Modelado | Fuera del alcance de esta entrega; se detalla en Próximos pasos |
| 5. Evaluación | KPIs definidos con línea base real, y auditoría de sesgos de los datos |
| 6. Despliegue | Fuera del alcance |

CRISP-DM es iterativo, no estrictamente lineal: varias decisiones de la preparación de datos
surgieron de hallazgos del análisis exploratorio que obligaron a revisar de nuevo la calidad de los
datos, en particular la forma en que el dataset repite canciones entre géneros.

---

## 5. Fuente de datos y herramientas

### Fuente de datos

Se utiliza el **Spotify Tracks Dataset**, publicado en Kaggle por el usuario *maharshipandya*
(https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). El autor lo construyó
consultando la Web API de Spotify: para cada uno de 114 géneros musicales recuperó cerca de 1.000
canciones junto con sus metadatos y características de audio.

| Atributo | Valor |
|---|---|
| Registros | 114.000 filas |
| Columnas | 21 |
| Unidad de observación | Una canción recuperada al buscar un género |
| Variable objetivo | `popularity` (entero de 0 a 100) |
| Origen | Web API de Spotify |
| Formato | CSV |

Un detalle de construcción que condiciona todo el análisis: como las canciones se recuperaron por
género, una misma canción puede aparecer varias veces si pertenece a más de un género. Esto se
verifica y se corrige en las secciones 6 y 7 de este informe.

**Licencia y restricciones de uso.** Los términos de desarrollador de Spotify, vigentes desde mayo
de 2023, prohíben usar su contenido para entrenar modelos de Machine Learning. Este proyecto es
estrictamente académico; un uso comercial del modelo resultante requeriría revisar esa licencia o
llegar a un acuerdo con Spotify. Además, desde el 27 de noviembre de 2024 Spotify restringió el
acceso de nuevas aplicaciones a los endpoints de características de audio, por lo que hoy no es
posible obtener estas mismas variables para canciones nuevas a través de la API pública — una
limitación real para un eventual despliegue del modelo.

### Estructura general de las variables

| Grupo | Cantidad | Variables |
|---|---|---|
| Identificadores | 2 | `track_id`, `Unnamed: 0` (índice exportado por el autor del dataset) |
| Texto libre | 3 | `artists`, `album_name`, `track_name` |
| Variable objetivo | 1 | `popularity` |
| Numéricas continuas | 10 | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` |
| Discretas / binarias | 4 | `explicit`, `mode`, `key`, `time_signature` |
| Categórica nominal | 1 | `track_genre` (114 categorías) |

El notebook incluye además un glosario completo con la definición y el rango documentado de cada
una de las 21 columnas, tomado de la documentación de la Web API de Spotify.

### Herramientas utilizadas

| Herramienta | Uso |
|---|---|
| Google Colab | Desarrollo y ejecución del notebook. El archivo de datos se sube manualmente al entorno de ejecución; el notebook no depende de ninguna estructura de carpetas local |
| Python (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`) | Manipulación, visualización y preparación de los datos |
| Markdown | Este informe y la documentación dentro del propio notebook |

---

## 6. Análisis exploratorio de los datos (EDA)

### 6.1 Calidad de los datos: problemas que no eran evidentes

Una revisión superficial del archivo sugería un dataset casi perfecto: solo una fila con valores
nulos, y cero filas duplicadas según el chequeo estándar de `pandas`. Ambas cosas eran engañosas.

- **Duplicados ocultos.** El dataset trae una columna de índice exportado (`Unnamed: 0`) que, al
  ser distinta en cada fila, hace que ninguna fila sea nunca idéntica a otra según `duplicated()`.
  Al excluir esa columna aparecen **450 filas exactamente duplicadas**.
- **Canciones repetidas entre géneros.** Como el dataset se construyó buscando canciones por
  género, una misma canción puede quedar registrada varias veces bajo etiquetas distintas. Se
  identificaron **24.259 filas adicionales** que repiten una canción ya presente; 16.299 canciones
  aparecen en 2 a 9 géneros distintos, siempre con el mismo audio y solo cambiando la etiqueta de
  género. Dos etiquetas —`singer-songwriter` y `songwriter`— resultaron ser exactamente la misma
  categoría escrita de dos formas.
- **Versiones distintas de una misma canción.** Además de las repeticiones por género, 13.054
  canciones tienen más de una versión con distinto `track_id` (un single, una edición de álbum, un
  recopilatorio), con el mismo artista, título y —en la mayoría de los casos— la misma duración
  exacta.
- **Valores fuera de rango.** 157 filas tienen `tempo = 0` (el 88% pertenece al género `sleep`,
  contenido sin estructura musical), 973 filas tienen un compás fuera del rango documentado, 90
  filas tienen un volumen (`loudness`) positivo, y 876 filas tienen `speechiness` sobre 0,66 —el
  umbral que la propia documentación de Spotify usa para señalar contenido casi enteramente
  hablado—, de las cuales el 87% pertenece al género `comedy`.

Ninguno de estos problemas se corrige en esta sección: aquí solo se diagnostican. El tratamiento
completo está en la sección 7.

### 6.2 Variable objetivo: `popularity`

Con una fila por canción, la popularidad tiene media 33,2, mediana 33 y desviación estándar 20,6,
con una asimetría casi nula, por lo que no necesita ninguna transformación. El rasgo más llamativo
es un pico aislado en el valor 0, que concentra al 10,5% de las canciones y no sigue la forma
continua del resto de la distribución.

Al investigar ese grupo, se encontró que el 30% de esas canciones (2.867) tiene otra versión con
popularidad positiva — por ejemplo, *"Buddy Holly"* de Weezer tiene 75 puntos en su álbum original
y 0 en un disco recopilatorio, con exactamente la misma duración. Spotify concentra las
reproducciones en una sola versión y deja la otra en cero, aunque la canción sí sea popular. El 70%
restante de los ceros no tiene ninguna versión alternativa y representa canciones genuinamente poco
escuchadas — la "cola larga" real del catálogo.

### 6.3 Relación entre las variables de audio y la popularidad

Ninguna característica de audio tiene una correlación relevante con la popularidad: la más alta es
`instrumentalness`, con un coeficiente de Spearman de apenas −0,12. Sin embargo, al observar la
popularidad media por decil de cada variable —agrupando las canciones en diez tramos de igual
tamaño y calculando el promedio de popularidad de cada tramo— aparecen relaciones que una
correlación lineal no puede detectar: `energy`, `danceability` y la duración muestran una forma de
**U invertida**, donde los valores intermedios son más populares que los extremos. Esto explica por
qué, por ejemplo, `energy` tiene una correlación lineal prácticamente nula (−0,02) pese a que la
diferencia de popularidad entre sus deciles llega a 7 puntos.

Esta relación no lineal es la principal razón para preferir, en la siguiente etapa, modelos capaces
de capturar curvas —como árboles de decisión o ensambles— por sobre un modelo puramente lineal.

### 6.4 El género domina, con matices

Para medir cuánto explica una variable categórica como el género, se usó la razón de correlación
**η² (eta cuadrado)**, que no debe confundirse con una correlación de Pearson: mientras la
correlación mide relaciones lineales entre variables numéricas, η² compara qué tan distintos son
los promedios de popularidad entre categorías, respecto a la variación total de todas las
canciones. Se interpreta directamente como la proporción de la varianza explicada por esa variable.

`track_genre` obtuvo un η² de 0,326 — explica el 32,6% de las diferencias de popularidad, muy por
sobre `time_signature` (0,005), `explicit` (0,003), `key` (0,0006) y `mode` (0,0003). La popularidad
media varía de 2,2 (`iranian`) a 59,3 (`pop-film`) entre los 113 géneros, con un gradiente continuo
y no solo un par de casos extremos.

### 6.5 Outliers y multicolinealidad

Usando el criterio del rango intercuartílico se detectaron numerosos valores atípicos, pero al
investigar qué había detrás de cada uno se confirmó que corresponden a tipos de contenido reales,
no a errores de captura: los outliers de `instrumentalness` reflejan que la variable es bimodal
(canciones con y sin voz), los de `speechiness` son comedia, los de `liveness` son grabaciones en
vivo, y los de `duration_ms` son sets de DJ o piezas clásicas extensas.

Sobre multicolinealidad —cuando dos variables llevan prácticamente la misma información—, se usó el
**VIF** (*Variance Inflation Factor*, factor de inflación de varianza), que estima cuánto se puede
predecir cada variable a partir de las demás: si una variable se puede predecir casi perfectamente
con el resto, es redundante y su VIF sube. Un VIF sobre 5 suele considerarse preocupante y sobre 10,
severo. El máximo obtenido entre las variables originales fue 4,24 (`energy`), por lo que no se
detectó multicolinealidad severa.

---

## 7. Preparación de los datos

El orden de esta sección sigue una regla: **todo paso que no aprende parámetros de los datos se
aplica antes de dividir en entrenamiento y prueba; todo paso que sí aprende parámetros —codificar,
transformar, escalar— se ajusta únicamente con el conjunto de entrenamiento.** Esto evita la
llamada **fuga de datos** (*data leakage*): si información del conjunto de prueba se filtrara en la
preparación, el modelo luego parecería mejor de lo que realmente es, porque ya habría tenido
contacto indirecto con esos datos.

### 7.1 Limpieza y deduplicación

| Acción | Filas afectadas |
|---|---|
| Eliminar `Unnamed: 0` | 1 columna |
| Eliminar la fila con valores nulos | −1 |
| Eliminar duplicados exactos | −450 |
| Unificar `songwriter` → `singer-songwriter` | −1.000 |
| Dejar una fila por canción (`track_id`) | −22.809 |

El paso de dejar una sola fila por canción merece una explicación aparte. El archivo está ordenado
alfabéticamente por género, así que quedarse con la primera aparición de cada canción habría
favorecido sistemáticamente a los géneros que empiezan con las primeras letras del alfabeto (por
ejemplo, `reggaeton` habría quedado con solo 74 canciones). En su lugar, el género que se conserva
para cada canción se eligió al azar, con una semilla fija para que el resultado sea reproducible; la
popularidad final de cada canción es el promedio entre sus copias.

Con esto, el dataset pasa de 113.999 a 89.740 canciones.

### 7.2 Valores inválidos y outliers

| Caso | Decisión | Justificación |
|---|---|---|
| `tempo = 0` (157 filas) | Eliminar | 88% es el género `sleep`; no es música en sentido estricto |
| `time_signature = 0` (5 filas) | Imputar con la moda (4) | Muy pocos casos; el compás 4/4 concentra el 89% del dataset |
| `time_signature = 1` (973 filas) | Conservar | Géneros sin métrica regular; se resume en una variable binaria |
| `loudness > 0 dB` (90 filas) | Conservar | Masterizaciones muy comprimidas; físicamente posible |
| `speechiness > 0,66` (868 filas) | Eliminar | Umbral de Spotify para contenido casi enteramente hablado (87% es `comedy`) |
| `popularity = 0` con versión popular (2.856 filas) | Eliminar | El 0 es un artefacto de conteo, no refleja el interés real de los oyentes |
| Outliers detectados por IQR | Conservar | Variación legítima entre géneros; eliminarlos sesgaría el dataset |

Con esto, el dataset pasa de 89.740 a **85.859 canciones**.

### 7.3 Ingeniería de características

Se crearon cinco variables nuevas a partir de la información ya disponible:

| Variable | Cómo se calcula | Qué aporta |
|---|---|---|
| `duracion_min` | `duration_ms / 60000` | Reemplaza `duration_ms` en una unidad interpretable |
| `n_artistas` | Conteo de artistas separados por `;` | Las colaboraciones (2–3 artistas) tienen en promedio 2 puntos más de popularidad |
| `n_generos` | Número de géneros en que aparecía la canción antes de deduplicar | Las canciones en 2 géneros promedian 41,3 puntos, frente a 33,3 en las de un solo género |
| `es_instrumental` | `instrumentalness > 0,5` (umbral de Spotify) | Resume una variable bimodal; ser instrumental resta 7,9 puntos en promedio |
| `compas_4_4` | `time_signature == 4` | Resume 5 categorías dispersas en una binaria; aporta 4,5 puntos en promedio |

También se evaluó una variable para marcar grabaciones en vivo (`liveness > 0,8`), pero se descartó
porque su diferencia de popularidad era de solo 1 punto y no agregaba información nueva.

### 7.4 Selección de variables

Con las variables originales y las nuevas ya disponibles, se decidió qué conservar para el modelo:

| Variable | Decisión | Justificación |
|---|---|---|
| `track_id`, `album_name` | Excluir | Identificador y texto libre sin valor predictivo |
| `artists`, `track_name` | Excluir del modelo | Se usan solo para agrupar la división train/test |
| `duration_ms`, `time_signature` | Reemplazar | Por `duracion_min` y `compas_4_4` |
| `instrumentalness` | Excluir | VIF superior a 13 junto con `es_instrumental` (redundancia directa) |
| `key` | Excluir | η² = 0,0006; su codificación one-hot agregaría 11 columnas sin aporte |
| `mode`, `energy`, `loudness`, `acousticness` | Conservar | VIF por debajo de 5; cada una aporta un matiz distinto |

El VIF final, recalculado sobre el conjunto de 16 variables resultante, tiene un máximo de 4,34
(`energy`): la multicolinealidad quedó controlada sin sacrificar información relevante.

### 7.5 División en entrenamiento y prueba

La división se hizo 80/20, pero no con un muestreo aleatorio simple, sino con
**`GroupShuffleSplit`** de scikit-learn. A diferencia de una división aleatoria, que reparte cada
fila al azar sin ninguna restricción, `GroupShuffleSplit` permite definir grupos de filas que deben
quedar completos en un mismo lado de la partición. Aquí el grupo se definió como "artista + título
de la canción", porque —como se documentó en la sección 6.1— existen 13.054 canciones con más de
una versión bajo distinto `track_id`. Sin esa restricción, una versión de una canción podría caer en
entrenamiento y otra en prueba, y el modelo terminaría siendo evaluado con música que, en la
práctica, ya conocía.

La división resultó en 68.704 canciones de entrenamiento y 17.155 de prueba, sin ningún grupo
compartido entre ambos conjuntos, y con una distribución de `popularity` prácticamente idéntica en
ambos (media 34,32 en entrenamiento y 34,67 en prueba).

### 7.6 Codificación, transformación y escalado

Estas son las únicas técnicas de esta entrega que "aprenden" parámetros de los datos, y por eso se
ajustan exclusivamente con el conjunto de entrenamiento.

| Método | Variables | Por qué |
|---|---|---|
| Target Encoding (con *cross-fitting*, 5 bloques) | `track_genre` | 113 categorías; una codificación one-hot generaría demasiadas columnas dispersas |
| Yeo-Johnson (`PowerTransformer`) | `loudness`, `speechiness`, `liveness`, `n_generos`, `duracion_min`, `n_artistas` | Asimetría superior a 1 en entrenamiento; a diferencia del logaritmo, admite valores en cero |
| `StandardScaler` | 11 variables continuas | Las variables tienen escalas muy distintas (milisegundos, decibelios, BPM, proporciones) |
| Sin transformar (0/1) | `explicit`, `mode`, `es_instrumental`, `compas_4_4` | Ya están en una escala comparable y conservan su interpretación directa |

**Target Encoding** reemplaza cada categoría por un número derivado del objetivo — en este caso, la
popularidad promedio de las canciones de ese género —, en vez de crear una columna binaria por cada
una de las 113 categorías. El riesgo de este método es que, si el promedio de un género se calcula
incluyendo la propia fila que se está codificando, se produce fuga de datos dentro del propio
entrenamiento. Para evitarlo se usa *cross-fitting*: el conjunto de entrenamiento se divide en 5
bloques, y el promedio que recibe cada fila se calcula siempre con datos de los otros 4 bloques,
nunca con el suyo propio. La variable resultante, `genero_codificado`, terminó siendo la que mejor
se correlaciona con la popularidad de todo el conjunto (0,590).

**Yeo-Johnson** es una transformación que busca automáticamente la potencia matemática que hace más
simétrica una distribución muy sesgada, de forma similar a lo que logra un logaritmo. Se prefirió
sobre el logaritmo porque este último no está definido en cero, y varias de las variables tratadas
—como `speechiness`— tienen muchos valores en cero.

### 7.7 Resultado final

| Paso | Filas restantes |
|---|---|
| Archivo original | 114.000 |
| Tras limpieza y deduplicación | 89.740 |
| Tras tratar valores inválidos | 88.715 |
| Tras eliminar popularidad 0 artificial | **85.859** |

El dataset final conserva el 75,3% de las filas originales — la reducción corresponde casi en su
totalidad a filas repetidas, no a pérdida real de información. Queda dividido en 68.704 canciones de
entrenamiento y 17.155 de prueba, con 16 variables numéricas, sin valores nulos y sin fuga de datos.
Los objetos ajustados durante la preparación —el codificador de género, el transformador y el
escalador— quedan disponibles en memoria para aplicarse de la misma forma sobre datos nuevos en la
siguiente etapa.

---

## 8. Análisis de sesgos

El análisis se realiza sobre los datos, antes de entrenar cualquier modelo, para identificar qué
sesgos ya están presentes y serían heredados por cualquier algoritmo entrenado con ellos.

### 8.1 Muestreo artificial por género

El dataset tiene exactamente 1.000 canciones por género, una distribución que no representa el
catálogo real de Spotify, donde géneros masivos como el pop tienen muchísimas más canciones y
oyentes que géneros de nicho. Al deduplicar, los géneros de mayor alcance comercial resultaron ser
los más afectados, por compartir más canciones con otros géneros: `reggaeton` perdió el 69% de sus
canciones, `reggae` el 68%, `alternative` y `edm` el 66%. El género `comedy` perdió el 77%, pero por
una razón distinta: la mayoría de sus filas se eliminó por ser contenido hablado, no por
solaparse con otros géneros.

### 8.2 Efecto superestrella: concentración por artista

Aplicando el mismo método de η² usado para el género, pero sobre el artista principal, se obtuvo un
valor de 0,713. Como muchos artistas tienen una sola canción en el dataset, parte de esa cifra es
mecánica: recalculando η² sobre la popularidad ordenada al azar, el artista aún "explica" un 0,204
solo por casualidad. Descontando ese efecto, el exceso del artista sobre el azar es de 0,51, frente
a 0,35 del género — saber quién interpreta una canción dice más sobre su popularidad que su género o
su sonido. El modelo planteado no usa el artista como variable directa, pero existe el riesgo de que
las características de audio funcionen como una "huella" del estilo de artistas consolidados,
reforzando indirectamente este efecto.

### 8.3 Factores no observados y alcance de Spotify

Dos limitaciones conceptuales que no se pueden corregir con estos datos:

- **Factores no observados.** La popularidad real también depende de marketing, presencia en
  playlists editoriales y fama previa del artista fuera de la plataforma — nada de esto está en el
  dataset, lo que pone un techo realista a cuánto puede explicar cualquier modelo entrenado solo
  con audio y metadatos.
- **Alcance de Spotify.** La variable `popularity` mide el éxito de una canción dentro de Spotify,
  no su éxito musical real. La penetración de mercado de la plataforma varía mucho entre países, así
  que una canción puede ser muy popular en su país de origen y aun así tener un valor bajo, si ahí
  la gente escucha música por otros medios.

---

## 9. Consideraciones de ética y privacidad

**Privacidad.** El dataset no contiene información de usuarios ni de oyentes; la popularidad es un
indicador agregado. Sí contiene nombres de artistas, que son figuras públicas cuyos datos
corresponden a su actividad profesional en la plataforma. Aun así, un modelo que califique a
artistas individuales como "poco populares" puede afectar su reputación, por lo que sus resultados
no deberían publicarse por artista sin contexto adicional.

**Licencia de los datos.** Como ya se mencionó en la sección 5, los términos de desarrollador de
Spotify restringen usar su contenido para entrenar modelos de Machine Learning; este proyecto es
académico y no está pensado para uso comercial.

**Desigualdad entre mercados y artistas.** Por los sesgos descritos en la sección 8, un uso
automático del modelo perjudicaría sistemáticamente a la música no anglosajona, a los géneros de
nicho y a los artistas emergentes.

**Uso responsable de `explicit`.** La variable `explicit` se asocia con una popularidad más alta,
pero esa asociación no debe interpretarse como una recomendación de producción: que el contenido
explícito se asocie a más popularidad no significa que producirlo la genere.

**Explicabilidad.** El modelo debe usarse como herramienta de apoyo a los equipos de A&R, nunca como
una decisión automática. Sus predicciones deberían acompañarse siempre del error esperado por
género —el KPI de equidad definido en la sección 3— y de las variables que más influyeron en cada
estimación.

---

## 10. Reproducibilidad y ejecución del notebook

El notebook está pensado para ejecutarse directamente en Google Colab, sin depender de ninguna
estructura de carpetas: basta con subir el archivo `Spotify_Tracks_Dataset.csv` al entorno de
ejecución (por ejemplo, al panel de archivos de Colab) y correr todas las celdas en orden.

- **Semilla fija.** Todo paso aleatorio del notebook —la selección del género que se conserva por
  canción, la división train/test, el Target Encoding— usa una única semilla (`SEMILLA = 42`), por
  lo que volver a ejecutar el notebook produce exactamente los mismos resultados.
- **Documentación en el propio notebook.** Cada decisión de preparación está explicada en una celda
  Markdown junto al código que la implementa, y cada análisis va seguido de su interpretación.
- **Sin archivos intermedios.** El notebook no guarda ni exporta datos a disco; los conjuntos
  preparados y los objetos ajustados quedan en memoria durante la sesión, listos para usarse si se
  continúa con el modelado en el mismo notebook.
- **Compatibilidad.** El notebook usa `sklearn.preprocessing.TargetEncoder`, disponible desde
  scikit-learn 1.3 en adelante; fue verificado en Google Colab con las versiones de librerías
  disponibles por defecto en el entorno.

---

## 11. Próximos pasos

La siguiente etapa del proyecto es la fase de modelado de CRISP-DM: entrenar y comparar tres
modelos candidatos sobre el conjunto de datos preparado en esta entrega.

| Modelo | Categoría | Por qué se incluye |
|---|---|---|
| Regresión Lineal con regularización Ridge / Lasso | Referencia (*baseline* interpretable) | Sirve como punto de comparación transparente y reduce el sobreajuste ante variables correlacionadas entre sí, como `energy` y `loudness` |
| Random Forest | *Bagging* (ensamble de árboles) | Puede capturar las relaciones no lineales detectadas en el EDA —como la forma de U en `energy` y en la duración— sin necesitar que los datos sigan una distribución particular |
| XGBoost | *Boosting* (ensamble secuencial) | Suele ofrecer el mayor potencial predictivo entre los tres, especialmente con variables categóricas ya codificadas como `track_genre` |

Los tres modelos se evaluarán con la misma partición de datos, el mismo preprocesamiento y los
mismos KPIs definidos en la sección 3, usando validación cruzada con K-Fold (k = 5) para asegurar
que el resultado sea estable y no dependa de una sola forma de dividir los datos. Otras tareas
pendientes para esa etapa:

- Verificar si `n_generos` aporta al modelo o si conviene excluirla, dado su posible origen como
  artefacto del proceso de recolección (sección 8.1).
- Analizar los residuos del modelo por género y por tamaño de catálogo del artista, para confirmar
  si el KPI de equidad se cumple de forma consistente.
- Evaluar fuentes de datos complementarias —como la fecha de lanzamiento o los seguidores del
  artista— y características de audio calculadas con librerías abiertas (por ejemplo `librosa`),
  dado que la API de Spotify ya no entrega estas variables para canciones nuevas.

---

## 12. Referencias

- maharshipandya. *Spotify Tracks Dataset*. Kaggle.
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Spotify for Developers. *Web API Reference — Get Track's Audio Features*.
- Spotify for Developers. *Spotify Developer Terms* (vigentes desde mayo de 2023) y anuncio de
  cambios en la Web API (27 de noviembre de 2024).
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
- Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in
  classification and prediction problems*. ACM SIGKDD Explorations, 3(1).
- Yeo, I.-K. y Johnson, R. A. (2000). *A new family of power transformations to improve normality
  or symmetry*. Biometrika, 87(4).
- Documentación de scikit-learn: `TargetEncoder`, `PowerTransformer`, `StandardScaler`,
  `GroupShuffleSplit`.

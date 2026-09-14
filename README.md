# Informe Evaluación Parcial 1: Inteligencia Musical y Predicción de Popularidad de Canciones

 
**Caso:** Caso C — Inteligencia musical y predicción de popularidad de canciones  

---

## 1. Descripción del contexto / problema de negocio
En la industria discográfica y en las plataformas de streaming como Spotify, predecir el impacto o popularidad de un tema musical antes o durante su lanzamiento es crítico para la toma de decisiones estratégicas. Artistas, sellos discográficos y equipos de marketing invierten recursos considerables en producción y promoción sin tener certeza del retorno en reproducciones o alcance. 

El presente proyecto busca desarrollar un modelo analítico capaz de predecir el índice de popularidad de una canción (escalado entre 0 y 100) en función de sus atributos audiométricos (como *danceability*, *energy*, *loudness*, *acousticness*) y metadatos asociados (como género musical y duración). Esto permitirá a los equipos de producción optimizar estrategias de lanzamiento, segmentación en playlists y decisiones artísticas basadas en datos.

---

## 2. Objetivos del proyecto

### Objetivo General
Desarrollar, evaluar y seleccionar un modelo de aprendizaje supervisado de Regresión que prediga de forma precisa la métrica continua de popularidad (`popularity`) de las canciones en Spotify a partir de sus características musicales y categóricas.

### Objetivos Específicos
* Realizar un Análisis Exploratorio de Datos (EDA) exhaustivo para identificar patrones, distribuciones y relaciones entre las variables audiométricas y la popularidad.
* Ejecutar un pipeline de preparación y preprocesamiento de datos que contemple la limpieza de inconsistencias, imputación de nulos y codificación de variables categóricas.
* Implementar y evaluar 3 algoritmos de Machine Learning de Regresión.
* Analizar los posibles sesgos del conjunto de datos e identificar las consideraciones éticas y de privacidad aplicables al tratamiento de datos musicales y de artistas.

---

## 3. Definición inicial de criterio de salida para los algoritmos
El criterio de salida establece el marco metodológico para evaluar el éxito de los modelos de Regresión desarrollados:
* **Separación de Datos:** División estricta en subconjuntos de Entrenamiento (80%) y Prueba (20%) garantizando reproducibilidad (`random_state`).
* **Validación Cruzada:** Evaluación de desempeño mediante *K-Fold Cross-Validation* ($k=5$) sobre el conjunto de entrenamiento para evitar el sobreajuste (*overfitting*).
* **Cumplimiento de Métricas:** Un algoritmo se considerará preliminarmente válido solo si supera los umbrales de error y capacidad explicativa fijados por los estándares de la industria.

---

## 4. Selección inicial de algoritmo (máximo 3)
Dado que la variable objetivo `popularity` es continua (rango de 0 a 100), se seleccionaron 3 algoritmos de Regresión complementarios en complejidad y arquitectura:

1. **Regresión Lineal Múltiple con Regularización Ridge / Lasso:**
   * *Justificación:* Sirve como modelo de línea base (*baseline*). Permite evaluar relaciones lineales directas entre los atributos audiométricos y previene el sobreajuste mediante la penalización de coeficientes ante la presencia de alta multicolinealidad.
2. **Random Forest Regressor (Ensamble por Bagging):**
   * *Justificación:* Modela eficazmente relaciones no lineales y complejas interacciones entre variables (ej. la combinación de *energy* y *loudness* en distintos géneros) sin requerir supuestos de normalidad estricta.
3. **XGBoost Regressor (Extreme Gradient Boosting):**
   * *Justificación:* Algoritmo de ensamble basado en árboles con *boosting* secuencial. Es altamente eficiente, optimiza el manejo de datos de alta dimensionalidad (tras la codificación de géneros) y suele ofrecer el mejor rendimiento predictivo en la industria para datos estructurados.

---

## 5. Determinación y justificación de las métricas seleccionadas
Para evaluar el rendimiento del modelo de regresión se utilizarán 3 métricas complementarias:

* **RMSE (Root Mean Squared Error / Raíz del Error Cuadrático Medio):**
   * *Justificación:* Es la métrica principal. Penaliza con mayor severidad los errores grandes (desviaciones drásticas en popularidad) y se expresa en la misma unidad que la variable objetivo (puntos de popularidad de 0 a 100).
* **MAE (Mean Absolute Error / Error Absoluto Medio):**
   * *Justificación:* Proporciona una medida lineal directa del error promedio en la predicción de puntos de popularidad, siendo menos sensible a valores atípicos (*outliers*) que el RMSE.
* **Coeficiente de Determinación ($R^2$):**
   * *Justificación:* Mide la proporción de la varianza total de la popularidad explicada por las variables del modelo. Permite comparar el desempeño global del modelo respecto a una predicción trivial media.

---

## 6. Criterio de Salida: Umbral de métricas seleccionadas
De acuerdo con los estándares de la industria para datos comportamentales y de entrenamiento musical (donde existe un alto grado de variabilidad estocástica y factores externos no registrados), se definen los siguientes umbrales de validez:

* **$R^2 \ge 0.55$:** El modelo debe explicar al menos el 55% de la variabilidad de la popularidad de las canciones.
* **$RMSE \le 14.0$:** El error cuadrático medio no debe superar los 14 puntos de popularidad dentro del rango de 0 a 100.
* **$MAE \le 10.0$:** En promedio, la desviación absoluta de la predicción respecto a la popularidad real no debe sobrepasar los 10 puntos.

---

## 7. Descripción de las fuentes de datos utilizadas
El dataset procesado consta de **114,000 registros** y **21 columnas** provenientes de la API de Spotify:

* **Variable Objetivo:** `popularity` (entero de 0 a 100, media de 33.24 y desviación estándar de 22.31).
* **Variables Audiométricas / Físicas (Continuas/Discretas):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `key`, `mode`, `time_signature`.
* **Variables Categóricas y Descriptivas:** `artists` (31,437 únicos), `album_name` (46,589 únicos), `track_name` (73,608 únicos), `track_genre` (114 géneros únicos con 1,000 registros cada uno), `track_id`.
* **Variable Booleana:** `explicit` (`False`: 104,253; `True`: 9,747).

---

## 8. Análisis Exploratorio de los Datos (EDA)
De acuerdo con la exploración realizada en el notebook `.ipynb`:

* **Calidad de Datos:** No existen registros duplicados. Se detectó únicamente 1 valor faltante en las columnas `artists`, `album_name` y `track_name`, lo que demuestra una calidad de captura casi óptima.
* **Balance de Géneros:** El dataset presenta una distribución perfectamente balanceada en términos de categorías musicales, incluyendo **114 géneros** con exactamente **1,000 canciones** registradas por género.
* **Comportamiento de la Popularidad:** La distribución de la popularidad presenta una mediana de `35.0` y un rango intercuartílico de `17.0` a `50.0`, observándose un sesgo hacia valores bajos debido a un número significativo de canciones con popularidad igual a `0`.
* **Concentración de Artistas:** Aunque existen más de 31,000 artistas únicos, se detectan concentraciones puntuales como *The Beatles* con 279 apariciones.

---

## 9. Preparación inicial de los Datos
Para garantizar la calidad del entrenamiento de los modelos, el plan de preprocesamiento incluye:

1. **Limpieza de Nulos:** Eliminación o imputación directa de la única fila con valores faltantes.
2. **Filtrado de Outliers:** Tratamiento de duraciones extremas inusuales (`duration_ms` iguales a 0 o superiores a 80 minutos) y temas con `tempo` igual a 0 BPM.
3. **Ingeniería de Características y Codificación:**
   * Aplicación de **One-Hot Encoding** o **Target Encoding** sobre los 114 géneros musicales (`track_genre`).
   * Conversión de la variable booleana `explicit` a formato binario (0/1).
4. **Escalado de Características:** Transformación de variables continuas (`duration_ms`, `loudness`, `tempo`) mediante *StandardScaler* o *RobustScaler* para estabilizar los algoritmos basados en gradientes y distancia.

---

## 10. Análisis de Sesgos
* **Sesgo por Factores de Causalidad No Observada:** La popularidad real de una canción está fuertemente influenciada por factores externos al archivo de audio (presupuesto de marketing, viralidad en redes sociales como TikTok, popularidad previa del artista). Un modelo basado solo en métricas de audio puede subestimar estos fenómenos exógenos, ya que ninguno de ellos está registrado en el dataset.
* **Sesgo de Alcance Limitado a Spotify:** El dataset solo contiene canciones que efectivamente fueron publicadas y distribuidas en Spotify, dejando fuera música no distribuida digitalmente, lanzamientos independientes sin llegada a plataformas de streaming, o contenido retirado del catálogo. Esto implica que el modelo no aprende qué hace popular a una canción en términos absolutos, sino qué hace popular a una canción **dentro del universo de música que ya logró entrar a Spotify** — su capacidad de generalizar fuera de ese universo es limitada.

---

## 11. Consideraciones de Ética y Privacidad
* **Privacidad de Datos y Artistas:** Los datos analizados corresponden a metadatos y métricas públicas expuestas por la API web de Spotify, por lo que no comprometen la privacidad ni contienen datos personales identificables (PII) de oyentes o usuarios. Sin embargo, la columna `artists` sí identifica a personas reales (31,437 artistas únicos); aunque sea información pública, se excluye explícitamente como variable predictiva para evitar que el modelo asocie la popularidad a la identidad del artista en lugar de a las características musicales del track.
* **Riesgo de Uso Indebido (Dual-Use):** Un modelo de este tipo puede usarse legítimamente como apoyo a decisiones de promoción, pero también podría emplearse para filtrar automáticamente qué artistas fichar o promocionar, replicando y amplificando los sesgos de género y mercado ya identificados sin intervención humana. Se recomienda que el modelo se utilice únicamente como herramienta de apoyo a la decisión, no como filtro automático de decisiones artísticas.
* **Explicabilidad como Responsabilidad Ética:** Dado que las predicciones pueden influir en decisiones que afectan el sustento de un artista, la elección del modelo final debería sopesar precisión contra interpretabilidad: un modelo como Regresión Lineal/Ridge es transparente en sus coeficientes, mientras que XGBoost, pese a ofrecer mayor precisión, es más difícil de auditar. Esta tensión debe considerarse explícitamente al seleccionar el modelo de producción.


---

## 12. Directrices de Manejo del Dataset y Precauciones Técnicas

Para asegurar que la manipulación de los datos no introduzca sesgos técnicos ni errores en el modelado, se establecen las siguientes directrices y resguardos:

### Precauciones y Buenas Prácticas de Data Leakage
* **Prevención de Data Leakage (Fuga de Datos):** Todo el proceso de transformación (escalado con `StandardScaler`, imputación de valores faltantes y codificación de variables categóricas) **debe ajustarse (*fit*) exclusivamente con el subconjunto de entrenamiento (Train)** y solo aplicarse (*transform*) sobre el conjunto de prueba (Test).
* **Filtro de Metadatos No Predictivos:** Variables de identificación única como `track_id` y `Unnamed: 0` deben eliminarse antes del modelado, ya que no aportan valor explicativo y podrían causar sobreajuste.

### Protocolo de Tratamiento de Anomalías (Outliers y Inconsistencias)
* **Valores Nulos:** Dado que solo existe 1 fila con valores faltantes en `artists`, `album_name` y `track_name`, la precaución estándar es eliminar esa fila directamente para no comprometer el código con imputaciones complejas innecesarias.
* **Canciones con Duración Anormal (`duration_ms`):** Filtrar o revisar registros con duraciones extremadamente cortas ($< 30.000\text{ ms}$) o excesivamente largas ($> 15\text{ minutos}$), ya que corresponden a podcasts, fragmentos incompletos o pistas de audio ambiental que distorsionan el aprendizaje del algoritmo.
* **Inconsistencias en Ritmo (`tempo = 0 BPM`):** Se deben imputar o remover los registros donde el tempo es exactamente 0.0 BPM, pues representan errores de captura de la API o pistas sin estructura rítmica clara.

### Precaución con la Variable Categórica `track_genre`
* **Alta Dimensionalidad:** La variable `track_genre` contiene 114 categorías. Al aplicar *One-Hot Encoding*, el dataset se expandirá sustancialmente en número de columnas. Se recomienda evaluar técnicas de reducción de dimensionalidad o *Target Encoding* si el tiempo de entrenamiento de modelos como *Random Forest* o *XGBoost* se eleva drásticamente.

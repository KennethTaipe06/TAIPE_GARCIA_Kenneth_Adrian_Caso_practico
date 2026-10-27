# Caso 1 — Detección de transacciones fraudulentas

Contenido del repositorio. La explicación del problema, la metodología y los resultados está en `informe/`.

## Estructura

```
TAIPE_GARCIA_Kenneth_Adrian_Caso_practico/
├── 01_EDA.ipynb … 06_evaluacion.ipynb   notebooks, en orden de ejecución
├── comun/                               código compartido por los notebooks
├── data/                                datos de entrada y datos preparados
├── resultados/                          modelos entrenados, métricas y predicciones
├── figuras/                             gráficas del análisis, de los modelos y del informe
├── informe/                             entregables: informe y presentación
├── app/                                 probador web de los modelos
├── Dockerfile                           imagen del probador web
└── compose.yaml                         levanta el probador con Docker Compose
```

## Notebooks

| Notebook | Contenido |
|---|---|
| `01_EDA.ipynb` | Carga, calidad de datos, análisis exploratorio, selección de variables y partición 80/20. Genera `data/dataset_preparado.csv`. |
| `02_regresion_logistica.ipynb` | Regresión Logística: búsqueda de hiperparámetros con validación cruzada. |
| `03_arbol_decision.ipynb` | Árbol de Decisión. |
| `04_random_forest.ipynb` | Random Forest. |
| `05_gradient_boosting.ipynb` | Gradient Boosting. |
| `06_evaluacion.ipynb` | Comparación de modelos, umbral, evaluación única en test, análisis de errores y conclusiones. |

Los notebooks 02 a 05 guardan su modelo y sus métricas en `resultados/`; el 06 los lee de ahí.

## Carpetas

**`comun/`**
- `comun.py`: rutas, semilla, listas de variables, particiones de validación cruzada, métricas, carga de train/test, preprocesamiento (`make_prep`) y funciones para guardar figuras y resultados.

**`data/`**
- `transacciones_fraude_bigdata.csv`: datos originales del caso.
- `dataset_preparado.csv`: datos sin `customer_id` ni variables redundantes, con la columna `conjunto` (`train` / `test`). Conserva `transaction_id` solo para rastrear errores; no es predictor.

**`resultados/`** (un archivo de cada tipo por modelo)
- `*.joblib`: modelo final entrenado (pipeline completo).
- `cv_*.csv`: métricas de validación cruzada (media y desviación).
- `oof_*.csv`: predicciones fuera de partición sobre train (`y`, `proba`).

**`figuras/`** (prefijo según su origen)
- `02_`–`09_`: análisis exploratorio (`01_EDA`).
- `lr_`, `arbol_`, `rf_`, `gb_`: cada modelo (notebooks 02–05).
- `eval_`: evaluación final (`06_evaluacion`).
- `exp_arbol_`: experimentos adicionales del Árbol de Decisión (generadas fuera de los notebooks).
- `informe_`: figuras hechas para el informe y la presentación (generadas fuera de los notebooks).

**`informe/`**
- `Informe_Caso1_Deteccion_Fraude.docx`: informe técnico.
- `Presentacion_Caso1_Deteccion_Fraude.pptx`: presentación para la exposición.

**`app/`** (probador web de los 4 modelos)
- `servidor.py`: servidor con la biblioteca estándar de Python (`http.server`); carga los `.joblib` de `resultados/`.
- `static/index.html`: página para elegir un modelo, cargar un caso de prueba o escribir los datos, y ver la probabilidad de fraude y la decisión.
- `datos/modelos.json`: por modelo, su umbral de decisión (el que maximiza F1 en train), métricas de validación cruzada y 5 casos reales del conjunto de prueba (2 fraudes detectados, 1 no detectado, 1 falsa alarma, 1 legítima correcta).
- `generar_casos.py`: regenera `datos/modelos.json` (`python app/generar_casos.py`).
- `requirements.txt`: mismas versiones con las que se entrenaron los modelos.

## Requisitos

**Probador web con Docker**
- Docker Desktop encendido y unos 700 MB libres (la imagen ocupa 650 MB).
- Puerto 8000 libre; si está ocupado, cambiarlo en `compose.yaml` (por ejemplo `"8080:8000"`).
- Internet solo para construir la imagen la primera vez; después funciona sin conexión.
- Sin GPU. En uso consume unos 110 MB de RAM y predice con los 4 modelos en unos 35 ms.

**Probador web sin Docker**
- Python 3.13 con las versiones exactas de `app/requirements.txt` (numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0).
- La versión de scikit-learn debe ser exactamente 1.9.1: los `.joblib` se guardaron con ella y con otra pueden no cargar o dar resultados distintos.

**Notebooks (volver a entrenar los modelos)**
- Python 3.13 y Jupyter.
- pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2 y seaborn 0.13.2.
- imbalanced-learn 0.14.2, solo para los experimentos con SMOTE.
- Solo CPU (no se usa GPU). Los datos ocupan unos 5 MB; un equipo con 8 GB de RAM es suficiente (estimación, no medida).
- El notebook más lento es `05_gradient_boosting.ipynb` por la búsqueda de hiperparámetros (varios minutos).

## Probador web

Con Docker Compose, desde la raíz del repositorio:

```
docker compose up -d --build
```

Se abre en http://localhost:8000. Para detenerlo: `docker compose down`. Sin Docker: `python app/servidor.py`.

## Reproducción

Ejecutar los notebooks en orden (01 → 06). Semilla fija: `RANDOM_STATE = 42`. Los notebooks regeneran `data/dataset_preparado.csv`, `resultados/` y las figuras con prefijo de notebook o modelo. Si se vuelven a entrenar los modelos, ejecutar después `python app/generar_casos.py` para actualizar los umbrales y casos del probador web.

"""Genera app/datos/modelos.json: por cada modelo, su umbral, métricas y 5 casos reales del conjunto de prueba.

Se ejecuta una vez desde la raíz del proyecto:  python app/generar_casos.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

RAIZ = Path(__file__).resolve().parent.parent
RES, DATA, SALIDA = RAIZ / "resultados", RAIZ / "data", Path(__file__).resolve().parent / "datos"

MODELOS = {
    "gradient_boosting": {
        "nombre": "Gradient Boosting", "puesto": 1,
        "config": "HistGradientBoosting · árboles de profundidad 1 (tocones) · tasa 0.03 · 400 iteraciones · hoja mínima 50 · peso del fraude ×10",
        "metricas": {"PR-AUC": 0.062, "ROC-AUC": 0.706, "Precision": 0.084, "Recall": 0.147, "F1": 0.107, "Fraudes en el top 10%": 0.318},
    },
    "random_forest": {
        "nombre": "Random Forest", "puesto": 2,
        "config": "300 árboles de profundidad 3 · 5% de las variables por corte · hoja mínima 50 · peso del fraude ×10",
        "metricas": {"PR-AUC": 0.057, "ROC-AUC": 0.701, "Precision": 0.078, "Recall": 0.154, "F1": 0.103, "Fraudes en el top 10%": 0.281},
    },
    "regresion_logistica": {
        "nombre": "Regresión Logística", "puesto": 3,
        "config": "C = 0.001 (regularización L2 fuerte) · pesos balanceados · logaritmo y escalado de las numéricas",
        "metricas": {"PR-AUC": 0.051, "ROC-AUC": 0.681, "Precision": 0.063, "Recall": 0.216, "F1": 0.097, "Fraudes en el top 10%": 0.277},
    },
    "arbol_decision": {
        "nombre": "Árbol de Decisión", "puesto": 4,
        "config": "Profundidad 4 · hoja mínima 50 · criterio Gini · peso del fraude ×10",
        "metricas": {"PR-AUC": 0.040, "ROC-AUC": 0.672, "Precision": 0.079, "Recall": 0.120, "F1": 0.095, "Fraudes en el top 10%": 0.255},
    },
}
TIPOS = {"VP": "Fraude real · el modelo lo detecta", "FN": "Fraude real · el modelo no lo detecta",
         "FP": "Transacción legítima · falsa alarma", "VN": "Transacción legítima · el modelo acierta"}


def umbral_f1(oof):
    """Mismo criterio de los notebooks: umbral que maximiza F1 en las predicciones fuera de partición de train."""
    p, r, u = precision_recall_curve(oof["y"], oof["proba"])
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(u[f1[:-1].argmax()])


def mediana(idx, proba):
    orden = idx[np.argsort(proba[idx])]
    return orden[len(orden) // 2]


test = pd.read_csv(DATA / "dataset_preparado.csv").query("conjunto == 'test'").reset_index(drop=True)
assert len(test) == 10_000 and test["fraud"].sum() == 182, "el conjunto de prueba no es el esperado"
salida = {}
for clave, meta in MODELOS.items():
    modelo = joblib.load(RES / f"{clave}.joblib")
    umbral = umbral_f1(pd.read_csv(RES / f"oof_{clave}.csv"))
    X = test[list(modelo.feature_names_in_)]
    proba = modelo.predict_proba(X)[:, 1]
    pred, y = proba >= umbral, test["fraud"].to_numpy() == 1
    grupos = {"VP": np.where(pred & y)[0], "FN": np.where(~pred & y)[0], "FP": np.where(pred & ~y)[0], "VN": np.where(~pred & ~y)[0]}
    elegidos = [("VP", grupos["VP"][proba[grupos["VP"]].argmax()]), ("VP", mediana(grupos["VP"], proba)),
                ("FN", mediana(grupos["FN"], proba)), ("FP", grupos["FP"][proba[grupos["FP"]].argmax()]),
                ("VN", mediana(grupos["VN"], proba))]
    casos = []
    for tipo, i in elegidos:
        fila = X.iloc[i].to_dict()
        casos.append({"id": test.loc[i, "transaction_id"], "tipo": tipo, "descripcion": TIPOS[tipo],
                      "real": int(test.loc[i, "fraud"]), "proba": round(float(proba[i]), 4),
                      "datos": {k: (v.item() if hasattr(v, "item") else v) for k, v in fila.items()}})
    salida[clave] = {**meta, "umbral": round(umbral, 4), "casos": casos}
    print(f"{meta['nombre']:<20} umbral {umbral:.3f} | casos: " + ", ".join(f"{c['tipo']} {c['proba']:.3f}" for c in casos))

SALIDA.mkdir(exist_ok=True)
(SALIDA / "modelos.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print("guardado:", SALIDA / "modelos.json")

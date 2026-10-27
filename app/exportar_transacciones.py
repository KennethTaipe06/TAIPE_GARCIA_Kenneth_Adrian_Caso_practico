"""Exporta a app/datos/transacciones_prueba.txt transacciones REALES del conjunto de prueba (no generadas),
mezclando fraudes y legítimas, con los valores tal como se escriben en el probador web.

Se ejecuta desde la raíz del proyecto:  python app/exportar_transacciones.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
APP = Path(__file__).resolve().parent
META = json.loads((APP / "datos" / "modelos.json").read_text(encoding="utf-8"))
YA_USADOS = {c["id"] for m in META.values() for c in m["casos"]}
CAMPOS = [("amount_usd", "Monto de la transacción (USD)"), ("amount_vs_avg", "Monto ÷ promedio de 30 días (veces)"),
          ("distance_home_km", "Distancia a la ubicación habitual (km)"), ("ip_risk_score", "Riesgo de la IP (0–100)"),
          ("hour", "Hora (0–23)"), ("day_of_week", "Día de la semana (0–6)"), ("transactions_24h", "Transacciones del cliente en 24 h"),
          ("account_age_days", "Antigüedad de la cuenta (días)"), ("failed_attempts_24h", "Intentos fallidos en 24 h"),
          ("channel", "Canal"), ("merchant_category", "Categoría del comercio"), ("country", "País"),
          ("device_type", "Tipo de dispositivo"), ("card_present", "Tarjeta presente"), ("new_device", "Dispositivo nuevo")]

test = pd.read_csv(RAIZ / "data" / "dataset_preparado.csv").query("conjunto == 'test'").reset_index(drop=True)
test = test[~test["transaction_id"].isin(YA_USADOS)].reset_index(drop=True)
X = test[[c for c, _ in CAMPOS]]
pred = {}
for clave, m in META.items():
    proba = joblib.load(RAIZ / "resultados" / f"{clave}.joblib").predict_proba(X)[:, 1]
    pred[clave] = (proba, proba >= m["umbral"])

gb = pred["gradient_boosting"][1]
fraude = test["fraud"].to_numpy() == 1
rng = np.random.default_rng(42)
grupos = [("fraude", fraude & gb, 3), ("fraude", fraude & ~gb, 2), ("legítima", ~fraude & gb, 2), ("legítima", ~fraude & ~gb, 3)]
elegidas = np.concatenate([rng.choice(np.where(mask)[0], n, replace=False) for _, mask, n in grupos])
rng.shuffle(elegidas)

lineas = ["TRANSACCIONES REALES PARA PROBAR LOS MODELOS",
          "Origen: data/dataset_preparado.csv, conjunto de prueba (10,000 transacciones que los modelos no vieron al entrenar).",
          "Son filas reales del dataset del caso; ningún valor fue generado. 5 fraudes y 5 legítimas, en orden mezclado.",
          "Cópielas en el probador (http://localhost:8000) y compare la predicción con la etiqueta real.",
          "Tarjeta presente / Dispositivo nuevo: 1 = Sí, 0 = No.", ""]
for n, i in enumerate(elegidas, 1):
    fila = test.loc[i]
    lineas += ["=" * 60, f"TRANSACCIÓN {n}   ({fila['transaction_id']})", "=" * 60]
    for var, etiqueta in CAMPOS:
        lineas.append(f"{etiqueta:<40} {var:<20} {fila[var]}")
    lineas.append(f"{'ETIQUETA REAL':<40} {'fraud':<20} {int(fila['fraud'])}  ({'FRAUDE' if fila['fraud'] == 1 else 'LEGÍTIMA'})")
    lineas.append("")

lineas += ["=" * 60, "RESULTADO DE CADA MODELO (para verificar en el probador)", "=" * 60,
           f"{'N.º':<5}{'Real':<10}" + "".join(f"{META[c]['nombre']:<22}" for c in META)]
for n, i in enumerate(elegidas, 1):
    # mismo redondeo que el probador (4 decimales y luego porcentaje con 1 decimal)
    celdas = "".join(f"{round(float(pred[c][0][i]), 4) * 100:.1f}% {'ALERTA' if pred[c][1][i] else 'no':<10}".ljust(22) for c in META)
    lineas.append(f"{n:<5}{'fraude' if fraude[i] else 'legítima':<10}{celdas}")
lineas += ["", "Umbrales: " + " · ".join(f"{m['nombre']} {m['umbral']:.3f}" for m in META.values())]

salida = APP / "datos" / "transacciones_prueba.txt"
salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
print(salida, "|", len(elegidas), "transacciones |", int(fraude[elegidas].sum()), "fraudes")

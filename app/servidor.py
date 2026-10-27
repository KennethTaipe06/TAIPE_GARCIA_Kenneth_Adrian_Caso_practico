"""Servidor de demostración para probar los 4 modelos de detección de fraude.

Solo biblioteca estándar (http.server) + las librerías que ya usan los modelos.
  GET  /               página de prueba (static/index.html)
  GET  /api/modelos    modelos, umbrales, métricas, casos de prueba y campos del formulario
  POST /api/predecir   {"modelo": "<clave>" | "todos", "datos": {15 variables}}
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd

APP = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTADOS", APP.parent / "resultados"))
PUERTO = int(os.environ.get("PORT", "8000"))

META = json.loads((APP / "datos" / "modelos.json").read_text(encoding="utf-8"))
MODELOS = {clave: joblib.load(RES / f"{clave}.joblib") for clave in META}
COLUMNAS = list(next(iter(MODELOS.values())).feature_names_in_)

# categorías conocidas por los modelos (las del One-Hot ajustado en entrenamiento)
_prep = MODELOS["gradient_boosting"].named_steps["prep"]
_cat = dict(zip(_prep.named_transformers_["cat"].feature_names_in_, _prep.named_transformers_["cat"].categories_))
CAMPOS = [  # (variable, etiqueta, tipo, mínimo, máximo, paso)
    ("amount_usd", "Monto de la transacción (USD)", "num", 0.5, 5000, 0.01),
    ("amount_vs_avg", "Monto ÷ promedio de 30 días (veces)", "num", 0.001, 600, 0.001),
    ("distance_home_km", "Distancia a la ubicación habitual (km)", "num", 0, 200, 0.01),
    ("ip_risk_score", "Riesgo de la IP (0–100)", "num", 0, 100, 0.1),
    ("hour", "Hora (0–23)", "int", 0, 23, 1),
    ("day_of_week", "Día de la semana (0–6)", "int", 0, 6, 1),
    ("transactions_24h", "Transacciones del cliente en 24 h", "int", 1, 50, 1),
    ("account_age_days", "Antigüedad de la cuenta (días)", "int", 0, 10000, 1),
    ("failed_attempts_24h", "Intentos fallidos en 24 h", "int", 0, 10, 1),
    ("channel", "Canal", "cat", None, None, None),
    ("merchant_category", "Categoría del comercio", "cat", None, None, None),
    ("country", "País", "cat", None, None, None),
    ("device_type", "Tipo de dispositivo", "cat", None, None, None),
    ("card_present", "Tarjeta presente", "bin", 0, 1, 1),
    ("new_device", "Dispositivo nuevo", "bin", 0, 1, 1),
]
assert sorted(c[0] for c in CAMPOS) == sorted(COLUMNAS), "los campos no coinciden con las variables del modelo"
FORMULARIO = [{"variable": v, "etiqueta": e, "tipo": t, "min": lo, "max": hi, "paso": p,
               "opciones": [str(x) for x in _cat[v]] if t == "cat" else None} for v, e, t, lo, hi, p in CAMPOS]
DATOS_API = json.dumps({"modelos": META, "campos": FORMULARIO}, ensure_ascii=False).encode("utf-8")


def validar(datos):
    """Convierte y valida las 15 variables; devuelve (fila, error)."""
    fila = {}
    for v, e, t, lo, hi, _ in CAMPOS:
        if v not in datos or datos[v] in ("", None):
            return None, f"Falta el campo «{e}»."
        valor = datos[v]
        if t == "cat":
            if str(valor) not in _cat[v]:
                return None, f"«{e}» debe ser uno de: {', '.join(map(str, _cat[v]))}."
            fila[v] = str(valor)
            continue
        try:
            num = float(valor)
        except (TypeError, ValueError):
            return None, f"«{e}» debe ser un número."
        if t in ("int", "bin"):
            if num != int(num):
                return None, f"«{e}» debe ser un número entero."
            num = int(num)
        if not lo <= num <= hi:
            return None, f"«{e}» debe estar entre {lo} y {hi}."
        fila[v] = num
    return pd.DataFrame([fila])[COLUMNAS], None


def predecir(clave, X):
    proba = float(MODELOS[clave].predict_proba(X)[0, 1])
    umbral = META[clave]["umbral"]
    return {"modelo": clave, "nombre": META[clave]["nombre"], "proba": round(proba, 4),
            "umbral": umbral, "fraude": proba >= umbral}


class Manejador(BaseHTTPRequestHandler):
    def _enviar(self, codigo, cuerpo, tipo="application/json; charset=utf-8"):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo, obj):
        self._enviar(codigo, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._enviar(200, (APP / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/modelos":
            self._enviar(200, DATOS_API)
        elif self.path == "/salud":
            self._json(200, {"estado": "ok", "modelos": list(MODELOS)})
        else:
            self._json(404, {"error": "No encontrado"})

    def do_POST(self):
        if self.path != "/api/predecir":
            return self._json(404, {"error": "No encontrado"})
        try:
            largo = int(self.headers.get("Content-Length", "0"))
            if largo > 10_000:
                return self._json(413, {"error": "Solicitud demasiado grande"})
            cuerpo = json.loads(self.rfile.read(largo) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "JSON inválido"})
        clave = cuerpo.get("modelo")
        if clave != "todos" and clave not in MODELOS:
            return self._json(400, {"error": "Modelo desconocido"})
        X, error = validar(cuerpo.get("datos") or {})
        if error:
            return self._json(400, {"error": error})
        claves = list(MODELOS) if clave == "todos" else [clave]
        self._json(200, {"resultados": [predecir(c, X) for c in claves]})

    def log_message(self, formato, *args):  # registro breve en consola
        print(f"{self.address_string()} {formato % args}")


if __name__ == "__main__":
    print(f"Modelos cargados: {', '.join(MODELOS)} | http://localhost:{PUERTO}")
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Manejador).serve_forever()

"""Configuración compartida por los notebooks de modelos (Caso 1: detección de fraude)."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent.parent  # raíz del proyecto (comun/ está un nivel abajo)
DATA, FIG, RES = BASE / "data", BASE / "figuras", BASE / "resultados"
TARGET = "fraud"

# Predictoras finales (01_EDA, sección 7: se excluyen is_international y avg_amount_30d)
LOG1P = ["amount_usd", "distance_home_km"]  # sesgo a la derecha
LOG = ["amount_vs_avg"]                      # cociente > 0: log lo deja simétrico alrededor de 0
NUM_RESTO = ["ip_risk_score", "hour", "day_of_week", "transactions_24h",
             "account_age_days", "failed_attempts_24h"]
CAT = ["channel", "merchant_category", "country", "device_type"]
BIN = ["card_present", "new_device"]
FEATURES = LOG1P + LOG + NUM_RESTO + CAT + BIN

# Mismos 5 folds estratificados para todos los modelos (paso 7 del PDF)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"precision": "precision", "recall": "recall", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}


def _cargar(conjunto):
    d = pd.read_csv(DATA / "dataset_preparado.csv")
    return d[d["conjunto"] == conjunto].reset_index(drop=True)


def cargar_train():
    d = _cargar("train")
    return d[FEATURES], d[TARGET]


def cargar_test():
    """Solo en 03_evaluacion.ipynb: el test se usa una única vez (paso 10 del PDF)."""
    d = _cargar("test")
    return d[FEATURES], d[TARGET], d["transaction_id"]


def guardar(nombre):
    FIG.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / f"{nombre}.png", dpi=200, bbox_inches="tight")
    plt.show()


def resumen_cv(grid):
    """Media y desviación de las 5 métricas en los folds, para la mejor combinación de hiperparámetros."""
    r, i = grid.cv_results_, grid.best_index_
    return pd.DataFrame({m: {"media": r[f"mean_test_{m}"][i], "std": r[f"std_test_{m}"][i]}
                         for m in SCORING}).T.rename_axis("métrica")


def guardar_resultados(nombre, modelo, resumen, y, oof):
    """Guarda el modelo ajustado en todo train, sus métricas de CV y sus probabilidades fuera de fold."""
    RES.mkdir(exist_ok=True)
    joblib.dump(modelo, RES / f"{nombre}.joblib")
    resumen.to_csv(RES / f"cv_{nombre}.csv")
    pd.DataFrame({"y": y, "proba": oof}).to_csv(RES / f"oof_{nombre}.csv", index=False)
    return sorted(p.name for p in RES.glob(f"*{nombre}*"))


def hora_features(X):
    """hour → seno y coseno (ciclo de 24 h) + marca de madrugada (23:00 a 04:00)."""
    h = np.asarray(X, dtype=float).ravel()
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            ((h >= 23) | (h <= 4)).astype(float)])


def nombres_hora(transformer, input_features):
    return ["hour_sin", "hour_cos", "madrugada"]


def make_prep(escalar):
    """escalar=True: Regresión Logística y red (log + StandardScaler).
    escalar=False: árboles (numéricas sin transformar)."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if not escalar:
        return ColumnTransformer([
            ("num", "passthrough", LOG1P + LOG + NUM_RESTO + BIN),
            ("cat", ohe, CAT),
        ], verbose_feature_names_out=False)

    def log_y_escala(f):
        return Pipeline([("log", FunctionTransformer(f, feature_names_out="one-to-one")),
                         ("sc", StandardScaler())])

    return ColumnTransformer([
        ("log1p", log_y_escala(np.log1p), LOG1P),
        ("log", log_y_escala(np.log), LOG),
        ("num", StandardScaler(), NUM_RESTO),
        ("cat", ohe, CAT),
        ("bin", "passthrough", BIN),
    ], verbose_feature_names_out=False)

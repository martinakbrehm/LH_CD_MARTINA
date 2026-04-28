"""
modeling.py
===========
Treinamento, avaliação e persistência de modelos de regressão para
previsão da nota IMDb.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TARGET = "IMDB_Rating"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Treinamento & avaliação
# ---------------------------------------------------------------------------

def build_models() -> dict[str, Any]:
    """Retorna um dicionário com os modelos candidatos instanciados."""
    return {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate_models(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series, pd.Series]:
    """Treina e avalia múltiplos modelos com validação cruzada + hold-out.

    Parameters
    ----------
    df_model : pd.DataFrame
        DataFrame com features e target (colunas incluindo ``IMDB_Rating``).

    Returns
    -------
    results_df : pd.DataFrame
        Tabela comparativa de métricas.
    trained_models : dict
        Dicionário com os modelos já treinados no conjunto de treino.
    X_test, y_test : pd.Series
        Conjunto de teste para análise posterior.
    """
    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = build_models()
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    records: list[dict] = []
    trained: dict[str, Any] = {}

    for name, mdl in models.items():
        # Cross-validation
        cv_rmse = -cross_val_score(
            mdl, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=kf,
        )
        cv_r2 = cross_val_score(mdl, X_train, y_train, scoring="r2", cv=kf)

        # Treino & hold-out
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        mae_test = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)

        records.append(
            {
                "Modelo": name,
                "CV RMSE (média)": round(cv_rmse.mean(), 4),
                "CV RMSE (std)": round(cv_rmse.std(), 4),
                "CV R² (média)": round(cv_r2.mean(), 4),
                "RMSE Teste": round(rmse_test, 4),
                "MAE Teste": round(mae_test, 4),
                "R² Teste": round(r2_test, 4),
            }
        )
        trained[name] = mdl
        logger.info("%s → RMSE=%.4f | R²=%.4f", name, rmse_test, r2_test)

    results_df = pd.DataFrame(records).sort_values("RMSE Teste")
    return results_df, trained, X_test, y_test


def get_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> pd.Series:
    """Extrai importância de features de um RandomForestRegressor.

    Parameters
    ----------
    model : RandomForestRegressor
    feature_names : list[str]

    Returns
    -------
    pd.Series ordenada de forma decrescente.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Serializa o modelo com joblib.

    Parameters
    ----------
    model : estimator scikit-learn
    path : str | Path
        Caminho do arquivo de saída (ex.: ``models/random_forest.pkl``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modelo salvo em: %s", path)


def load_model(path: str | Path) -> Any:
    """Carrega um modelo salvo com joblib.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Estimador scikit-learn.
    """
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Predição pontual
# ---------------------------------------------------------------------------

def predict_single_film(
    model: Any,
    runtime: float,
    no_of_votes: int,
    meta_score: float,
    has_famous_star: bool,
) -> float:
    """Prevê a nota IMDb de um único filme.

    Parameters
    ----------
    model : estimador treinado
    runtime : float
        Duração em minutos.
    no_of_votes : int
        Número de votos no IMDb.
    meta_score : float
        Pontuação no Metacritic.
    has_famous_star : bool
        Presença de estrela no top-10.

    Returns
    -------
    float : nota predita.
    """
    features = pd.DataFrame(
        {
            "Runtime": [runtime],
            "Log_No_of_Votes": [np.log1p(no_of_votes)],
            "Meta_score": [meta_score],
            "Has_Famous_Star": [int(has_famous_star)],
        }
    )
    return float(model.predict(features)[0])

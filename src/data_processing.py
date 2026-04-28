"""
data_processing.py
==================
Funções de carregamento, limpeza e engenharia de features para o
dataset de filmes IMDb.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Carrega o CSV do IMDb e retorna um DataFrame bruto.

    Parameters
    ----------
    filepath : str | Path
        Caminho para o arquivo CSV.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, index_col=0)
    logger.info("Dataset carregado: %d linhas × %d colunas", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as etapas de limpeza no dataset.

    Etapas realizadas
    -----------------
    - Remove coluna `Unnamed: 0` (índice residual do CSV).
    - `Gross`      → numérico (remove vírgulas); NaN → 0.
    - `Runtime`    → inteiro (remove ' min').
    - `Released_Year` → inteiro nullable; linhas sem ano são descartadas.
    - `Certificate`   → NaN → 'Unrated'.
    - `Meta_score`    → NaN → média da coluna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto retornado por :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame limpo, sem modificar o original.
    """
    df = df.copy()

    # Gross
    df["Gross"] = (
        df["Gross"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
    )
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce").fillna(0)

    # Runtime
    df["Runtime"] = (
        df["Runtime"]
        .astype(str)
        .str.replace("min", "", regex=False)
        .str.strip()
    )
    df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")

    # Released_Year
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df = df.dropna(subset=["Released_Year"])
    df["Released_Year"] = df["Released_Year"].astype(int)

    # Certificate
    df["Certificate"] = df["Certificate"].fillna("Unrated")

    # Meta_score
    df["Meta_score"] = df["Meta_score"].fillna(df["Meta_score"].mean())

    logger.info("Após limpeza: %d linhas × %d colunas", *df.shape)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engenharia de features
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Cria features derivadas usadas na modelagem e nas hipóteses.

    Features criadas
    ----------------
    - ``Has_Famous_Star``       : bool – algum dos 4 atores está no top-N.
    - ``Has_Famous_Director``   : bool – diretor está no top-N.
    - ``Certificate_Category``  : str  – classificação agrupada do certificado.
    - ``Runtime_Category``      : str  – curto / médio / longo.
    - ``Log_No_of_Votes``       : float – log1p do número de votos.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        Quantos atores/diretores compõem o "hall da fama". Padrão 10.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # ── Estrelas famosas ────────────────────────────────────────────────────
    actor_counts = pd.concat(
        [df["Star1"], df["Star2"], df["Star3"], df["Star4"]]
    ).value_counts()
    top_actors = set(actor_counts.head(top_n).index)

    df["Has_Famous_Star"] = (
        df["Star1"].isin(top_actors)
        | df["Star2"].isin(top_actors)
        | df["Star3"].isin(top_actors)
        | df["Star4"].isin(top_actors)
    )

    # ── Diretores famosos ───────────────────────────────────────────────────
    director_counts = df["Director"].value_counts()
    top_directors = set(director_counts.head(top_n).index)
    df["Has_Famous_Director"] = df["Director"].isin(top_directors)

    # ── Certificado agrupado ────────────────────────────────────────────────
    less_restrictive = {"G", "U", "Passed", "Approved"}
    moderate = {"PG", "PG-13", "UA", "U/A"}

    def _cert_category(cert: str) -> str:
        if cert in less_restrictive:
            return "Menos restritivo"
        if cert in moderate:
            return "Moderadamente restritivo"
        return "Mais restritivo"

    df["Certificate_Category"] = df["Certificate"].apply(_cert_category)

    # ── Categoria de duração ────────────────────────────────────────────────
    def _runtime_category(rt: float) -> str:
        if rt < 90:
            return "Curto (<90 min)"
        if rt <= 120:
            return "Médio (90-120 min)"
        return "Longo (>120 min)"

    df["Runtime_Category"] = df["Runtime"].apply(_runtime_category)

    # ── Log-votos ───────────────────────────────────────────────────────────
    df["Log_No_of_Votes"] = np.log1p(df["No_of_Votes"])

    return df


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna um DataFrame com uma linha por gênero (explode na coluna Genre).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_genres = df.copy()
    df_genres["Genre"] = df_genres["Genre"].str.split(", ")
    return df_genres.explode("Genre").reset_index(drop=True)


def prepare_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e prepara as features para o modelo de regressão.

    Usa apenas filmes únicos (1 linha por título) e as variáveis:
    Runtime, Log_No_of_Votes, Meta_score, Has_Famous_Star.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame já processado por :func:`add_features`.

    Returns
    -------
    pd.DataFrame com colunas [features + 'IMDB_Rating'].
    """
    features = ["Runtime", "Log_No_of_Votes", "Meta_score", "Has_Famous_Star", "IMDB_Rating"]
    df_model = (
        df[["Series_Title"] + features]
        .drop_duplicates(subset=["Series_Title"])
        .drop(columns=["Series_Title"])
        .dropna()
    )
    df_model["Has_Famous_Star"] = df_model["Has_Famous_Star"].astype(int)
    return df_model.reset_index(drop=True)

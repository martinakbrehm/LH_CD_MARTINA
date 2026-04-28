"""
visualization.py
================
Funções de visualização reutilizáveis para o projeto de dados cinematográficos.
Todas as funções retornam o objeto Figure para facilitar customização adicional
e permitir salvar imagens de forma programática.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Paleta padrão do projeto
PALETTE = "viridis"
TITLE_SIZE = 14
LABEL_SIZE = 11


def _style() -> None:
    """Aplica estilo visual padrão do projeto."""
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)


# ---------------------------------------------------------------------------
# EDA Univariada
# ---------------------------------------------------------------------------

def plot_numeric_distribution(
    series: pd.Series,
    title: str,
    xlabel: str,
    color: str = "steelblue",
    bins: int = 30,
) -> Figure:
    """Histograma + KDE + Boxplot lado a lado para uma variável numérica."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")

    sns.histplot(series, kde=True, bins=bins, color=color, ax=axes[0])
    axes[0].set_xlabel(xlabel, fontsize=LABEL_SIZE)
    axes[0].set_ylabel("Frequência", fontsize=LABEL_SIZE)
    axes[0].set_title("Distribuição")

    sns.boxplot(y=series, color=color, ax=axes[1])
    axes[1].set_ylabel(xlabel, fontsize=LABEL_SIZE)
    axes[1].set_title("Boxplot")

    fig.tight_layout()
    return fig


def plot_categorical_frequency(
    series: pd.Series,
    title: str,
    xlabel: str,
    top_n: int | None = None,
) -> Figure:
    """Gráfico de barras para variáveis categóricas."""
    _style()
    counts = series.value_counts()
    if top_n:
        counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=counts.index, y=counts.values, palette=PALETTE, ax=ax)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequência", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EDA Bivariada / Multivariada
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Mapa de calor da matriz de correlação para variáveis numéricas."""
    _style()
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação", fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_boxplot_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_labels: dict | None = None,
    order: list | None = None,
) -> Figure:
    """Boxplot de uma variável numérica agrupada por categoria."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x=x, y=y, data=df, palette=PALETTE, ax=ax, order=order, showfliers=False
    )
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if x_labels:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels.get(t, t) for t in ticks])
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    alpha: float = 0.5,
    add_trend: bool = True,
) -> Figure:
    """Scatter plot com linha de tendência opcional (regressão linear)."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=x, y=y, data=df, alpha=alpha, ax=ax, color="steelblue")
    if add_trend:
        sns.regplot(
            x=x,
            y=y,
            data=df,
            scatter=False,
            color="crimson",
            line_kws={"linewidth": 2},
            ax=ax,
        )
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Modelagem
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list[str], importances: np.ndarray
) -> Figure:
    """Gráfico de barras horizontal para importância de features."""
    _style()
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE, len(feature_names))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors,
    )
    ax.set_title("Importância das Features — Random Forest", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Importância Média (Gini)", fontsize=LABEL_SIZE)
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Modelo"
) -> Figure:
    """Gráfico de dispersão entre valores reais e preditos."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color="steelblue", edgecolors="white", linewidth=0.4)
    lims = [
        min(y_true.min(), y_pred.min()) - 0.1,
        max(y_true.max(), y_pred.max()) + 0.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Predição perfeita")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"Predito vs. Real — {model_name}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Nota Real (IMDb)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Nota Predita", fontsize=LABEL_SIZE)
    ax.legend()
    fig.tight_layout()
    return fig

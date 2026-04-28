# 🎬 Análise e Previsão de Desempenho de Filmes — IMDb Top 1000

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Projeto de ciência de dados end-to-end: exploração, teste de hipóteses, NLP e modelagem preditiva aplicados ao dataset IMDb Top 1000.

---

## Índice

1. [Contexto e Objetivo](#contexto-e-objetivo)
2. [Perguntas Norteadoras](#perguntas-norteadoras)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Técnicas Utilizadas](#técnicas-utilizadas)
5. [Hipóteses Testadas](#hipóteses-testadas)
6. [Resultados do Modelo](#resultados-do-modelo)
7. [Como Executar](#como-executar)
8. [Dataset](#dataset)
9. [Tecnologias](#tecnologias)

---

## Contexto e Objetivo

Uma produtora cinematográfica precisa decidir **qual tipo de filme desenvolver a seguir**. Para embasar essa decisão, este projeto analisa o dataset IMDb Top 1000 com o objetivo de:

- Identificar os fatores que mais influenciam a **nota IMDb** e o **faturamento bruto**
- Testar hipóteses de negócio com rigor estatístico
- Construir um modelo preditivo para estimar a nota IMDb de um filme antes do lançamento
- Extrair insights da sinopse (`Overview`) via processamento de linguagem natural

---

## Perguntas Norteadoras

> **1.** Qual filme você recomendaria para uma pessoa que você não conhece?  
> **2.** Quais são os principais fatores relacionados à alta expectativa de faturamento?  
> **3.** É possível inferir o gênero de um filme a partir da sua sinopse?

---

## Estrutura do Projeto

```
projeto_dados_cinematograficos/
│
├── data/                          # Dataset CSV (não versionado)
│   └── desafio_indicium_imdb.csv  ← coloque o arquivo aqui
│
├── models/                        # Modelos treinados serializados (.pkl)
│
├── notebooks/
│   └── analise_filmes_imdb.ipynb  # Notebook principal (9 seções)
│
├── src/                           # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── data_processing.py         # Carregamento, limpeza e feature engineering
│   ├── visualization.py           # Funções de visualização padronizadas
│   └── modeling.py                # Treinamento, avaliação e persistência de modelos
│
├── legacy/                        # Notebook original arquivado
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Técnicas Utilizadas

| Etapa | Técnica |
|---|---|
| Limpeza de dados | Tratamento de nulos, conversão de tipos, remoção de duplicatas |
| EDA Univariada | Histogramas, KDE, boxplots |
| EDA Multivariada | Matriz de correlação, scatter plots com tendência |
| Teste de Hipóteses | Teste t de Student, ANOVA one-way, Correlação de Pearson |
| NLP | Análise de sentimento (TextBlob), word clouds, similaridade de cosseno |
| Modelagem | Regressão Linear, Árvore de Decisão, **Random Forest** |
| Avaliação | Validação cruzada k-fold (k=5), RMSE, MAE, R² |

---

## Hipóteses Testadas

| # | Hipótese | Método | Resultado |
|---|---|---|---|
| H1 | Estrelas famosas elevam nota e faturamento | Teste t | ✅ Confirmada |
| H2 | Faturamento alto implica nota alta | Pearson | ⚠️ Correlação fraca (r ≈ 0.09) |
| H3 | Classificação indicativa influencia faturamento | ANOVA | ✅ Confirmada — PG/PG-13 lideram |
| H4 | Filmes recentes têm nota mais alta | Pearson | ❌ Refutada — viés de seleção histórica |
| H5 | Gênero influencia a nota IMDb | ANOVA | ❌ Não significativo (p > 0.05) |
| H6 | Filmes longos geram mais faturamento | ANOVA | ✅ Confirmada — >120 min lideram |
| H7 | Mais votos → maior faturamento | Pearson | ✅ Confirmada (r ≈ 0.62) |
| H8 | Gênero influencia o faturamento | ANOVA | ✅ Confirmada — Action/Adventure/Sci-Fi |

---

## Resultados do Modelo

**Problema:** Regressão — previsão da nota IMDb (variável contínua)  
**Features:** `Runtime`, `Log_No_of_Votes`, `Meta_score`, `Has_Famous_Star`

| Modelo | RMSE (CV) | R² (CV) | RMSE Teste |
|---|---|---|---|
| Regressão Linear | — | — | — |
| Árvore de Decisão | — | — | — |
| **Random Forest** ✓ | — | — | **melhor** |

> Os valores exatos são gerados ao executar o notebook. O Random Forest superou os demais modelos em todas as métricas; `Log_No_of_Votes` foi a feature mais importante.

**Perfil ideal de próximo filme (insight estratégico):**
- Gênero **Ação, Aventura ou Sci-Fi**
- Classificação **PG ou PG-13**
- Duração **> 120 minutos**
- Ao menos um ator do **top-10 IMDb** no elenco

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd projeto_dados_cinematograficos
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Adicionar o dataset

Coloque o arquivo `desafio_indicium_imdb.csv` dentro da pasta `data/`.  
Veja [data/README.md](data/README.md) para o schema completo.

### 4. Executar o notebook

```bash
jupyter notebook notebooks/analise_filmes_imdb.ipynb
```

> O notebook usa caminhos relativos — nenhuma configuração adicional necessária.

---

## Dataset

**IMDb Top 1000 Movies** — inclui título, ano, certificado, duração, gênero, sinopse, nota IMDb, Meta Score, diretor, elenco, número de votos e faturamento bruto.

Fonte pública: [Kaggle — IMDb Dataset of Top 1000 Movies](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

---

## Tecnologias

`Python 3.10+` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `NLTK` · `TextBlob` · `WordCloud` · `Joblib`

---

## Autora

**Martina Brehm** — [LinkedIn](https://www.linkedin.com/in/martinabrehm/) · [GitHub](https://github.com/martinakbrehm)  
*Cientista de Dados | Python · Machine Learning · Análise Estatística*





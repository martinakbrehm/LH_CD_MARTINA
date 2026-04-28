# data/

Esta pasta deve conter o arquivo CSV do dataset IMDb Top 1000.

## Arquivo esperado

`desafio_indicium_imdb.csv`

## Colunas do dataset

| Coluna | Tipo | Descrição |
|---|---|---|
| `Series_Title` | str | Título do filme |
| `Released_Year` | int | Ano de lançamento |
| `Certificate` | str | Classificação indicativa |
| `Runtime` | str | Duração (`"142 min"`) |
| `Genre` | str | Gênero(s) separados por vírgula |
| `IMDB_Rating` | float | Nota no IMDb (0–10) |
| `Overview` | str | Sinopse do filme |
| `Meta_score` | float | Pontuação no Metacritic (0–100) |
| `Director` | str | Nome do diretor |
| `Star1–Star4` | str | Atores principais |
| `No_of_Votes` | int | Número de votos no IMDb |
| `Gross` | str | Faturamento bruto (`"28,341,469"`) |

> ⚠️ O arquivo CSV **não está versionado** no repositório. Solicite o arquivo ao responsável pelo projeto ou use o dataset público do Kaggle: [IMDb Movies Dataset](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows).

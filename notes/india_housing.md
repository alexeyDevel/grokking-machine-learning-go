# India Housing

Этот пример показывает Go-аналог подхода `Turi Create` для прогнозирования цен на жильё:

1. Загрузить CSV.
2. Разделить данные на `train` и `test`.
3. Закодировать числовые и категориальные признаки.
4. Обучить линейную регрессию.
5. Посчитать `RMSE` и `MAE`.
6. Сделать прогноз для новой квартиры.

## Важно про данные

Файл `data/india_housing_sample.csv` - учебный мини-датасет. Он нужен, чтобы отработать ML-пайплайн на Go без Python и без Turi Create.

Для реального прогноза этот CSV нужно заменить на настоящий датасет объявлений, например с колонками:

- `city`
- `area_sqft`
- `bedrooms`
- `bathrooms`
- `age_years`
- `near_metro`
- `furnished`
- `price_lakhs`

## Почему это похоже на Turi Create

В Turi Create код часто выглядит как высокоуровневый pipeline:

```text
load data -> split data -> train model -> evaluate -> predict
```

В Go мы делаем то же самое явно:

```bash
go run ./cmd/india_housing
```

Здесь нет магии фреймворка, зато хорошо видно, что происходит:

- `LoadCSV` читает таблицу.
- `TrainTestSplit` отделяет тестовые данные.
- `Encoder` превращает город и furnishing в числовые признаки.
- `TrainLinearRegression` обучает веса модели.
- `Evaluate` считает ошибку.


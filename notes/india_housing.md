# India Housing

Этот пример повторяет логику notebook `House_price_predictions.ipynb` из 3 главы,
но на Go:

1. Загрузить CSV.
2. Разделить данные на `train` и `test`.
3. Масштабировать числовые признаки `Area` и `No. of Bedrooms`.
4. Закодировать `Location` через one-hot признаки.
5. Обучить `LinearRegression` через least squares.
6. Посчитать `RMSE` и `MAE`.
7. Сделать прогноз для новой квартиры.

## Важно про данные

Файл `internal/indiahousing/Hyderabad.csv` - тот же формат данных, который
используется в notebook:

- `Price` - цена в рупиях, это целевая переменная.
- `Area` - площадь.
- `Location` - район, который превращается в one-hot признаки.
- `No. of Bedrooms` - количество спален.
- остальные колонки - бинарные признаки удобств: `Gymnasium`, `SwimmingPool`,
  `LiftAvailable` и так далее.

## Почему это похоже на notebook

В Python-версии используются `pandas`, `numpy` и `sklearn.LinearRegression`.
В Go-версии эти шаги разложены по файлам:

- `LoadCSV` читает таблицу как `pd.read_csv`.
- `Encoder` делает scaling и one-hot encoding.
- `TrainLinearRegression` решает задачу least squares через `gonum/mat` SVD.
- `Evaluate` считает ошибку прогноза.

Запуск:

```bash
go run ./cmd/india_housing
```

package chapter03

import (
	"errors"
	"math/rand"
)

// TrainingResult хранит обученную модель и историю ошибки по эпохам.
// TrainingResult stores the trained model and the error history by epoch.
type TrainingResult struct {
	// Model - финальная модель после всех эпох обучения.
	// Model is the final model after all training epochs.
	Model Model

	// Errors - значения RMSE перед каждым шагом обновления модели.
	// Errors are RMSE values before each model update step.
	Errors []float64
}

// LinearRegression обучает модель линейной регрессии через stochastic gradient descent.
// На каждой эпохе она выбирает случайную точку и применяет SquareTrick.
// LinearRegression trains a linear regression model with stochastic gradient descent.
// On each epoch, it picks a random point and applies SquareTrick.
func LinearRegression(dataset Dataset, learningRate float64, epochs int, rng *rand.Rand) (TrainingResult, error) {
	if len(dataset.RoomCounts) == 0 {
		return TrainingResult{}, errors.New("room counts must not be empty")
	}
	if len(dataset.RoomCounts) != len(dataset.ActualPrices) {
		return TrainingResult{}, errors.New("room counts and actual prices must have the same length")
	}
	if epochs < 0 {
		return TrainingResult{}, errors.New("epochs must not be negative")
	}
	if rng == nil {
		// Если генератор не передали, создаём фиксированный, чтобы результат был повторяемым.
		// If no generator is passed, create a fixed one so the result is reproducible.
		rng = rand.New(rand.NewSource(0))
	}

	model := Model{
		// Начальный наклон выбирается случайно в диапазоне [0.0, 1.0).
		// The initial slope is chosen randomly in the range [0.0, 1.0).
		PricePerRoom: rng.Float64(),

		// Начальная базовая цена тоже выбирается случайно в диапазоне [0.0, 1.0).
		// The initial base price is also chosen randomly in the range [0.0, 1.0).
		BasePrice: rng.Float64(),
	}

	// errorsByEpoch заранее получает ёмкость epochs, потому что мы добавим одну ошибку на эпоху.
	// errorsByEpoch gets capacity epochs in advance because we append one error per epoch.
	errorsByEpoch := make([]float64, 0, epochs)

	for range epochs {
		// Считаем ошибку всей модели на всех точках до очередного обновления.
		// Calculate the model error on all points before the next update.
		currentRMSE, err := RMSE(dataset.ActualPrices, model.PredictAll(dataset.RoomCounts))
		if err != nil {
			return TrainingResult{}, err
		}
		errorsByEpoch = append(errorsByEpoch, currentRMSE)

		// i - случайный индекс точки из датасета. Для 6 точек это число от 0 до 5.
		// i is a random dataset point index. For 6 points, it is a number from 0 to 5.
		i := rng.Intn(len(dataset.RoomCounts))

		// Обновляем модель только по одной случайной точке: это и есть stochastic gradient descent.
		// Update the model using only one random point: this is stochastic gradient descent.
		model = SquareTrick(
			model,
			dataset.RoomCounts[i],
			dataset.ActualPrices[i],
			learningRate,
		)
	}

	return TrainingResult{
		Model:  model,
		Errors: errorsByEpoch,
	}, nil
}

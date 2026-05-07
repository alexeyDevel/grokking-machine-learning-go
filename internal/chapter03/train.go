package chapter03

import (
	"errors"
	"math/rand"
)

// TrainingResult хранит обученную модель и историю ошибки по эпохам.
type TrainingResult struct {
	Model  Model
	Errors []float64
}

// LinearRegression обучает модель линейной регрессии через stochastic gradient descent.
// На каждой эпохе она выбирает случайную точку и применяет SquareTrick.
func LinearRegression(dataset Dataset, learningRate float64, epochs int, rng *rand.Rand) (TrainingResult, error) {
	if len(dataset.Features) == 0 {
		return TrainingResult{}, errors.New("features must not be empty")
	}
	if len(dataset.Features) != len(dataset.Labels) {
		return TrainingResult{}, errors.New("features and labels must have the same length")
	}
	if epochs < 0 {
		return TrainingResult{}, errors.New("epochs must not be negative")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}

	model := Model{
		PricePerRoom: rng.Float64(),
		BasePrice:    rng.Float64(),
	}
	errorsByEpoch := make([]float64, 0, epochs)

	for range epochs {
		errorValue, err := RMSE(dataset.Labels, model.PredictAll(dataset.Features))
		if err != nil {
			return TrainingResult{}, err
		}
		errorsByEpoch = append(errorsByEpoch, errorValue)

		i := rng.Intn(len(dataset.Features))
		model = SquareTrick(
			model,
			dataset.Features[i],
			dataset.Labels[i],
			learningRate,
		)
	}

	return TrainingResult{
		Model:  model,
		Errors: errorsByEpoch,
	}, nil
}

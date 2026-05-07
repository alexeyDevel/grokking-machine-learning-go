package chapter03

import (
	"errors"
	"math"
	"math/rand"
)

type Dataset struct {
	Features []float64
	Labels   []float64
}

type Model struct {
	PricePerRoom float64
	BasePrice    float64
}

type TrainingResult struct {
	Model  Model
	Errors []float64
}

func HousingDataset() Dataset {
	return Dataset{
		Features: []float64{1, 2, 3, 5, 6, 7},
		Labels:   []float64{155, 197, 244, 356, 407, 448},
	}
}

func (m Model) Predict(numRooms float64) float64 {
	return m.BasePrice + m.PricePerRoom*numRooms
}

func (m Model) PredictAll(features []float64) []float64 {
	predictions := make([]float64, len(features))
	for i, feature := range features {
		predictions[i] = m.Predict(feature)
	}

	return predictions
}

func SimpleTrick(model Model, numRooms, price float64, rng *rand.Rand) Model {
	smallRandom1 := rng.Float64() * 0.1
	smallRandom2 := rng.Float64() * 0.1
	predictedPrice := model.Predict(numRooms)

	if price > predictedPrice && numRooms > 0 {
		model.PricePerRoom += smallRandom1
		model.BasePrice += smallRandom2
	}
	if price > predictedPrice && numRooms < 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}
	if price < predictedPrice && numRooms > 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice -= smallRandom2
	}
	if price < predictedPrice && numRooms < 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}

	return model
}

func AbsoluteTrick(model Model, numRooms, price, learningRate float64) Model {
	predictedPrice := model.Predict(numRooms)
	if price > predictedPrice {
		model.PricePerRoom += learningRate * numRooms
		model.BasePrice += learningRate
	} else {
		model.PricePerRoom -= learningRate * numRooms
		model.BasePrice -= learningRate
	}

	return model
}

func SquareTrick(model Model, numRooms, price, learningRate float64) Model {
	predictedPrice := model.Predict(numRooms)
	errorValue := price - predictedPrice

	model.PricePerRoom += learningRate * numRooms * errorValue
	model.BasePrice += learningRate * errorValue

	return model
}

func RMSE(labels, predictions []float64) (float64, error) {
	if len(labels) == 0 {
		return 0, errors.New("labels must not be empty")
	}
	if len(labels) != len(predictions) {
		return 0, errors.New("labels and predictions must have the same length")
	}

	var sumSquares float64
	for i := range labels {
		diff := labels[i] - predictions[i]
		sumSquares += diff * diff
	}

	return math.Sqrt(sumSquares / float64(len(labels))), nil
}

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

func OrdinaryLeastSquares(dataset Dataset) (Model, error) {
	if len(dataset.Features) == 0 {
		return Model{}, errors.New("features must not be empty")
	}
	if len(dataset.Features) != len(dataset.Labels) {
		return Model{}, errors.New("features and labels must have the same length")
	}

	var sumX, sumY float64
	for i := range dataset.Features {
		sumX += dataset.Features[i]
		sumY += dataset.Labels[i]
	}

	n := float64(len(dataset.Features))
	meanX := sumX / n
	meanY := sumY / n

	var numerator, denominator float64
	for i := range dataset.Features {
		xDiff := dataset.Features[i] - meanX
		yDiff := dataset.Labels[i] - meanY
		numerator += xDiff * yDiff
		denominator += xDiff * xDiff
	}
	if denominator == 0 {
		return Model{}, errors.New("cannot fit a line when all features are equal")
	}

	pricePerRoom := numerator / denominator
	return Model{
		PricePerRoom: pricePerRoom,
		BasePrice:    meanY - pricePerRoom*meanX,
	}, nil
}

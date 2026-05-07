package chapter03

import "errors"

// OrdinaryLeastSquares находит оптимальную прямую по закрытой формуле.
// Это аналог результата, который notebook получает через scikit-learn LinearRegression.
// OrdinaryLeastSquares finds the optimal line with a closed-form formula.
// This is the Go equivalent of the result produced by scikit-learn LinearRegression in the notebook.
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

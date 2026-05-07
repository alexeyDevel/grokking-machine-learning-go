package chapter03

import (
	"errors"
	"math"
)

// RMSE считает корень из средней квадратичной ошибки между настоящими и предсказанными ценами.
// RMSE calculates the root mean square error between actual and predicted prices.
func RMSE(actualPrices, predictedPrices []float64) (float64, error) {
	if len(actualPrices) == 0 {
		return 0, errors.New("actual prices must not be empty")
	}
	if len(actualPrices) != len(predictedPrices) {
		return 0, errors.New("actual prices and predicted prices must have the same length")
	}

	// sumSquares хранит сумму квадратов ошибок.
	// sumSquares stores the sum of squared errors.
	var sumSquares float64
	for i := range actualPrices {
		// predictionError - ошибка для одной точки: правильная цена минус предсказанная цена.
		// predictionError is the error for one point: actual price minus predicted price.
		predictionError := actualPrices[i] - predictedPrices[i]

		// Квадрат ошибки делает отрицательные и положительные ошибки одинаково важными.
		// Squaring the error makes negative and positive errors equally important.
		sumSquares += predictionError * predictionError
	}

	// Делим на количество точек и берём корень, чтобы вернуться к масштабу цены.
	// Divide by the number of points and take the square root to return to the price scale.
	return math.Sqrt(sumSquares / float64(len(actualPrices))), nil
}

package chapter03

import (
	"errors"
	"math"
)

// RMSE считает корень из средней квадратичной ошибки между ответами и предсказаниями.
// RMSE calculates the root mean square error between labels and predictions.
func RMSE(labels, predictions []float64) (float64, error) {
	if len(labels) == 0 {
		return 0, errors.New("labels must not be empty")
	}
	if len(labels) != len(predictions) {
		return 0, errors.New("labels and predictions must have the same length")
	}

	// sumSquares хранит сумму квадратов ошибок.
	// sumSquares stores the sum of squared errors.
	var sumSquares float64
	for i := range labels {
		// diff - ошибка для одной точки: правильная цена минус предсказанная цена.
		// diff is the error for one point: actual price minus predicted price.
		diff := labels[i] - predictions[i]

		// Квадрат ошибки делает отрицательные и положительные ошибки одинаково важными.
		// Squaring the error makes negative and positive errors equally important.
		sumSquares += diff * diff
	}

	// Делим на количество точек и берём корень, чтобы вернуться к масштабу цены.
	// Divide by the number of points and take the square root to return to the price scale.
	return math.Sqrt(sumSquares / float64(len(labels))), nil
}

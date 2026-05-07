package chapter03

import (
	"errors"
	"math"
)

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

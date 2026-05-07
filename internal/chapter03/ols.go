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

	// sumX и sumY нужны, чтобы найти средние значения признаков и цен.
	// sumX and sumY are used to find the average feature value and average price.
	var sumX, sumY float64
	for i := range dataset.Features {
		sumX += dataset.Features[i]
		sumY += dataset.Labels[i]
	}

	// n - количество точек в датасете.
	// n is the number of points in the dataset.
	n := float64(len(dataset.Features))

	// meanX - среднее количество комнат, meanY - средняя цена.
	// meanX is the average number of rooms, meanY is the average price.
	meanX := sumX / n
	meanY := sumY / n

	// numerator и denominator - части формулы для наклона прямой.
	// numerator and denominator are parts of the formula for the line slope.
	var numerator, denominator float64
	for i := range dataset.Features {
		// xDiff показывает, насколько текущий x отличается от среднего x.
		// xDiff shows how far the current x is from the average x.
		xDiff := dataset.Features[i] - meanX

		// yDiff показывает, насколько текущая цена отличается от средней цены.
		// yDiff shows how far the current price is from the average price.
		yDiff := dataset.Labels[i] - meanY

		// numerator накапливает совместное изменение x и y.
		// numerator accumulates how x and y change together.
		numerator += xDiff * yDiff

		// denominator накапливает разброс x относительно среднего.
		// denominator accumulates the spread of x around its average.
		denominator += xDiff * xDiff
	}
	if denominator == 0 {
		return Model{}, errors.New("cannot fit a line when all features are equal")
	}

	// pricePerRoom - найденный наклон прямой: ожидаемая прибавка цены за одну комнату.
	// pricePerRoom is the fitted slope: expected price increase for one room.
	pricePerRoom := numerator / denominator
	return Model{
		PricePerRoom: pricePerRoom,

		// BasePrice выбирается так, чтобы прямая проходила через точку (meanX, meanY).
		// BasePrice is chosen so the line passes through the point (meanX, meanY).
		BasePrice: meanY - pricePerRoom*meanX,
	}, nil
}

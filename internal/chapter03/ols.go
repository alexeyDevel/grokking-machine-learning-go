package chapter03

import "errors"

// OrdinaryLeastSquares находит оптимальную прямую по закрытой формуле.
// Это аналог результата, который notebook получает через scikit-learn LinearRegression.
// OrdinaryLeastSquares finds the optimal line with a closed-form formula.
// This is the Go equivalent of the result produced by scikit-learn LinearRegression in the notebook.
func OrdinaryLeastSquares(dataset Dataset) (Model, error) {
	if len(dataset.RoomCounts) == 0 {
		return Model{}, errors.New("room counts must not be empty")
	}
	if len(dataset.RoomCounts) != len(dataset.ActualPrices) {
		return Model{}, errors.New("room counts and actual prices must have the same length")
	}

	// sumX и sumY нужны, чтобы найти средние значения признаков и цен.
	// sumX and sumY are used to find the average feature value and average price.
	var sumX, sumY float64
	for i := range dataset.RoomCounts {
		sumX += dataset.RoomCounts[i]
		sumY += dataset.ActualPrices[i]
	}

	// n - количество точек в датасете.
	// n is the number of points in the dataset.
	n := float64(len(dataset.RoomCounts))

	// meanRoomCount - среднее количество комнат, meanActualPrice - средняя цена.
	// meanRoomCount is the average number of rooms, meanActualPrice is the average price.
	meanRoomCount := sumX / n
	meanActualPrice := sumY / n

	// numerator и denominator - части формулы для наклона прямой.
	// numerator and denominator are parts of the formula for the line slope.
	var numerator, denominator float64
	for i := range dataset.RoomCounts {
		// roomCountDiff показывает, насколько текущее количество комнат отличается от среднего.
		// roomCountDiff shows how far the current rooms count is from the average rooms count.
		roomCountDiff := dataset.RoomCounts[i] - meanRoomCount

		// actualPriceDiff показывает, насколько текущая цена отличается от средней цены.
		// actualPriceDiff shows how far the current price is from the average price.
		actualPriceDiff := dataset.ActualPrices[i] - meanActualPrice

		// numerator накапливает совместное изменение x и y.
		// numerator accumulates how x and y change together.
		numerator += roomCountDiff * actualPriceDiff

		// denominator накапливает разброс x относительно среднего.
		// denominator accumulates the spread of x around its average.
		denominator += roomCountDiff * roomCountDiff
	}
	if denominator == 0 {
		return Model{}, errors.New("cannot fit a line when all room counts are equal")
	}

	// pricePerRoom - найденный наклон прямой: ожидаемая прибавка цены за одну комнату.
	// pricePerRoom is the fitted slope: expected price increase for one room.
	pricePerRoom := numerator / denominator
	return Model{
		PricePerRoom: pricePerRoom,

		// BasePrice выбирается так, чтобы прямая проходила через точку (meanRoomCount, meanActualPrice).
		// BasePrice is chosen so the line passes through the point (meanRoomCount, meanActualPrice).
		BasePrice: meanActualPrice - pricePerRoom*meanRoomCount,
	}, nil
}

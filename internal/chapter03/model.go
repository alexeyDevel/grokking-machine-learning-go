package chapter03

// Model описывает прямую линейной регрессии:
// price = BasePrice + PricePerRoom * roomCount.
// Model describes a linear regression line:
// price = BasePrice + PricePerRoom * roomCount.
type Model struct {
	// PricePerRoom - наклон прямой: насколько меняется цена при добавлении одной комнаты.
	// PricePerRoom is the slope: how much the price changes when one room is added.
	PricePerRoom float64

	// BasePrice - свободный член: базовая цена, когда количество комнат равно 0.
	// BasePrice is the intercept: the base price when the number of rooms is 0.
	BasePrice float64
}

// Predict вычисляет цену для одного значения признака.
// Predict calculates the price for one feature value.
func (m Model) Predict(roomCount float64) float64 {
	// Формула прямой: y = b + wx, где x - комнаты, w - цена за комнату, b - базовая цена.
	// Line formula: y = b + wx, where x is rooms, w is price per room, and b is base price.
	return m.BasePrice + m.PricePerRoom*roomCount
}

// PredictAll вычисляет предсказания модели для набора значений "количество комнат".
// PredictAll calculates model predictions for a set of room-count values.
func (m Model) PredictAll(roomCounts []float64) []float64 {
	// Создаём срез такой же длины, чтобы для каждого дома было одно предсказание.
	// Create a slice of the same length, so every house has one prediction.
	predictedPrices := make([]float64, len(roomCounts))
	for i, roomCount := range roomCounts {
		// i - индекс текущего дома, roomCount - количество комнат в нём.
		// i is the current house index, roomCount is its number of rooms.
		predictedPrices[i] = m.Predict(roomCount)
	}

	return predictedPrices
}

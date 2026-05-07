package chapter03

// Model описывает прямую линейной регрессии:
// price = BasePrice + PricePerRoom * numRooms.
// Model describes a linear regression line:
// price = BasePrice + PricePerRoom * numRooms.
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
func (m Model) Predict(numRooms float64) float64 {
	// Формула прямой: y = b + wx, где x - комнаты, w - цена за комнату, b - базовая цена.
	// Line formula: y = b + wx, where x is rooms, w is price per room, and b is base price.
	return m.BasePrice + m.PricePerRoom*numRooms
}

// PredictAll вычисляет предсказания модели для набора признаков.
// PredictAll calculates model predictions for a set of features.
func (m Model) PredictAll(features []float64) []float64 {
	// Создаём срез такой же длины, чтобы для каждого входного значения было одно предсказание.
	// Create a slice of the same length, so every input value has one prediction.
	predictions := make([]float64, len(features))
	for i, feature := range features {
		// i - индекс текущего значения, feature - само значение признака.
		// i is the current index, feature is the feature value itself.
		predictions[i] = m.Predict(feature)
	}

	return predictions
}

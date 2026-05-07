package chapter03

// Model описывает прямую линейной регрессии:
// price = BasePrice + PricePerRoom * numRooms.
// Model describes a linear regression line:
// price = BasePrice + PricePerRoom * numRooms.
type Model struct {
	PricePerRoom float64
	BasePrice    float64
}

// Predict вычисляет цену для одного значения признака.
// Predict calculates the price for one feature value.
func (m Model) Predict(numRooms float64) float64 {
	return m.BasePrice + m.PricePerRoom*numRooms
}

// PredictAll вычисляет предсказания модели для набора признаков.
// PredictAll calculates model predictions for a set of features.
func (m Model) PredictAll(features []float64) []float64 {
	predictions := make([]float64, len(features))
	for i, feature := range features {
		predictions[i] = m.Predict(feature)
	}

	return predictions
}

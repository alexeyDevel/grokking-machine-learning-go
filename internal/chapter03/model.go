package chapter03

// Model описывает прямую линейной регрессии:
// price = BasePrice + PricePerRoom * numRooms.
type Model struct {
	PricePerRoom float64
	BasePrice    float64
}

// Predict вычисляет цену для одного значения признака.
func (m Model) Predict(numRooms float64) float64 {
	return m.BasePrice + m.PricePerRoom*numRooms
}

// PredictAll вычисляет предсказания модели для набора признаков.
func (m Model) PredictAll(features []float64) []float64 {
	predictions := make([]float64, len(features))
	for i, feature := range features {
		predictions[i] = m.Predict(feature)
	}

	return predictions
}

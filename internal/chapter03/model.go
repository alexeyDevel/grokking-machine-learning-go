package chapter03

type Model struct {
	PricePerRoom float64
	BasePrice    float64
}

func (m Model) Predict(numRooms float64) float64 {
	return m.BasePrice + m.PricePerRoom*numRooms
}

func (m Model) PredictAll(features []float64) []float64 {
	predictions := make([]float64, len(features))
	for i, feature := range features {
		predictions[i] = m.Predict(feature)
	}

	return predictions
}

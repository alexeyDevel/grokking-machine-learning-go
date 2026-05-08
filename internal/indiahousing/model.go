package indiahousing

import "fmt"

// LinearModel хранит веса линейной регрессии.
// LinearModel stores linear regression weights.
type LinearModel struct {
	Encoder      Encoder
	FeatureNames []string
	Weights      []float64
	Bias         float64
}

// Predict оценивает цену жилья в лакхах рупий.
// Predict estimates the house price in lakh rupees.
func (m LinearModel) Predict(house House) float64 {
	features := m.Encoder.Encode(house)

	priceLakhs := m.Bias
	for i, featureValue := range features {
		priceLakhs += m.Weights[i] * featureValue
	}

	return priceLakhs
}

// ExplainTopWeights показывает признаки с самым большим влиянием на прогноз.
// ExplainTopWeights shows features with the strongest impact on the prediction.
func (m LinearModel) ExplainTopWeights(limit int) []string {
	if limit > len(m.Weights) {
		limit = len(m.Weights)
	}

	used := make([]bool, len(m.Weights))
	result := make([]string, 0, limit)
	for range limit {
		bestIndex := -1
		bestAbs := -1.0
		for i, weight := range m.Weights {
			if used[i] {
				continue
			}
			absWeight := weight
			if absWeight < 0 {
				absWeight = -absWeight
			}
			if absWeight > bestAbs {
				bestAbs = absWeight
				bestIndex = i
			}
		}
		if bestIndex == -1 {
			break
		}
		used[bestIndex] = true
		result = append(result, fmt.Sprintf("%s: %.2f", m.FeatureNames[bestIndex], m.Weights[bestIndex]))
	}

	return result
}

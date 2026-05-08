package indiahousing

import (
	"errors"
	"math"
	"math/rand"
)

// TrainingOptions хранит настройки обучения.
// TrainingOptions stores training settings.
type TrainingOptions struct {
	LearningRate float64
	Epochs       int
}

// Evaluation хранит метрики качества модели.
// Evaluation stores model quality metrics.
type Evaluation struct {
	RMSE float64
	MAE  float64
}

// TrainTestSplit делит данные на train и test, как это делал бы ML-фреймворк.
// TrainTestSplit splits data into train and test sets, like an ML framework would.
func TrainTestSplit(houses []House, testRatio float64, rng *rand.Rand) ([]House, []House, error) {
	if len(houses) < 2 {
		return nil, nil, errors.New("need at least two houses")
	}
	if testRatio <= 0 || testRatio >= 1 {
		return nil, nil, errors.New("testRatio must be between 0 and 1")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}

	shuffled := append([]House(nil), houses...)
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	testSize := int(float64(len(shuffled)) * testRatio)
	if testSize < 1 {
		testSize = 1
	}

	test := shuffled[:testSize]
	train := shuffled[testSize:]
	return train, test, nil
}

// TrainLinearRegression обучает линейную регрессию через batch gradient descent.
// TrainLinearRegression trains linear regression with batch gradient descent.
func TrainLinearRegression(houses []House, options TrainingOptions) (LinearModel, error) {
	if len(houses) == 0 {
		return LinearModel{}, errors.New("houses must not be empty")
	}
	if options.LearningRate <= 0 {
		return LinearModel{}, errors.New("learning rate must be positive")
	}
	if options.Epochs <= 0 {
		return LinearModel{}, errors.New("epochs must be positive")
	}

	encoder, err := NewEncoder(houses)
	if err != nil {
		return LinearModel{}, err
	}

	featureNames := encoder.FeatureNames()
	model := LinearModel{
		Encoder:      encoder,
		FeatureNames: featureNames,
		Weights:      make([]float64, len(featureNames)),
	}

	for range options.Epochs {
		weightGradients := make([]float64, len(model.Weights))
		var biasGradient float64

		for _, house := range houses {
			features := encoder.Encode(house)
			predictedPrice := model.Predict(house)
			predictionError := predictedPrice - house.PriceLakhs

			biasGradient += predictionError
			for i, featureValue := range features {
				weightGradients[i] += predictionError * featureValue
			}
		}

		n := float64(len(houses))
		model.Bias -= options.LearningRate * biasGradient / n
		for i := range model.Weights {
			model.Weights[i] -= options.LearningRate * weightGradients[i] / n
		}
	}

	return model, nil
}

// Evaluate считает RMSE и MAE на наборе домов.
// Evaluate calculates RMSE and MAE on a set of houses.
func Evaluate(model LinearModel, houses []House) (Evaluation, error) {
	if len(houses) == 0 {
		return Evaluation{}, errors.New("houses must not be empty")
	}

	var sumSquares float64
	var sumAbsolute float64
	for _, house := range houses {
		predictionError := house.PriceLakhs - model.Predict(house)
		sumSquares += predictionError * predictionError
		if predictionError < 0 {
			predictionError = -predictionError
		}
		sumAbsolute += predictionError
	}

	n := float64(len(houses))
	return Evaluation{
		RMSE: math.Sqrt(sumSquares / n),
		MAE:  sumAbsolute / n,
	}, nil
}

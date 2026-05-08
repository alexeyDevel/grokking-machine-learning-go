package indiahousing

import (
	"errors"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Evaluation хранит метрики качества модели.
// Evaluation stores model quality metrics.
type Evaluation struct {
	RMSE float64
	MAE  float64
}

// TrainTestSplit делит данные на train и test.
// TrainTestSplit splits data into train and test sets.
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

// TrainLinearRegression обучает линейную регрессию как sklearn.LinearRegression.
// Вместо итераций мы решаем задачу least squares через SVD.
// TrainLinearRegression trains linear regression like sklearn.LinearRegression.
// Instead of iterations, it solves the least-squares problem with SVD.
func TrainLinearRegression(houses []House) (LinearModel, error) {
	if len(houses) == 0 {
		return LinearModel{}, errors.New("houses must not be empty")
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

	rowCount := len(houses)
	columnCount := len(featureNames) + 1
	designMatrix := mat.NewDense(rowCount, columnCount, nil)
	targetPrices := mat.NewDense(rowCount, 1, nil)

	for rowIndex, house := range houses {
		features := encoder.Encode(house)

		// Первый столбец равен 1: так модель учит intercept/bias.
		// The first column is 1: this lets the model learn the intercept/bias.
		designMatrix.Set(rowIndex, 0, 1)
		for featureIndex, featureValue := range features {
			designMatrix.Set(rowIndex, featureIndex+1, featureValue)
		}
		targetPrices.Set(rowIndex, 0, house.PriceRupees)
	}

	var svd mat.SVD
	if ok := svd.Factorize(designMatrix, mat.SVDThin); !ok {
		return LinearModel{}, errors.New("could not factorize feature matrix")
	}

	rank := svd.Rank(1e-12)
	coefficients := mat.NewDense(columnCount, 1, nil)
	svd.SolveTo(coefficients, targetPrices, rank)

	model.Bias = coefficients.At(0, 0)
	for featureIndex := range model.Weights {
		model.Weights[featureIndex] = coefficients.At(featureIndex+1, 0)
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
		predictionError := house.PriceRupees - model.Predict(house)
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

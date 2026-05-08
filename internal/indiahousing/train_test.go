package indiahousing

import (
	"math"
	"math/rand"
	"testing"
)

func TestIndiaHousingPipeline(t *testing.T) {
	houses, err := LoadCSV("../../data/india_housing_sample.csv")
	if err != nil {
		t.Fatal(err)
	}

	train, test, err := TrainTestSplit(houses, 0.25, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := TrainLinearRegression(train, TrainingOptions{
		LearningRate: 0.03,
		Epochs:       12000,
	})
	if err != nil {
		t.Fatal(err)
	}

	evaluation, err := Evaluate(model, test)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(evaluation.RMSE) || math.IsInf(evaluation.RMSE, 0) {
		t.Fatalf("RMSE must be finite, got %f", evaluation.RMSE)
	}
	if evaluation.RMSE <= 0 || evaluation.RMSE > 80 {
		t.Fatalf("RMSE = %.2f, want a positive and reasonable value", evaluation.RMSE)
	}

	predictedPrice := model.Predict(House{
		City:      "Bengaluru",
		AreaSqFt:  1420,
		Bedrooms:  3,
		Bathrooms: 2,
		AgeYears:  3,
		NearMetro: true,
		Furnished: "Semi",
	})
	if predictedPrice <= 0 {
		t.Fatalf("predicted price must be positive, got %.2f", predictedPrice)
	}
}

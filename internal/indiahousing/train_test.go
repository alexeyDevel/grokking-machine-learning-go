package indiahousing

import (
	"math"
	"math/rand"
	"testing"
)

func TestIndiaHousingPipeline(t *testing.T) {
	houses, err := LoadCSV("Hyderabad.csv")
	if err != nil {
		t.Fatal(err)
	}

	train, test, err := TrainTestSplit(houses, 0.25, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := TrainLinearRegression(train)
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
	if evaluation.RMSE <= 0 || evaluation.RMSE > 10000000 {
		t.Fatalf("RMSE = %.2f, want a positive and reasonable value", evaluation.RMSE)
	}

	predictedPrice := model.Predict(House{
		Location: "Hitech City",
		AreaSqFt: 1420,
		Bedrooms: 3,
		Amenities: map[string]float64{
			"24X7Security":  1,
			"CarParking":    1,
			"Gymnasium":     1,
			"LiftAvailable": 1,
			"PowerBackup":   1,
			"SwimmingPool":  1,
		},
	})
	if predictedPrice <= 0 {
		t.Fatalf("predicted price must be positive, got %.2f", predictedPrice)
	}
}

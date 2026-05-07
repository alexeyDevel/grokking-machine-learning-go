package chapter03

import (
	"math"
	"math/rand"
	"testing"
)

func TestOrdinaryLeastSquares(t *testing.T) {
	model, err := OrdinaryLeastSquares(HousingDataset())
	if err != nil {
		t.Fatal(err)
	}

	assertClose(t, model.PricePerRoom, 50.39285714285714, 1e-9)
	assertClose(t, model.BasePrice, 99.59523809523819, 1e-9)
	assertClose(t, model.Predict(4), 301.16666666666674, 1e-9)
}

func TestLinearRegressionWithSquareTrick(t *testing.T) {
	dataset := HousingDataset()
	result, err := LinearRegression(dataset, 0.01, 10000, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Errors) != 10000 {
		t.Fatalf("len(result.Errors) = %d, want 10000", len(result.Errors))
	}

	initialError := result.Errors[0]
	finalError, err := RMSE(dataset.Labels, result.Model.PredictAll(dataset.Features))
	if err != nil {
		t.Fatal(err)
	}
	if finalError >= initialError {
		t.Fatalf("final RMSE %.2f should be lower than initial RMSE %.2f", finalError, initialError)
	}
	if finalError > 6 {
		t.Fatalf("final RMSE %.2f is unexpectedly high", finalError)
	}
}

func TestRMSE(t *testing.T) {
	got, err := RMSE(
		[]float64{155, 197, 244},
		[]float64{150, 200, 250},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertClose(t, got, math.Sqrt((25+9+36)/3.0), 1e-9)
}

func assertClose(t *testing.T, got, want, tolerance float64) {
	t.Helper()

	if math.Abs(got-want) > tolerance {
		t.Fatalf("got %.12f, want %.12f", got, want)
	}
}

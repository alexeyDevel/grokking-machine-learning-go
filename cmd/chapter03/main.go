package main

import (
	"fmt"
	"math/rand"

	"github.com/alexeyDevel/grokking-machine-learning-go/internal/chapter03"
)

func main() {
	dataset := chapter03.HousingDataset()

	trained, err := chapter03.LinearRegression(
		dataset,
		0.01,
		10000,
		rand.New(rand.NewSource(0)),
	)
	if err != nil {
		panic(err)
	}

	exact, err := chapter03.OrdinaryLeastSquares(dataset)
	if err != nil {
		panic(err)
	}

	trainedRMSE, err := chapter03.RMSE(dataset.Labels, trained.Model.PredictAll(dataset.Features))
	if err != nil {
		panic(err)
	}
	exactRMSE, err := chapter03.RMSE(dataset.Labels, exact.PredictAll(dataset.Features))
	if err != nil {
		panic(err)
	}

	fmt.Println("Chapter 3: Linear regression")
	fmt.Printf("features: %.0f\n", dataset.Features)
	fmt.Printf("labels:   %.0f\n\n", dataset.Labels)

	fmt.Println("Square trick SGD")
	fmt.Printf("price per room: %.6f\n", trained.Model.PricePerRoom)
	fmt.Printf("base price:     %.6f\n", trained.Model.BasePrice)
	fmt.Printf("RMSE:           %.6f\n", trainedRMSE)
	fmt.Printf("prediction for 4 rooms: %.6f\n\n", trained.Model.Predict(4))

	fmt.Println("Ordinary least squares")
	fmt.Printf("price per room: %.6f\n", exact.PricePerRoom)
	fmt.Printf("base price:     %.6f\n", exact.BasePrice)
	fmt.Printf("RMSE:           %.6f\n", exactRMSE)
	fmt.Printf("prediction for 4 rooms: %.6f\n", exact.Predict(4))
}

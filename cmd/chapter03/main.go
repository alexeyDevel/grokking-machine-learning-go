package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/alexeyDevel/grokking-machine-learning-go/internal/chapter03"
)

func main() {
	// dataset - маленькая таблица из главы: количество комнат -> цены.
	// dataset is the small chapter table: rooms -> prices.
	dataset := chapter03.HousingDataset()

	// trained - модель, обученная постепенными случайными шагами.
	// trained is a model trained by small random update steps.
	trained, err := chapter03.LinearRegressionWithSnapshots(
		dataset,
		0.01,                        // learningRate: размер одного шага обучения / one training step size.
		10000,                       // epochs: сколько раз обновлять модель / how many times to update the model.
		rand.New(rand.NewSource(0)), // rng: фиксированный генератор случайных чисел / fixed random number generator.
		1000,                        // snapshotEvery: сохранять прямую каждые 1000 эпох / save a line every 1000 epochs.
	)
	if err != nil {
		panic(err)
	}

	// exact - модель, найденная напрямую по формуле без случайного обучения.
	// exact is a model found directly by formula, without random training.
	exact, err := chapter03.OrdinaryLeastSquares(dataset)
	if err != nil {
		panic(err)
	}

	// trainedRMSE - ошибка модели, обученной через SquareTrick.
	// trainedRMSE is the error of the model trained with SquareTrick.
	trainedRMSE, err := chapter03.RMSE(dataset.ActualPrices, trained.Model.PredictAll(dataset.RoomCounts))
	if err != nil {
		panic(err)
	}

	// exactRMSE - ошибка точной модели OrdinaryLeastSquares.
	// exactRMSE is the error of the exact OrdinaryLeastSquares model.
	exactRMSE, err := chapter03.RMSE(dataset.ActualPrices, exact.PredictAll(dataset.RoomCounts))
	if err != nil {
		panic(err)
	}

	fmt.Println("Chapter 3: Linear regression")
	fmt.Printf("room counts:  %.0f\n", dataset.RoomCounts)
	fmt.Printf("actual prices: %.0f\n\n", dataset.ActualPrices)

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

	outputDir := filepath.Join("visualizations", "chapter03")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		panic(err)
	}

	regressionPath := filepath.Join(outputDir, "regression-training.svg")
	regressionSVG := chapter03.RegressionTrainingSVG(dataset, trained.Snapshots, exact)
	if err := os.WriteFile(regressionPath, []byte(regressionSVG), 0o644); err != nil {
		panic(err)
	}

	rmsePath := filepath.Join(outputDir, "rmse-training.svg")
	rmseSVG := chapter03.RMSETrainingSVG(trained.Errors)
	if err := os.WriteFile(rmsePath, []byte(rmseSVG), 0o644); err != nil {
		panic(err)
	}

	fmt.Printf("\nVisualizations saved:\n")
	fmt.Printf("- %s\n", regressionPath)
	fmt.Printf("- %s\n", rmsePath)
}

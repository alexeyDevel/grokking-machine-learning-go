package main

import (
	"fmt"
	"math/rand"

	"github.com/alexeyDevel/grokking-machine-learning-go/internal/indiahousing"
)

func main() {
	houses, err := indiahousing.LoadCSV("data/india_housing_sample.csv")
	if err != nil {
		panic(err)
	}

	train, test, err := indiahousing.TrainTestSplit(
		houses,
		0.25,
		rand.New(rand.NewSource(7)),
	)
	if err != nil {
		panic(err)
	}

	model, err := indiahousing.TrainLinearRegression(train, indiahousing.TrainingOptions{
		LearningRate: 0.03,
		Epochs:       12000,
	})
	if err != nil {
		panic(err)
	}

	trainEvaluation, err := indiahousing.Evaluate(model, train)
	if err != nil {
		panic(err)
	}
	testEvaluation, err := indiahousing.Evaluate(model, test)
	if err != nil {
		panic(err)
	}

	example := indiahousing.House{
		City:      "Bengaluru",
		AreaSqFt:  1420,
		Bedrooms:  3,
		Bathrooms: 2,
		AgeYears:  3,
		NearMetro: true,
		Furnished: "Semi",
	}
	predictedPrice := model.Predict(example)

	fmt.Println("India housing price prediction")
	fmt.Printf("rows: %d train / %d test\n\n", len(train), len(test))

	fmt.Println("Metrics, lakhs INR")
	fmt.Printf("train RMSE: %.2f, train MAE: %.2f\n", trainEvaluation.RMSE, trainEvaluation.MAE)
	fmt.Printf("test  RMSE: %.2f, test  MAE: %.2f\n\n", testEvaluation.RMSE, testEvaluation.MAE)

	fmt.Println("Example prediction")
	fmt.Printf("%s, %.0f sqft, %.0f bedrooms, %.0f bathrooms, %.0f years old, metro=%t, furnished=%s\n",
		example.City,
		example.AreaSqFt,
		example.Bedrooms,
		example.Bathrooms,
		example.AgeYears,
		example.NearMetro,
		example.Furnished,
	)
	fmt.Printf("predicted price: %.2f lakhs INR\n\n", predictedPrice)

	fmt.Println("Strongest learned weights")
	for _, line := range model.ExplainTopWeights(8) {
		fmt.Println("-", line)
	}
}

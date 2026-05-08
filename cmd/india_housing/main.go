package main

import (
	"fmt"
	"math/rand"

	"github.com/alexeyDevel/grokking-machine-learning-go/internal/indiahousing"
)

func main() {
	houses, err := indiahousing.LoadCSV("internal/indiahousing/Hyderabad.csv")
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

	model, err := indiahousing.TrainLinearRegression(train)
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
	}
	predictedPrice := model.Predict(example)

	fmt.Println("Hyderabad housing price prediction")
	fmt.Printf("rows: %d train / %d test\n\n", len(train), len(test))

	fmt.Println("Metrics, INR")
	fmt.Printf("train RMSE: %.2f, train MAE: %.2f\n", trainEvaluation.RMSE, trainEvaluation.MAE)
	fmt.Printf("test  RMSE: %.2f, test  MAE: %.2f\n\n", testEvaluation.RMSE, testEvaluation.MAE)

	fmt.Println("Example prediction")
	fmt.Printf("%s, %.0f sqft, %.0f bedrooms, amenities: gym=%t, pool=%t, parking=%t\n",
		example.Location,
		example.AreaSqFt,
		example.Bedrooms,
		example.Amenities["Gymnasium"] == 1,
		example.Amenities["SwimmingPool"] == 1,
		example.Amenities["CarParking"] == 1,
	)
	fmt.Printf("predicted price: %.2f INR, about %.2f lakhs INR\n\n", predictedPrice, predictedPrice/indiahousing.RupeesInLakh)

	fmt.Println("Strongest learned weights")
	for _, line := range model.ExplainTopWeights(8) {
		fmt.Println("-", line)
	}
}

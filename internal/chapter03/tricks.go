package chapter03

import "math/rand"

func SimpleTrick(model Model, numRooms, price float64, rng *rand.Rand) Model {
	smallRandom1 := rng.Float64() * 0.1
	smallRandom2 := rng.Float64() * 0.1
	predictedPrice := model.Predict(numRooms)

	if price > predictedPrice && numRooms > 0 {
		model.PricePerRoom += smallRandom1
		model.BasePrice += smallRandom2
	}
	if price > predictedPrice && numRooms < 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}
	if price < predictedPrice && numRooms > 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice -= smallRandom2
	}
	if price < predictedPrice && numRooms < 0 {
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}

	return model
}

func AbsoluteTrick(model Model, numRooms, price, learningRate float64) Model {
	predictedPrice := model.Predict(numRooms)
	if price > predictedPrice {
		model.PricePerRoom += learningRate * numRooms
		model.BasePrice += learningRate
	} else {
		model.PricePerRoom -= learningRate * numRooms
		model.BasePrice -= learningRate
	}

	return model
}

func SquareTrick(model Model, numRooms, price, learningRate float64) Model {
	predictedPrice := model.Predict(numRooms)
	errorValue := price - predictedPrice

	model.PricePerRoom += learningRate * numRooms * errorValue
	model.BasePrice += learningRate * errorValue

	return model
}

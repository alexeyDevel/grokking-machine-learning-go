package chapter03

import "math/rand"

// SimpleTrick слегка меняет наклон и свободный член в случайную сторону,
// если предсказание оказалось выше или ниже настоящей цены.
// SimpleTrick slightly changes the slope and intercept in a random direction
// when the prediction is above or below the actual price.
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

// AbsoluteTrick двигает прямую к точке на фиксированный шаг learningRate.
// AbsoluteTrick moves the line toward the point by a fixed learningRate step.
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

// SquareTrick обновляет параметры пропорционально ошибке предсказания.
// Чем дальше точка от прямой, тем сильнее модель сдвигается к ней.
// SquareTrick updates the parameters in proportion to the prediction error.
// The farther the point is from the line, the stronger the model moves toward it.
func SquareTrick(model Model, numRooms, price, learningRate float64) Model {
	predictedPrice := model.Predict(numRooms)
	errorValue := price - predictedPrice

	model.PricePerRoom += learningRate * numRooms * errorValue
	model.BasePrice += learningRate * errorValue

	return model
}

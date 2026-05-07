package chapter03

import "math/rand"

// SimpleTrick слегка меняет наклон и свободный член в случайную сторону,
// если предсказание оказалось выше или ниже настоящей цены.
// SimpleTrick slightly changes the slope and intercept in a random direction
// when the prediction is above or below the actual price.
func SimpleTrick(model Model, roomCount, actualPrice float64, rng *rand.Rand) Model {
	// smallRandom1 и smallRandom2 - маленькие случайные шаги в диапазоне [0.0, 0.1).
	// smallRandom1 and smallRandom2 are small random steps in the range [0.0, 0.1).
	smallRandom1 := rng.Float64() * 0.1
	smallRandom2 := rng.Float64() * 0.1

	// predictedPrice - цена, которую текущая модель предсказывает для roomCount.
	// predictedPrice is the price predicted by the current model for roomCount.
	predictedPrice := model.Predict(roomCount)

	if actualPrice > predictedPrice && roomCount > 0 {
		// Настоящая цена выше предсказания, значит поднимаем прямую.
		// The actual price is above the prediction, so we move the line upward.
		model.PricePerRoom += smallRandom1
		model.BasePrice += smallRandom2
	}
	if actualPrice > predictedPrice && roomCount < 0 {
		// В учебной формуле учтён случай отрицательного x, хотя в нашем датасете комнат меньше 0 нет.
		// The book formula handles negative x, although our room dataset has no negative values.
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}
	if actualPrice < predictedPrice && roomCount > 0 {
		// Настоящая цена ниже предсказания, значит опускаем прямую.
		// The actual price is below the prediction, so we move the line downward.
		model.PricePerRoom -= smallRandom1
		model.BasePrice -= smallRandom2
	}
	if actualPrice < predictedPrice && roomCount < 0 {
		// Ещё один случай для отрицательного x из общей идеи simple trick.
		// One more negative-x case from the general simple trick idea.
		model.PricePerRoom -= smallRandom1
		model.BasePrice += smallRandom2
	}

	return model
}

// AbsoluteTrick двигает прямую к точке на фиксированный шаг learningRate.
// AbsoluteTrick moves the line toward the point by a fixed learningRate step.
func AbsoluteTrick(model Model, roomCount, actualPrice, learningRate float64) Model {
	// learningRate - размер шага обучения: чем он больше, тем сильнее меняются параметры.
	// learningRate is the training step size: the larger it is, the stronger the parameter update.
	predictedPrice := model.Predict(roomCount)
	if actualPrice > predictedPrice {
		// Если модель занизила цену, увеличиваем наклон и базовую цену.
		// If the model predicted too low, increase the slope and base price.
		model.PricePerRoom += learningRate * roomCount
		model.BasePrice += learningRate
	} else {
		// Если модель завысила цену, уменьшаем наклон и базовую цену.
		// If the model predicted too high, decrease the slope and base price.
		model.PricePerRoom -= learningRate * roomCount
		model.BasePrice -= learningRate
	}

	return model
}

// SquareTrick обновляет параметры пропорционально ошибке предсказания.
// Чем дальше точка от прямой, тем сильнее модель сдвигается к ней.
// SquareTrick updates the parameters in proportion to the prediction error.
// The farther the point is from the line, the stronger the model moves toward it.
func SquareTrick(model Model, roomCount, actualPrice, learningRate float64) Model {
	// predictedPrice - текущее предсказание модели для выбранной точки.
	// predictedPrice is the current model prediction for the selected point.
	predictedPrice := model.Predict(roomCount)

	// predictionError - разница между настоящей ценой и предсказанием.
	// Если значение положительное, модель предсказала слишком низко.
	// predictionError is the difference between the actual price and the prediction.
	// If it is positive, the model predicted too low.
	predictionError := actualPrice - predictedPrice

	// Обновляем наклон: ошибка умножается на roomCount, потому что наклон отвечает за влияние признака.
	// Update the slope: the error is multiplied by roomCount because slope controls the feature impact.
	model.PricePerRoom += learningRate * roomCount * predictionError

	// Обновляем базовую цену: она двигает всю прямую вверх или вниз.
	// Update the base price: it moves the whole line up or down.
	model.BasePrice += learningRate * predictionError

	return model
}

package chapter03

// Dataset хранит учебные данные для задачи линейной регрессии.
// Dataset stores training data for a linear regression task.
type Dataset struct {
	// RoomCounts - входные значения модели: количество комнат в каждом доме.
	// RoomCounts are model inputs: the number of rooms in each house.
	RoomCounts []float64

	// ActualPrices - правильные ответы: настоящие цены домов.
	// ActualPrices are correct answers: the real house prices.
	ActualPrices []float64
}

// HousingDataset возвращает учебный набор данных из главы:
// количество комнат в доме и соответствующая цена.
// HousingDataset returns the chapter's training dataset:
// the number of rooms in a house and the corresponding price.
func HousingDataset() Dataset {
	return Dataset{
		// 1, 2, 3, 5, 6, 7 - количество комнат.
		// 1, 2, 3, 5, 6, 7 are room counts.
		RoomCounts: []float64{1, 2, 3, 5, 6, 7},

		// 155, 197, ... - цены для домов с соответствующим количеством комнат.
		// 155, 197, ... are prices for houses with the corresponding room counts.
		ActualPrices: []float64{155, 197, 244, 356, 407, 448},
	}
}

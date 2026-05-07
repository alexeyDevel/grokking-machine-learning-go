package chapter03

// Dataset хранит признаки и правильные ответы для задачи линейной регрессии.
// Dataset stores features and labels for a linear regression task.
type Dataset struct {
	// Features - входные значения модели: в этой главе это количество комнат.
	// Features are model inputs: in this chapter, the number of rooms.
	Features []float64

	// Labels - правильные ответы: в этой главе это цены домов.
	// Labels are correct answers: in this chapter, house prices.
	Labels []float64
}

// HousingDataset возвращает учебный набор данных из главы:
// количество комнат в доме и соответствующая цена.
// HousingDataset returns the chapter's training dataset:
// the number of rooms in a house and the corresponding price.
func HousingDataset() Dataset {
	return Dataset{
		// 1, 2, 3, 5, 6, 7 - количество комнат.
		// 1, 2, 3, 5, 6, 7 are room counts.
		Features: []float64{1, 2, 3, 5, 6, 7},

		// 155, 197, ... - цены для домов с соответствующим количеством комнат.
		// 155, 197, ... are prices for houses with the corresponding room counts.
		Labels: []float64{155, 197, 244, 356, 407, 448},
	}
}

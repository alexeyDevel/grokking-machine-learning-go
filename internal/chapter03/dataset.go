package chapter03

// Dataset хранит признаки и правильные ответы для задачи линейной регрессии.
// Dataset stores features and labels for a linear regression task.
type Dataset struct {
	Features []float64
	Labels   []float64
}

// HousingDataset возвращает учебный набор данных из главы:
// количество комнат в доме и соответствующая цена.
// HousingDataset returns the chapter's training dataset:
// the number of rooms in a house and the corresponding price.
func HousingDataset() Dataset {
	return Dataset{
		Features: []float64{1, 2, 3, 5, 6, 7},
		Labels:   []float64{155, 197, 244, 356, 407, 448},
	}
}

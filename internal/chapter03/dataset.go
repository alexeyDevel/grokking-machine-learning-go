package chapter03

type Dataset struct {
	Features []float64
	Labels   []float64
}

func HousingDataset() Dataset {
	return Dataset{
		Features: []float64{1, 2, 3, 5, 6, 7},
		Labels:   []float64{155, 197, 244, 356, 407, 448},
	}
}

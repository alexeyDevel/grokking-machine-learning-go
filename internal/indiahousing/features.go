package indiahousing

import (
	"fmt"
	"math"
	"sort"
)

type featureStats struct {
	Mean float64
	Std  float64
}

// Encoder превращает House в числовые признаки для линейной модели.
// Encoder converts House values into numeric features for a linear model.
type Encoder struct {
	Cities      []string
	Furnishings []string

	AreaStats     featureStats
	BedroomStats  featureStats
	BathroomStats featureStats
	AgeStats      featureStats
}

// NewEncoder создаёт кодировщик по обучающему набору данных.
// NewEncoder creates an encoder from the training dataset.
func NewEncoder(houses []House) (Encoder, error) {
	if len(houses) == 0 {
		return Encoder{}, fmt.Errorf("houses must not be empty")
	}

	citySet := map[string]bool{}
	furnishingSet := map[string]bool{}
	areas := make([]float64, len(houses))
	bedrooms := make([]float64, len(houses))
	bathrooms := make([]float64, len(houses))
	ages := make([]float64, len(houses))

	for i, house := range houses {
		citySet[house.City] = true
		furnishingSet[house.Furnished] = true
		areas[i] = house.AreaSqFt
		bedrooms[i] = house.Bedrooms
		bathrooms[i] = house.Bathrooms
		ages[i] = house.AgeYears
	}

	return Encoder{
		Cities:        sortedKeys(citySet),
		Furnishings:   sortedKeys(furnishingSet),
		AreaStats:     calculateStats(areas),
		BedroomStats:  calculateStats(bedrooms),
		BathroomStats: calculateStats(bathrooms),
		AgeStats:      calculateStats(ages),
	}, nil
}

// FeatureNames возвращает названия признаков в том же порядке, что и Encode.
// FeatureNames returns feature names in the same order as Encode.
func (e Encoder) FeatureNames() []string {
	names := []string{"area_sqft", "bedrooms", "bathrooms", "age_years", "near_metro"}
	for _, city := range e.Cities[1:] {
		names = append(names, "city_"+city)
	}
	for _, furnishing := range e.Furnishings[1:] {
		names = append(names, "furnished_"+furnishing)
	}
	return names
}

// Encode превращает один дом в числовой вектор признаков.
// Encode converts one house into a numeric feature vector.
func (e Encoder) Encode(house House) []float64 {
	features := []float64{
		normalize(house.AreaSqFt, e.AreaStats),
		normalize(house.Bedrooms, e.BedroomStats),
		normalize(house.Bathrooms, e.BathroomStats),
		normalize(house.AgeYears, e.AgeStats),
		boolAsFloat(house.NearMetro),
	}

	// Первую категорию пропускаем: она становится базовой.
	// Skip the first category: it becomes the baseline.
	for _, city := range e.Cities[1:] {
		features = append(features, boolAsFloat(house.City == city))
	}
	for _, furnishing := range e.Furnishings[1:] {
		features = append(features, boolAsFloat(house.Furnished == furnishing))
	}

	return features
}

func sortedKeys(values map[string]bool) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func calculateStats(values []float64) featureStats {
	var sum float64
	for _, value := range values {
		sum += value
	}
	mean := sum / float64(len(values))

	var squaredDiffs float64
	for _, value := range values {
		diff := value - mean
		squaredDiffs += diff * diff
	}
	std := math.Sqrt(squaredDiffs / float64(len(values)))
	if std == 0 {
		std = 1
	}

	return featureStats{Mean: mean, Std: std}
}

func normalize(value float64, stats featureStats) float64 {
	return (value - stats.Mean) / stats.Std
}

func boolAsFloat(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

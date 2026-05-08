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

// Encoder превращает строки Hyderabad.csv в числовые признаки для линейной модели.
// Encoder converts Hyderabad.csv rows into numeric features for a linear model.
type Encoder struct {
	Locations []string
	Amenities []string

	AreaStats    featureStats
	BedroomStats featureStats
}

// NewEncoder создаёт кодировщик по обучающему набору данных.
// NewEncoder creates an encoder from the training dataset.
func NewEncoder(houses []House) (Encoder, error) {
	if len(houses) == 0 {
		return Encoder{}, fmt.Errorf("houses must not be empty")
	}

	locationSet := map[string]bool{}
	amenitySeen := map[string]bool{}
	amenities := []string{}
	areas := make([]float64, len(houses))
	bedrooms := make([]float64, len(houses))

	for i, house := range houses {
		locationSet[house.Location] = true
		areas[i] = house.AreaSqFt
		bedrooms[i] = house.Bedrooms

		for amenity := range house.Amenities {
			if !amenitySeen[amenity] {
				amenitySeen[amenity] = true
				amenities = append(amenities, amenity)
			}
		}
	}
	sort.Strings(amenities)

	return Encoder{
		Locations:    sortedKeys(locationSet),
		Amenities:    amenities,
		AreaStats:    calculateStats(areas),
		BedroomStats: calculateStats(bedrooms),
	}, nil
}

// FeatureNames возвращает названия признаков в том же порядке, что и Encode.
// FeatureNames returns feature names in the same order as Encode.
func (e Encoder) FeatureNames() []string {
	names := []string{"Area", "No. of Bedrooms"}
	for _, amenity := range e.Amenities {
		names = append(names, amenity)
	}
	for _, location := range e.Locations[1:] {
		names = append(names, "Location_"+location)
	}
	return names
}

// Encode превращает один объект недвижимости в числовой вектор признаков.
// Encode converts one property into a numeric feature vector.
func (e Encoder) Encode(house House) []float64 {
	features := []float64{
		normalize(house.AreaSqFt, e.AreaStats),
		normalize(house.Bedrooms, e.BedroomStats),
	}

	for _, amenity := range e.Amenities {
		features = append(features, house.Amenities[amenity])
	}
	// Первую location пропускаем: она становится базовой категорией.
	// Skip the first location: it becomes the baseline category.
	for _, location := range e.Locations[1:] {
		features = append(features, boolAsFloat(house.Location == location))
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

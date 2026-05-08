package indiahousing

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
)

// House хранит одну строку датасета с индийской недвижимостью.
// House stores one row from the Indian housing dataset.
type House struct {
	City       string
	AreaSqFt   float64
	Bedrooms   float64
	Bathrooms  float64
	AgeYears   float64
	NearMetro  bool
	Furnished  string
	PriceLakhs float64
}

// LoadCSV загружает данные о жилье из CSV-файла.
// LoadCSV loads housing data from a CSV file.
func LoadCSV(path string) ([]House, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	if len(header) != 8 {
		return nil, fmt.Errorf("expected 8 columns, got %d", len(header))
	}

	var houses []House
	for rowNumber := 2; ; rowNumber++ {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row %d: %w", rowNumber, err)
		}

		house, err := parseHouse(record)
		if err != nil {
			return nil, fmt.Errorf("parse row %d: %w", rowNumber, err)
		}
		houses = append(houses, house)
	}

	return houses, nil
}

func parseHouse(record []string) (House, error) {
	if len(record) != 8 {
		return House{}, fmt.Errorf("expected 8 values, got %d", len(record))
	}

	areaSqFt, err := strconv.ParseFloat(record[1], 64)
	if err != nil {
		return House{}, err
	}
	bedrooms, err := strconv.ParseFloat(record[2], 64)
	if err != nil {
		return House{}, err
	}
	bathrooms, err := strconv.ParseFloat(record[3], 64)
	if err != nil {
		return House{}, err
	}
	ageYears, err := strconv.ParseFloat(record[4], 64)
	if err != nil {
		return House{}, err
	}
	nearMetro, err := strconv.ParseBool(record[5])
	if err != nil {
		return House{}, err
	}
	priceLakhs, err := strconv.ParseFloat(record[7], 64)
	if err != nil {
		return House{}, err
	}

	return House{
		City:       record[0],
		AreaSqFt:   areaSqFt,
		Bedrooms:   bedrooms,
		Bathrooms:  bathrooms,
		AgeYears:   ageYears,
		NearMetro:  nearMetro,
		Furnished:  record[6],
		PriceLakhs: priceLakhs,
	}, nil
}

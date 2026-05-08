package indiahousing

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
)

const RupeesInLakh = 100000.0

// House хранит одну строку Hyderabad.csv.
// House stores one row from Hyderabad.csv.
type House struct {
	Location    string
	AreaSqFt    float64
	Bedrooms    float64
	Amenities   map[string]float64
	PriceRupees float64
}

// LoadCSV загружает Hyderabad.csv так же, как pd.read_csv в notebook.
// LoadCSV loads Hyderabad.csv similarly to pd.read_csv in the notebook.
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
	if len(header) < 4 {
		return nil, fmt.Errorf("expected at least 4 columns, got %d", len(header))
	}

	columns := mapColumns(header)
	requiredColumns := []string{"Price", "Area", "Location", "No. of Bedrooms"}
	for _, column := range requiredColumns {
		if _, ok := columns[column]; !ok {
			return nil, fmt.Errorf("missing required column %q", column)
		}
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

		house, err := parseHyderabadHouse(header, columns, record)
		if err != nil {
			return nil, fmt.Errorf("parse row %d: %w", rowNumber, err)
		}
		houses = append(houses, house)
	}

	return houses, nil
}

func mapColumns(header []string) map[string]int {
	columns := make(map[string]int, len(header))
	for i, column := range header {
		columns[column] = i
	}
	return columns
}

func parseHyderabadHouse(header []string, columns map[string]int, record []string) (House, error) {
	if len(record) != len(header) {
		return House{}, fmt.Errorf("expected %d values, got %d", len(header), len(record))
	}

	priceRupees, err := parseFloat(record[columns["Price"]])
	if err != nil {
		return House{}, fmt.Errorf("price: %w", err)
	}
	areaSqFt, err := parseFloat(record[columns["Area"]])
	if err != nil {
		return House{}, fmt.Errorf("area: %w", err)
	}
	bedrooms, err := parseFloat(record[columns["No. of Bedrooms"]])
	if err != nil {
		return House{}, fmt.Errorf("bedrooms: %w", err)
	}

	amenities := map[string]float64{}
	for _, column := range header[4:] {
		value, err := parseFloat(record[columns[column]])
		if err != nil {
			return House{}, fmt.Errorf("%s: %w", column, err)
		}
		amenities[column] = value
	}

	return House{
		Location:    record[columns["Location"]],
		AreaSqFt:    areaSqFt,
		Bedrooms:    bedrooms,
		Amenities:   amenities,
		PriceRupees: priceRupees,
	}, nil
}

func parseFloat(value string) (float64, error) {
	return strconv.ParseFloat(value, 64)
}

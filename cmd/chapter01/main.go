package main

import (
	"fmt"

	"github.com/alexeyDevel/grokking-machine-learning-go/internal/chapter01"
)

func main() {
	point := chapter01.Point{X: 2, Y: 3}
	prediction := chapter01.ClassifyByLine(point, chapter01.Line{
		Slope:     1,
		Intercept: 0,
	})

	fmt.Printf("point: (%.1f, %.1f)\n", point.X, point.Y)
	fmt.Printf("prediction: %s\n", prediction)
}

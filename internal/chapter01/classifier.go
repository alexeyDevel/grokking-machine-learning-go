package chapter01

type Point struct {
	X float64
	Y float64
}

type Line struct {
	Slope     float64
	Intercept float64
}

func ClassifyByLine(point Point, line Line) string {
	yOnLine := line.Slope*point.X + line.Intercept
	if point.Y >= yOnLine {
		return "above"
	}

	return "below"
}

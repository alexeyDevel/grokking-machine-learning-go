package chapter03

import (
	"fmt"
	"html"
	"math"
	"strings"
)

// RegressionTrainingSVG строит SVG с точками датасета и несколькими прямыми обучения.
// RegressionTrainingSVG builds an SVG with dataset points and several training lines.
func RegressionTrainingSVG(dataset Dataset, snapshots []Model, exact Model) string {
	const (
		width   = 760.0
		height  = 520.0
		left    = 72.0
		right   = 32.0
		top     = 42.0
		bottom  = 66.0
		minX    = 0.0
		maxX    = 8.0
		minY    = 0.0
		maxY    = 500.0
		lineMin = 0.0
		lineMax = 8.0
	)

	plotWidth := width - left - right
	plotHeight := height - top - bottom
	xScale := func(roomCount float64) float64 {
		return left + (roomCount-minX)/(maxX-minX)*plotWidth
	}
	yScale := func(price float64) float64 {
		return top + (maxY-price)/(maxY-minY)*plotHeight
	}

	var b strings.Builder
	writeSVGHeader(&b, width, height, "Chapter 3: Linear regression training")
	writeAxes(&b, left, top, plotWidth, plotHeight, "room count", "price")

	for i, model := range snapshots {
		opacity := 0.12 + 0.55*float64(i)/math.Max(1, float64(len(snapshots)-1))
		color := "#6b7280"
		if i == len(snapshots)-1 {
			color = "#2563eb"
			opacity = 0.95
		}
		x1 := xScale(lineMin)
		y1 := yScale(model.Predict(lineMin))
		x2 := xScale(lineMax)
		y2 := yScale(model.Predict(lineMax))
		fmt.Fprintf(&b, `<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="2" opacity="%.2f"/>`, x1, y1, x2, y2, color, opacity)
	}

	fmt.Fprintf(
		&b,
		`<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="7 5"/>`,
		xScale(lineMin),
		yScale(exact.Predict(lineMin)),
		xScale(lineMax),
		yScale(exact.Predict(lineMax)),
	)

	for i := range dataset.RoomCounts {
		fmt.Fprintf(
			&b,
			`<circle cx="%.2f" cy="%.2f" r="5" fill="#111827"/>`,
			xScale(dataset.RoomCounts[i]),
			yScale(dataset.ActualPrices[i]),
		)
	}

	writeLegend(&b, width-258, 70, []legendItem{
		{Color: "#6b7280", Label: "intermediate SGD lines"},
		{Color: "#2563eb", Label: "final SGD line"},
		{Color: "#dc2626", Label: "ordinary least squares"},
		{Color: "#111827", Label: "actual prices"},
	})
	b.WriteString(`</svg>`)
	return b.String()
}

// RMSETrainingSVG строит SVG-график падения RMSE по эпохам.
// RMSETrainingSVG builds an SVG chart showing how RMSE changes by epoch.
func RMSETrainingSVG(errors []float64) string {
	const (
		width  = 760.0
		height = 420.0
		left   = 72.0
		right  = 32.0
		top    = 42.0
		bottom = 66.0
	)

	plotWidth := width - left - right
	plotHeight := height - top - bottom
	maxError := maxFloat(errors)
	if maxError == 0 {
		maxError = 1
	}

	xScale := func(epoch int) float64 {
		if len(errors) <= 1 {
			return left
		}
		return left + float64(epoch)/float64(len(errors)-1)*plotWidth
	}
	yScale := func(errorValue float64) float64 {
		return top + (maxError-errorValue)/maxError*plotHeight
	}

	var b strings.Builder
	writeSVGHeader(&b, width, height, "Chapter 3: RMSE during training")
	writeAxes(&b, left, top, plotWidth, plotHeight, "epoch", "RMSE")

	b.WriteString(`<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="`)
	step := 1
	if len(errors) > 600 {
		step = len(errors) / 600
	}
	for i := 0; i < len(errors); i += step {
		fmt.Fprintf(&b, "%.2f,%.2f ", xScale(i), yScale(errors[i]))
	}
	if len(errors) > 0 {
		last := len(errors) - 1
		fmt.Fprintf(&b, "%.2f,%.2f", xScale(last), yScale(errors[last]))
	}
	b.WriteString(`"/>`)

	if len(errors) > 0 {
		fmt.Fprintf(&b, `<text x="%.2f" y="%.2f" font-size="13" fill="#111827">start RMSE: %.2f</text>`, left+18, top+24, errors[0])
		fmt.Fprintf(&b, `<text x="%.2f" y="%.2f" font-size="13" fill="#111827">final RMSE: %.2f</text>`, left+18, top+44, errors[len(errors)-1])
	}

	b.WriteString(`</svg>`)
	return b.String()
}

type legendItem struct {
	Color string
	Label string
}

func writeSVGHeader(b *strings.Builder, width, height float64, title string) {
	fmt.Fprintf(b, `<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">`, width, height, width, height)
	b.WriteString(`<rect width="100%" height="100%" fill="#ffffff"/>`)
	fmt.Fprintf(b, `<text x="24" y="26" font-size="18" font-family="Arial, sans-serif" font-weight="700" fill="#111827">%s</text>`, html.EscapeString(title))
}

func writeAxes(b *strings.Builder, left, top, plotWidth, plotHeight float64, xLabel, yLabel string) {
	b.WriteString(`<g font-family="Arial, sans-serif" fill="#374151">`)
	fmt.Fprintf(b, `<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#f9fafb" stroke="#d1d5db"/>`, left, top, plotWidth, plotHeight)
	for i := 0; i <= 4; i++ {
		x := left + float64(i)/4*plotWidth
		y := top + float64(i)/4*plotHeight
		fmt.Fprintf(b, `<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#e5e7eb"/>`, x, top, x, top+plotHeight)
		fmt.Fprintf(b, `<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#e5e7eb"/>`, left, y, left+plotWidth, y)
	}
	fmt.Fprintf(b, `<text x="%.2f" y="%.2f" text-anchor="middle" font-size="14">%s</text>`, left+plotWidth/2, top+plotHeight+44, html.EscapeString(xLabel))
	fmt.Fprintf(b, `<text x="20" y="%.2f" transform="rotate(-90 20 %.2f)" text-anchor="middle" font-size="14">%s</text>`, top+plotHeight/2, top+plotHeight/2, html.EscapeString(yLabel))
	b.WriteString(`</g>`)
}

func writeLegend(b *strings.Builder, x, y float64, items []legendItem) {
	b.WriteString(`<g font-family="Arial, sans-serif" font-size="13" fill="#111827">`)
	for i, item := range items {
		rowY := y + float64(i)*22
		fmt.Fprintf(b, `<rect x="%.2f" y="%.2f" width="14" height="14" fill="%s"/>`, x, rowY-11, item.Color)
		fmt.Fprintf(b, `<text x="%.2f" y="%.2f">%s</text>`, x+22, rowY, html.EscapeString(item.Label))
	}
	b.WriteString(`</g>`)
}

func maxFloat(values []float64) float64 {
	var maxValue float64
	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}

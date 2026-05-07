package chapter01

import "testing"

func TestClassifyByLine(t *testing.T) {
	line := Line{Slope: 1, Intercept: 0}

	tests := []struct {
		name  string
		point Point
		want  string
	}{
		{name: "above line", point: Point{X: 2, Y: 3}, want: "above"},
		{name: "on line", point: Point{X: 2, Y: 2}, want: "above"},
		{name: "below line", point: Point{X: 2, Y: 1}, want: "below"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyByLine(tt.point, line)
			if got != tt.want {
				t.Fatalf("ClassifyByLine(%v, %v) = %q, want %q", tt.point, line, got, tt.want)
			}
		})
	}
}

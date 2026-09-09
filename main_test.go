package main

import (
	"math"
	"testing"
)

func TestCoverGradientMatchesLogLoss(t *testing.T) {
	r := []float64{1.1, 0.9, 1.2}
	u := []float64{0.2, 0.3, 0.5}
	epsilon := 1e-6
	gradient := coverGradient(r, u)
	for i := range u {
		plus := append([]float64(nil), u...)
		minus := append([]float64(nil), u...)
		plus[i] += epsilon
		minus[i] -= epsilon
		finiteDifference := (-math.Log(dot(r, plus)) + math.Log(dot(r, minus))) / (2 * epsilon)
		if math.Abs(gradient[i]-finiteDifference) > 1e-6 {
			t.Fatalf("gradient[%d] = %v, finite difference = %v", i, gradient[i], finiteDifference)
		}
	}
}

func TestProjectInterior(t *testing.T) {
	projected := projectInterior([]float64{2, -1, 0}, 1e-6)
	if math.Abs(sum(projected)-1) > 1e-12 {
		t.Fatalf("projected weights sum to %v", sum(projected))
	}
	for i, value := range projected {
		if value <= 0 || !isFinite(value) {
			t.Fatalf("projected[%d] = %v", i, value)
		}
	}
}

func TestDONSUpdateIsCausal(t *testing.T) {
	dons := NewDONS(2, NStock, BetaStock)
	predicted := dons.Predict(10)
	if math.Abs(predicted[0]-0.5) > 1e-12 || math.Abs(predicted[1]-0.5) > 1e-12 {
		t.Fatalf("unexpected initial prediction: %v", predicted)
	}
	dons.Update([]float64{1.6, 0.5}, 10)
	next := dons.Predict(10)
	if math.Abs(sum(next)-1) > 1e-12 || next[0] <= 0 || next[1] <= 0 {
		t.Fatalf("invalid post-update prediction: %v", next)
	}
}

func TestMetaPredictionNormalizes(t *testing.T) {
	meta := NewMetaDONS(3, 8, EtaStock)
	predicted := meta.Predict(0)
	if math.Abs(sum(predicted)-1) > 1e-12 {
		t.Fatalf("meta prediction sums to %v", sum(predicted))
	}
	meta.Update([]float64{1.1, 0.95, 1.02}, 0)
	if len(meta.experts) == 0 {
		t.Fatal("meta did not create an expert")
	}
}

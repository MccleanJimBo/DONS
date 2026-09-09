package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
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

func TestExecutionMode(t *testing.T) {
	portfolio := []float64{0.5, 0.3, 0.2}
	if got := dot(topK(portfolio, 2), []float64{1, 1, 1}); math.Abs(got-1) > 1e-12 {
		t.Fatalf("top-k portfolio is not normalized: %v", got)
	}
	if got := dot(portfolio, []float64{1, 1, 1}); math.Abs(got-1) > 1e-12 {
		t.Fatalf("dense portfolio changed unexpectedly: %v", got)
	}
}

func TestDONSParametersPropagateToExperts(t *testing.T) {
	meta := NewMetaDONS(2, 8, EtaStock)
	meta.SetDONSParameters(1.5, 0.1)
	meta.Predict(0)
	if len(meta.experts) != 1 {
		t.Fatal("expected one initial expert")
	}
	if meta.experts[0].dons.gamma != 1.5 || meta.experts[0].dons.momentum != 0.1 {
		t.Fatalf("parameters did not propagate: gamma=%v momentum=%v", meta.experts[0].dons.gamma, meta.experts[0].dons.momentum)
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

func TestMetaUsesOverlappingGeometricIntervals(t *testing.T) {
	meta := NewMetaDONS(3, 16, EtaStock)
	meta.Predict(2)
	if len(meta.experts) != 3 {
		t.Fatalf("expected experts at starts 0, 1, and 2, got %d", len(meta.experts))
	}
	active := meta.activeExperts(2)
	if len(active) != 2 {
		t.Fatalf("expected two active overlapping experts at t=2, got %d", len(active))
	}
	for _, expert := range active {
		if expert.start > 2 || expert.end <= 2 {
			t.Fatalf("expert interval [%d, %d) does not cover t=2", expert.start, expert.end)
		}
	}
}

func TestBarrierAdaptsAndStaysFinite(t *testing.T) {
	dons := NewDONS(3, NStock, BetaStock)
	initial := append([]float64(nil), dons.p...)
	dons.updateP([]float64{0.8, 0.1, 0.1})
	if dons.p[0] <= initial[0] {
		t.Fatalf("barrier coordinate did not adapt: before %v, after %v", initial[0], dons.p[0])
	}
	nt := dons.computeNt(100)
	gradient := barrierGrad([]float64{0.8, 0.1, 0.1}, nt)
	hessian := barrierHessian([]float64{0.8, 0.1, 0.1}, nt)
	for i := range gradient {
		if !isFinite(nt[i]) || !isFinite(gradient[i]) || !isFinite(hessian[i][i]) || hessian[i][i] <= 0 {
			t.Fatalf("invalid barrier derivative at %d: nt=%v gradient=%v hessian=%v", i, nt[i], gradient[i], hessian[i][i])
		}
	}
}

func TestSolveConstrainedNewtonRespectsSimplexConstraint(t *testing.T) {
	H := mat.NewDense(3, 3, []float64{
		2, 0, 0,
		0, 3, 0,
		0, 0, 4,
	})
	grad := []float64{1, -1, 0}
	delta, err := solveConstrainedNewton(H, grad)
	if err != nil {
		t.Fatalf("solveConstrainedNewton returned error: %v", err)
	}
	if len(delta) != 3 {
		t.Fatalf("unexpected delta length: %d", len(delta))
	}
	if math.Abs(sum(delta)) > 1e-8 {
		t.Fatalf("constrained Newton step violates simplex feasibility: sum(delta)=%v", sum(delta))
	}
	for i := range delta {
		if !isFinite(delta[i]) {
			t.Fatalf("non-finite Newton step component %d: %v", i, delta[i])
		}
	}
}

func TestMetaRetiresExpiredExperts(t *testing.T) {
	meta := NewMetaDONS(3, 16, EtaStock)
	meta.ensureExperts(8)
	if len(meta.experts) == 0 {
		t.Fatal("expected experts to be created")
	}
	for _, e := range meta.experts {
		e.end = 8
	}
	meta.retireExpiredExperts(8)
	if len(meta.experts) != 0 {
		t.Fatalf("expected all expired experts to be retired, but kept %d", len(meta.experts))
	}
}

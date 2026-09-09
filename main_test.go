package main

import (
	"math"
	"testing"
	"time"

	"dons/data"

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

func TestCheckedCoverGradientRejectsInvalidReturn(t *testing.T) {
	if _, err := checkedCoverGradient([]float64{0, 0}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("expected zero portfolio return to be rejected")
	}
	if _, err := checkedCoverGradient([]float64{math.NaN(), 1}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("expected non-finite return to be rejected")
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
	selected := topK(portfolio, 2)
	if got := dot(selected, []float64{1, 1, 1}); math.Abs(got-1) > 1e-12 {
		t.Fatalf("top-k portfolio is not normalized: %v", got)
	}
	positive := 0
	for _, weight := range selected {
		if weight > 0 {
			positive++
		}
	}
	if positive != 2 || selected[2] != 0 {
		t.Fatalf("top-k portfolio did not keep exactly two assets: %v", selected)
	}
	if got := dot(portfolio, []float64{1, 1, 1}); math.Abs(got-1) > 1e-12 {
		t.Fatalf("dense portfolio changed unexpectedly: %v", got)
	}
}

func TestTopKZeroWeightsFallsBackToUniformSelection(t *testing.T) {
	selected := topK([]float64{0, 0, 0}, 2)
	if math.Abs(sum(selected)-1) > 1e-12 {
		t.Fatalf("zero-weight top-k fallback is not normalized: %v", selected)
	}
	if selected[0] != 0.5 || selected[1] != 0.5 || selected[2] != 0 {
		t.Fatalf("unexpected zero-weight top-k fallback: %v", selected)
	}
}

func TestBuildReturnMatrixAuditsForwardFillAndDates(t *testing.T) {
	day1 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	day2 := day1.AddDate(0, 0, 1)
	day3 := day1.AddDate(0, 0, 2)
	prices := map[string][]data.PricesStruct{
		"A": {{Date: day1, Prices: []float64{100}}, {Date: day3, Prices: []float64{200}}},
		"B": {{Date: day1, Prices: []float64{100}}, {Date: day2, Prices: []float64{50}}, {Date: day3, Prices: []float64{75}}},
	}
	returns, dates, audit, err := BuildReturnMatrix(prices, data.Universe{Tickers: []string{"A", "B"}})
	if err != nil {
		t.Fatalf("BuildReturnMatrix returned error: %v", err)
	}
	if len(returns) != 2 || len(dates) != 2 || !dates[0].Equal(day2) || !dates[1].Equal(day3) {
		t.Fatalf("returns are not aligned to endpoint dates: returns=%v dates=%v", returns, dates)
	}
	if audit.ForwardFilledPrices != 1 {
		t.Fatalf("expected one forward-filled price, got %+v", audit)
	}
	if math.Abs(returns[0][0]-1) > 1e-12 || math.Abs(returns[1][0]-1.5) > 1e-12 {
		t.Fatalf("unexpected aligned returns: %v", returns)
	}
	if audit.ClampedHighReturns != 1 {
		t.Fatalf("expected one high-return clamp, got %+v", audit)
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

func TestConfigValidatesAndPropagates(t *testing.T) {
	config := DefaultConfig()
	if config.BarrierStrength != 0.25 || config.CurvatureResetThreshold != 1e5 || config.QuadraticDecay != 0.99 || config.QuadraticUpdateScale != 1.0 {
		t.Fatalf("unexpected optimizer defaults: barrier=%v curvature reset=%v quadratic decay=%v update scale=%v", config.BarrierStrength, config.CurvatureResetThreshold, config.QuadraticDecay, config.QuadraticUpdateScale)
	}
	config.BarrierStrength = 4
	config.Beta = 0.25
	config.QuadraticDecay = 0.9
	config.QuadraticUpdateScale = 0.5
	config.Eta = 0.2
	config.Gamma = 1.5
	config.Momentum = 0.1
	config.TopK = 2
	if err := config.Validate(); err != nil {
		t.Fatalf("default-derived config should validate: %v", err)
	}
	meta := NewMetaDONSWithConfig(3, 8, config)
	meta.Predict(0)
	if len(meta.experts) != 1 {
		t.Fatal("expected configured meta learner to create an expert")
	}
	expert := meta.experts[0]
	if expert.dons.n != config.BarrierStrength || expert.dons.beta != config.Beta || expert.dons.quadraticDecay != config.QuadraticDecay || expert.dons.quadraticUpdateScale != config.QuadraticUpdateScale {
		t.Fatalf("config did not reach expert: n=%v beta=%v decay=%v scale=%v", expert.dons.n, expert.dons.beta, expert.dons.quadraticDecay, expert.dons.quadraticUpdateScale)
	}
	invalid := config
	invalid.Gamma = 0
	if err := invalid.Validate(); err == nil {
		t.Fatal("expected zero gamma to be rejected")
	}
	invalid = config
	invalid.QuadraticDecay = 0
	if err := invalid.Validate(); err == nil {
		t.Fatal("expected zero quadratic decay to be rejected")
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

func TestDONSAsymmetricReturnProducesFeasibleMovement(t *testing.T) {
	dons := NewDONS(3, NStock, BetaStock)
	dons.Predict(10)
	dons.Update([]float64{1.2, 0.9, 1.0}, 10)
	next := dons.Predict(10)
	if math.Abs(sum(next)-1) > 1e-12 {
		t.Fatalf("updated portfolio is not normalized: %v", next)
	}
	if math.Abs(next[0]-next[1]) < 1e-12 {
		t.Fatalf("asymmetric return did not produce portfolio movement: %v", next)
	}
	if dons.lastProjectedGradientNorm <= 0 {
		t.Fatalf("projected gradient was not recorded: %v", dons.lastProjectedGradientNorm)
	}
}

func TestDONSUpdateClearsPendingPrediction(t *testing.T) {
	dons := NewDONS(2, NStock, BetaStock)
	dons.Predict(10)
	dons.Update([]float64{1.1, 0.9}, 10)
	if dons.pending != nil {
		t.Fatal("DONS update retained a stale pending prediction")
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
	if len(meta.experts) != 2 {
		t.Fatalf("expected active-lifetime experts at starts 1 and 2, got %d", len(meta.experts))
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

func TestMetaExpertLifecycleAndLocalHorizon(t *testing.T) {
	meta := NewMetaDONS(3, 16, EtaStock)
	meta.Predict(0)
	if len(meta.experts) != 1 || meta.experts[0].state != ExpertActive {
		t.Fatalf("expected initial expert to be active: %+v", meta.experts)
	}
	if meta.experts[0].localHorizon != 2 {
		t.Fatalf("unexpected initial expert horizon: %d", meta.experts[0].localHorizon)
	}
	meta.Predict(2)
	if meta.retired != 1 {
		t.Fatalf("expected one retired expert at t=2, got %d", meta.retired)
	}
	for _, expert := range meta.activeExperts(2) {
		if expert.state != ExpertActive || expert.localHorizon != expert.end-expert.start {
			t.Fatalf("invalid active expert lifecycle: %+v", expert)
		}
		if !isFinite(expert.priorLogWeight) || !isFinite(expert.logWeight) {
			t.Fatalf("invalid expert prior: %+v", expert)
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

func TestConstrainedNewtonReportsResiduals(t *testing.T) {
	H := mat.NewDense(2, 2, []float64{
		2, 0,
		0, 3,
	})
	result, err := solveConstrainedNewtonDetailed(H, []float64{1, -1})
	if err != nil {
		t.Fatalf("solveConstrainedNewtonDetailed returned error: %v", err)
	}
	if result.kktResidual > 1e-10 {
		t.Fatalf("KKT residual is too large: %v", result.kktResidual)
	}
	if result.relativeKKTResidual > 1e-10 {
		t.Fatalf("relative KKT residual is too large: %v", result.relativeKKTResidual)
	}
	if result.constraintResidual > 1e-10 {
		t.Fatalf("constraint residual is too large: %v", result.constraintResidual)
	}
}

func TestDONSUpdateTracksNewtonDiagnostics(t *testing.T) {
	dons := NewDONS(3, NStock, BetaStock)
	dons.Predict(10)
	dons.Update([]float64{1.1, 0.95, 1.02}, 10)
	if !isFinite(dons.lastNewtonDecrement) || dons.lastNewtonDecrement < 0 {
		t.Fatalf("invalid Newton decrement: %v", dons.lastNewtonDecrement)
	}
	if !isFinite(dons.lastKKTResidual) || !isFinite(dons.lastConstraintResidual) {
		t.Fatalf("invalid Newton residuals: KKT=%v constraint=%v", dons.lastKKTResidual, dons.lastConstraintResidual)
	}
	if !isFinite(dons.lastRelativeKKTResidual) {
		t.Fatalf("invalid relative KKT residual: %v", dons.lastRelativeKKTResidual)
	}
	if !isFinite(dons.lastHessianRegularization) || dons.lastHessianRegularization < 0 {
		t.Fatalf("invalid Hessian regularization: %v", dons.lastHessianRegularization)
	}
	if !isFinite(dons.lastGradientNorm) || !isFinite(dons.lastProjectedGradientNorm) ||
		!isFinite(dons.lastBarrierHessianNorm) || !isFinite(dons.lastQuadraticHessianNorm) ||
		!isFinite(dons.lastCombinedHessianNorm) {
		t.Fatalf("invalid gradient/Hessian scales: gradient=%v projected=%v barrier=%v quadratic=%v combined=%v",
			dons.lastGradientNorm, dons.lastProjectedGradientNorm, dons.lastBarrierHessianNorm,
			dons.lastQuadraticHessianNorm, dons.lastCombinedHessianNorm)
	}
}

func TestObjectiveComponentsSumToTotal(t *testing.T) {
	w := []float64{0.5, 0.3, 0.2}
	nt := []float64{2.0, 1.5, 1.0}
	q := mat.NewDense(3, 3, []float64{
		0.2, 0.02, 0.01,
		0.02, 0.3, 0.01,
		0.01, 0.01, 0.4,
	})
	linear := []float64{0.2, -0.3, 0.1}
	offset := []float64{0.05, 0.1, -0.05}
	obj := objectiveComponents(w, nt, q, offset, linear)
	if !isFinite(obj.Barrier) || !isFinite(obj.LinearLoss) || !isFinite(obj.Quadratic) || !isFinite(obj.Offset) || !isFinite(obj.Total) {
		t.Fatalf("objective components are not finite: %+v", obj)
	}
	if math.Abs(obj.Total-(obj.Barrier+obj.LinearLoss+obj.Quadratic+obj.Offset)) > 1e-8 {
		t.Fatalf("objective decomposition does not sum to total: total=%v components=%+v", obj.Total, obj)
	}
}

func TestDiagnosticsCaptureFinalRowExpertsBeforeAdvance(t *testing.T) {
	meta := NewMetaDONS(3, 8, EtaStock)
	meta.ensureExperts(2)
	if len(meta.experts) == 0 {
		t.Fatal("expected expert set to exist before meauring final-row diagnostics")
	}
	beforeLastUpdate := meta.activeExperts(2)
	if len(beforeLastUpdate) == 0 {
		t.Fatal("final-row expert snapshot was empty before the last update")
	}
	meta.Update([]float64{0.9, 1.1, 1.05}, 2)
	if len(meta.activeExperts(2)) == 0 {
		t.Fatal("final-row expert snapshot was lost after the last update")
	}
	if len(beforeLastUpdate) == 0 {
		t.Fatal("final-row experts were not retained for the diagnostic row")
	}
}

func TestInteriorStepCapAndLogWealth(t *testing.T) {
	step := maxInteriorStep([]float64{0.1, 0.9}, []float64{-1, 1}, 1e-3)
	if step <= 0 || step >= 0.1 {
		t.Fatalf("interior step cap is invalid: %v", step)
	}
	if got := wealthFromLog(math.Log(2)); math.Abs(got-2) > 1e-12 {
		t.Fatalf("wealth conversion failed: %v", got)
	}
	if got := wealthFromLog(math.Log(math.MaxFloat64) + 1); got != math.MaxFloat64 {
		t.Fatalf("wealth overflow was not capped: %v", got)
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

func syntheticMetaBacktest(returns [][]float64) ([]float64, [][]float64) {
	meta := NewMetaDONS(len(returns[0]), len(returns), EtaStock)
	wealth := 1.0
	wealthPath := []float64{wealth}
	portfolios := make([][]float64, 0, len(returns))

	for t, currentReturn := range returns {
		portfolio := meta.Predict(t)
		portfolios = append(portfolios, append([]float64(nil), portfolio...))
		wealth *= dot(currentReturn, portfolio)
		wealthPath = append(wealthPath, wealth)
		meta.Update(currentReturn, t)
	}
	return wealthPath, portfolios
}

func assertSyntheticBacktestValid(t *testing.T, wealth []float64, portfolios [][]float64) {
	t.Helper()
	if len(wealth) != len(portfolios)+1 {
		t.Fatalf("wealth path length %d does not match portfolio count %d", len(wealth), len(portfolios))
	}
	for i, value := range wealth {
		if !isFinite(value) || value <= 0 {
			t.Fatalf("wealth[%d] = %v", i, value)
		}
	}
	for period, portfolio := range portfolios {
		if math.Abs(sum(portfolio)-1) > 1e-10 {
			t.Fatalf("portfolio[%d] sums to %v", period, sum(portfolio))
		}
		for i, weight := range portfolio {
			if !isFinite(weight) || weight <= 0 {
				t.Fatalf("portfolio[%d][%d] = %v", period, i, weight)
			}
		}
	}
}

func TestSyntheticConstantReturnBacktest(t *testing.T) {
	returns := make([][]float64, 12)
	for t := range returns {
		returns[t] = []float64{1, 1, 1}
	}
	wealth, portfolios := syntheticMetaBacktest(returns)
	assertSyntheticBacktestValid(t, wealth, portfolios)
	if math.Abs(wealth[len(wealth)-1]-1) > 1e-10 {
		t.Fatalf("constant-return wealth changed: %v", wealth[len(wealth)-1])
	}
}

func TestSyntheticAlternatingWinnerBacktestIsCausalAndRepeatable(t *testing.T) {
	returns := [][]float64{
		{1.20, 0.90, 1.00},
		{0.90, 1.20, 1.00},
		{1.20, 0.90, 1.00},
		{0.90, 1.20, 1.00},
		{1.20, 0.90, 1.00},
		{0.90, 1.20, 1.00},
		{1.20, 0.90, 1.00},
		{0.90, 1.20, 1.00},
	}
	wealth, portfolios := syntheticMetaBacktest(returns)
	repeatedWealth, repeatedPortfolios := syntheticMetaBacktest(returns)
	assertSyntheticBacktestValid(t, wealth, portfolios)
	assertSyntheticBacktestValid(t, repeatedWealth, repeatedPortfolios)

	if math.Abs(sum(portfolios[0])-1) > 1e-10 || math.Abs(portfolios[0][0]-1.0/3) > 1e-10 {
		t.Fatalf("first portfolio was not selected before observing returns: %v", portfolios[0])
	}
	if math.Abs(wealth[len(wealth)-1]-repeatedWealth[len(repeatedWealth)-1]) > 1e-12 {
		t.Fatalf("repeated synthetic backtests disagree: %v versus %v", wealth[len(wealth)-1], repeatedWealth[len(repeatedWealth)-1])
	}
	for period := range portfolios {
		for i := range portfolios[period] {
			if math.Abs(portfolios[period][i]-repeatedPortfolios[period][i]) > 1e-12 {
				t.Fatalf("portfolio[%d] is not repeatable: %v versus %v", period, portfolios[period], repeatedPortfolios[period])
			}
		}
	}
}

func TestRunTestAlignsFirstReturnWithFirstDate(t *testing.T) {
	dates := []time.Time{
		time.Date(2025, 1, 2, 0, 0, 0, 0, time.UTC),
		time.Date(2025, 1, 3, 0, 0, 0, 0, time.UTC),
	}
	returns := [][]float64{{1.1}, {1.2}}
	prices := map[string][]data.PricesStruct{
		"A": {
			{Date: dates[0].AddDate(0, 0, -1), Prices: []float64{100}},
			{Date: dates[0], Prices: []float64{110}},
		},
	}
	meta := NewMetaDONS(1, len(returns), EtaStock)
	wealth, _, _, _, _, _ := runTest(
		returns,
		dates,
		[]string{"A"},
		prices,
		[]float64{1},
		[]float64{1},
		meta,
		0,
	)
	if len(wealth) != len(returns)+1 {
		t.Fatalf("expected initial wealth plus one point per return, got %d", len(wealth))
	}
	if math.Abs(wealth[1]-1.1) > 1e-12 {
		t.Fatalf("first return was not applied at the first endpoint: %v", wealth)
	}
	if math.Abs(wealth[2]-1.32) > 1e-12 {
		t.Fatalf("wealth does not include both aligned returns: %v", wealth)
	}
}

package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	// Your data loader package
	"dons/data"
)

const (
	NStock    = 2.0  // barrier strength
	BetaStock = 0.5  // quadratic regularization
	EtaStock  = 0.15 // meta learning rate
)

type Config struct {
	BarrierStrength         float64
	BarrierDecay            float64
	BarrierResetThreshold   float64
	CurvatureResetThreshold float64
	Beta                    float64
	QuadraticDecay          float64
	QuadraticUpdateScale    float64
	Eta                     float64
	Gamma                   float64
	Momentum                float64
	TopK                    int
	TransactionCost         float64
}

func DefaultConfig() Config {
	return Config{
		BarrierStrength:         0.25,
		BarrierDecay:            1.0,
		BarrierResetThreshold:   1e-8,
		CurvatureResetThreshold: 1e5,
		Beta:                    BetaStock,
		QuadraticDecay:          0.99,
		QuadraticUpdateScale:    1.0,
		Eta:                     EtaStock,
		Gamma:                   3.0,
		Momentum:                0.0,
		TopK:                    0,
		TransactionCost:         0.0,
	}
}

func (c Config) Validate() error {
	if c.BarrierStrength <= 0 || !isFinite(c.BarrierStrength) {
		return errors.New("barrier strength must be finite and positive")
	}
	if c.BarrierDecay <= 0 || c.BarrierDecay > 1 || !isFinite(c.BarrierDecay) {
		return errors.New("barrier decay must be finite and in (0, 1]")
	}
	if c.BarrierResetThreshold < 0 || !isFinite(c.BarrierResetThreshold) {
		return errors.New("barrier reset threshold must be finite and non-negative")
	}
	if c.CurvatureResetThreshold <= 0 || !isFinite(c.CurvatureResetThreshold) {
		return errors.New("curvature reset threshold must be finite and positive")
	}
	if c.Beta < 0 || !isFinite(c.Beta) {
		return errors.New("beta must be finite and non-negative")
	}
	if c.QuadraticDecay <= 0 || c.QuadraticDecay > 1 || !isFinite(c.QuadraticDecay) {
		return errors.New("quadratic decay must be finite and in (0, 1]")
	}
	if c.QuadraticUpdateScale < 0 || !isFinite(c.QuadraticUpdateScale) {
		return errors.New("quadratic update scale must be finite and non-negative")
	}
	if c.Eta < 0 || !isFinite(c.Eta) {
		return errors.New("eta must be finite and non-negative")
	}
	if c.Gamma <= 0 || !isFinite(c.Gamma) {
		return errors.New("gamma must be finite and positive")
	}
	if c.Momentum < 0 || c.Momentum >= 1 || !isFinite(c.Momentum) {
		return errors.New("momentum must be finite, at least 0, and less than 1")
	}
	if c.TopK < 0 {
		return errors.New("top-k must be non-negative")
	}
	if c.TransactionCost < 0 || !isFinite(c.TransactionCost) {
		return errors.New("transaction cost must be finite and non-negative")
	}
	return nil
}

// ─────────────────────────────────────────────
// Expert struct
// ─────────────────────────────────────────────
type ExpertState int

const (
	ExpertSleeping ExpertState = iota
	ExpertActive
	ExpertRetired
)

type Expert struct {
	dons              *DONS
	start             int
	end               int
	localHorizon      int
	state             ExpertState
	priorLogWeight    float64
	logWeight         float64
	portfolio         []float64
	numericalFailures int
}

type MetaDONS struct {
	d        int
	T        int
	eta      float64
	gamma    float64
	momentum float64
	config   Config
	experts  []*Expert
	retired  int
}

// ─────────────────────────────────────────────
// DONS struct
// ─────────────────────────────────────────────
type BarrierState struct {
	p []float64
}

func (b *BarrierState) update(u []float64) {
	if b == nil {
		return
	}
	if len(b.p) != len(u) {
		b.p = make([]float64, len(u))
	}
	for i := range u {
		if 2*b.p[i] < u[i] {
			// p_{t+1,i} = max{p_{t,i}, u_{t,i}} with the paper's adaptive barrier threshold.
			b.p[i] = math.Max(u[i], 1e-12)
		}
	}
}

func (b *BarrierState) nt(T float64, n float64) []float64 {
	if b == nil {
		return nil
	}
	base := math.Max(1.0/float64(len(b.p)), 1e-12)
	nt := make([]float64, len(b.p))
	for i := range nt {
		// n_{t,i} = n * exp(log(p_{t,i}/p0) * log(T+1)), where p0 = 1/d is the reference barrier scale.
		factor := math.Max(b.p[i], 1e-12) / base
		if T <= 1 {
			nt[i] = n
			continue
		}
		logNt := math.Log(math.Max(n, 1e-300)) + math.Log(factor)*math.Log1p(T)
		logNt = math.Min(logNt, math.Log(math.MaxFloat64))
		nt[i] = math.Exp(logNt)
	}
	return nt
}

type DONS struct {
	d                             int
	n                             float64
	beta                          float64
	gamma                         float64
	momentum                      float64
	barrierDecay                  float64
	barrierResetThreshold         float64
	curvatureResetThreshold       float64
	quadraticDecay                float64
	quadraticUpdateScale          float64
	epsilon                       float64
	w                             []float64
	p                             []float64
	barrier                       *BarrierState
	quadHess                      *mat.Dense
	quadOffset                    []float64
	last                          []float64
	pending                       []float64
	lastNewtonDecrement           float64
	lastKKTResidual               float64
	lastRelativeKKTResidual       float64
	lastConstraintResidual        float64
	lastHessianRegularization     float64
	lastNewtonAccepted            bool
	lastRawStepNorm               float64
	lastAcceptedStepNorm          float64
	lastStepSize                  float64
	lastBacktrackingAttempts      int
	lastObjectiveBefore           float64
	lastObjectiveAfter            float64
	lastBarrierObjective          float64
	lastLinearLossObjective       float64
	lastQuadraticObjective        float64
	lastOffsetObjective           float64
	lastTotalObjective            float64
	lastProjectionFallback        bool
	lastMaxWeightChange           float64
	lastGradientNorm              float64
	lastProjectedGradientNorm     float64
	lastBarrierHessianNorm        float64
	lastQuadraticHessianNorm      float64
	lastCombinedHessianNorm       float64
	lastQuadraticMemoryBeforeNorm float64
	lastQuadraticUpdateNorm       float64
	lastQuadraticMemoryAfterNorm  float64
	lastQuadraticDecay            float64
	lastCurvatureReset            bool
}

func NewMetaDONS(d, T int, eta float64) *MetaDONS {
	config := DefaultConfig()
	config.Eta = eta
	return NewMetaDONSWithConfig(d, T, config)
}

func NewMetaDONSWithConfig(d, T int, config Config) *MetaDONS {
	return &MetaDONS{
		d:        d,
		T:        T,
		eta:      config.Eta,
		gamma:    config.Gamma,
		momentum: config.Momentum,
		config:   config,
		experts:  []*Expert{},
	}
}

func (m *MetaDONS) SetDONSParameters(gamma, momentum float64) {
	m.gamma = gamma
	m.momentum = momentum
	m.config.Gamma = gamma
	m.config.Momentum = momentum
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────
func newBarrierState(d int) *BarrierState {
	p := make([]float64, d)
	base := 1.0 / float64(d)
	for i := range p {
		p[i] = 0.5 * base
	}
	return &BarrierState{p: p}
}

func NewDONS(d int, n, beta float64) *DONS {
	return NewDONSWithConfig(d, n, beta, DefaultConfig())
}

func NewDONSWithConfig(d int, n, beta float64, config Config) *DONS {
	w1 := make([]float64, d)
	for i := range w1 {
		w1[i] = 1.0 / float64(d)
	}

	barrier := newBarrierState(d)
	return &DONS{
		d:                       d,
		n:                       n,
		beta:                    beta,
		gamma:                   3.0,
		momentum:                0.0,
		barrierDecay:            config.BarrierDecay,
		barrierResetThreshold:   config.BarrierResetThreshold,
		curvatureResetThreshold: config.CurvatureResetThreshold,
		quadraticDecay:          config.QuadraticDecay,
		quadraticUpdateScale:    config.QuadraticUpdateScale,
		epsilon:                 1e-9,
		w:                       w1,
		p:                       append([]float64(nil), barrier.p...),
		barrier:                 barrier,
		quadHess:                mat.NewDense(d, d, nil),
		quadOffset:              make([]float64, d),
	}
}

func (m *MetaDONS) newExpert(start, end int) *Expert {
	localHorizon := maxInt(1, end-start)
	priorLogWeight := -math.Log(float64(localHorizon))
	dons := NewDONSWithConfig(m.d, m.config.BarrierStrength, m.config.Beta, m.config)
	dons.gamma = m.gamma
	dons.momentum = m.momentum
	return &Expert{
		dons:           dons,
		start:          start,
		end:            end,
		localHorizon:   localHorizon,
		state:          ExpertSleeping,
		priorLogWeight: priorLogWeight,
		logWeight:      priorLogWeight,
	}
}

func (m *MetaDONS) intervalEnd(start int) int {
	if start == 0 {
		return minInt(m.T, 2)
	}
	return minInt(m.T, 3*start)
}

func (m *MetaDONS) ensureExperts(t int) {
	seen := make(map[int]bool, len(m.experts))
	for _, e := range m.experts {
		seen[e.start] = true
	}

	if !seen[0] && m.intervalEnd(0) > t {
		e := m.newExpert(0, m.intervalEnd(0))
		m.experts = append(m.experts, e)
		seen[0] = true
	}

	for start := 1; start <= t && start < m.T; start *= 2 {
		if seen[start] {
			continue
		}
		e := m.newExpert(start, m.intervalEnd(start))
		if e.end <= t {
			continue
		}
		m.experts = append(m.experts, e)
		seen[start] = true
	}

	sort.Slice(m.experts, func(i, j int) bool {
		return m.experts[i].start < m.experts[j].start
	})
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ─────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────

func add(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func scale(a []float64, c float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * c
	}
	return out
}

// ─────────────────────────────────────────────
// Simplex projection
// ─────────────────────────────────────────────

func projectSimplex(v []float64) []float64 {
	n := len(v)
	u := make([]float64, n)
	copy(u, v)

	// sort descending
	sort.Slice(u, func(i, j int) bool { return u[i] > u[j] })

	// find rho
	var rho int
	var sum float64
	for i := 0; i < n; i++ {
		sum += u[i]
		if u[i]+(1.0-sum)/float64(i+1) > 0 {
			rho = i
		}
	}

	// compute theta
	sum = 0
	for i := 0; i <= rho; i++ {
		sum += u[i]
	}
	theta := (sum - 1.0) / float64(rho+1)

	// project
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = math.Max(v[i]-theta, 0)
	}
	return w
}

func projectInterior(v []float64, epsilon float64) []float64 {
	if epsilon*float64(len(v)) >= 1 {
		epsilon = 1 / (2 * float64(len(v)))
	}
	shifted := make([]float64, len(v))
	for i, value := range v {
		shifted[i] = value - epsilon
	}
	projected := projectSimplex(shifted)
	for i := range projected {
		projected[i] = epsilon + (1-float64(len(v))*epsilon)*projected[i]
	}
	return projected
}

// ─────────────────────────────────────────────
// Cover gradient
// ─────────────────────────────────────────────

func coverGradient(r, u []float64) []float64 {
	den := dot(r, u)
	g := make([]float64, len(r))
	for i := range r {
		g[i] = -r[i] / den
	}
	return g
}

func checkedCoverGradient(r, u []float64) ([]float64, error) {
	if len(r) == 0 || len(r) != len(u) {
		return nil, errors.New("return and portfolio dimensions do not match")
	}
	for _, value := range append(append([]float64(nil), r...), u...) {
		if !isFinite(value) {
			return nil, errors.New("return or portfolio contains a non-finite value")
		}
	}
	den := dot(r, u)
	if !isFinite(den) || den <= 1e-12 {
		return nil, errors.New("portfolio return is too small for a stable gradient")
	}
	gradient := make([]float64, len(r))
	for i := range r {
		gradient[i] = -r[i] / den
	}
	return gradient, nil
}

func validBarrierInputs(w, nt []float64) bool {
	if len(w) == 0 || len(w) != len(nt) {
		return false
	}
	for i := range w {
		if !isFinite(w[i]) || w[i] <= 0 || !isFinite(nt[i]) || nt[i] <= 0 {
			return false
		}
	}
	return true
}

// ─────────────────────────────────────────────
// Barrier + quadratic terms
// ─────────────────────────────────────────────

func barrierGrad(u, nt []float64) []float64 {
	g := make([]float64, len(u))
	for i := range u {
		g[i] = -nt[i] / u[i]
	}
	return g
}

func barrierHessian(u, nt []float64) [][]float64 {
	H := make([][]float64, len(u))
	for i := range u {
		H[i] = make([]float64, len(u))
		H[i][i] = nt[i] / (u[i] * u[i])
	}
	return H
}

func addQuadraticTerm(dons *DONS, g, w []float64) {
	dons.lastQuadraticMemoryBeforeNorm = matrixInfinityNorm(dons.quadHess)
	dons.lastQuadraticDecay = dons.quadraticDecay
	if dons.quadraticDecay != 1 {
		for i := 0; i < dons.d; i++ {
			for j := 0; j < dons.d; j++ {
				dons.quadHess.Set(i, j, dons.quadHess.At(i, j)*dons.quadraticDecay)
			}
			dons.quadOffset[i] *= dons.quadraticDecay
		}
	}
	update := mat.NewDense(dons.d, dons.d, nil)
	for i := 0; i < dons.d; i++ {
		anchor := 0.0
		for j := 0; j < dons.d; j++ {
			coefficient := (dons.beta * dons.quadraticUpdateScale / 4.0) * g[i] * g[j]
			dons.quadHess.Set(i, j, dons.quadHess.At(i, j)+coefficient)
			update.Set(i, j, coefficient)
			anchor += coefficient * w[j]
		}
		dons.quadOffset[i] -= anchor
	}
	dons.lastQuadraticUpdateNorm = matrixInfinityNorm(update)
	dons.lastQuadraticMemoryAfterNorm = matrixInfinityNorm(dons.quadHess)
}

func (m *MetaDONS) Predict(t int) []float64 {
	m.retireExpiredExperts(t)
	m.ensureExperts(t)
	active := m.activeExperts(t)
	if len(active) == 0 {
		uniform := make([]float64, m.d)
		for i := range uniform {
			uniform[i] = 1.0 / float64(m.d)
		}
		return uniform
	}

	// Meta aggregation:
	//   w_t = sum_{E in active} p(E | t) * u_t^E, with p(E | t) proportional to exp(log w_t(E)).
	// This is the geometric-interval expert mixture used to combine active experts.
	w := make([]float64, m.d)
	logZ := math.Inf(-1)
	for _, e := range active {
		e.portfolio = e.dons.Predict(e.localHorizon)
		logZ = logAdd(logZ, e.logWeight)
	}
	for _, e := range active {
		weight := math.Exp(e.logWeight - logZ)
		if !isFinite(weight) || weight < 0 {
			weight = 0
		}
		w = add(w, scale(e.portfolio, weight))
	}
	if sum(w) <= 0 {
		for i := range w {
			w[i] = 1.0 / float64(m.d)
		}
		return w
	}
	return projectInterior(w, 1e-9)
}

func (m *MetaDONS) Update(r []float64, t int) {
	m.retireExpiredExperts(t)
	for _, e := range m.activeExperts(t) {
		if len(e.portfolio) == 0 {
			e.portfolio = make([]float64, m.d)
			for i := range e.portfolio {
				e.portfolio[i] = 1.0 / float64(m.d)
			}
		}
		value := dot(r, e.portfolio)
		if value <= 0 || !isFinite(value) {
			e.numericalFailures++
			if e.numericalFailures >= 3 {
				e.state = ExpertRetired
				e.end = t
			}
			continue
		}
		// Expert weight update in the paper's notation:
		//   log w_{t+1}(E) = log w_t(E) - eta * ell_t(u_t^E)
		// with ell_t(u) = -log <r_t, u>.
		e.logWeight -= m.eta * (-math.Log(value))
		e.dons.Update(r, e.localHorizon)
	}
}

func (m *MetaDONS) retireExpiredExperts(t int) {
	kept := m.experts[:0]
	for _, e := range m.experts {
		if e.end <= t {
			e.state = ExpertRetired
			m.retired++
			continue
		}
		kept = append(kept, e)
	}
	m.experts = kept
}

func (m *MetaDONS) activeExperts(t int) []*Expert {
	active := make([]*Expert, 0, len(m.experts))
	for _, e := range m.experts {
		if e.state == ExpertRetired {
			continue
		}
		if e.start <= t && t < e.end {
			e.state = ExpertActive
			active = append(active, e)
		} else {
			e.state = ExpertSleeping
		}
	}
	return active
}

func logAdd(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}
func topK(w []float64, K int) []float64 {
	if K <= 0 || len(w) == 0 {
		return make([]float64, len(w))
	}
	if K > len(w) {
		K = len(w)
	}
	// Top‑K filtering
	type pair struct {
		idx int
		val float64
	}
	arr := make([]pair, len(w))
	for i := range w {
		arr[i] = pair{i, w[i]}
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].val > arr[j].val
	})

	// Keep only top K
	wSparse := make([]float64, len(w))
	for i := 0; i < K; i++ {
		wSparse[arr[i].idx] = arr[i].val
	}

	// Normalize ONLY top‑K entries
	sumTop := 0.0
	for _, v := range wSparse {
		sumTop += v
	}
	for i := range wSparse {
		if wSparse[i] > 0 {
			wSparse[i] /= sumTop
		}
	}
	if sumTop <= 0 || !isFinite(sumTop) {
		for i := 0; i < K; i++ {
			wSparse[arr[i].idx] = 1 / float64(K)
		}
	}

	return wSparse

}

// ─────────────────────────────────────────────
// DONS Step
// ─────────────────────────────────────────────
func (d *DONS) Step(r []float64, t, T int) []float64 {
	// Predict before observing the current return, then update the model using r.
	portfolio := d.Predict(T)
	d.Update(r, T)
	return portfolio
}

func (d *DONS) Predict(T int) []float64 {
	base := append([]float64(nil), d.w...)
	if d.momentum != 0 && d.last != nil {
		for i := range base {
			base[i] += d.momentum * (d.w[i] - d.last[i])
		}
	}
	d.last = append([]float64(nil), d.w...)
	d.pending = projectInterior(smoothed(base, d.d, T), d.epsilon)
	return append([]float64(nil), d.pending...)
}

func smoothed(w []float64, d, T int) []float64 {
	result := make([]float64, len(w))
	for i := range w {
		result[i] = (1-1/float64(T))*w[i] + 1/(float64(d)*float64(T))
	}
	return result
}

type newtonSolveResult struct {
	delta               []float64
	kktResidual         float64
	relativeKKTResidual float64
	constraintResidual  float64
}

func solveConstrainedNewtonDetailed(H *mat.Dense, grad []float64) (newtonSolveResult, error) {
	d := H.RawMatrix().Cols
	kkt := mat.NewDense(d+1, d+1, nil)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			kkt.Set(i, j, H.At(i, j))
		}
		kkt.Set(i, d, 1)
		kkt.Set(d, i, 1)
	}
	rhs := mat.NewVecDense(d+1, nil)
	for i := range grad {
		rhs.SetVec(i, -grad[i])
	}
	rhs.SetVec(d, 0)

	var solution mat.VecDense
	if err := solution.SolveVec(kkt, rhs); err != nil {
		return newtonSolveResult{}, err
	}

	delta := make([]float64, d)
	for i := range delta {
		delta[i] = solution.AtVec(i)
	}

	residual := make([]float64, d+1)
	for i := 0; i < d; i++ {
		value := grad[i] + solution.AtVec(d)
		for j := 0; j < d; j++ {
			value += H.At(i, j) * delta[j]
		}
		residual[i] = value
	}
	residual[d] = sum(delta)

	residualNorm := vectorNorm(residual[:d])
	gradientNorm := vectorNorm(grad)
	hessianNorm := matrixInfinityNorm(H)
	deltaNorm := vectorNorm(delta)
	relativeResidual := residualNorm / (1 + gradientNorm + hessianNorm*deltaNorm)

	return newtonSolveResult{
		delta:               delta,
		kktResidual:         residualNorm,
		relativeKKTResidual: relativeResidual,
		constraintResidual:  math.Abs(residual[d]),
	}, nil
}

func solveConstrainedNewton(H *mat.Dense, grad []float64) ([]float64, error) {
	result, err := solveConstrainedNewtonDetailed(H, grad)
	if err != nil {
		return nil, err
	}
	return result.delta, nil
}

func vectorNorm(values []float64) float64 {
	norm := 0.0
	for _, value := range values {
		norm += value * value
	}
	return math.Sqrt(norm)
}

func matrixInfinityNorm(matrix *mat.Dense) float64 {
	rows, cols := matrix.Dims()
	norm := 0.0
	for i := 0; i < rows; i++ {
		rowSum := 0.0
		for j := 0; j < cols; j++ {
			rowSum += math.Abs(matrix.At(i, j))
		}
		norm = math.Max(norm, rowSum)
	}
	return norm
}

func diagonalHessianNorm(hessian [][]float64) float64 {
	norm := 0.0
	for i := range hessian {
		rowSum := 0.0
		for _, value := range hessian[i] {
			rowSum += math.Abs(value)
		}
		norm = math.Max(norm, rowSum)
	}
	return norm
}

func (d *DONS) stabilizedQuadraticHessian() (*mat.Dense, float64) {
	q := mat.NewDense(d.d, d.d, nil)
	maxDiag := 0.0
	minDiag := math.Inf(1)
	minGershgorin := math.Inf(1)
	invalid := false

	for i := 0; i < d.d; i++ {
		rowSum := 0.0
		for j := 0; j < d.d; j++ {
			value := 0.5 * (d.quadHess.At(i, j) + d.quadHess.At(j, i))
			if !isFinite(value) {
				invalid = true
			}
			q.Set(i, j, value)
			if i != j {
				rowSum += math.Abs(value)
			}
		}
		diagonal := q.At(i, i)
		if !isFinite(diagonal) || diagonal < 0 {
			invalid = true
		}
		maxDiag = math.Max(maxDiag, diagonal)
		minDiag = math.Min(minDiag, diagonal)
		minGershgorin = math.Min(minGershgorin, diagonal-rowSum)
	}

	if invalid {
		d.quadHess = mat.NewDense(d.d, d.d, nil)
		d.quadOffset = make([]float64, d.d)
		return d.quadHess, 0
	}

	regularization := 0.0
	if maxDiag > 0 && (minDiag <= 0 || maxDiag/minDiag > 1e10) {
		regularization = math.Max(1e-10, maxDiag*1e-8)
	}
	if minGershgorin <= 0 {
		regularization = math.Max(regularization, -minGershgorin+1e-10)
	}
	if regularization > 0 {
		for i := 0; i < d.d; i++ {
			d.quadHess.Set(i, i, d.quadHess.At(i, i)+regularization)
			q.Set(i, i, q.At(i, i)+regularization)
		}
	}
	return q, regularization
}

type ObjectiveComponents struct {
	Barrier    float64
	LinearLoss float64
	Quadratic  float64
	Offset     float64
	Total      float64
}

func objectiveComponents(w, nt []float64, q *mat.Dense, offset, linear []float64) ObjectiveComponents {
	components := ObjectiveComponents{}
	for i := range w {
		if w[i] <= 0 || !isFinite(w[i]) {
			components.Total = math.Inf(1)
			return components
		}
		components.Barrier -= nt[i] * math.Log(w[i])
		components.LinearLoss += linear[i] * w[i]
		components.Offset += offset[i] * w[i]
		for j := range w {
			components.Quadratic += 0.5 * w[i] * q.At(i, j) * w[j]
		}
	}
	components.Total = components.Barrier + components.LinearLoss + components.Quadratic + components.Offset
	return components
}

func quadraticObjective(w, nt []float64, q *mat.Dense, offset, linear []float64) float64 {
	return objectiveComponents(w, nt, q, offset, linear).Total
}

func maxInteriorStep(w, delta []float64, epsilon float64) float64 {
	step := math.Inf(1)
	for i := range w {
		if delta[i] < 0 {
			step = math.Min(step, (w[i]-epsilon)/-delta[i])
		}
	}
	if math.IsInf(step, 1) {
		return 1
	}
	return math.Max(0, 0.99*step)
}

func (d *DONS) Update(r []float64, T int) {
	d.lastCurvatureReset = false
	if d.barrier != nil && len(d.p) > 0 {
		barrierScale := 0.0
		for i := range d.p {
			barrierScale = math.Max(barrierScale, math.Abs(d.p[i]))
		}
		if barrierScale > d.barrierResetThreshold && d.lastCombinedHessianNorm > 0 && d.lastCombinedHessianNorm > d.curvatureResetThreshold {
			d.quadHess = mat.NewDense(d.d, d.d, nil)
			d.quadOffset = make([]float64, d.d)
			d.lastCurvatureReset = true
		}
	}
	d.lastNewtonDecrement = 0
	d.lastKKTResidual = math.Inf(1)
	d.lastRelativeKKTResidual = math.Inf(1)
	d.lastConstraintResidual = math.Inf(1)
	d.lastHessianRegularization = 0
	d.lastNewtonAccepted = false
	d.lastRawStepNorm = 0
	d.lastAcceptedStepNorm = 0
	d.lastStepSize = 0
	d.lastBacktrackingAttempts = 0
	d.lastObjectiveBefore = math.Inf(1)
	d.lastObjectiveAfter = math.Inf(1)
	d.lastBarrierObjective = math.NaN()
	d.lastLinearLossObjective = math.NaN()
	d.lastQuadraticObjective = math.NaN()
	d.lastOffsetObjective = math.NaN()
	d.lastTotalObjective = math.NaN()
	d.lastProjectionFallback = false
	d.lastMaxWeightChange = 0
	d.lastGradientNorm = 0
	d.lastProjectedGradientNorm = 0
	d.lastBarrierHessianNorm = 0
	d.lastQuadraticHessianNorm = 0
	d.lastCombinedHessianNorm = 0
	d.lastQuadraticMemoryBeforeNorm = 0
	d.lastQuadraticUpdateNorm = 0
	d.lastQuadraticMemoryAfterNorm = 0
	d.lastQuadraticDecay = d.quadraticDecay

	u := d.pending
	if u == nil {
		u = d.Predict(T)
	}
	d.pending = nil
	g, err := checkedCoverGradient(r, u)
	if err != nil {
		return
	}
	addQuadraticTerm(d, g, d.w)
	d.updateP(u)
	nt := d.computeNt(float64(T))
	if !validBarrierInputs(d.w, nt) {
		return
	}
	q, regularization := d.stabilizedQuadraticHessian()
	d.lastHessianRegularization = regularization
	grad := barrierGrad(d.w, nt)
	for i := range grad {
		grad[i] += g[i]
		for j := 0; j < d.d; j++ {
			grad[i] += q.At(i, j) * d.w[j]
		}
		grad[i] += d.quadOffset[i]
	}
	d.lastGradientNorm = vectorNorm(grad)
	projectedGradient := append([]float64(nil), grad...)
	projectedMean := sum(projectedGradient) / float64(len(projectedGradient))
	for i := range projectedGradient {
		projectedGradient[i] -= projectedMean
	}
	d.lastProjectedGradientNorm = vectorNorm(projectedGradient)

	// The Newton step solves the constrained KKT system for the barrier-plus-quadratic objective.
	// This is kept structurally close to the paper's damped-update form, while remaining
	// numerically stable in a practical simplex implementation.
	H := mat.NewDense(d.d, d.d, nil)
	barrierH := barrierHessian(d.w, nt)
	for i := 0; i < d.d; i++ {
		for j := 0; j < d.d; j++ {
			H.Set(i, j, barrierH[i][j]+q.At(i, j))
		}
	}
	d.lastBarrierHessianNorm = diagonalHessianNorm(barrierH)
	d.lastQuadraticHessianNorm = matrixInfinityNorm(q)
	d.lastCombinedHessianNorm = matrixInfinityNorm(H)
	if d.lastCombinedHessianNorm > d.curvatureResetThreshold {
		d.quadHess = mat.NewDense(d.d, d.d, nil)
		d.quadOffset = make([]float64, d.d)
		d.lastCurvatureReset = true
		q, _ = d.stabilizedQuadraticHessian()
		H = mat.NewDense(d.d, d.d, nil)
		for i := 0; i < d.d; i++ {
			for j := 0; j < d.d; j++ {
				H.Set(i, j, barrierH[i][j]+q.At(i, j))
			}
		}
		d.lastCombinedHessianNorm = matrixInfinityNorm(H)
	}

	result, err := solveConstrainedNewtonDetailed(H, grad)
	for attempt := 0; err != nil && attempt < 3; attempt++ {
		hessianScale := 0.0
		for i := 0; i < d.d; i++ {
			hessianScale = math.Max(hessianScale, math.Abs(H.At(i, i)))
		}
		ridge := math.Max(1e-12, hessianScale*1e-8) * math.Pow10(attempt)
		for i := 0; i < d.d; i++ {
			d.quadHess.Set(i, i, d.quadHess.At(i, i)+ridge)
			q.Set(i, i, q.At(i, i)+ridge)
			H.Set(i, i, H.At(i, i)+ridge)
		}
		d.lastHessianRegularization += ridge
		result, err = solveConstrainedNewtonDetailed(H, grad)
	}
	if err != nil {
		return
	}
	d.lastKKTResidual = result.kktResidual
	d.lastRelativeKKTResidual = result.relativeKKTResidual
	d.lastConstraintResidual = result.constraintResidual
	decrement := math.Max(0, -dot(grad, result.delta))
	d.lastNewtonDecrement = decrement
	den := 1 + 4*math.Sqrt(decrement)
	rawStep := d.gamma / den
	step := math.Min(rawStep, maxInteriorStep(d.w, result.delta, d.epsilon))
	d.lastRawStepNorm = rawStep * vectorNorm(result.delta)
	d.lastStepSize = step
	objectiveBefore := objectiveComponents(d.w, nt, q, d.quadOffset, g)
	d.lastObjectiveBefore = objectiveBefore.Total
	d.lastBarrierObjective = objectiveBefore.Barrier
	d.lastLinearLossObjective = objectiveBefore.LinearLoss
	d.lastQuadraticObjective = objectiveBefore.Quadratic
	d.lastOffsetObjective = objectiveBefore.Offset
	d.lastTotalObjective = objectiveBefore.Total
	for attempt := 0; attempt < 12; attempt++ {
		candidate := make([]float64, d.d)
		for i := range candidate {
			candidate[i] = d.w[i] + step*result.delta[i]
		}
		next := candidate
		if math.Abs(sum(candidate)-1) > 1e-10 || !allInterior(candidate, d.epsilon) {
			next = projectInterior(candidate, d.epsilon)
			d.lastProjectionFallback = true
		}
		newObjective := objectiveComponents(next, nt, q, d.quadOffset, g)
		if isFinite(newObjective.Total) && newObjective.Total <= objectiveBefore.Total+1e-12*(1+math.Abs(objectiveBefore.Total)) {
			previousWeights := append([]float64(nil), d.w...)
			d.w = next
			d.lastNewtonAccepted = true
			stepDifference := make([]float64, len(next))
			for i := range next {
				stepDifference[i] = next[i] - previousWeights[i]
			}
			d.lastAcceptedStepNorm = vectorNorm(stepDifference)
			d.lastBacktrackingAttempts = attempt
			d.lastObjectiveAfter = newObjective.Total
			d.lastBarrierObjective = newObjective.Barrier
			d.lastLinearLossObjective = newObjective.LinearLoss
			d.lastQuadraticObjective = newObjective.Quadratic
			d.lastOffsetObjective = newObjective.Offset
			d.lastTotalObjective = newObjective.Total
			d.lastMaxWeightChange = maxWeightChange(previousWeights, next)
			return
		}
		step *= 0.5
		d.lastStepSize = step
	}
	d.lastBacktrackingAttempts = 12
}

func allInterior(values []float64, epsilon float64) bool {
	for _, value := range values {
		if !isFinite(value) || value < epsilon {
			return false
		}
	}
	return true
}

func maxWeightChange(before, after []float64) float64 {
	if len(before) != len(after) {
		return math.Inf(1)
	}
	maximum := 0.0
	for i := range before {
		maximum = math.Max(maximum, math.Abs(after[i]-before[i]))
	}
	return maximum
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

func (d *DONS) updateP(u []float64) {
	if d.barrier == nil {
		d.barrier = newBarrierState(d.d)
	}
	if len(d.barrier.p) != len(u) {
		d.barrier.p = make([]float64, len(u))
		copy(d.barrier.p, d.p)
	}
	// Paper notation: update p_t,i only when 2p_t,i < u_t,i, preserving the barrier's
	// adaptive recursion while keeping the iterates strictly interior.
	d.barrier.update(u)
	if d.barrierDecay > 0 && d.barrierDecay < 1 {
		for i := range d.barrier.p {
			d.barrier.p[i] *= d.barrierDecay
		}
	}
	if d.barrierResetThreshold > 0 {
		base := 1.0 / float64(len(d.barrier.p))
		for i := range d.barrier.p {
			if d.barrier.p[i] < math.Max(d.barrierResetThreshold, base*1e-3) {
				d.barrier.p[i] = math.Max(d.barrierResetThreshold, base*1e-3)
			}
		}
	}
	d.p = append([]float64(nil), d.barrier.p...)
}

func (d *DONS) computeNt(T float64) []float64 {
	if d.barrier == nil {
		d.barrier = newBarrierState(d.d)
	}
	// n_{t,i} = n * exp(log(p_{t,i}/p̄) * log(T+1)), with p̄ = 1/d as the baseline.
	return d.barrier.nt(T, d.n)
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func writeCSV(filename string, records [][]string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	for _, rec := range records {
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	return nil
}

func plotSeries(filename string, series []float64, title string) error {
	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = "t"
	p.Y.Label.Text = title

	pts := make(plotter.XYs, len(series))
	for i := range series {
		pts[i].X = float64(i + 1)
		pts[i].Y = series[i]
	}

	l, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	p.Add(l)

	return p.Save(8*vg.Inch, 4*vg.Inch, filename)
}

type PreprocessingAudit struct {
	ForwardFilledPrices  int
	LeadingMissingPrices int
	InvalidPricePairs    int
	NonFiniteReturns     int
	ClampedLowReturns    int
	ClampedHighReturns   int
}

// BuildReturnMatrix: prices -> aligned multiplicative returns
func BuildReturnMatrix(all map[string][]data.PricesStruct, universe data.Universe) ([][]float64, []time.Time, PreprocessingAudit, error) {
	tickers := universe.Tickers
	d := len(tickers)
	audit := PreprocessingAudit{}
	if d == 0 {
		return nil, nil, audit, errors.New("universe has no tickers")
	}
	for _, ticker := range tickers {
		if len(all[ticker]) == 0 {
			return nil, nil, audit, fmt.Errorf("no prices for ticker %s", ticker)
		}
	}

	// Collect all dates
	dateMap := make(map[time.Time]bool)
	for _, prices := range all {
		for _, dp := range prices {
			dateMap[dp.Date] = true
		}
	}

	dates := make([]time.Time, 0, len(dateMap))
	for dt := range dateMap {
		dates = append(dates, dt)
	}
	sort.Slice(dates, func(i, j int) bool { return dates[i].Before(dates[j]) })

	// Build price matrix
	priceMatrix := make([][]float64, len(dates))
	for i := range priceMatrix {
		priceMatrix[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			priceMatrix[i][j] = math.NaN()
		}
	}

	for j, ticker := range tickers {
		series := all[ticker]
		idx := 0
		for _, dp := range series {
			for idx < len(dates) && dates[idx].Before(dp.Date) {
				idx++
			}
			if idx < len(dates) && dates[idx].Equal(dp.Date) {
				priceMatrix[idx][j] = dp.Prices[0]
			}
		}
	}

	// Forward-fill missing prices
	for j := 0; j < d; j++ {
		var last float64
		haveLast := false
		for i := 0; i < len(dates); i++ {
			if !math.IsNaN(priceMatrix[i][j]) {
				last = priceMatrix[i][j]
				haveLast = true
			} else if haveLast {
				priceMatrix[i][j] = last
				audit.ForwardFilledPrices++
			} else {
				audit.LeadingMissingPrices++
			}
		}
	}

	// Multiplicative returns
	R := make([][]float64, len(dates)-1)
	for t := 1; t < len(dates); t++ {
		R[t-1] = make([]float64, d)
		for j := 0; j < d; j++ {

			p0 := priceMatrix[t-1][j]
			p1 := priceMatrix[t][j]

			// invalid prices → neutral multiplier
			if p0 <= 0 || p1 <= 0 || math.IsNaN(p0) || math.IsNaN(p1) {
				R[t-1][j] = 1.0
				audit.InvalidPricePairs++
				continue
			}

			r := p1 / p0

			// sanitize only NaN/Inf
			if math.IsNaN(r) || math.IsInf(r, 0) {
				R[t-1][j] = 1.0
				audit.NonFiniteReturns++
				continue
			}

			// IMPORTANT: clamp extreme values
			if r < 0.5 {
				r = 0.5
				audit.ClampedLowReturns++
			}
			if r > 1.5 {
				r = 1.5
				audit.ClampedHighReturns++
			}

			R[t-1][j] = r
		}
	}

	return R, dates[1:], audit, nil
}

func comparatorSPY(universe data.Universe) []float64 {
	d := len(universe.Tickers)
	u := make([]float64, d)
	for i, t := range universe.Tickers {
		if t == "SPY" {
			u[i] = 1.0
			return u
		}
	}
	panic("SPY not found in universe")
}

func bestSingleAssetInHindsight(R [][]float64) []float64 {
	T := len(R)
	d := len(R[0])

	wealth := make([]float64, d)
	for i := 0; i < d; i++ {
		w := 1.0
		for t := 0; t < T; t++ {
			w *= R[t][i]
		}
		wealth[i] = w
	}

	best := 0
	for i := 1; i < d; i++ {
		if wealth[i] > wealth[best] {
			best = i
		}
	}

	u := make([]float64, d)
	u[best] = 1.0
	return u
}
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}
func sum(w []float64) float64 {
	s := 0.0
	for _, x := range w {
		s += x
	}
	return s
}

type BacktestResult struct {
	AlgoWealth    []float64
	DenseWealth   []float64
	SPYWealth     []float64
	BestWealth    []float64
	Records       [][]string
	WeightRecords [][]string
	Trades        [][]string
	Diagnostics   [][]string
}

func priceLookup(allPrices map[string][]data.PricesStruct) map[string]map[time.Time]float64 {
	lookup := make(map[string]map[time.Time]float64, len(allPrices))
	for ticker, prices := range allPrices {
		lookup[ticker] = make(map[time.Time]float64, len(prices))
		for _, price := range prices {
			if len(price.Prices) > 0 && price.Prices[0] > 0 && isFinite(price.Prices[0]) {
				lookup[ticker][price.Date] = price.Prices[0]
			}
		}
	}
	return lookup
}

func runTestDetailed(
	Rtest [][]float64,
	datesTest []time.Time,
	tickers []string,
	allPrices map[string][]data.PricesStruct,
	uSPY []float64,
	uBest []float64,
	meta *MetaDONS,
	config Config,
) BacktestResult {
	T := len(Rtest)
	d := len(tickers)
	if T == 0 {
		return BacktestResult{}
	}
	if len(datesTest) != T {
		return BacktestResult{}
	}
	if len(uSPY) != d || len(uBest) != d {
		return BacktestResult{}
	}
	for _, row := range Rtest {
		if len(row) != d {
			return BacktestResult{}
		}
	}

	// Rtest[t] is the return ending on datesTest[t]. Keep index 0 as the
	// initial wealth before the first test-period return is applied.
	algoWealth := make([]float64, T+1)
	denseWealth := make([]float64, T+1)
	spyWealth := make([]float64, T+1)
	bestWealth := make([]float64, T+1)
	algoLogWealth := make([]float64, T+1)
	spyLogWealth := make([]float64, T+1)
	bestLogWealth := make([]float64, T+1)
	denseLogWealth := make([]float64, T+1)

	algoWealth[0] = 1.0
	spyWealth[0] = 1.0
	bestWealth[0] = 1.0
	denseWealth[0] = 1.0

	records := [][]string{{"t", "date", "ret_executed_net", "ret_dense", "ret_spy", "ret_best", "turnover", "wealth_executed_net", "wealth_dense", "wealth_spy", "wealth_best"}}

	weightRecords := [][]string{}
	header := append([]string{"t", "date"}, tickers...)
	weightRecords = append(weightRecords, header)

	trades := [][]string{{"t", "date", "ticker", "entry", "exit", "weight"}}
	diagnostics := [][]string{{"t", "date", "turnover", "active_experts", "retired_experts", "invalid_expert_updates", "max_newton_decrement", "max_relative_kkt_residual", "max_constraint_residual", "max_hessian_regularization", "accepted_newton_steps", "max_raw_step_norm", "max_accepted_step_norm", "max_step_size", "mean_step_size", "max_backtracking_attempts", "min_objective_before", "max_objective_after", "mean_objective_before", "max_weight_change", "mean_weight_change", "sum_abs_weight_change", "projection_fallbacks", "max_gradient_norm", "mean_gradient_norm", "max_projected_gradient_norm", "mean_projected_gradient_norm", "max_barrier_hessian_norm", "max_quadratic_hessian_norm", "max_combined_hessian_norm", "max_quadratic_memory_before_norm", "max_quadratic_update_norm", "max_quadratic_memory_after_norm", "quadratic_decay", "quadratic_update_scale", "max_barrier_objective", "mean_barrier_objective", "max_linear_loss_objective", "mean_linear_loss_objective", "max_quadratic_objective", "mean_quadratic_objective", "max_offset_objective", "mean_offset_objective", "max_total_objective", "mean_total_objective", "curvature_resets"}}

	pricesByDate := priceLookup(allPrices)
	prevPrices := make(map[string]float64, d)
	previousExecuted := make([]float64, d)
	for j := 0; j < d; j++ {
		previousExecuted[j] = 1 / float64(d)
	}

	w := make([]float64, d)
	iter := 0

	for t := 0; t < T; t++ {
		rToday := Rtest[t]
		wealthIndex := t + 1

		// Select the portfolio before observing today's return.
		w = meta.Predict(iter)
		// Top-k is an execution overlay. Keep the dense learner portfolio in w,
		// but record and trade the sparse, renormalized portfolio.
		executed := w
		if config.TopK > 0 {
			executed = topK(w, config.TopK)
		}
		row := []string{fmt.Sprintf("%d", wealthIndex), datesTest[t].Format("2006-01-02")}
		for _, weight := range executed {
			row = append(row, fmt.Sprintf("%.15g", weight))
		}
		weightRecords = append(weightRecords, row)
		tickersOut := []string{}
		entriesOut := []string{}
		exitsOut := []string{}
		weightsOut := []string{}
		for j := 0; j < d; j++ {
			if executed[j] <= 1e-12 {
				continue
			}
			currPrice, ok := pricesByDate[tickers[j]][datesTest[t]]
			if !ok {
				continue
			}
			tickersOut = append(tickersOut, tickers[j])
			entryPrice := currPrice
			if previousPrice, exists := prevPrices[tickers[j]]; exists {
				entryPrice = previousPrice
			}
			entriesOut = append(entriesOut, fmt.Sprintf("%.4f", entryPrice))
			exitsOut = append(exitsOut, fmt.Sprintf("%.4f", currPrice))
			weightsOut = append(weightsOut, fmt.Sprintf("%.15g", executed[j]))
			prevPrices[tickers[j]] = currPrice
		}
		trades = append(trades, []string{
			fmt.Sprintf("%d", wealthIndex),
			datesTest[t].Format("2006-01-02"),
			strings.Join(tickersOut, " "),
			strings.Join(entriesOut, " "),
			strings.Join(exitsOut, " "),
			strings.Join(weightsOut, " "),
		})
		// DAILY COMPOUNDING. Zero means execute the dense learner output.
		turnover := 0.5 * l1Distance(executed, previousExecuted)
		retDense := dotSafe(rToday, w)
		retAlgo := dotSafe(rToday, executed)
		retAlgo *= math.Max(0, 1-config.TransactionCost*turnover)
		previousExecuted = append(previousExecuted[:0], executed...)

		retSPY := dotSafe(rToday, uSPY)
		retBest := dotSafe(rToday, uBest)
		diagnosticExperts := meta.activeExperts(t)
		meta.Update(rToday, iter)
		iter++
		// guard
		if retAlgo <= 0 || math.IsNaN(retAlgo) || math.IsInf(retAlgo, 0) {
			retAlgo = 1.0
		}
		if retDense <= 0 || math.IsNaN(retDense) || math.IsInf(retDense, 0) {
			retDense = 1.0
		}
		if retSPY <= 0 || math.IsNaN(retSPY) || math.IsInf(retSPY, 0) {
			retSPY = 1.0
		}
		if retBest <= 0 || math.IsNaN(retBest) || math.IsInf(retBest, 0) {
			retBest = 1.0
		}
		algoLogWealth[wealthIndex] = algoLogWealth[wealthIndex-1] + math.Log(retAlgo)
		denseLogWealth[wealthIndex] = denseLogWealth[wealthIndex-1] + math.Log(retDense)
		spyLogWealth[wealthIndex] = spyLogWealth[wealthIndex-1] + math.Log(retSPY)
		bestLogWealth[wealthIndex] = bestLogWealth[wealthIndex-1] + math.Log(retBest)
		algoWealth[wealthIndex] = wealthFromLog(algoLogWealth[wealthIndex])
		denseWealth[wealthIndex] = wealthFromLog(denseLogWealth[wealthIndex])
		spyWealth[wealthIndex] = wealthFromLog(spyLogWealth[wealthIndex])
		bestWealth[wealthIndex] = wealthFromLog(bestLogWealth[wealthIndex])
		records = append(records, []string{
			fmt.Sprintf("%d", wealthIndex),
			datesTest[t].Format("2006-01-02"),
			fmt.Sprintf("%.6f", retAlgo),
			fmt.Sprintf("%.6f", retDense),
			fmt.Sprintf("%.6f", retSPY),
			fmt.Sprintf("%.6f", retBest),
			fmt.Sprintf("%.6f", turnover),
			fmt.Sprintf("%.6f", algoWealth[wealthIndex]),
			fmt.Sprintf("%.6f", denseWealth[wealthIndex]),
			fmt.Sprintf("%.6f", spyWealth[wealthIndex]),
			fmt.Sprintf("%.6f", bestWealth[wealthIndex]),
		})

		maxDecrement := 0.0
		maxRelativeKKT := 0.0
		maxConstraint := 0.0
		maxRegularization := 0.0
		acceptedSteps := 0
		invalidUpdates := 0
		maxRawStepNorm := 0.0
		maxAcceptedStepNorm := 0.0
		maxStepSize := 0.0
		meanStepSize := 0.0
		maxBacktracking := 0
		minObjectiveBefore := math.Inf(1)
		maxObjectiveAfter := math.Inf(-1)
		meanObjectiveBefore := 0.0
		maxWeightChangeValue := 0.0
		meanWeightChangeValue := 0.0
		sumAbsWeightChange := 0.0
		projectionFallbacks := 0
		maxGradientNorm := 0.0
		meanGradientNorm := 0.0
		maxProjectedGradientNorm := 0.0
		meanProjectedGradientNorm := 0.0
		maxBarrierHessianNorm := 0.0
		maxQuadraticHessianNorm := 0.0
		maxCombinedHessianNorm := 0.0
		maxQuadraticMemoryBeforeNorm := 0.0
		maxQuadraticUpdateNorm := 0.0
		maxQuadraticMemoryAfterNorm := 0.0
		quadraticDecay := 0.0
		quadraticUpdateScale := 0.0
		maxBarrierObjective := math.Inf(-1)
		meanBarrierObjective := 0.0
		maxLinearLossObjective := math.Inf(-1)
		meanLinearLossObjective := 0.0
		maxQuadraticObjective := math.Inf(-1)
		meanQuadraticObjective := 0.0
		maxOffsetObjective := math.Inf(-1)
		meanOffsetObjective := 0.0
		maxTotalObjective := math.Inf(-1)
		meanTotalObjectiveValue := 0.0
		nActive := 0
		for _, expert := range diagnosticExperts {
			nActive++
			maxDecrement = math.Max(maxDecrement, expert.dons.lastNewtonDecrement)
			maxRelativeKKT = math.Max(maxRelativeKKT, expert.dons.lastRelativeKKTResidual)
			maxConstraint = math.Max(maxConstraint, expert.dons.lastConstraintResidual)
			maxRegularization = math.Max(maxRegularization, expert.dons.lastHessianRegularization)
			if expert.dons.lastNewtonAccepted {
				acceptedSteps++
			}
			maxRawStepNorm = math.Max(maxRawStepNorm, expert.dons.lastRawStepNorm)
			maxAcceptedStepNorm = math.Max(maxAcceptedStepNorm, expert.dons.lastAcceptedStepNorm)
			maxStepSize = math.Max(maxStepSize, expert.dons.lastStepSize)
			meanStepSize += expert.dons.lastStepSize
			maxBacktracking = maxInt(maxBacktracking, expert.dons.lastBacktrackingAttempts)
			minObjectiveBefore = math.Min(minObjectiveBefore, expert.dons.lastObjectiveBefore)
			maxObjectiveAfter = math.Max(maxObjectiveAfter, expert.dons.lastObjectiveAfter)
			meanObjectiveBefore += expert.dons.lastObjectiveBefore
			maxWeightChangeValue = math.Max(maxWeightChangeValue, expert.dons.lastMaxWeightChange)
			meanWeightChangeValue += expert.dons.lastMaxWeightChange
			sumAbsWeightChange += expert.dons.lastAcceptedStepNorm
			if expert.dons.lastProjectionFallback {
				projectionFallbacks++
			}
			maxGradientNorm = math.Max(maxGradientNorm, expert.dons.lastGradientNorm)
			meanGradientNorm += expert.dons.lastGradientNorm
			maxProjectedGradientNorm = math.Max(maxProjectedGradientNorm, expert.dons.lastProjectedGradientNorm)
			meanProjectedGradientNorm += expert.dons.lastProjectedGradientNorm
			maxBarrierHessianNorm = math.Max(maxBarrierHessianNorm, expert.dons.lastBarrierHessianNorm)
			maxQuadraticHessianNorm = math.Max(maxQuadraticHessianNorm, expert.dons.lastQuadraticHessianNorm)
			maxCombinedHessianNorm = math.Max(maxCombinedHessianNorm, expert.dons.lastCombinedHessianNorm)
			maxQuadraticMemoryBeforeNorm = math.Max(maxQuadraticMemoryBeforeNorm, expert.dons.lastQuadraticMemoryBeforeNorm)
			maxQuadraticUpdateNorm = math.Max(maxQuadraticUpdateNorm, expert.dons.lastQuadraticUpdateNorm)
			maxQuadraticMemoryAfterNorm = math.Max(maxQuadraticMemoryAfterNorm, expert.dons.lastQuadraticMemoryAfterNorm)
			quadraticDecay = math.Max(quadraticDecay, expert.dons.lastQuadraticDecay)
			quadraticUpdateScale = math.Max(quadraticUpdateScale, expert.dons.quadraticUpdateScale)
			maxBarrierObjective = math.Max(maxBarrierObjective, expert.dons.lastBarrierObjective)
			meanBarrierObjective += expert.dons.lastBarrierObjective
			maxLinearLossObjective = math.Max(maxLinearLossObjective, expert.dons.lastLinearLossObjective)
			meanLinearLossObjective += expert.dons.lastLinearLossObjective
			maxQuadraticObjective = math.Max(maxQuadraticObjective, expert.dons.lastQuadraticObjective)
			meanQuadraticObjective += expert.dons.lastQuadraticObjective
			maxOffsetObjective = math.Max(maxOffsetObjective, expert.dons.lastOffsetObjective)
			meanOffsetObjective += expert.dons.lastOffsetObjective
			maxTotalObjective = math.Max(maxTotalObjective, expert.dons.lastTotalObjective)
			meanTotalObjectiveValue += expert.dons.lastTotalObjective
			invalidUpdates += expert.numericalFailures
		}
		if nActive > 0 {
			meanStepSize = meanStepSize / float64(nActive)
			meanObjectiveBefore = meanObjectiveBefore / float64(nActive)
			meanWeightChangeValue = meanWeightChangeValue / float64(nActive)
			meanGradientNorm = meanGradientNorm / float64(nActive)
			meanProjectedGradientNorm = meanProjectedGradientNorm / float64(nActive)
			meanBarrierObjective = meanBarrierObjective / float64(nActive)
			meanLinearLossObjective = meanLinearLossObjective / float64(nActive)
			meanQuadraticObjective = meanQuadraticObjective / float64(nActive)
			meanOffsetObjective = meanOffsetObjective / float64(nActive)
			meanTotalObjectiveValue = meanTotalObjectiveValue / float64(nActive)
		} else {
			meanStepSize = 0
			meanObjectiveBefore = 0
			meanWeightChangeValue = 0
			meanGradientNorm = 0
			meanProjectedGradientNorm = 0
			meanBarrierObjective = 0
			meanLinearLossObjective = 0
			meanQuadraticObjective = 0
			meanOffsetObjective = 0
			meanTotalObjectiveValue = 0
		}
		if math.IsInf(minObjectiveBefore, 1) {
			minObjectiveBefore = 0
		}
		if math.IsInf(maxObjectiveAfter, -1) {
			maxObjectiveAfter = 0
		}
		if math.IsInf(maxBarrierObjective, 1) {
			maxBarrierObjective = 0
		}
		if math.IsInf(maxLinearLossObjective, 1) {
			maxLinearLossObjective = 0
		}
		if math.IsInf(maxQuadraticObjective, 1) {
			maxQuadraticObjective = 0
		}
		if math.IsInf(maxOffsetObjective, 1) {
			maxOffsetObjective = 0
		}
		if math.IsInf(maxTotalObjective, 1) {
			maxTotalObjective = 0
		}
		resetCount := 0
		for _, expert := range diagnosticExperts {
			if expert.dons.lastCurvatureReset {
				resetCount++
			}
		}
		diagnostics = append(diagnostics, []string{
			fmt.Sprintf("%d", wealthIndex), datesTest[t].Format("2006-01-02"),
			fmt.Sprintf("%.6f", turnover), fmt.Sprintf("%d", len(diagnosticExperts)),
			fmt.Sprintf("%d", meta.retired), fmt.Sprintf("%d", invalidUpdates),
			fmt.Sprintf("%.6g", maxDecrement), fmt.Sprintf("%.6g", maxRelativeKKT),
			fmt.Sprintf("%.6g", maxConstraint), fmt.Sprintf("%.6g", maxRegularization),
			fmt.Sprintf("%d", acceptedSteps),
			fmt.Sprintf("%.6g", maxRawStepNorm), fmt.Sprintf("%.6g", maxAcceptedStepNorm),
			fmt.Sprintf("%.6g", maxStepSize), fmt.Sprintf("%.6g", meanStepSize), fmt.Sprintf("%d", maxBacktracking),
			fmt.Sprintf("%.6g", minObjectiveBefore), fmt.Sprintf("%.6g", maxObjectiveAfter), fmt.Sprintf("%.6g", meanObjectiveBefore),
			fmt.Sprintf("%.6g", maxWeightChangeValue), fmt.Sprintf("%.6g", meanWeightChangeValue), fmt.Sprintf("%.6g", sumAbsWeightChange), fmt.Sprintf("%d", projectionFallbacks),
			fmt.Sprintf("%.6g", maxGradientNorm), fmt.Sprintf("%.6g", meanGradientNorm), fmt.Sprintf("%.6g", maxProjectedGradientNorm), fmt.Sprintf("%.6g", meanProjectedGradientNorm),
			fmt.Sprintf("%.6g", maxBarrierHessianNorm), fmt.Sprintf("%.6g", maxQuadraticHessianNorm), fmt.Sprintf("%.6g", maxCombinedHessianNorm),
			fmt.Sprintf("%.6g", maxQuadraticMemoryBeforeNorm), fmt.Sprintf("%.6g", maxQuadraticUpdateNorm), fmt.Sprintf("%.6g", maxQuadraticMemoryAfterNorm), fmt.Sprintf("%.6g", quadraticDecay), fmt.Sprintf("%.6g", quadraticUpdateScale),
			fmt.Sprintf("%.6g", maxBarrierObjective), fmt.Sprintf("%.6g", meanBarrierObjective), fmt.Sprintf("%.6g", maxLinearLossObjective), fmt.Sprintf("%.6g", meanLinearLossObjective),
			fmt.Sprintf("%.6g", maxQuadraticObjective), fmt.Sprintf("%.6g", meanQuadraticObjective), fmt.Sprintf("%.6g", maxOffsetObjective), fmt.Sprintf("%.6g", meanOffsetObjective),
			fmt.Sprintf("%.6g", maxTotalObjective), fmt.Sprintf("%.6g", meanTotalObjectiveValue),
			fmt.Sprintf("%d", resetCount),
		})

	}

	return BacktestResult{
		AlgoWealth: algoWealth, DenseWealth: denseWealth, SPYWealth: spyWealth,
		BestWealth: bestWealth, Records: records, WeightRecords: weightRecords,
		Trades: trades, Diagnostics: diagnostics,
	}
}

func runTest(Rtest [][]float64, datesTest []time.Time, tickers []string, allPrices map[string][]data.PricesStruct, uSPY, uBest []float64, meta *MetaDONS, executionK int) ([]float64, []float64, []float64, [][]string, [][]string, [][]string) {
	config := DefaultConfig()
	config.TopK = executionK
	result := runTestDetailed(Rtest, datesTest, tickers, allPrices, uSPY, uBest, meta, config)
	return result.AlgoWealth, result.SPYWealth, result.BestWealth, result.Records, result.WeightRecords, result.Trades
}

func l1Distance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	distance := 0.0
	for i := range a {
		distance += math.Abs(a[i] - b[i])
	}
	return distance
}
func dotSafe(r, w []float64) float64 {
	sum := 0.0
	for i := range r {
		if math.IsNaN(r[i]) || math.IsInf(r[i], 0) {
			continue
		}
		sum += w[i] * r[i]
	}
	return sum
}

func wealthFromLog(logWealth float64) float64 {
	if logWealth >= math.Log(math.MaxFloat64) {
		return math.MaxFloat64
	}
	if logWealth <= math.Log(math.SmallestNonzeroFloat64) {
		return 0
	}
	return math.Exp(logWealth)
}

func periodicReturns(wealth []float64) []float64 {
	if len(wealth) < 2 {
		return nil
	}
	returns := make([]float64, 0, len(wealth)-1)
	for i := 1; i < len(wealth); i++ {
		if wealth[i-1] <= 0 || !isFinite(wealth[i-1]) || !isFinite(wealth[i]) {
			continue
		}
		returns = append(returns, wealth[i]/wealth[i-1]-1)
	}
	return returns
}

func volatility(returns []float64) float64 {
	if len(returns) < 2 {
		return 0
	}
	mean := 0.0
	for _, value := range returns {
		mean += value
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, value := range returns {
		difference := value - mean
		variance += difference * difference
	}
	return math.Sqrt(variance / float64(len(returns)-1))
}

func sharpe(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	mean := 0.0
	for _, value := range returns {
		mean += value
	}
	mean /= float64(len(returns))
	vol := volatility(returns)
	if vol == 0 {
		return 0
	}
	return mean / vol
}

func maxDrawdown(wealth []float64) float64 {
	if len(wealth) == 0 {
		return 0
	}
	peak := wealth[0]
	maxDD := 0.0
	for _, value := range wealth {
		if value > peak {
			peak = value
		}
		if peak <= 0 || !isFinite(value) {
			continue
		}
		drawdown := 1 - value/peak
		if drawdown > maxDD {
			maxDD = drawdown
		}
	}
	return maxDD
}

func main() {
	config := DefaultConfig()
	executionK := flag.Int("top-k", config.TopK, "execute only the top K assets; 0 keeps the dense portfolio")
	barrierStrength := flag.Float64("barrier-strength", config.BarrierStrength, "barrier strength for the DONS log barrier")
	gamma := flag.Float64("gamma", config.Gamma, "Newton damping multiplier")
	momentum := flag.Float64("momentum", config.Momentum, "heuristic momentum overlay; 0 disables it")
	barrierDecay := flag.Float64("barrier-decay", config.BarrierDecay, "optional multiplicative decay applied to the barrier state each update; 1 keeps it fixed")
	barrierResetThreshold := flag.Float64("barrier-reset-threshold", config.BarrierResetThreshold, "minimum barrier floor before the adaptive barrier is reset")
	curvatureResetThreshold := flag.Float64("curvature-reset-threshold", config.CurvatureResetThreshold, "reset quadratic curvature when the combined Hessian exceeds this scale")
	quadraticDecay := flag.Float64("quadratic-decay", config.QuadraticDecay, "multiplicative decay applied to accumulated quadratic memory each update")
	quadraticUpdateScale := flag.Float64("quadratic-update-scale", config.QuadraticUpdateScale, "scale applied to each new quadratic outer-product update")
	transactionCost := flag.Float64("transaction-cost", config.TransactionCost, "proportional cost per unit turnover")
	flag.Parse()
	config.TopK = *executionK
	config.BarrierStrength = *barrierStrength
	config.Gamma = *gamma
	config.Momentum = *momentum
	config.BarrierDecay = *barrierDecay
	config.BarrierResetThreshold = *barrierResetThreshold
	config.CurvatureResetThreshold = *curvatureResetThreshold
	config.QuadraticDecay = *quadraticDecay
	config.QuadraticUpdateScale = *quadraticUpdateScale
	config.TransactionCost = *transactionCost
	if err := config.Validate(); err != nil {
		log.Fatal(err)
	}

	// --- Load universe ---
	universe, err := data.LoadUniverse()
	if err != nil {
		log.Fatalf("load universe: %v", err)
	}

	// --- Load prices (daily OR weekly OR monthly depending on fetcher) ---
	allPrices, err := data.LoadAllYahooPrices(universe)
	if err != nil {
		log.Fatalf("load prices: %v", err)
	}

	// --- Build generic return matrix ---
	R, dates, audit, err := BuildReturnMatrix(allPrices, universe)
	if err != nil {
		log.Fatalf("build returns: %v", err)
	}

	// --- Train/test split ---
	trainEnd := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

	var Rtest [][]float64
	var datesTest []time.Time

	for i, dt := range dates {
		if !dt.Before(trainEnd) {
			Rtest = append(Rtest, R[i])
			datesTest = append(datesTest, dt)
		}
	}

	d := len(universe.Tickers)
	Ttest := len(Rtest)
	if d == 0 || Ttest == 0 {
		log.Fatal("no test data remains after building the return matrix")
	}

	// --- MetaDONS parameters ---

	meta := NewMetaDONSWithConfig(d, Ttest, config)

	// --- Comparator vectors ---
	uSPY := comparatorSPY(universe)
	uBest := bestSingleAssetInHindsight(Rtest)

	// --- Run the generic backtest engine ---
	backtest := runTestDetailed(
		Rtest,
		datesTest,
		universe.Tickers,
		allPrices,
		uSPY,
		uBest,
		meta,
		config,
	)
	algoWealth := backtest.AlgoWealth
	spyWealth := backtest.SPYWealth
	bestWealth := backtest.BestWealth
	records := backtest.Records
	weightRecords := backtest.WeightRecords
	trades := backtest.Trades

	// --- Print summary ---
	log.Printf("Test period: %s to %s",
		datesTest[0].Format("2006-01-02"),
		datesTest[len(datesTest)-1].Format("2006-01-02"),
	)

	log.Printf("Final wealth: Algo=%.4f, SPY=%.4f, BestSingle=%.4f",
		algoWealth[len(algoWealth)-1],
		spyWealth[len(spyWealth)-1],
		bestWealth[len(bestWealth)-1],
	)
	log.Printf("Final wealth: Dense=%.4f, ExecutedNet=%.4f, turnover cost=%.6f",
		backtest.DenseWealth[len(backtest.DenseWealth)-1],
		backtest.AlgoWealth[len(backtest.AlgoWealth)-1],
		config.TransactionCost,
	)

	algoReturns := periodicReturns(algoWealth)
	spyReturns := periodicReturns(spyWealth)
	bestReturns := periodicReturns(bestWealth)
	log.Printf("Diagnostics (daily, non-annualized):")
	log.Printf("  Algo:       volatility=%.6f, Sharpe=%.6f, max drawdown=%.2f%%",
		volatility(algoReturns), sharpe(algoReturns), 100*maxDrawdown(algoWealth))
	log.Printf("  SPY:        volatility=%.6f, Sharpe=%.6f, max drawdown=%.2f%%",
		volatility(spyReturns), sharpe(spyReturns), 100*maxDrawdown(spyWealth))
	log.Printf("  BestSingle: volatility=%.6f, Sharpe=%.6f, max drawdown=%.2f%%",
		volatility(bestReturns), sharpe(bestReturns), 100*maxDrawdown(bestWealth))

	// --- Save CSV outputs ---
	if err := writeCSV("analysis/backtest_test.csv", records); err != nil {
		log.Fatalf("CSV error: %v", err)
	}

	if err := writeCSV("analysis/weights.csv", weightRecords); err != nil {
		log.Fatalf("CSV error: %v", err)
	}

	if err := writeCSV("analysis/trades.csv", trades); err != nil {
		log.Fatalf("CSV error: %v", err)
	}
	if err := writeCSV("analysis/diagnostics.csv", backtest.Diagnostics); err != nil {
		log.Fatalf("CSV error: %v", err)
	}
	auditRecords := [][]string{{"metric", "count"},
		{"forward_filled_prices", fmt.Sprintf("%d", audit.ForwardFilledPrices)},
		{"leading_missing_prices", fmt.Sprintf("%d", audit.LeadingMissingPrices)},
		{"invalid_price_pairs", fmt.Sprintf("%d", audit.InvalidPricePairs)},
		{"non_finite_returns", fmt.Sprintf("%d", audit.NonFiniteReturns)},
		{"clamped_low_returns", fmt.Sprintf("%d", audit.ClampedLowReturns)},
		{"clamped_high_returns", fmt.Sprintf("%d", audit.ClampedHighReturns)},
	}
	if err := writeCSV("analysis/preprocessing_audit.csv", auditRecords); err != nil {
		log.Fatalf("CSV error: %v", err)
	}
	if err := plotSeries("analysis/strategy.png", algoWealth, "Strategy"); err != nil {
		fmt.Println("plotSeries error:", err)
	}
	if err := plotSeries("analysis/spy.png", spyWealth, "SPY"); err != nil {
		fmt.Println("plotSeries error:", err)
	}
	regretSeries := make([]float64, len(algoWealth))
	regret := 0.0
	for i := range algoWealth {
		regret += math.Log(spyWealth[i] / algoWealth[i])
		regretSeries[i] = regret
	}
	if err := plotSeries("analysis/regret.png", regretSeries, "Regret sum(log(spy/algo))"); err != nil {
		fmt.Println("regret plot error:", err)
	}

	fmt.Println("Backtest complete.")
}

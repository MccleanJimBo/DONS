package main

import (
	"encoding/csv"
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

// ─────────────────────────────────────────────
// Expert struct
// ─────────────────────────────────────────────
type Expert struct {
	dons      *DONS
	start     int
	end       int
	logWeight float64
	portfolio []float64
}

type MetaDONS struct {
	d        int
	T        int
	eta      float64
	gamma    float64
	momentum float64
	experts  []*Expert
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
		nt[i] = n * math.Exp(math.Log(factor)*math.Log(T+1))
	}
	return nt
}

type DONS struct {
	d          int
	n          float64
	beta       float64
	gamma      float64
	momentum   float64
	epsilon    float64
	w          []float64
	p          []float64
	barrier    *BarrierState
	quadHess   *mat.Dense
	quadOffset []float64
	last       []float64
	pending    []float64
}

func NewMetaDONS(d, T int, eta float64) *MetaDONS {
	return &MetaDONS{
		d:        d,
		T:        T,
		eta:      eta,
		gamma:    3.0,
		momentum: 0.0,
		experts:  []*Expert{},
	}
}

func (m *MetaDONS) SetDONSParameters(gamma, momentum float64) {
	m.gamma = gamma
	m.momentum = momentum
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
	w1 := make([]float64, d)
	for i := range w1 {
		w1[i] = 1.0 / float64(d)
	}

	barrier := newBarrierState(d)
	return &DONS{
		d:          d,
		n:          n,
		beta:       beta,
		gamma:      3.0,
		momentum:   0.0,
		epsilon:    1e-9,
		w:          w1,
		p:          append([]float64(nil), barrier.p...),
		barrier:    barrier,
		quadHess:   mat.NewDense(d, d, nil),
		quadOffset: make([]float64, d),
	}
}

func (m *MetaDONS) ensureExperts(t int) {
	seen := make(map[int]bool, len(m.experts))
	for _, e := range m.experts {
		seen[e.start] = true
	}

	if !seen[0] {
		e := &Expert{
			dons:      NewDONS(m.d, NStock, BetaStock),
			start:     0,
			end:       minInt(m.T, 2),
			logWeight: 0,
		}
		e.dons.gamma = m.gamma
		e.dons.momentum = m.momentum
		m.experts = append(m.experts, e)
		seen[0] = true
	}

	for start := 1; start <= t && start < m.T; start *= 2 {
		if seen[start] {
			continue
		}
		end := minInt(m.T, 3*start)
		e := &Expert{
			dons:      NewDONS(m.d, NStock, BetaStock),
			start:     start,
			end:       end,
			logWeight: 0,
		}
		e.dons.gamma = m.gamma
		e.dons.momentum = m.momentum
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
	for i := 0; i < dons.d; i++ {
		anchor := 0.0
		for j := 0; j < dons.d; j++ {
			coefficient := (dons.beta / 4.0) * g[i] * g[j]
			dons.quadHess.Set(i, j, dons.quadHess.At(i, j)+coefficient)
			anchor += coefficient * w[j]
		}
		dons.quadOffset[i] -= anchor
	}
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
		e.portfolio = e.dons.Predict(m.T)
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
			value = 1e-12
		}
		// Expert weight update in the paper's notation:
		//   log w_{t+1}(E) = log w_t(E) - eta * ell_t(u_t^E)
		// with ell_t(u) = -log <r_t, u>.
		e.logWeight -= m.eta * (-math.Log(value))
		e.dons.Update(r, m.T)
	}
}

func (m *MetaDONS) retireExpiredExperts(t int) {
	kept := m.experts[:0]
	for _, e := range m.experts {
		if e.end <= t {
			continue
		}
		kept = append(kept, e)
	}
	m.experts = kept
}

func (m *MetaDONS) activeExperts(t int) []*Expert {
	active := make([]*Expert, 0, len(m.experts))
	for _, e := range m.experts {
		if e.start <= t && t < e.end {
			active = append(active, e)
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

func solveConstrainedNewton(H *mat.Dense, grad []float64) ([]float64, error) {
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
		return nil, err
	}

	delta := make([]float64, d)
	for i := range delta {
		delta[i] = solution.AtVec(i)
	}
	return delta, nil
}

func (d *DONS) Update(r []float64, T int) {
	u := d.pending
	if u == nil {
		u = d.Predict(T)
	}
	g := coverGradient(r, u)
	addQuadraticTerm(d, g, d.w)
	d.updateP(u)
	nt := d.computeNt(float64(T))
	grad := barrierGrad(d.w, nt)
	for i := range grad {
		for j := 0; j < d.d; j++ {
			grad[i] += d.quadHess.At(i, j) * d.w[j]
		}
		grad[i] += d.quadOffset[i]
	}

	// The Newton step solves the constrained KKT system for the barrier-plus-quadratic objective.
	// This is kept structurally close to the paper's damped-update form, while remaining
	// numerically stable in a practical simplex implementation.
	H := mat.NewDense(d.d, d.d, nil)
	for i := 0; i < d.d; i++ {
		for j := 0; j < d.d; j++ {
			H.Set(i, j, barrierHessian(d.w, nt)[i][j]+d.quadHess.At(i, j))
		}
	}

	delta, err := solveConstrainedNewton(H, grad)
	if err != nil {
		return
	}
	decrement := math.Max(0, dot(grad, delta))
	den := 1 + 4*math.Sqrt(decrement)
	next := make([]float64, d.d)
	for i := range next {
		next[i] = d.w[i] + d.gamma*delta[i]/den
	}
	d.w = projectInterior(next, d.epsilon)
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

// BuildReturnMatrix: prices -> aligned multiplicative returns
func BuildReturnMatrix(all map[string][]data.PricesStruct, universe data.Universe) ([][]float64, []time.Time, error) {
	tickers := universe.Tickers
	d := len(tickers)

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
				continue
			}

			r := p1 / p0

			// sanitize only NaN/Inf
			if math.IsNaN(r) || math.IsInf(r, 0) {
				R[t-1][j] = 1.0
				continue
			}

			// IMPORTANT: clamp extreme values
			if r < 0.5 {
				r = 0.5
			}
			if r > 1.5 {
				r = 1.5
			}

			R[t-1][j] = r
		}
	}

	return R, dates[1:], nil
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

func runTest(
	Rtest [][]float64,
	datesTest []time.Time,
	tickers []string,
	allPrices map[string][]data.PricesStruct,
	uSPY []float64,
	uBest []float64,
	meta *MetaDONS,
	executionK int,
) (
	[]float64,
	[]float64,
	[]float64,
	[][]string,
	[][]string,
	[][]string,
) {
	T := len(Rtest)
	d := len(tickers)

	algoWealth := make([]float64, T)
	spyWealth := make([]float64, T)
	bestWealth := make([]float64, T)

	algoWealth[0] = 1.0
	spyWealth[0] = 1.0
	bestWealth[0] = 1.0

	records := [][]string{{"t", "date", "ret_algo", "ret_spy", "ret_best", "wealth_algo", "wealth_spy", "wealth_best"}}

	weightRecords := [][]string{}
	header := append([]string{"t", "date"}, tickers...)
	weightRecords = append(weightRecords, header)

	trades := [][]string{{"t", "date", "ticker", "entry", "exit", "weight"}}

	prevPrices := make([]float64, d)
	for j := 0; j < d; j++ {
		prevPrices[j] = allPrices[tickers[j]][0].Prices[0]
	}

	w := make([]float64, d)
	iter := 0

	for t := 1; t < T; t++ {
		rToday := Rtest[t]

		// Select the portfolio before observing today's return.
		w = meta.Predict(iter)
		row := []string{fmt.Sprintf("%d", t), datesTest[t].Format("2006-01-02")}
		for _, weight := range w {
			row = append(row, fmt.Sprintf("%.6f", weight))
		}
		weightRecords = append(weightRecords, row)
		tickersOut := []string{}
		entriesOut := []string{}
		exitsOut := []string{}
		weightsOut := []string{}
		for j := 0; j < d; j++ {
			if w[j] <= 1e-12 {
				continue
			}
			currPrice := allPrices[tickers[j]][t].Prices[0]
			tickersOut = append(tickersOut, tickers[j])
			entriesOut = append(entriesOut, fmt.Sprintf("%.4f", prevPrices[j]))
			exitsOut = append(exitsOut, fmt.Sprintf("%.4f", currPrice))
			weightsOut = append(weightsOut, fmt.Sprintf("%.6f", w[j]))
			prevPrices[j] = currPrice
		}
		trades = append(trades, []string{
			fmt.Sprintf("%d", t),
			datesTest[t].Format("2006-01-02"),
			strings.Join(tickersOut, " "),
			strings.Join(entriesOut, " "),
			strings.Join(exitsOut, " "),
			strings.Join(weightsOut, " "),
		})
		// DAILY COMPOUNDING. Zero means execute the dense learner output.
		executed := w
		if executionK > 0 {
			executed = topK(w, executionK)
		}
		retAlgo := dotSafe(rToday, executed)

		retSPY := dotSafe(rToday, uSPY)
		retBest := dotSafe(rToday, uBest)
		meta.Update(rToday, iter)
		iter++
		// guard
		if retAlgo <= 0 || math.IsNaN(retAlgo) || math.IsInf(retAlgo, 0) {
			retAlgo = 1.0
		}
		if retSPY <= 0 || math.IsNaN(retSPY) || math.IsInf(retSPY, 0) {
			retSPY = 1.0
		}
		if retBest <= 0 || math.IsNaN(retBest) || math.IsInf(retBest, 0) {
			retBest = 1.0
		}
		records = append(records, []string{
			fmt.Sprintf("%d", t),
			datesTest[t].Format("2006-01-02"),
			fmt.Sprintf("%.6f", retAlgo),
			fmt.Sprintf("%.6f", retSPY),
			fmt.Sprintf("%.6f", retBest),
		})

		algoWealth[t] = algoWealth[t-1] * retAlgo
		spyWealth[t] = spyWealth[t-1] * retSPY
		bestWealth[t] = bestWealth[t-1] * retBest
		records = append(records, []string{
			fmt.Sprintf("%d", t),
			datesTest[t].Format("2006-01-02"),
			fmt.Sprintf("%.6f", retAlgo),
			fmt.Sprintf("%.6f", retSPY),
			fmt.Sprintf("%.6f", retBest),
			fmt.Sprintf("%.6f", algoWealth[t]),
			fmt.Sprintf("%.6f", spyWealth[t]),
			fmt.Sprintf("%.6f", bestWealth[t]),
		})

	}

	return algoWealth, spyWealth, bestWealth, records, weightRecords, trades
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
	executionK := flag.Int("top-k", 0, "execute only the top K assets; 0 keeps the dense portfolio")
	gamma := flag.Float64("gamma", 3.0, "Newton damping multiplier")
	momentum := flag.Float64("momentum", 0.0, "heuristic momentum overlay; 0 disables it")
	flag.Parse()
	if *executionK < 0 {
		log.Fatal("top-k must be non-negative")
	}
	if *gamma <= 0 {
		log.Fatal("gamma must be positive")
	}
	if *momentum < 0 || *momentum >= 1 {
		log.Fatal("momentum must be at least 0 and less than 1")
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
	R, dates, err := BuildReturnMatrix(allPrices, universe)
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

	// --- MetaDONS parameters ---

	eta := 0.15

	meta := NewMetaDONS(d, Ttest, eta)
	meta.SetDONSParameters(*gamma, *momentum)

	// --- Comparator vectors ---
	uSPY := comparatorSPY(universe)
	uBest := bestSingleAssetInHindsight(Rtest)

	// --- Run the generic backtest engine ---
	algoWealth, spyWealth, bestWealth,
		records, weightRecords, trades :=
		runTest(
			Rtest,
			datesTest,
			universe.Tickers,
			allPrices,
			uSPY,
			uBest,
			meta,
			*executionK,
		)

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

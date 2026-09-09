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
type DONS struct {
	d          int
	n          float64
	beta       float64
	gamma      float64
	momentum   float64
	epsilon    float64
	w          []float64
	p          []float64
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
func NewDONS(d int, n, beta float64) *DONS {
	w1 := make([]float64, d)
	for i := range w1 {
		w1[i] = 1.0 / float64(d)
	}

	p0 := make([]float64, d)
	for i := range p0 {
		p0[i] = 1.0 / float64(d)
	}

	return &DONS{
		d:          d,
		n:          n,
		beta:       beta,
		gamma:      3.0,
		momentum:   0.0,
		epsilon:    1e-9,
		w:          w1,
		p:          p0,
		quadHess:   mat.NewDense(d, d, nil),
		quadOffset: make([]float64, d),
	}
}

func (m *MetaDONS) ensureExperts(t int) {
	if len(m.experts) == 0 {
		e := &Expert{
			dons:      NewDONS(m.d, NStock, BetaStock),
			start:     0,
			end:       minInt(m.T, 2),
			logWeight: 0,
		}
		e.dons.gamma = m.gamma
		e.dons.momentum = m.momentum
		m.experts = append(m.experts, e)
	}

	for start := 1; start <= t && start < m.T; start *= 2 {
		known := false
		for _, e := range m.experts {
			if e.start == start {
				known = true
				break
			}
		}
		if !known {
			e := &Expert{
				dons:      NewDONS(m.d, NStock, BetaStock),
				start:     t,
				end:       minInt(m.T, start+2*start),
				logWeight: 0,
			}
			e.dons.gamma = m.gamma
			e.dons.momentum = m.momentum
			e.start = start
			m.experts = append(m.experts, e)
		}
		if start > m.T/2 {
			break
		}
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ─────────────────────────────────────────────
// Geometric start times: 1,2,4,8,16,...
// ─────────────────────────────────────────────

func geometricStarts(t int) []int {
	var starts []int
	k := 1
	for k <= t {
		starts = append(starts, k)
		k *= 2
	}
	return starts
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────

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

func sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
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
	m.ensureExperts(t)
	active := m.activeExperts(t)
	w := make([]float64, m.d)
	logZ := math.Inf(-1)
	for _, e := range active {
		e.portfolio = e.dons.Predict(m.T)
		logZ = logAdd(logZ, e.logWeight)
	}
	for _, e := range active {
		weight := math.Exp(e.logWeight - logZ)
		w = add(w, scale(e.portfolio, weight))
	}
	return projectInterior(w, 1e-9)
}

func (m *MetaDONS) Update(r []float64, t int) {
	for _, e := range m.activeExperts(t) {
		value := dot(r, e.portfolio)
		if value <= 0 || !isFinite(value) {
			value = 1e-12
		}
		e.logWeight -= m.eta * (-math.Log(value))
		e.dons.Update(r, m.T)
	}
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
	H := mat.NewDense(d.d+1, d.d+1, nil)
	for i := 0; i < d.d; i++ {
		for j := 0; j < d.d; j++ {
			H.Set(i, j, barrierHessian(d.w, nt)[i][j]+d.quadHess.At(i, j))
		}
		H.Set(i, d.d, 1)
		H.Set(d.d, i, 1)
	}
	rhs := mat.NewVecDense(d.d+1, nil)
	for i := range grad {
		rhs.SetVec(i, -grad[i])
	}
	var solution mat.VecDense
	if err := solution.SolveVec(H, rhs); err != nil {
		return
	}
	newton := make([]float64, d.d)
	for i := range newton {
		newton[i] = solution.AtVec(i)
	}
	decrement := math.Max(0, dot(grad, newton))
	den := 1 + 4*math.Sqrt(decrement)
	next := make([]float64, d.d)
	for i := range next {
		next[i] = d.w[i] + d.gamma*solution.AtVec(i)/den
	}
	d.w = projectInterior(next, d.epsilon)
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

func (d *DONS) updateP(u []float64) {
	for i := range d.p {
		if 2*d.p[i] < u[i] {
			d.p[i] = u[i]
		}
	}
}

func (d *DONS) computeNt(T float64) []float64 {
	nt := make([]float64, d.d)
	for i := range nt {
		nt[i] = d.n * math.Exp(math.Log(d.p[i]/float64(d.d))*math.Log(T))
	}
	return nt
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func volatility(xs []float64) float64 {
	n := float64(len(xs))
	var mean float64
	for _, x := range xs {
		mean += x
	}
	mean /= n

	var varSum float64
	for _, x := range xs {
		diff := x - mean
		varSum += diff * diff
	}
	return math.Sqrt(varSum / (n - 1))
}

func sharpe(xs []float64) float64 {
	n := float64(len(xs))
	var mean float64
	for _, x := range xs {
		mean += x
	}
	mean /= n
	return mean / volatility(xs)
}

func maxDrawdown(wealth []float64) float64 {
	maxPeak := wealth[0]
	maxDD := 0.0
	for _, w := range wealth {
		if w > maxPeak {
			maxPeak = w
		}
		dd := 1.0 - w/maxPeak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

func rollingRegret(lossAlgo, lossComp []float64, window int) []float64 {
	T := len(lossAlgo)
	rr := make([]float64, T)
	var sumA, sumC float64

	for t := 0; t < T; t++ {
		sumA += lossAlgo[t]
		sumC += lossComp[t]
		if t >= window {
			sumA -= lossAlgo[t-window]
			sumC -= lossComp[t-window]
		}
		rr[t] = sumA - sumC
	}
	return rr
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
func cleanSeries(xs []float64) []float64 {
	out := make([]float64, len(xs))
	for i, v := range xs {
		if !isFinite(v) {
			out[i] = 0.0
		} else {
			out[i] = v
		}
	}
	return out
}
func isMonthBoundary(prev, curr time.Time) bool {
	return prev.Month() != curr.Month()
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

func max(w []float64) float64 {
	m := w[0]
	for _, x := range w {
		if x > m {
			m = x
		}
	}
	return m
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
		if datesTest[t-1].Month() != datesTest[t].Month() {
			row := []string{
				fmt.Sprintf("%d", t),
				datesTest[t].Format("2006-01-02"),
			}
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
				exitsOut = append(exitsOut, fmt.Sprintf("%.4f", allPrices[tickers[j]][t].Prices[0]))
				weightsOut = append(weightsOut, fmt.Sprintf("%.6f", w[j]))

				prevPrices[j] = currPrice
			}
			// ONE LINE PER REBALANCE
			trades = append(trades, []string{
				fmt.Sprintf("%d", t),
				datesTest[t].Format("2006-01-02"),
				strings.Join(tickersOut, " "),

				strings.Join(entriesOut, " "),
				strings.Join(exitsOut, " "),
				strings.Join(weightsOut, " "),
			})

		}
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
func normalize(w []float64) {
	sum := 0.0
	for _, v := range w {
		if v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) {
			sum += v
		}
	}
	if sum == 0 {
		return
	}
	for i := range w {
		if w[i] > 0 {
			w[i] /= sum
		} else {
			w[i] = 0
		}
	}
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

	var Rtrain, Rtest [][]float64
	var datesTrain, datesTest []time.Time

	for i, dt := range dates {
		if dt.Before(trainEnd) {
			Rtrain = append(Rtrain, R[i])
			datesTrain = append(datesTrain, dt)
		} else {
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

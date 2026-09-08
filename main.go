package main

import (
	"encoding/csv"
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
	dons   *DONS
	start  int
	weight float64
	T      int
}

type MetaDONS struct {
	d       int
	T       int
	eta     float64
	experts []*Expert
}

// ─────────────────────────────────────────────
// DONS struct
// ─────────────────────────────────────────────
type DONS struct {
	d         int
	n         float64
	beta      float64
	w         [][]float64 // weight history
	p         []float64   // barrier parameter vector
	G         []float64   // gradient accumulator
	gsHistory [][]float64 // gradient history
	wsHistory [][]float64 // weight history for quadratic term
}

func NewMetaDONS(d, T int, eta float64) *MetaDONS {
	return &MetaDONS{
		d:       d,
		T:       T,
		eta:     eta,
		experts: []*Expert{},
	}
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
		p0[i] = float64(d)
	}

	return &DONS{
		d:         d,
		n:         n,
		beta:      beta,
		w:         [][]float64{w1},
		p:         p0,
		G:         make([]float64, d),
		gsHistory: [][]float64{},
		wsHistory: [][]float64{},
	}
}

func (m *MetaDONS) ensureExperts(t int) {
	if len(m.experts) == 0 {
		// first expert at t=0
		e := &Expert{
			dons:   NewDONS(m.d, NStock, BetaStock),
			start:  t,
			weight: 1.0,
			T:      m.T,
		}
		m.experts = append(m.experts, e)
		return
	}

	// geometric schedule based on number of experts
	if len(m.experts) < 20 { // cap to avoid explosion
		if t == 1<<len(m.experts) {
			e := &Expert{
				dons:   NewDONS(m.d, NStock, BetaStock),
				start:  t,
				weight: 1.0,
				T:      m.T,
			}
			m.experts = append(m.experts, e)
		}
	}
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

// ─────────────────────────────────────────────
// Cover gradient
// ─────────────────────────────────────────────

func coverGradient(r, u []float64) []float64 {
	den := dot(r, u)
	g := make([]float64, len(r))
	for i := range r {
		g[i] = r[i] / den
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

func quadraticGrad(u []float64, gsHistory [][]float64, wsHistory [][]float64, beta float64) []float64 {
	d := len(u)
	grad := make([]float64, d)
	for s := range gsHistory {
		gs := gsHistory[s]
		ws := wsHistory[s]
		diff := sub(u, ws)
		inner := dot(gs, diff)
		for i := 0; i < d; i++ {
			grad[i] += (beta / 4.0) * inner * gs[i]
		}
	}
	return grad
}

func quadraticHessian(gsHistory [][]float64, beta float64, d int) [][]float64 {
	H := make([][]float64, d)
	for i := 0; i < d; i++ {
		H[i] = make([]float64, d)
	}
	for s := range gsHistory {
		gs := gsHistory[s]
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				H[i][j] += (beta / 4.0) * gs[i] * gs[j]
			}
		}
	}
	return H
}

func (m *MetaDONS) Step(r []float64, t int) []float64 {
	m.ensureExperts(t)

	active := []*Expert{}
	for _, e := range m.experts {
		if e.start <= t {
			active = append(active, e)
		}
	}

	portfolios := make([][]float64, len(active))
	losses := make([]float64, len(active))

	for i, e := range active {
		uRaw := e.dons.Step(r, t, m.T)
		u := projectSimplex(uRaw)
		portfolios[i] = u

		val := dot(r, u)
		if val <= 0 {
			losses[i] = 1000
		} else {
			losses[i] = -math.Log(val)
		}
	}

	var Z float64
	for i, e := range active {
		e.weight *= math.Exp(-m.eta * losses[i])
		Z += e.weight
	}
	for _, e := range active {
		e.weight /= Z
	}

	w := make([]float64, m.d)
	for i, e := range active {
		w = add(w, scale(portfolios[i], e.weight))
	}
	wSparse := topK(w, 3)
	return wSparse
}
func topK(w []float64, K int) []float64 {
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
	wt := d.w[len(d.w)-1]

	u := make([]float64, d.d)
	for i := range u {
		u[i] = (1.0-1.0/float64(T))*wt[i] + 1.0/(float64(d.d)*float64(T))
	}
	// MOMENTUM BIAS
	momentum := 0.10
	if len(d.wsHistory) > 0 {
		// PREVIOUS WEIGHT
		wtPrev := d.wsHistory[len(d.wsHistory)-1]

		for i := range u {
			u[i] += momentum * (wt[i] - wtPrev[i])
		}
	}
	u = projectSimplex(u)
	g := coverGradient(r, u)

	d.gsHistory = append(d.gsHistory, g)
	d.wsHistory = append(d.wsHistory, wt)

	d.updateP(u)
	nt := d.computeNt(float64(T))

	Vgrad := barrierGrad(wt, nt)
	Vhess := barrierHessian(wt, nt)

	Qgrad := quadraticGrad(wt, d.gsHistory, d.wsHistory, d.beta)
	Qhess := quadraticHessian(d.gsHistory, d.beta, d.d)

	totalGrad := add(Vgrad, Qgrad)

	newtonDir := make([]float64, d.d)
	H := mat.NewDense(d.d, d.d, nil)
	for i := 0; i < d.d; i++ {
		for j := 0; j < d.d; j++ {
			H.Set(i, j, Vhess[i][j]+Qhess[i][j])
		}
	}

	epsilon := 1e-6
	for i := 0; i < d.d; i++ {
		H.Set(i, i, H.At(i, i)+epsilon)
	}

	gradVec := mat.NewVecDense(d.d, totalGrad)

	var Hinv mat.Dense
	if err := Hinv.Inverse(H); err == nil {
		HinvGrad := mat.NewVecDense(d.d, nil)
		HinvGrad.MulVec(&Hinv, gradVec)
		for i := 0; i < d.d; i++ {
			newtonDir[i] = HinvGrad.AtVec(i)
		}
	} else {
		for i := 0; i < d.d; i++ {
			newtonDir[i] = totalGrad[i]
		}
	}
	gamma := 3.0
	den := 1.0 + 4.0*math.Sqrt(dot(totalGrad, newtonDir))
	step := scale(newtonDir, gamma/den)

	wNext := make([]float64, d.d)
	for i := range wNext {
		wNext[i] = wt[i] - step[i]
	}

	wNext = projectSimplex(wNext)

	bad := false
	for i := range wNext {
		if math.IsNaN(wNext[i]) || math.IsInf(wNext[i], 0) {
			bad = true
			break
		}
	}

	if bad {
		wNext = make([]float64, d.d)
		for i := range wNext {
			wNext[i] = 1.0 / float64(d.d)
		}
	}

	d.w = append(d.w, wNext)
	return projectSimplex(wNext)

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

		// DAILY expert update (DONS + MetaDONS)
		w = meta.Step(rToday, iter)
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
		iter++

		// DAILY COMPOUNDING
		retAlgo := dotSafe(rToday, w)

		retSPY := dotSafe(rToday, uSPY)
		retBest := dotSafe(rToday, uBest)
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
	fmt.Println("Backtest complete.")
}

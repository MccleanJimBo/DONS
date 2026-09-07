package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
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
	start  int     // when this expert begins
	dons   *DONS   // its own DONS instance
	weight float64 // meta-weight
}

// ─────────────────────────────────────────────
// Meta‑DONS struct
// ─────────────────────────────────────────────

type MetaDONS struct {
	d       int
	T       int
	n       float64
	beta    float64
	eta     float64
	experts []*Expert
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────

func NewMetaDONS(d, T int, n, beta, eta float64) *MetaDONS {
	return &MetaDONS{
		d:       d,
		T:       T,
		n:       n,
		beta:    beta,
		eta:     eta,
		experts: []*Expert{},
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
// Ensure experts exist for this time t
// ─────────────────────────────────────────────

func (m *MetaDONS) ensureExperts(t int) {
	needed := geometricStarts(t)

	existing := make(map[int]bool)
	for _, e := range m.experts {
		existing[e.start] = true
	}

	for _, s := range needed {
		if !existing[s] {
			m.experts = append(m.experts, &Expert{
				start:  s,
				dons:   NewDONS(m.d, m.n, m.beta),
				weight: 1.0,
			})
		}
	}
}

// ─────────────────────────────────────────────
// Meta‑DONS Step
// ─────────────────────────────────────────────

func (m *MetaDONS) Step(r []float64, t int) []float64 {
	m.ensureExperts(t)

	// Active experts = those whose start <= t
	active := []*Expert{}
	for _, e := range m.experts {
		if e.start <= t {
			active = append(active, e)
		}
	}

	portfolios := make([][]float64, len(active))
	losses := make([]float64, len(active))

	// Each expert runs its own DONS step
	for i, e := range active {
		u := e.dons.Step(r, t, m.T)
		portfolios[i] = u
		losses[i] = -math.Log(dot(r, u))
	}

	// Update expert weights (exponential weights)
	var Z float64
	for i, e := range active {
		e.weight *= math.Exp(-m.eta * losses[i])
		Z += e.weight
	}
	for _, e := range active {
		e.weight /= Z
	}

	// Mixture portfolio
	w := make([]float64, m.d)
	for i, e := range active {
		w = add(w, scale(portfolios[i], e.weight))
	}

	return projectSimplex(w)
}

// ─────────────────────────────────────────────
// DONS struct
// ─────────────────────────────────────────────

type DONS struct {
	d         int
	n         float64
	beta      float64
	w         [][]float64
	p         []float64
	G         []float64
	gsHistory [][]float64
	wsHistory [][]float64
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

func projectSimplex(x []float64) []float64 {
	n := len(x)
	u := make([]float64, n)
	copy(u, x)

	sort.Slice(u, func(i, j int) bool { return u[i] > u[j] })

	var sum float64
	rho := -1
	for i := 0; i < n; i++ {
		sum += u[i]
		t := (sum - 1.0) / float64(i+1)
		if u[i] > t {
			rho = i
		}
	}

	sum = 0
	for i := 0; i <= rho; i++ {
		sum += u[i]
	}
	theta := (sum - 1.0) / float64(rho+1)

	y := make([]float64, n)
	for i := 0; i < n; i++ {
		y[i] = math.Max(x[i]-theta, 0.0)
	}
	return y
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

// ─────────────────────────────────────────────
// DONS Step
// ─────────────────────────────────────────────

func (d *DONS) Step(r []float64, t, T int) []float64 {
	wt := d.w[len(d.w)-1]

	u := make([]float64, d.d)
	for i := range u {
		u[i] = (1.0-1.0/float64(T))*wt[i] + 1.0/(float64(d.d)*float64(T))
	}

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
	// newtonDir MUST be declared before the inverse attempt
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

	// try Newton; if it fails, fall back to gradient step
	var Hinv mat.Dense
	if err := Hinv.Inverse(H); err == nil {
		HinvGrad := mat.NewVecDense(d.d, nil)
		HinvGrad.MulVec(&Hinv, gradVec)
	} else {
		// fallback: simple gradient direction
		for i := 0; i < d.d; i++ {
			newtonDir[i] = totalGrad[i]
		}
	}

	HinvGrad := mat.NewVecDense(d.d, nil)
	HinvGrad.MulVec(&Hinv, gradVec)

	for i := 0; i < d.d; i++ {
		newtonDir[i] = HinvGrad.AtVec(i)
	}

	den := 1.0 + 4.0*math.Sqrt(dot(totalGrad, newtonDir))
	step := scale(newtonDir, 1.0/den)

	wNext := make([]float64, d.d)
	for i := range wNext {
		wNext[i] = wt[i] - step[i]
	}

	wNext = projectSimplex(wNext)
	// sanity check: no NaN, no Inf
	bad := false
	for i := range wNext {
		if math.IsNaN(wNext[i]) || math.IsInf(wNext[i], 0) {
			bad = true
			break
		}
	}

	if bad {
		// fallback: keep previous weights or go to uniform
		wNext = make([]float64, d.d)
		for i := range wNext {
			wNext[i] = 1.0 / float64(d.d)
		}
	}

	d.w = append(d.w, wNext)

	return u
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

// BuildReturnMatrix: prices -> aligned returns
func BuildReturnMatrix(all map[string][]data.DailyPrices, universe data.Universe) ([][]float64, []time.Time, error) {
	tickers := universe.Tickers
	d := len(tickers)

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

	R := make([][]float64, len(dates)-1)
	for t := 1; t < len(dates); t++ {
		R[t-1] = make([]float64, d)
		for j := 0; j < d; j++ {
			R[t-1][j] = priceMatrix[t][j] / priceMatrix[t-1][j]
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
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}
func main() {

	universe, err := data.LoadUniverse()
	if err != nil {
		log.Fatalf("load universe: %v", err)
	}
	weightRecords := [][]string{}
	header := []string{"t", "date"}
	for _, ticker := range universe.Tickers {
		header = append(header, ticker)
	}
	weightRecords = append(weightRecords, header)

	allPrices, err := data.LoadAllYahooPrices(universe)
	if err != nil {
		log.Fatalf("load prices: %v", err)
	}

	R, dates, err := BuildReturnMatrix(allPrices, universe)
	if err != nil {
		log.Fatalf("build returns: %v", err)
	}

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

	// tuned-ish params for stocks (refine via grid search)
	n := 2.0
	beta := 0.5
	eta := 0.15

	meta := NewMetaDONS(d, Ttest, n, beta, eta)

	uSPY := comparatorSPY(universe)
	uBest := bestSingleAssetInHindsight(Rtest)

	algoWealth := make([]float64, Ttest)
	spyWealth := make([]float64, Ttest)
	bestWealth := make([]float64, Ttest)

	lossAlgo := make([]float64, Ttest)
	lossSPY := make([]float64, Ttest)
	lossBest := make([]float64, Ttest)

	logRetAlgo := make([]float64, Ttest)
	logRetSPY := make([]float64, Ttest)

	wa, ws, wb := 1.0, 1.0, 1.0
	var cumLossAlgo, cumLossSPY, cumLossBest float64

	records := [][]string{{"t", "date", "loss_algo", "loss_spy", "loss_best", "regret_spy", "regret_best", "wealth_algo", "wealth_spy", "wealth_best"}}
	// previous month's weights (initialized to uniform)
	wPrev := make([]float64, len(universe.Tickers))
	for i := range wPrev {
		wPrev[i] = 1.0 / float64(len(wPrev))
	}
	for t := 1; t <= Ttest; t++ {
		r := Rtest[t-1]
		w := meta.Step(r, t) // pre-sharpening: amplify differences before softmax

		// bias weights using recent returns
		for i := range w {
			w[i] *= math.Exp(2.0 * r[i])
		}
		gamma := 3.0
		for i := range w {
			w[i] = math.Pow(w[i], gamma)
		}
		w = projectSimplex(w)
		// monthly rebalance: only update weights on the first trading day of each month
		rebalance := false
		if t == 1 {
			rebalance = true
		} else {
			prev := datesTest[t-2]
			curr := datesTest[t-1]
			if prev.Month() != curr.Month() {
				rebalance = true
			}
		}

		// keep previous weights for non-rebalance days
		if !rebalance {
			// use last month's weights
			w = wPrev
		} else {
			// select top-K stocks before softmax
			K := 3

			type pair struct {
				idx int
				val float64
			}

			pairs := make([]pair, len(w))
			for i := range w {
				pairs[i] = pair{i, w[i]}
			}

			sort.Slice(pairs, func(i, j int) bool {
				return pairs[i].val > pairs[j].val
			})

			// zero out everything except top-K
			wK := make([]float64, len(w))
			for i := 0; i < K; i++ {
				wK[pairs[i].idx] = pairs[i].val
			}

			// renormalize
			sum := 0.0
			for _, v := range wK {
				sum += v
			}
			for i := range wK {
				wK[i] /= sum
			}

			// now apply softmax to wK
			w = wK
			// softmax sharpening
			alpha := 6.0 // try 2, 4, 8 for more concentration
			maxW := w[0]
			for _, v := range w {
				if v > maxW {
					maxW = v
				}
			}

			// numerically stable softmax
			expW := make([]float64, len(w))
			sumExp := 0.0
			for i := range w {
				expW[i] = math.Exp(alpha * (w[i] - maxW))
				sumExp += expW[i]
			}

			wSharp := make([]float64, len(w))
			for i := range w {
				wSharp[i] = expW[i] / sumExp
			}

			// store for next days
			wPrev = wSharp
			w = wSharp
		}
		for i := range w {
			if !isFinite(w[i]) {
				w[i] = 0.0
			}
		}
		retAlgo := dot(r, w)
		if !isFinite(retAlgo) || retAlgo <= 0 {
			// fallback: treat as no gain/loss
			retAlgo = 1.0
		}

		retSPY := dot(r, uSPY)
		if !isFinite(retSPY) || retSPY <= 0 {
			retSPY = 1.0
		}

		retBest := dot(r, uBest)
		if !isFinite(retBest) || retBest <= 0 {
			retBest = 1.0
		}

		wa *= retAlgo
		ws *= retSPY
		wb *= retBest

		algoWealth[t-1] = wa
		spyWealth[t-1] = ws
		bestWealth[t-1] = wb

		la := -math.Log(retAlgo)
		ls := -math.Log(retSPY)
		lb := -math.Log(retBest)

		lossAlgo[t-1] = la
		lossSPY[t-1] = ls
		lossBest[t-1] = lb

		cumLossAlgo += la
		cumLossSPY += ls
		cumLossBest += lb

		regretSPY := cumLossAlgo - cumLossSPY
		regretBest := cumLossAlgo - cumLossBest

		logRetAlgo[t-1] = math.Log(retAlgo)
		logRetSPY[t-1] = math.Log(retSPY)
		if !isFinite(logRetAlgo[t-1]) {
			logRetAlgo[t-1] = 0.0
		}
		if !isFinite(logRetSPY[t-1]) {
			logRetSPY[t-1] = 0.0
		}
		records = append(records, []string{
			fmt.Sprintf("%d", t),
			datesTest[t-1].Format("2006-01-02"),
			fmt.Sprintf("%.6f", la),
			fmt.Sprintf("%.6f", ls),
			fmt.Sprintf("%.6f", lb),
			fmt.Sprintf("%.6f", regretSPY),
			fmt.Sprintf("%.6f", regretBest),
			fmt.Sprintf("%.6f", wa),
			fmt.Sprintf("%.6f", ws),
			fmt.Sprintf("%.6f", wb),
		})
		row := []string{
			fmt.Sprintf("%d", t),
			datesTest[t-1].Format("2006-01-02"),
		}

		for _, weight := range w {
			if !isFinite(weight) {
				weight = 0.0
			}
			row = append(row, fmt.Sprintf("%.6f", weight))
		}

		weightRecords = append(weightRecords, row)
	}

	rrSPY := rollingRegret(lossAlgo, lossSPY, 60)

	ddAlgo := maxDrawdown(algoWealth)
	ddSPY := maxDrawdown(spyWealth)

	shAlgo := sharpe(logRetAlgo)
	shSPY := sharpe(logRetSPY)

	volAlgo := volatility(logRetAlgo)
	volSPY := volatility(logRetSPY)

	log.Printf("Test period: %s to %s", datesTest[0].Format("2006-01-02"), datesTest[len(datesTest)-1].Format("2006-01-02"))
	log.Printf("Final wealth: Algo=%.4f, SPY=%.4f, BestSingle=%.4f", wa, ws, wb)
	log.Printf("Max drawdown: Algo=%.4f, SPY=%.4f", ddAlgo, ddSPY)
	log.Printf("Sharpe: Algo=%.4f, SPY=%.4f", shAlgo, shSPY)
	log.Printf("Volatility: Algo=%.4f, SPY=%.4f", volAlgo, volSPY)
	if err := writeCSV("analysis/weights.csv", weightRecords); err != nil {
		log.Fatalf("weights CSV error: %v", err)
	}

	if err := writeCSV("analysis/backtest_test_stock.csv", records); err != nil {
		log.Fatalf("CSV error: %v", err)
	}
	if err := plotSeries("analysis/regret_spy_stock.png", rrSPY, "Rolling Regret vs SPY (60d)"); err != nil {
		log.Fatalf("Plot error: %v", err)
	}
	if err := plotSeries("analysis/wealth_algo_stock.png", cleanSeries(algoWealth), "Algo Wealth (Stocks)"); err != nil {
		log.Fatalf("Plot error: %v", err)
	}

	if err := plotSeries("analysis/wealth_spy_stock.png", cleanSeries(spyWealth), "SPY Wealth"); err != nil {
		log.Fatalf("Plot error: %v", err)
	}

	fmt.Println("Stock backtest complete. CSV: analysis/backtest_test_stock.csv")
}

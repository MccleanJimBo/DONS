package data

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"

	// time is required for Yahoo timestamp conversion and cache dates.
	"time"
)

// PricesStruct holds raw periodic close prices for one date.
type PricesStruct struct {
	Date   time.Time
	Prices []float64
}

// Universe defines the tickers you track.
type Universe struct {
	Tickers []string
	Assets  []string
}

const priceCacheDirectory = "csv"

//
// ────────────────────────────────────────────────────────────────
//   UNIVERSE LOADING
// ────────────────────────────────────────────────────────────────
//

// LoadUniverse returns the ORCA universe.
func LoadUniverse() (Universe, error) {
	return Universe{

		Tickers: []string{
			"SPY", // keep first
			"AAPL", "MSFT", "NVDA", "AVGO", "CRM",
			"LLY", "JNJ", "UNH", "ABBV", "MRK",
			"JPM", "BAC", "WFC", "GS", "MS",
			"XOM", "CVX", "COP", "EOG", "SLB",
			"AMZN", "TSLA", "HD", "MCD", "NKE",
			"PG", "KO", "PEP", "WMT", "COST",
			"HON", "UNP", "CAT", "GE", "RTX",
			"LIN", "SHW", "ECL", "APD", "NEM",
			"AMT", "PLD", "EQIX", "SPG", "WELL",
			"NEE", "DUK", "SO", "AEP", "SRE",
			"GOOGL", "META", "NFLX", "TMUS", "DIS", "tlt",
		},
	}, nil
}

// FetchYahooCSV downloads Yahoo Finance CSV for a ticker.
func FetchYahooCSV(ticker string, start, end time.Time) (map[string]any, error) {
	host := []string{"query2", "query1"}[time.Now().Unix()%2]
	client := &http.Client{}
	freq := "1d"
	period1 := start.Unix()
	period2 := end.Unix()
	startStr := strconv.FormatInt(period1, 10)
	endStr := strconv.FormatInt(period2, 10)
	url := fmt.Sprintf("https://%s.finance.yahoo.com/v8/finance/chart/%s?&interval=%s&period1=%s&period2=%s&events=div%%7Csplit&includeAdjustedClose=true&corsDomain=finance.yahoo.com", host, ticker, freq, startStr, endStr)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("Error creating request: %v", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Error: %v", err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error: %v", err)
	}

	var result map[string]any
	err = json.Unmarshal(respBody, &result)
	if err != nil {
		return nil, fmt.Errorf("Error unmarshalling JSON: %v", err)
	}

	chart, ok := result["chart"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("invalid response format")
	}
	res, ok := chart["result"].([]any)
	if !ok || len(res) == 0 {
		return nil, fmt.Errorf("no result in response")
	}
	return res[0].(map[string]any), nil
}

// toTimeSlice converts Yahoo timestamp values to time.Time values.
func toTimeSlice(slice []any) []time.Time {
	result := make([]time.Time, len(slice))
	for i, value := range slice {
		result[i] = time.Unix(int64(value.(float64)), 0)
	}
	return result
}

// ParseYahooCSV converts a Yahoo chart response into periodic close prices.
func ParseYahooCSV(response map[string]any) ([]PricesStruct, error) {
	indicators := response["indicators"].(map[string]any)
	quote := indicators["quote"].([]any)[0].(map[string]any)
	closeValues := quote["close"].([]any)
	if adjusted, ok := indicators["adjclose"].([]any); ok && len(adjusted) > 0 {
		adjustedValues := adjusted[0].(map[string]any)["adjclose"].([]any)
		if len(adjustedValues) > 0 {
			closeValues = adjustedValues
		}
	}
	timestamps := response["timestamp"].([]any)

	type rawPrice struct {
		date  time.Time
		value *float64
	}
	raw := make([]rawPrice, 0, len(timestamps))
	for i := 0; i < len(timestamps) && i < len(closeValues); i++ {
		var closePrice *float64
		if value, ok := closeValues[i].(float64); ok && !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0 {
			closePrice = &value
		}
		raw = append(raw, rawPrice{date: time.Unix(int64(timestamps[i].(float64)), 0), value: closePrice})
	}

	out := make([]PricesStruct, 0, len(raw))
	pending := make([]time.Time, 0, 5)
	var previous float64
	hasPrevious := false
	flushPending := func() {
		if hasPrevious && len(pending) <= 5 {
			for _, date := range pending {
				out = append(out, PricesStruct{Date: date, Prices: []float64{previous}})
			}
		}
		pending = pending[:0]
	}
	for _, point := range raw {
		if point.value == nil {
			if hasPrevious {
				pending = append(pending, point.date)
			}
			continue
		}
		flushPending()
		previous = *point.value
		hasPrevious = true
		out = append(out, PricesStruct{Date: point.date, Prices: []float64{previous}})
	}
	flushPending()
	return out, nil
}
func cachedPricePath(ticker, stamp string) string {
	return filepath.Join(priceCacheDirectory, fmt.Sprintf("%s_%s.csv", ticker, stamp))
}

func loadCachedPrices(path string) ([]PricesStruct, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read cache header: %w", err)
	}
	if len(header) != 2 || header[0] != "date" || header[1] != "adjusted_close" {
		return nil, fmt.Errorf("invalid cache header in %s", path)
	}

	prices := make([]PricesStruct, 0)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read cache row: %w", err)
		}
		if len(record) != 2 {
			return nil, fmt.Errorf("invalid cache row in %s", path)
		}
		date, err := time.Parse(time.RFC3339Nano, record[0])
		if err != nil {
			return nil, fmt.Errorf("parse cached date %q: %w", record[0], err)
		}
		closePrice, err := strconv.ParseFloat(record[1], 64)
		if err != nil || closePrice <= 0 || math.IsNaN(closePrice) || math.IsInf(closePrice, 0) {
			return nil, fmt.Errorf("invalid cached close %q", record[1])
		}
		prices = append(prices, PricesStruct{Date: date, Prices: []float64{closePrice}})
	}
	if len(prices) == 0 {
		return nil, fmt.Errorf("cache file %s is empty", path)
	}
	return prices, nil
}

func saveCachedPrices(path string, prices []PricesStruct) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create cache directory: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create cache file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write([]string{"date", "adjusted_close"}); err != nil {
		return fmt.Errorf("write cache header: %w", err)
	}
	for _, row := range prices {
		if len(row.Prices) == 0 {
			continue
		}
		if err := writer.Write([]string{
			row.Date.Format(time.RFC3339Nano),
			strconv.FormatFloat(row.Prices[0], 'g', -1, 64),
		}); err != nil {
			return fmt.Errorf("write cache row: %w", err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush cache file: %w", err)
	}
	return nil
}

func LoadAllYahooPrices(universe Universe) (map[string][]PricesStruct, error) {
	out := make(map[string][]PricesStruct)
	start := time.Date(2016, time.January, 1, 0, 0, 0, 0, time.UTC) // fixed training start plus the 2024-present test period
	end := time.Now()
	stamp := fmt.Sprintf("2016-present-%s", time.Now().Format("2016-01-02"))
	/*
		start := time.Now().AddDate(-17, 0, 0) // 17 years of history: 15 years train + test period
		end := time.Now().AddDate(-14, 0, 0)    // today
	*/
	for _, ticker := range universe.Tickers {
		cachePath := cachedPricePath(ticker, stamp)
		cached, cacheErr := loadCachedPrices(cachePath)
		if cacheErr == nil {
			log.Printf("loaded %s from %s", ticker, cachePath)
			out[ticker] = cached
			continue
		}
		if !os.IsNotExist(cacheErr) {
			return nil, fmt.Errorf("failed to load cached %s: %v", ticker, cacheErr)
		}

		rows, err := FetchYahooCSV(ticker, start, end)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch %s: %v", ticker, err)
		}

		parsed, err := ParseYahooCSV(rows)
		if err != nil {
			return nil, fmt.Errorf("failed to parse %s: %v", ticker, err)
		}
		if err := saveCachedPrices(cachePath, parsed); err != nil {
			return nil, fmt.Errorf("failed to cache %s: %v", ticker, err)
		}
		log.Printf("saved %s to %s", ticker, cachePath)

		out[ticker] = parsed
	}

	return out, nil
}

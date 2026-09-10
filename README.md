# DONS Portfolio Engine

This project is a Go implementation of a portfolio-selection prototype inspired by the DONS algorithm from the paper:

Damped Online Newton Step for Portfolio Selection
Zakaria Mhammedi and Alexander Rakhlin (2022)

## What the code does

The program loads market return data, builds a return matrix, and runs a backtest over a portfolio universe. It maintains a dense portfolio allocation and optionally applies a sparse top-k execution layer for trading.

The core learning loop is built around:

- a portfolio vector on the simplex
- a barrier-style update for the DONS objective
- a constrained Newton step with a simplex-affine KKT system
- a geometric sleeping-expert meta-allocation across multiple expert learners

The code is designed as a practical research prototype, not as a theorem-verified reproduction of the paper's exact algorithm.

## Main components

### DONS learner

The `DONS` type stores the current portfolio, a barrier state, quadratic curvature terms, and the pending portfolio to be used in the next update.

Key routines include:

- `Predict` — produces the next portfolio before observing the current return
- `Update` — consumes the return vector and updates the learner state
- `solveConstrainedNewton` — solves the constrained Newton system with simplex feasibility
- `updateP` and `computeNt` — maintain the barrier recursion and time-varying barrier strength

### Meta learner

The `MetaDONS` type aggregates multiple expert learners over geometric intervals. It keeps a set of active experts and combines their weighted portfolios into one dense meta-portfolio.

Key routines include:

- `ensureExperts` — creates the initial and geometric follow-on experts
- `retireExpiredExperts` — removes expired experts outside their active interval
- `Predict` — combines the active experts into a single portfolio
- `Update` — updates expert weights using the negative log-loss

### Execution layer

The project can take the dense meta portfolio and optionally apply a sparse `topK` execution policy, which keeps only the top-k assets before normalizing the executed portfolio.

## Typical usage

Run the backtest:

```bash
go run .
```

Run with sparse top-3 execution:

```bash
go run . -top-k 3
```

Run with a higher Newton damping and transaction cost:

```bash
go run . -gamma 10 -transaction-cost 0.001
```

## Command-line arguments

The program uses Go's standard `flag` package. The supported arguments are:

- `-top-k` (default: `0`)
  - execute only the top `K` assets; `0` keeps the dense portfolio as-is.

- `-barrier-strength` (default: `0.25`)
  - strength of the DONS log-barrier term used in the barrier-style Newton update.

- `-gamma` (default: `3.0`)
  - Newton damping multiplier. Higher values increase the step size scaling before backtracking.

- `-momentum` (default: `0.0`)
  - heuristic momentum overlay applied to the portfolio; `0` disables it. Values must stay in `[0, 1)`.

- `-barrier-decay` (default: `1.0`)
  - multiplicative decay applied to the barrier state each update; `1.0` keeps it fixed.

- `-barrier-reset-threshold` (default: `1e-8`)
  - lower floor for the barrier state before the adaptive barrier is reset.

- `-curvature-reset-threshold` (default: `1e5`)
  - reset quadratic curvature when the combined Hessian exceeds this scale.

- `-quadratic-decay` (default: `0.99`)
  - multiplicative decay applied to accumulated quadratic memory each update.

- `-quadratic-update-scale` (default: `1.0`)
  - scale applied to each new quadratic outer-product update.

- `-transaction-cost` (default: `0.0`)
  - proportional cost per unit turnover. A value of `0.001` means roughly 0.1% cost per turnover unit.

Example with several overrides:

```bash
go run . -top-k 10 -gamma 5 -momentum 0.1 -barrier-strength 0.5 -transaction-cost 0.001
```

The configuration is validated at startup; invalid values such as non-positive barrier strength or negative costs will cause the program to exit with an error.

## Data flow

1. Price data is loaded from the `csv` directory.
2. Multiplicative returns are constructed for the universe.
3. The algorithm selects a portfolio before the next return is revealed.
4. The return is applied to the selected portfolio.
5. The learner updates its internal state.
6. Wealth paths and summary metrics are produced for evaluation.

## Notes

- The project keeps the dense meta portfolio and the sparse executed portfolio distinct.
- The dense portfolio is used for learning; the sparse portfolio is an execution overlay.
- The implementation is intended to be a practical, causal portfolio engine, not a strict claim of the paper's formal regret guarantees.

# DONS Implementation Review

## Paper Identification

The relevant paper is *Damped Online Newton Step for Portfolio Selection* by Zakaria Mhammedi and Alexander Rakhlin (2022), arXiv:2202.07574.

ArXiv:2202.07556 is an unrelated paper and should not be used as the source for the DONS claims.

## Summary Assessment

The supplied high-level description of the paper is broadly accurate: DONS uses damped Newton updates to implement a time-varying barrier-based method efficiently, and the meta-algorithm avoids Ada-BARRONS-style FTRL restart computations through a geometric expert construction.

The current Go program is not a faithful implementation of the paper's Algorithms 1-3. It is a backtesting prototype inspired by those ideas, so the paper's regret and complexity guarantees do not apply to it as written.

## Observations

- **Round causality is violated.** `DONS.Step` receives the current return vector, updates its state, and returns `wNext`. `MetaDONS.Step` evaluates that returned portfolio with the same return vector. A portfolio for round `t` must be selected before observing `r_t`.

- **The loss-gradient sign disagrees with the stated loss.** For $\ell_t(u) = -\log\langle r_t, u\rangle$, the gradient is $-r_t / \langle r_t, u\rangle$. `coverGradient` returns the positive version.

- **Barrier adaptation never occurs.** Every `p[i]` is initialized to `d`, while `updateP` requires `2 * p[i] < u[i]`. Since every simplex coordinate is at most `1`, this condition cannot hold for the current initialization. Therefore `computeNt` always uses a fixed barrier strength rather than the paper's adaptive recursion.

- **Simplex projection conflicts with the log barrier.** `projectSimplex` may return zero coordinates. The barrier gradient and Hessian divide by `u[i]` and `u[i]^2`, so a projected boundary point can produce infinities or NaNs. A barrier method requires iterates in the strict simplex interior.

- **The Newton step differs materially from DONS.** The code uses an unconstrained inverse-Hessian direction, an arbitrary diagonal jitter, hard-coded `gamma := 3`, an undocumented momentum term, and Euclidean projection. These are not the constrained, self-concordant, time-varying DONS update analyzed in the paper.

- **The meta-algorithm is not the required sleeping-expert interval construction.** Experts start at `0, 2, 4, 8, ...`, rather than the stated `1, 2, 4, ...`; existing experts are never retired; and the resulting schedule does not provide the geometric interval coverage needed for adaptive interval regret.

- **Top-$3$ filtering changes the learning problem.** `topK(w, 3)` converts the aggregate into a sparse three-asset portfolio. This operation is not part of DONS and invalidates a direct application of the theorem.

- **The actual complexity grows with elapsed time.** Every step reconstructs the quadratic gradient and Hessian by scanning the complete history. The base method therefore has time and memory usage that grow with the horizon, rather than the intended horizon-independent $O(d^3)$ base-step cost.

- **The quadratic coefficient is consistent.** For a regularizer term $\frac{\beta}{8}(g_s^\top(u-w_s))^2$, the gradient and Hessian coefficients are $\beta/4$, matching `quadraticGrad` and `quadraticHessian`.

## Validation

`go test ./...` completed successfully. Both `dons` and `dons/data` have no test files, so there are currently no executable checks for causal ordering, strict interior feasibility, barrier adaptation, or regret behavior.

## Follow-up Observations

- **SPY is the practical comparator.** Comparing strategy wealth with SPY is more meaningful than comparing it with the best single asset selected in hindsight. The latter is an oracle that was not available when portfolios were selected.

- **Dense and sparse execution are now selectable.** `go run .` uses the dense meta portfolio. `go run . -top-k 3` applies top-$3$ filtering only to the executed portfolio, leaving the learner's dense state unchanged.

- **Gamma and momentum are active configuration options.** `gamma` controls the damped Newton step and defaults to `3.0`. Momentum is a strategy heuristic, defaults to `0.0`, and can be enabled with `-momentum`; `-momentum 0.10` activates the previously used setting.

- **Observed run with dense execution.** With `gamma = 3.0` and `momentum = 0.10`, the strategy finished with wealth `3.21` versus `2.57` for SPY. This is approximately 24.9% higher terminal wealth than SPY for that run.

- **Observed run with top-$3$ execution.** With the same gamma and momentum settings, top-$3$ execution finished with wealth `3.32` versus `2.57` for SPY, approximately 29.2% higher terminal wealth. This is an observed result for the current test period, not evidence that top-$3$ is generally superior.

- **Hyperparameter selection remains exposed to test-period bias.** Gamma, momentum, and top-$K$ were varied after inspecting the same evaluation period. A trustworthy performance claim requires selecting them on a validation period and reporting results on a later untouched test period.

- **No public Python reference implementation was found.** GitHub searches for arXiv `2202.07574`, the paper title, and DONS portfolio code found no matching public Python implementation. The paper and its algorithms remain the authoritative reference.

## Implementation Plan

The implementation should preserve top-$K$ selection, `gamma`, and momentum as explicit strategy overlays while correcting the DONS core. With these overlays enabled, the program can be a causal portfolio engine but cannot claim the paper's formal regret guarantee.

1. **Split prediction from learning.** Replace `Step(r, t)` with a method that produces $u_t$ before returns are observed and a separate update method that consumes $r_t$. Update the backtest to calculate wealth from the pre-return portfolio, then update the algorithm. Add a test that catches same-round look-ahead.

2. **Maintain a strictly interior simplex.** Remove Euclidean simplex projection from the DONS core. Use an interior-safe simplex operation that preserves $\sum_i w_i = 1$ and enforces $w_i \geq \epsilon$. Retain the smoothed output $u_t = (1 - 1/T)w_t + \mathbf{1}/(dT)$, and validate finite barrier derivatives and positive coordinates at every step.

3. **Correct the loss-gradient convention.** Define $\ell_t(u) = -\log\langle u, r_t\rangle$ in one place and make `coverGradient` return $g_t = -r_t / \langle u_t, r_t\rangle$. Keep any paper-specific modified gradient separate from the raw loss gradient. Add a finite-difference gradient test.

4. **Implement the barrier recursion exactly.** Replace the current `p` initialization and impossible update with the threshold and recursion specified in the paper. Use a purpose-named barrier state type and add a deterministic test that exercises a coordinate update and verifies the resulting $n_{t,i}$ values.

5. **Rebuild the constrained Newton update.** Construct the paper's time-varying barrier-plus-quadratic objective and solve its Newton system using a Gonum factorization rather than an explicit inverse. Enforce the simplex affine constraint directly. Retain the customized dampening rule

	$$
	w_{t+1} = w_t - \frac{\gamma}{1 + 4\lambda_t} H_t^{-1}\nabla f_t(w_t),
	$$

	exposing `gamma` as configuration and documenting that a value different from the paper's setting is a theoretical deviation.

6. **Keep momentum as a deliberate heuristic.** Add momentum to configuration, use a paper-compatible default of `0`, and retain `0.10` as the strategy setting. Apply it only to a valid pre-return iterate, then restore strict interior feasibility. Record raw and momentum-adjusted portfolios separately.

7. **Implement the geometric sleeping-expert meta-algorithm.** Replace the current start-only schedule with the paper's interval construction, activate and retire experts according to their intervals, and maintain only $O(\log T)$ active experts. Use log-domain weights for stable exp-concave aggregation. Test interval coverage and normalization.

8. **Treat top-$K$ as execution, not core DONS.** Retain dense meta output $w_t$ for the core algorithm and derive a separate tradeable portfolio $q_t = \operatorname{topK}(w_t, K)$. Backtest and report raw DONS/meta performance separately from sparse executed performance.

9. **Make the state and work incremental.** Update the quadratic curvature using rank-one additions based on $g_tg_t^\top$ instead of rebuilding it from history. Bound retained state according to the expert schedule. Add a benchmark over varying $T$ and $d$.

10. **Add focused coverage and diagnostics.** Test strict simplex interior and finite values, causal ordering, gradient correctness, barrier scheduling, Newton decrement/damping, meta-expert lifecycle, and aggregate normalization. Add a deterministic small-return regression case.

Implement in two passes: first the causal, strictly interior base DONS with tests; then the interval meta-algorithm and separate sparse-execution layer. This keeps failures local and clearly distinguishes paper-compatible results from strategy-specific results.

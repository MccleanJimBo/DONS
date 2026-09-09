# DONS Review Notes

## Paper and scope

- Relevant paper: Damped Online Newton Step for Portfolio Selection by Zakaria Mhammedi and Alexander Rakhlin (2022), arXiv:2202.07574.
- arXiv:2202.07556 is unrelated and should not be used as the source for DONS claims.
- The current Go program is a backtesting prototype inspired by the paper, not a faithful Algorithm 1-3 implementation.

## High-level assessment

- The high-level description is broadly accurate: DONS uses damped Newton updates and a time-varying barrier-based objective.
- The meta-algorithm is not a faithful reproduction of the paper’s geometric expert schedule and regret analysis.
- Formal regret and complexity guarantees do not apply to the current implementation as written.

## Core issues identified

### 1. Causal ordering is violated

- DONS.Step receives the current return vector, updates internal state, and returns the next portfolio.
- MetaDONS.Step evaluates that same return vector immediately afterward.
- A portfolio for round t must be selected before observing r_t.

### 2. Gradient sign is inconsistent with the stated loss

- For the loss \(\ell_t(u) = -\log\langle r_t, u\rangle\), the gradient is \(-r_t / \langle r_t, u\rangle\).
- The current code returns the positive version instead.

### 3. Barrier adaptation never occurs

- Every p[i] is initialized to d.
- The update rule requires 2 * p[i] < u[i].
- Since simplex coordinates are at most 1, this condition never holds with the current initialization.
- The adaptive barrier recursion from the paper is never actually used.

### 4. Simplex projection conflicts with the log barrier

- Projecting a vector onto the simplex can create zero coordinates.
- The barrier gradient and Hessian divide by u[i] and u[i]^2.
- This can create infinities or NaNs at the boundary.
- A valid barrier method requires a strictly interior simplex iterate.

### 5. Newton step is materially different from DONS

- The current code uses an unconstrained inverse-Hessian direction.
- It adds arbitrary diagonal jitter.
- It hard-codes gamma := 3.
- It includes an undocumented momentum term.
- It applies Euclidean projection instead of the paper’s constrained self-concordant update.

### 6. Meta-algorithm does not match the required sleeping-expert construction

- Experts start at 0, 2, 4, 8, ... instead of 1, 2, 4, ...
- Existing experts are not retired.
- The resulting interval schedule does not provide the geometric coverage needed for adaptive interval regret.

### 7. Top-k filtering changes the optimization problem

- topK(w, 3) transforms the dense portfolio into a sparse three-asset portfolio.
- This is not part of DONS and invalidates a direct theorem application.

### 8. Complexity grows with time horizon

- The implementation rebuilds the quadratic gradient and Hessian from complete history on each step.
- This makes run time and memory grow with horizon instead of the intended near-horizon-independent base cost.

### 9. Quadratic coefficient is consistent

- For the regularizer term \(\frac{\beta}{8}(g_s^\top(u - w_s))^2\), the gradient and Hessian coefficients are \(\beta/4\), matching the code.

## Validation status

- go test ./... completed successfully.
- There are no executable tests for:
  - causal ordering,
  - strict interior feasibility,
  - barrier adaptation,
  - regret behavior,
  - meta-expert scheduling.

## Follow-up observations

- SPY is a more meaningful practical benchmark than a hindsight best single asset.
- Dense and sparse execution are selectable: go run . uses the dense meta portfolio; go run . -top-k 3 applies top-k filtering only to execution.
- Gamma and momentum are active configuration options.
- With gamma = 3.0 and momentum = 0.10, one observed dense run ended at wealth 3.21 versus SPY at 2.57.
- A top-3 run under the same settings reached wealth 3.32 versus SPY at 2.57.
- These are observed test-period outcomes, not general evidence of superiority.
- Hyperparameter choices were selected using the same evaluation period, which introduces test-period bias.
- No public Python reference implementation was found for the paper.

## Implementation plan

The implementation should keep the strategy overlays while correcting the DONS core.

1. Split prediction from learning.
   - Produce u_t before returns are observed.
   - Update only after receiving r_t.
   - Backtest wealth using the pre-return portfolio.
   - Add a regression test for look-ahead.

2. Maintain a strictly interior simplex.
   - Remove Euclidean simplex projection from the DONS core.
   - Use an interior-safe update that preserves the simplex and enforces lower bounds.
   - Keep the smoothed output u_t = (1 - 1/T) w_t + 1/(dT) * 1.
   - Validate finite barrier derivatives and positive entries.

3. Correct the loss-gradient convention.
   - Define \(\ell_t(u) = -\log\langle u, r_t\rangle\) in one place.
   - Make the raw loss gradient g_t = -r_t / \langle u_t, r_t\rangle\.
   - Keep any paper-specific modified gradient separate.
   - Add a finite-difference gradient test.

4. Implement the barrier recursion exactly.
   - Replace the current p initialization and impossible update logic.
   - Use the paper’s threshold and recursion in a dedicated barrier-state type.
   - Add tests that exercise a coordinate update and verify values of n_{t,i}.

5. Rebuild the constrained Newton update.
   - Construct the time-varying barrier-plus-quadratic objective.
   - Solve the Newton system with a Gonum factorization.
   - Enforce the simplex affine constraint directly.
   - Keep the dampened update with gamma exposed as configuration.

6. Keep momentum as an explicit heuristic.
   - Default momentum to 0.
   - Retain 0.10 as a strategy setting.
   - Apply it only to a valid pre-return iterate.
   - Restore strict interior feasibility afterward.

7. Implement the geometric sleeping-expert meta-algorithm.
   - Replace the current schedule with the paper’s interval construction.
   - Activate and retire experts by interval.
   - Keep only O(log T) active experts.
   - Use log-domain weighting for stability.

8. Treat top-k as execution, not core DONS.
   - Keep the dense meta output w_t for learning.
   - Derive a separate tradeable portfolio q_t = topK(w_t, K).
   - Report raw DONS performance separately from sparse execution.

9. Make state and work incremental.
   - Update curvature via rank-one additions instead of rebuilding from history.
   - Bound retained state according to the expert schedule.
   - Add scalability benchmarks over T and d.

10. Add focused tests and diagnostics.
   - Test strict simplex interior and finite values.
   - Verify causal ordering.
   - Validate gradient correctness.
   - Check barrier scheduling.
   - Check Newton decrement and damping.
   - Check meta-expert lifecycle and normalization.
   - Add deterministic small-return regressions.

## Recommended implementation order

1. Base causal, strictly interior DONS with tests.
2. Interval meta-algorithm.
3. Sparse execution layer.

This keeps failures local and cleanly distinguishes paper-compatible core behavior from strategy overlays.

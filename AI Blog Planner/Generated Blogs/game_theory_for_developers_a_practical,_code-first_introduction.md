# Game Theory for Developers: A Practical, Code-First Introduction

## Intro: Why game theory matters for engineers

Engineers build systems where users, services, and components make decisions that affect one another. Game theory gives tools to model those interactions so you can design protocols, incentives, and interfaces that lead to predictable, robust outcomes.

Concrete problem — ad-auction bidding:
- Players: two bidders competing for a single ad slot.
- Actions: each bidder i picks a bid b_i ≥ 0.
- Payoffs: bidder i has private value v_i; if i wins at price p, utility = v_i − p, otherwise 0. Auction rule maps bids (b_1,b_2) → winner and price.

When to use game-theoretic framing (checklist):
- (1) Multiple decision-makers? ✓
- (2) Payoffs depend on others' actions? ✓
- (3) Information asymmetry exists? ✓

How equilibria guide design:
- Compute equilibrium strategies (or simulate best responses) to predict behavior under a protocol; use mechanism design to choose rules (e.g., reserve price, payment rule) that align individual incentives with your objective. Trade-offs: adding complexity can improve welfare but increases implementation and computation cost. Track one measurable outcome — average bid (or social welfare, or regret) — and monitor deviations from predicted equilibrium to detect collusion or model mismatch. Best practice: simulate equilibrium under realistic distributions before deployment (why: exposes failure modes cheaply).

## Core concepts: players, strategies, payoffs, best responses, and Nash equilibrium

Players — formal: agents with decision sets in a game. Code: players = ['Row','Col'].

Pure strategy — formal: a single action choice from a player's action set. Code: actions = {'Row': ['U','D'], 'Col': ['L','R']}.

Mixed strategy — formal: a probability distribution over a player's pure strategies. Code: sigma = {'Row': [p_U,p_D], 'Col': [q_L,q_R]} (vectors summing to 1).

Payoff function — formal: u_i(s1,...,sn) maps strategy profile to a real payoff for player i. Code: payoff_row = [[3,0],[1,2]]; payoff_col = [[2,1],[0,3]] (rows = U/D, cols = L/R).

Best response — formal: any strategy that maximizes a player's payoff given others' strategies. Code: best_response(payoff_matrix, opp_strategy) → argmax action(s).

Nash equilibrium — formal: a profile where every player's chosen strategy is a best response to the others. Code check: for each i, current_strategy_i in best_response_i(others).

Compute a best response (2-player finite)
Algorithm:
1. For each candidate action a_i of player i:
2. If opponent plays pure a_j: payoff = payoff_matrix[a_i][a_j].
   If opponent plays mixed q: payoff = sum_j q[j]*payoff_matrix[a_i][j].
3. Select argmax action(s).

Hand-worked example (Row vs Col, payoff_row above):
- If Col plays L: Row(U)=3, Row(D)=1 → best = U.
- If Col plays R: Row(U)=0, Row(D)=2 → best = D.

Mixed strategies and checking violations
Expected payoff of pure action a_i vs opponent mixed q: E = sum_j q[j]*payoff[a_i][j].
Vice versa: expected payoff of opponent pure vs your mixed p: E = sum_i p[i]*payoff[i][a_j].
Best-response violation: if current strategy s gives E_current but some a' has E_a' > E_current + eps, violation (use small eps for floating errors).

Quick test (unit-test)
Input:
- actions as above
- payoff_row = [[3,0],[1,2]]
- Col mixes q=[0.6,0.4]
Expect:
- best_response(Row, pure L) == ['U']
- best_response(Row, pure R) == ['D']
- best_response(Row, mix q) == ['U']

Tiny helper (Python-like):
```py
def expected(payoff, my_idx, opp_mix):
    return sum(opp_mix[j]*payoff[my_idx][j] for j in range(len(opp_mix)))
```

## Solve simple games: dominance, elimination, and a brute-force Nash solver (with code sketch)

Dominance elimination is the simplest reduction. Two concrete 2×2 payoff matrices (row player, column player). Rows = {Top, Bottom}, Columns = {Left, Right}.

1) Prisoner’s Dilemma (Bottom = Defect dominates Top = Cooperate)
Row payoffs:
- Top vs Left = 3, Top vs Right = 0
- Bottom vs Left = 5, Bottom vs Right = 1
Matrix (row player payoffs):
[[3, 0],
 [5, 1]]
Compare Bottom vs Top:
- If column plays Left: Bottom 5 > Top 3
- If column plays Right: Bottom 1 > Top 0
So Bottom strictly dominates Top. Eliminate Top -> reduced game is both players play Bottom (pure (Bottom, Right)).

2) Coordination (no dominance)
Row payoffs:
[[2, 0],
 [0, 1]]
No row strictly dominates the other (each is better against a different column), so no elimination.

MWE: brute-force 2×2 pure + simple mixed (grid) solver in Python
- Represent payoffs as 2×2 lists; p = prob row plays Top; q = prob column plays Left.
- For each grid p,q (step 0.05) check best-response conditions: every pure action with positive probability must be a best response; no excluded pure action should strictly improve payoff.

```python
import numpy as np

def find_equilibria(A, B, step=0.05, tol=1e-6):
    eqs = []
    ps = np.arange(0, 1+step/2, step)
    for p in ps:
        for q in ps:
            # expected payoff of row's pure actions against column(q)
            UR_top  = q*A[0][0] + (1-q)*A[0][1]
            UR_bot  = q*A[1][0] + (1-q)*A[1][1]
            # expected payoff of column's pure actions against row(p)
            UC_left = p*B[0][0] + (1-p)*B[1][0]
            UC_right= p*B[0][1] + (1-p)*B[1][1]

            # which actions are best responses (within tol)
            br_row = []
            if UR_top >= UR_bot - tol: br_row.append('Top')
            if UR_bot >= UR_top - tol: br_row.append('Bottom')
            br_col = []
            if UC_left >= UC_right - tol: br_col.append('Left')
            if UC_right >= UC_left - tol: br_col.append('Right')

            # support implied by (p,q)
            supp_row = ('Top' if p>tol else '') + (',Bottom' if p<1-tol else '')
            supp_row_set = set()
            if p>tol: supp_row_set.add('Top')
            if p<1-tol: supp_row_set.add('Bottom')
            if p<=tol: supp_row_set = {'Bottom'}
            if p>=1-tol: supp_row_set = {'Top'}

            supp_col_set = set()
            if q>tol: supp_col_set.add('Left')
            if q<1-tol: supp_col_set.add('Right')
            if q<=tol: supp_col_set = {'Right'}
            if q>=1-tol: supp_col_set = {'Left'}

            if supp_row_set.issubset(set(br_row)) and supp_col_set.issubset(set(br_col)):
                eqs.append((round(p,3), round(q,3)))
    return sorted(set(eqs))

# Prisoner's Dilemma (row and col symmetric)
A = [[3,0],[5,1]]
B = [[3,5],[0,1]]  # column's payoff matrix arranged columns as Left/Right, rows Top/Bottom

print(find_equilibria(A,B,step=0.25))
```

Expected output (grid includes pure D/D = (p=0,q=0)):
[(0.0, 0.0)]
(Interpretation: both defect — pure equilibrium.)

How to verify a candidate Nash (unit-test assertions)
- For a candidate mixed profile (p,q): for every player, for every pure deviation action a', expected_payoff(player, candidate) >= expected_payoff(player, a' vs opponents) - tol.
- For mixed strategies, assert all pure actions in the strategy’s support yield equal expected payoff (within tol).
- Assert probabilities are in [0,1] and sum to 1 (for larger supports).
Why: these checks ensure no unilateral profitable deviation and internal consistency.

Extensibility note
- n×m brute-force: enumerate supports and probability grids; complexity grows combinatorially (exponential in actions). Feasible for small (<4×4) only.
- For larger games use iterative methods (fictitious play, best-response dynamics), Lemke–Howson for 2-player nondegenerate games, or linear programming for zero-sum games. Trade-offs: grid brute-force is simple and reliable but costly; LP/algorithms are faster but require more complex implementation and numerical care.

## Common mistakes and how to avoid them

- Mistake: treating equilibrium as a prediction without considering dynamics  
  Mitigation: run agent-based simulations and measure convergence rate. Reproducible check: run asynchronous best-response updates from many random initial policies and record final states; if different runs end in different states, equilibrium is not a unique dynamical prediction.

  Example (Python, two players, two actions):
  ```python
  import random
  payoff = {('A','A'):(2,2),('A','B'):(0,3),('B','A'):(3,0),('B','B'):(1,1)}
  def best_response(other): return max(['A','B'], key=lambda a: payoff[(a,other)][0])
  def run(seed): 
      random.seed(seed); s=(random.choice(['A','B']),random.choice(['A','B']))
      for _ in range(100):
          i=random.choice([0,1])
          s=list(s); s[i]=best_response(s[1-i] if i==0 else s[0]); s=tuple(s)
      return s
  results={run(s) for s in range(50)}
  print(results)  # check multiple outcomes
  ```

- Mistake: forgetting mixed or continuous strategies  
  Mitigation: add tests that evaluate best responses to randomized opponent policies and use numeric tolerance for floats. Example: compute expected payoff against opponent distribution and assert best-response payoff >= alternative - atol (use numpy.isclose with atol=1e-6).

- Mistake: mis-specifying payoffs (units or incentives)  
  Mitigation: payoff-validation checklist:
  1. Unit consistency (same units across payoffs).  
  2. Monotonicity: higher-valued outcomes should increase payoff.  
  3. Sanity: verify existence/non-existence of dominated strategies.  
  Apply these to your example and fail fast on violations.

- Mistake: assuming unique equilibrium  
  Mitigation: diagnostic steps:
  1. Enumerate all pure/mixed equilibria (small games).  
  2. Plot payoff landscape for 2x2: parametrize strategies p,q∈[0,1], compute expected payoffs, visualize contours.  
  3. Test stability: perturb payoffs by ±ε and re-solve; report how equilibria move.  

Trade-offs: simulations add cost and stochasticity but reveal dynamics; tighter tolerances reduce false positives but risk rejecting correct near-ties.

## Applying game theory in practice: mechanism design, simulations, and trade-offs

Design a simple sealed‑bid auction
- Strategy space: each player i has valuation v_i ∈ [0,Vmax] and chooses bid b_i ∈ B (e.g., continuous [0,v_i] or discrete grid G).
- Allocation rule: highest bid wins; ties broken uniformly at random among top bidders.
- Payment rules:
  - First‑price (FPSB): winner pays p = b_winner.
  - Second‑price (Vickrey): winner pays p = max_{j≠w} b_j.
- Objectives:
  - Auctioneer: maximize expected revenue R = E[p].
  - Social welfare: maximize expected welfare W = E[sum of valuations of allocated items] (single item → v_winner).
- Player payoff (common formula):
  u_i = { v_i − p if i wins; 0 otherwise }.
  For Vickrey, p = max_{j≠i} b_j; for FPSB, p = b_i.

Actionable simulation plan
- Setup: choose N, T, valuation distribution F (e.g., Uniform[0,1]), strategy family s(v; θ) (e.g., shading α: b=α·v).
- Loop: for t in 1..T
  1. sample v_1..v_N ∼ F
  2. compute b_i = s(v_i; θ)
  3. determine winner, payment, record revenue, welfare, individual payoffs
- Exact metrics to collect and report:
  - average revenue (mean, std)
  - allocation efficiency = E[v_winner] / E[max_i v_i] (mean, std)
  - per‑agent average regret (approx): regret_i ≈ max_{b'∈G} u_i(b', b_{−i}) − u_i(b_i, b_{−i})
  - bid distribution, win rates by valuation decile
- Plots to produce:
  - revenue vs θ (line + CI)
  - efficiency vs θ
  - regret histogram / CDF
  - bid CDF by valuation bin
- Minimal code sketch:
```python
for t in range(T):
  v = rng.sample(N)
  b = s(v, theta)
  winner = argmax(b); payment = payment_rule(b, winner)
  record(revenue=payment, welfare=v[winner], payoffs=...)
```

Edge cases & one test
- Test list: asymmetric information (different F_i), collusion, bid ties, zero/identical valuations.
- Example test: two agents coordinate to submit b≈0 while others act truthfully → expected: revenue drops sharply vs baseline; detect by comparing revenue percentiles and increased win concentration.

Trade-offs & performance
- Per trial cost: sorting bids O(N log N). Computing regret by brute‑force over grid G costs O(N^2·G) per trial.
- Scaling: total cost ≈ O(T·(N log N + N^2 G)). For N>50 or G>1000 or T large (>1e5), switch from brute‑force to:
  - Monte‑Carlo sampling of opponent profiles,
  - Importance sampling for rare events,
  - Mean‑field / analytical approximations for symmetric large N.
- Also parallelize trials and vectorize bid computation to cut wall time. Tradeoffs: sampling reduces exactness; mean‑field gains speed but loses finite‑N effects.

## Testing, observability, and next steps: checklist and resources

Production-readiness checklist
1. Payoff validation tests — assert payoffs are within expected ranges, unit-test extreme inputs, and include property tests (e.g., sum of zero-sum payoffs == 0).
2. Equilibrium enumeration and stability checks — enumerate equilibria (pure/mixed) and verify stability by perturbing strategies and checking return-to-equilibrium behavior.
3. Unit tests for best-response calculations — test best-response on small canonical games (Prisoner’s Dilemma, Matching Pennies) with deterministic inputs.
4. Simulation reproducibility — fix RNG seeds, snapshot seeds/configs in outputs, and run full replay in CI (e.g., pytest + fixed seed matrix).

Observability and debugging tips
- Emit per-agent metrics: utility distribution (histogram), rolling mean/std of utility, counts of strategy changes, and convergence diagnostics (distance-to-equilibrium).
- Logs: timestamped events of strategy updates and perturbations; avoid verbose state dumps in prod.
- Detect non-convergence: compute variance over last-k utilities; flag if > threshold or if strategy-change-rate stays > r for t steps.

Example (Python):
```python
import numpy as np
def non_converged(last_k_utils, var_thresh=1e-3):
    return np.var(last_k_utils) > var_thresh
```

Next steps & resources
- Implement replicator dynamics; observe time-averaged frequencies.
- Study mixed-strategy computation via LP/QP (simplex, CVXOPT).
- Try Nashpy or Gambit to compute equilibria.
Exercise: run replicator dynamics on Rock-Paper-Scissors and use Nashpy to compute NE; expected: Nashpy NE ≈ [1/3,1/3,1/3], time-averaged strategies ≈ that vector.

Security/privacy checklist
- Never log raw user IDs or PII.
- Aggregate/anonymize agent traces (k-anonymity or simple hashing + salt).
- Apply access controls and retention limits; use DP mechanisms if traces feed ML.

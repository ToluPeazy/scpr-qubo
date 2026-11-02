# SCPR to QUBO

An exact encoding of the Set Covering Problem with Reasons (SCPR) as a Quadratic Unconstrained Binary Optimisation (QUBO) problem, with Beasley pre-reduction and post-QUBO variable fixing. The encoder emits a plain binary quadratic model, so any `dimod` sampler accepts it, including D-Wave's Leap hybrid solvers and the Advantage QPU.

## The problem

An SCPR instance has a universe `U`, a set of reasons `R`, and pairs `(A_i, R_i)` with `A_i ⊆ U` and `R_i ⊆ R`. A pair activates once you select every reason in `R_i`, and an active pair covers the elements of `A_i`. You look for the smallest `S ⊆ R` whose active pairs cover `U`. Pairs may overlap, and one reason can serve several pairs.

SCPR generalises set cover and is NP-hard. It appears in SMT and cylindrical algebraic decomposition (variable orderings as reasons), cancer genomics (gene markers), and cryptographic failure diagnosis (failure conditions).

## The encoding

| Component | Term |
|---|---|
| Variables | one binary `y_k` per reason |
| Objective | `+1` on each `y_k`, so the minimum favours small covers |
| Activation | pairwise AND gadget `P(ab - 2az - 2bz + 3z)`, chained for pairs with three or more reasons |
| Coverage | `P(Σ acts - 1 - s)²` with `s` in `⌈log₂ t⌉` slack bits for an element reachable by `t ≥ 2` pairs; `P(1 - act)` for `t = 1` |
| Penalty | `P = \|R\| + 1` by default |

The AND gadget equals zero exactly when `z = a ∧ b`. The slack lets an element take any number of covering pairs above one at no cost, which keeps overlapping optima feasible. The objective spans `[0, |R|]`, so a single violated penalty of at least `P` outweighs any objective saving. A feasible assignment encoding cover `S` has energy `|S|`.

The QUBO uses `|R| + Σ_i (|R_i| - 1) + Σ_u ⌈log₂ t_u⌉` variables. Beasley reduction runs first when you want fewer: Rule 1 forces the reasons of any pair that alone covers some element, and Rule 2 removes dominated pairs.

The pipeline compares five routes to the optimum:

1. brute force over subsets of `R`
2. Beasley reduction, then brute force on the reduced instance
3. QUBO on the full instance
4. Beasley reduction, then QUBO on the reduced instance (pre-Q)
5. QUBO on the full instance, then fixing the Rule 1 reasons to 1 (post-Q)

## Verification

`python -m scpr_qubo.harness` checks every stage against two independent exact oracles, brute force and an integer program solved with `scipy.optimize.milp`. It then confirms that dimod's `ExactSolver` and `SimulatedAnnealingSampler` reach the same minimum energy as the enumeration oracle. The output below comes from `results/harness_output.txt` (Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, dimod 0.12.23). The ILP serves as a development cross-check only, and the paper in `paper/` uses brute force as its sole reference oracle.

| Instance | `\|U\|` | `\|R\|` | `\|E\|` | Optimum | Stage 3 vars | Stage 4 vars | Stage 5 vars |
|---|---|---|---|---|---|---|---|
| Small_4x4 | 4 | 4 | 4 | 3 | 11 | 11 | 11 |
| Reduction_Demo | 5 | 4 | 4 | 3 | 10 | 8 | 9 |
| Thesis_1.5.2 | 16 | 8 | 4 | 6 | 21 | 0 | 15 |

All five stages return an optimal cover on all three instances. On Thesis_1.5.2, Beasley reduction resolves the whole instance, and Stage 4 builds no QUBO. On Reduction_Demo, the Stage 4 energy reads 2.000: the reduced QUBO scores the reduced subproblem, and the forced reason adds the third.

The randomised self-test draws 4,000 feasible instances with up to 5 elements, 5 reasons and 7 pairs (seed 1):

| Check | Count |
|---|---|
| Instances tested | 4,000 |
| Brute force and ILP disagree | 0 |
| Instances with QUBO stages 3 to 5 checked | 3,356 |
| Instances with QUBO stages skipped (over 20 QUBO variables) | 644 |
| Instances containing duplicate pairs | 872 |
| Stage failures | 0 |

The enumeration oracle caps the QUBO checks at 20 variables. Stage 2 runs on all 4,000 instances.

## Install and run

```bash
git clone https://github.com/ToluPeazy/scpr-qubo.git
cd scpr-qubo
pip install -e ".[test]"
python -m scpr_qubo.harness          # full run, about 4 to 5 minutes
pytest                               # unit tests and a 300-instance self-test
```

`--n` sets the number of random instances and `--seed` sets their seed.

```python
from scpr_qubo import SCPRInstance, build_qubo
from scpr_qubo.dimod_backend import to_bqm

inst = SCPRInstance(U=[0, 1, 2], R=["a", "b"],
                    E=[({0, 1}, {"a"}), ({1, 2}, {"b"})])
bqm = to_bqm(build_qubo(inst))       # pass to any dimod sampler
```

## Layout

```
src/scpr_qubo/
  instance.py       SCPR instance and coverage check
  beasley.py        Rule 1 and Rule 2 reduction
  qubo.py           encoder, exact enumeration solver, variable fixing
  dimod_backend.py  BinaryQuadraticModel conversion, ExactSolver and SA
  pipeline.py       the five stages
  oracles.py        brute force and ILP reference solvers
  benchmarks.py     benchmark instances
  harness.py        verification harness
tests/              pytest suite
results/            harness output
```

## Citation

Cite the thesis that defines SCPR and the Beasley reduction:

> A. T. Babatunde. *Introducing and Analysing the Set Covering Problem with Reasons.* PhD thesis, Coventry University, 2025.

`CITATION.cff` holds the citation for this repository.

## Licence

Apache License 2.0. See `LICENSE`.

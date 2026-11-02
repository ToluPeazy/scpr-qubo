import dimod
import numpy as np
import pytest

from scpr_qubo import benchmarks
from scpr_qubo.qpu_experiment import (
    build_problems, chain_strength_values, main, summarise,
)


def _by_variant(problems):
    return {(p.instance, p.variant): p for p in problems}


def test_problems_match_harness_sizes_and_targets():
    probs = _by_variant(build_problems([benchmarks.reduction_demo(),
                                        benchmarks.thesis_1_5_2()]))
    rd_full = probs[("Reduction_Demo", "full")]
    rd_pre = probs[("Reduction_Demo", "pre-q")]
    rd_post = probs[("Reduction_Demo", "post-q")]
    assert (rd_full.n_vars, rd_full.target_energy) == (10, 3)
    assert (rd_pre.n_vars, rd_pre.target_energy) == (8, 2)   # scores |S'| only
    assert (rd_post.n_vars, rd_post.target_energy) == (9, 3)
    assert (probs[("Thesis_1.5.2", "full")].n_vars,
            probs[("Thesis_1.5.2", "full")].target_energy) == (21, 6)
    assert probs[("Thesis_1.5.2", "post-q")].n_vars == 15
    # Beasley resolves Thesis_1.5.2 outright, so pre-Q has no QUBO to send.
    assert ("Thesis_1.5.2", "pre-q") not in probs


def test_variants_identical_to_full_are_skipped():
    probs = _by_variant(build_problems([benchmarks.small_4x4()]))
    assert set(probs) == {("Small_4x4", "full")}


@pytest.mark.parametrize("variant", ["full", "pre-q", "post-q"])
def test_decode_returns_optimal_cover_of_original_instance(variant):
    from scpr_qubo.qubo import solve_qubo_exact
    inst = benchmarks.reduction_demo()
    p = _by_variant(build_problems([inst]))[("Reduction_Demo", variant)]
    _, x = solve_qubo_exact(p.qubo)
    S = p.decode({i: int(v) for i, v in enumerate(x)})
    assert inst.check_coverage(S)
    assert len(S) == p.optimum_size == 3


def _problem_with_known_samples():
    inst = benchmarks.reduction_demo()
    p = _by_variant(build_problems([inst]))[("Reduction_Demo", "full")]
    from scpr_qubo.qubo import solve_qubo_exact
    _, x_opt = solve_qubo_exact(p.qubo)
    x_all = np.ones(p.n_vars, dtype=int)          # every reason on: feasible, size 4
    x_all[: p.qubo.n_reason] = 1
    # make the all-on assignment penalty-free: let the exact solver fix aux/slack
    from scpr_qubo.qubo import fix_variables
    fixed = {k: 1 for k in range(p.qubo.n_reason)}
    sub, keep, remap = fix_variables(p.qubo, fixed)
    _, xs = solve_qubo_exact(sub)
    for old in keep:
        x_all[old] = xs[remap[old]]
    x_none = np.zeros(p.n_vars, dtype=int)        # nothing selected: infeasible
    samples = np.array([x_opt, x_all, x_none])
    energies = [p.bqm.energy(dict(enumerate(s))) for s in samples]
    ss = dimod.SampleSet.from_samples(
        (samples, list(range(p.n_vars))), vartype="BINARY", energy=energies,
        num_occurrences=[5, 3, 2], chain_break_fraction=[0.0, 0.1, 0.5])
    return p, ss


def test_summarise_counts_success_feasibility_and_chain_breaks():
    p, ss = _problem_with_known_samples()
    row = summarise(p, ss)
    assert row["num_reads"] == 10
    assert row["success_prob"] == pytest.approx(0.5)
    assert row["feasible_frac"] == pytest.approx(0.8)
    assert row["mean_chain_break_frac"] == pytest.approx(0.13)
    assert row["best_feasible_size"] == 3
    assert row["best_energy"] == pytest.approx(3.0)


def test_chain_strength_values_scale_with_largest_bias():
    p = build_problems([benchmarks.reduction_demo()])[0]
    biggest = max(max(abs(v) for v in p.bqm.linear.values()),
                  max(abs(v) for v in p.bqm.quadratic.values()))
    vals = chain_strength_values(p.bqm, ["default", 0.5, 1.0])
    assert vals[0] == ("default", None)
    assert vals[1] == (0.5, pytest.approx(0.5 * biggest))
    assert vals[2] == (1.0, pytest.approx(biggest))


def test_qpu_run_refuses_without_yes(monkeypatch):
    import scpr_qubo.qpu_experiment as qe
    def boom(*a, **k):
        raise AssertionError("must not connect to the QPU without --yes")
    monkeypatch.setattr(qe, "_qpu_sampler", boom)
    assert main(["--sampler", "qpu"]) == 2


def test_mock_end_to_end(tmp_path):
    pytest.importorskip("dwave.system")
    out = tmp_path / "mock.csv"
    rc = main(["--sampler", "mock", "--reads", "20", "--chain-scales", "1.0",
               "--instances", "Reduction_Demo", "--seed", "1", "--out", str(out)])
    assert rc == 0
    import csv
    rows = list(csv.DictReader(out.open()))
    assert {r["variant"] for r in rows} == {"full", "pre-q", "post-q"}
    for r in rows:
        assert 0.0 <= float(r["success_prob"]) <= 1.0
        assert int(r["physical_qubits"]) >= int(r["logical_vars"])

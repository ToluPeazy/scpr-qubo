import itertools
import random

import pytest

from scpr_qubo import SCPRInstance, beasley, brute_force_optimum, build_qubo, solve_qubo_exact
from scpr_qubo import benchmarks
from scpr_qubo.harness import self_test, verify_instance
from scpr_qubo.qubo import QUBO


def test_and_gadget_truth_table():
    P = 3.0
    Q = {(0, 1): P, (0, 2): -2 * P, (1, 2): -2 * P, (2, 2): 3 * P}
    g = QUBO(Q, 0.0, 3, 0, [])
    for a, b, z in itertools.product((0, 1), repeat=3):
        e = g.energy([a, b, z])
        if z == (a & b):
            assert e == 0
        else:
            assert e >= P


def test_objective_sign_single_switch():
    # One reason covers everything. A -1 objective would also select the spare.
    inst = SCPRInstance([0, 1], ["a", "b"], [({0, 1}, {"a"}), ({0}, {"b"})])
    e, x = solve_qubo_exact(build_qubo(inst))
    assert build_qubo(inst).recover_reasons(x) == {"a"}
    assert e == pytest.approx(1.0)


def test_overlap_costs_nothing():
    # The optimum covers element 1 twice; an exact-cover penalty would reject it.
    inst = SCPRInstance([0, 1, 2], ["a", "b"], [({0, 1}, {"a"}), ({1, 2}, {"b"})])
    e, x = solve_qubo_exact(build_qubo(inst))
    assert e == pytest.approx(2.0)
    assert inst.check_coverage(build_qubo(inst).recover_reasons(x))


def test_feasible_energy_equals_cover_size():
    inst = benchmarks.reduction_demo()
    q = build_qubo(inst)
    _, S = brute_force_optimum(inst)
    e, _ = solve_qubo_exact(q)
    assert e == pytest.approx(len(S))


def test_rule2_keeps_one_duplicate():
    inst = SCPRInstance([0], ["a"], [({0}, {"a"}), ({0}, {"a"})])
    reduced, S_star, _ = beasley(inst)
    assert S_star == {"a"} or len(reduced.E) == 1


@pytest.mark.parametrize("make", benchmarks.ALL)
def test_benchmarks_verify(make):
    assert verify_instance(make(), verbose=False)


def test_random_self_test_small():
    c = self_test(n=300, seed=7, verbose=False)
    assert c["failures"] == 0 and c["oracle_mismatch"] == 0
    assert c["qubo_checked"] > 0

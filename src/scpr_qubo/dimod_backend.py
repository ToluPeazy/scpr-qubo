"""dimod solver path. The same QUBO goes to dimod's ExactSolver (exact) and the
reference SimulatedAnnealingSampler (heuristic). LeapHybridSampler and the QPU
take the same BinaryQuadraticModel; this repository does not call them."""

import numpy as np

from .oracles import brute_force_optimum
from .qubo import build_qubo, solve_qubo_exact


def to_bqm(qubo):
    import dimod
    bqm = dimod.BinaryQuadraticModel("BINARY")
    bqm.add_variables_from((i, 0.0) for i in range(qubo.n_vars))
    for (i, j), c in qubo.Q.items():
        if i == j:
            bqm.add_linear(i, c)
        else:
            bqm.add_quadratic(i, j, c)
    bqm.offset = qubo.offset
    return bqm


def solve_qubo_dimod_exact(qubo):
    import dimod
    s = dimod.ExactSolver().sample(to_bqm(qubo)).first
    x = np.array([int(s.sample[i]) for i in range(qubo.n_vars)])
    return float(s.energy), x


def solve_qubo_dimod_sa(qubo, num_reads=200):
    from dimod.reference.samplers import SimulatedAnnealingSampler
    s = SimulatedAnnealingSampler().sample(to_bqm(qubo), num_reads=num_reads).first
    x = np.array([int(s.sample[i]) for i in range(qubo.n_vars)])
    return float(s.energy), x


def cross_check_dimod(inst, P=None, sa_reads=200):
    """Energy agreement between the enumeration oracle, ExactSolver and SA.
    The QUBO can hold several optimal covers at equal energy, so the check is
    energy equality plus feasibility, not assignment identity."""
    q = build_qubo(inst, P)
    e_fs, x_fs = solve_qubo_exact(q)
    e_ex, _ = solve_qubo_dimod_exact(q)
    e_sa, x_sa = solve_qubo_dimod_sa(q, num_reads=sa_reads)
    bf, _ = brute_force_optimum(inst)
    return {
        "fs_energy": e_fs, "exact_energy": e_ex, "sa_energy": e_sa,
        "fs==exact": abs(e_fs - e_ex) < 1e-6,
        "sa_optimal": abs(e_sa - bf) < 1e-6 and inst.check_coverage(q.recover_reasons(x_sa)),
        "energy_is_opt": abs(e_fs - bf) < 1e-6 and inst.check_coverage(q.recover_reasons(x_fs)),
    }

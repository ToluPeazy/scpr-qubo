"""Exact reference solvers.

brute_force_optimum is the primary oracle. ilp_optimum is an independent second
oracle used only to cross-check brute force in the verification harness; it is
a test instrument and not part of the SCPR -> QUBO method.
"""

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def brute_force_optimum(inst):
    """Smallest S subset of R covering U, by enumeration in order of size."""
    for k in range(len(inst.R) + 1):
        for sub in itertools.combinations(inst.R, k):
            if inst.check_coverage(set(sub)):
                return k, set(sub)
    return None, None


def ilp_optimum(inst):
    """Exact optimum via integer LP: min sum y_k subject to per-element coverage
    >= 1, with z_i = AND of a multi-reason pair's reasons (McCormick)."""
    R, E, U = inst.R, inst.E, inst.U
    r = len(R)
    r2i = {rk: k for k, rk in enumerate(R)}

    pair_act = {}
    multi = []
    counter = r
    for i, (A, Ri) in enumerate(E):
        idxs = [r2i[x] for x in Ri if x in r2i]
        if len(idxs) == 0:
            pair_act[i] = None
        elif len(idxs) == 1:
            pair_act[i] = idxs[0]
        else:
            zi = counter
            counter += 1
            multi.append((i, idxs, zi))
            pair_act[i] = zi
    nv = counter

    c = np.zeros(nv)
    c[:r] = 1.0
    rows, lb, ub = [], [], []

    for (_, idxs, zi) in multi:
        for k in idxs:                                   # z - y_k <= 0
            row = np.zeros(nv); row[zi] = 1; row[k] = -1
            rows.append(row); lb.append(-np.inf); ub.append(0.0)
        row = np.zeros(nv)                               # sum y_k - z <= |R_i|-1
        for k in idxs:
            row[k] += 1
        row[zi] -= 1
        rows.append(row); lb.append(-np.inf); ub.append(len(idxs) - 1)

    for u in U:
        covering = [i for i, (A, Ri) in enumerate(E) if u in A]
        if any(pair_act[i] is None for i in covering):
            continue
        row = np.zeros(nv)
        has = False
        for i in covering:
            row[pair_act[i]] += 1
            has = True
        if not has:
            return None, None
        rows.append(-row); lb.append(-np.inf); ub.append(-1.0)

    cons = LinearConstraint(np.array(rows), np.array(lb), np.array(ub)) if rows else None
    res = milp(c, integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=cons)
    if not res.success:
        return None, None
    S = {R[k] for k in range(r) if res.x[k] > 0.5}
    return int(round(res.fun)), S

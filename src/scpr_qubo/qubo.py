"""SCPR -> QUBO encoding.

Objective:  +1 on each reason variable y_k (minimise |S|).
Activation: a pair with m >= 2 reasons uses m-1 chained pairwise AND gadgets,
            P(ab - 2az - 2bz + 3z), each zero exactly when z = a AND b.
Coverage:   at-least-one per element. An element reachable by t >= 2 pairs adds
            P(sum acts - 1 - s)^2 with s in ceil(log2 t) slack bits; an element
            reachable by one pair adds P(1 - act).
Penalty:    P = |R| + 1 by default, the smallest integer above the objective range.

Q is stored as an upper-triangular dict {(i, j): coeff} with i <= j; (i, i)
holds linear terms. offset holds the constant so that a feasible assignment
encoding cover S has energy exactly |S|.
"""

from collections import defaultdict

import numpy as np


class QUBO:
    def __init__(self, Q, offset, n_vars, n_reason, reason_list):
        self.Q = Q
        self.offset = offset
        self.n_vars = n_vars
        self.n_reason = n_reason
        self.reason_list = reason_list

    def energy(self, x):
        e = self.offset
        for (i, j), c in self.Q.items():
            e += c * x[i] * x[j]
        return e

    def recover_reasons(self, x):
        return {self.reason_list[k] for k in range(self.n_reason) if x[k] == 1}


def _addQ(Q, i, j, c):
    if c == 0:
        return
    if i > j:
        i, j = j, i
    Q[(i, j)] = Q.get((i, j), 0.0) + c


def build_qubo(inst, P=None):
    R, E, U = inst.R, inst.E, inst.U
    r = len(R)
    if P is None:
        P = r + 1
    r2i = {rk: k for k, rk in enumerate(R)}

    Q = {}
    offset = 0.0
    counter = r

    for k in range(r):
        _addQ(Q, k, k, 1.0)

    activation = {}
    for i, (A, Ri) in enumerate(E):
        idxs = [r2i[x] for x in Ri if x in r2i]
        if len(idxs) == 0:
            activation[i] = None
        elif len(idxs) == 1:
            activation[i] = idxs[0]
        else:
            prev = idxs[0]
            for t in range(1, len(idxs)):
                a, b = prev, idxs[t]
                z = counter
                counter += 1
                _addQ(Q, a, b, P)
                _addQ(Q, a, z, -2 * P)
                _addQ(Q, b, z, -2 * P)
                _addQ(Q, z, z, 3 * P)
                prev = z
            activation[i] = prev

    for u in U:
        covering = [i for i, (A, Ri) in enumerate(E) if u in A]
        if any(activation[i] is None for i in covering):
            continue
        acts = sorted({activation[i] for i in covering if activation[i] is not None})
        t = len(acts)
        if t == 0:
            offset += P * (r + 1)
            continue
        if t == 1:
            a = acts[0]
            _addQ(Q, a, a, -P)
            offset += P
        else:
            L = (t - 1).bit_length()
            slack = []
            for _ in range(L):
                slack.append(counter)
                counter += 1
            coeffs = {a: 1.0 for a in acts}
            for b in range(L):
                coeffs[slack[b]] = -(2.0 ** b)
            items = list(coeffs.items())
            offset += P
            for v, cv in items:
                _addQ(Q, v, v, P * (cv * cv - 2 * cv))
            for x in range(len(items)):
                for y in range(x + 1, len(items)):
                    (vi, ci), (vj, cj) = items[x], items[y]
                    _addQ(Q, vi, vj, P * 2 * ci * cj)

    return QUBO(Q, offset, counter, r, R)


def solve_qubo_exact(qubo, cap=22):
    """Exact minimum by vectorised enumeration of all 2^n assignments."""
    n = qubo.n_vars
    if n > cap:
        raise ValueError(f"n={n} exceeds enumeration cap {cap}")
    d = np.zeros(n)
    Qu = np.zeros((n, n))
    for (i, j), c in qubo.Q.items():
        if i == j:
            d[i] += c
        else:
            Qu[i, j] += c
    best_E = np.inf
    best_x = None
    total = 1 << n
    step = 1 << min(n, 18)
    bitpos = np.arange(n)
    for start in range(0, total, step):
        idx = np.arange(start, min(start + step, total), dtype=np.int64)
        X = ((idx[:, None] >> bitpos[None, :]) & 1).astype(np.float64)
        Ev = X @ d + ((X @ Qu) * X).sum(1)
        m = int(Ev.argmin())
        if Ev[m] < best_E:
            best_E = float(Ev[m])
            best_x = X[m].astype(int).copy()
    return best_E + qubo.offset, best_x


def fix_variables(qubo, fixed):
    """Substitute fixed[idx] = value into the QUBO. Couplings between two fixed
    variables go to the offset, so energies stay comparable."""
    newQ = defaultdict(float)
    off = qubo.offset
    for (i, j), c in qubo.Q.items():
        fi, fj = i in fixed, j in fixed
        if i == j:
            if fi:
                off += c * fixed[i]
            else:
                newQ[(i, i)] += c
        else:
            if fi and fj:
                off += c * fixed[i] * fixed[j]
            elif fi:
                newQ[(j, j)] += c * fixed[i]
            elif fj:
                newQ[(i, i)] += c * fixed[j]
            else:
                newQ[(i, j)] += c
    keep = sorted(set(range(qubo.n_vars)) - set(fixed))
    remap = {old: new for new, old in enumerate(keep)}
    Q2 = {}
    for (i, j), c in newQ.items():
        a, b = remap[i], remap[j]
        lo, hi = (a, b) if a <= b else (b, a)
        Q2[(lo, hi)] = Q2.get((lo, hi), 0.0) + c
    sub = QUBO(Q2, off, len(keep), 0, qubo.reason_list)
    return sub, keep, remap

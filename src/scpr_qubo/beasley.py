"""Beasley-style reduction for SCPR (Algorithm 2 of Babatunde, 2025).

Rule 1: an element covered by exactly one pair forces that pair's reasons.
Rule 2: (A_i, R_i) dominates (A_j, R_j) when A_i >= A_j and R_i <= R_j; the
dominated pair is removed. Duplicate pairs dominate each other, so Rule 2 keeps
the earliest member of each duplicate class.
"""

from .instance import SCPRInstance


def beasley(inst):
    """Return (reduced instance, forced reasons S*, rule counts)."""
    U = set(inst.U)
    R = set(inst.R)
    E = [(set(A), set(Ri)) for A, Ri in inst.E]
    S_star = set()
    stats = {"rule1": 0, "rule2": 0}

    changed = True
    while changed:
        changed = False

        for u in list(U):
            cov = [(i, A, Ri) for i, (A, Ri) in enumerate(E) if u in A]
            if len(cov) == 1:
                i, A, Ri = cov[0]
                if Ri - S_star:
                    S_star |= Ri
                    R -= Ri
                    U -= A
                    E.pop(i)
                    E = [(Aa - A, rr - Ri) for Aa, rr in E]
                    E = [(Aa, rr) for Aa, rr in E if Aa]
                    stats["rule1"] += 1
                    changed = True
                    break
        if changed:
            continue

        remove = set()
        for j, (Aj, Rj) in enumerate(E):
            for i, (Ai, Ri) in enumerate(E):
                if i == j or i in remove:
                    continue
                if Ai.issuperset(Aj) and Ri.issubset(Rj):
                    duplicate = (Ai == Aj and Ri == Rj)
                    if duplicate and i > j:
                        continue
                    remove.add(j)
                    break
        if remove:
            for j in sorted(remove, reverse=True):
                E.pop(j)
            stats["rule2"] += len(remove)
            changed = True

    reduced = SCPRInstance(sorted(U, key=str), sorted(R, key=str), E, inst.name + "'")
    return reduced, S_star, stats

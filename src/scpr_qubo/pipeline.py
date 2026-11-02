"""Five stages: (1) SCPR brute force, (2) Beasley + brute force, (3) QUBO with no
reduction, (4) Beasley before building the QUBO, (5) full QUBO then fixing the
Rule-1 forced reason variables."""

from .beasley import beasley
from .oracles import brute_force_optimum
from .qubo import build_qubo, fix_variables, solve_qubo_exact


def stage1(inst):
    size, S = brute_force_optimum(inst)
    return {"S": S, "size": size, "qvars": None, "energy": None}


def stage2(inst):
    reduced, S_star, _ = beasley(inst)
    if not reduced.U:
        S = set(S_star)
    else:
        _, Sp = brute_force_optimum(reduced)
        S = (Sp | S_star) if Sp is not None else None
    return {"S": S, "size": (len(S) if S is not None else None),
            "qvars": None, "energy": None}


def stage3(inst, P=None):
    q = build_qubo(inst, P)
    energy, x = solve_qubo_exact(q)
    S = q.recover_reasons(x)
    return {"S": S, "size": len(S), "qvars": q.n_vars, "energy": energy,
            "energy_target": len(S)}


def stage4(inst, P=None):
    reduced, S_star, _ = beasley(inst)
    if not reduced.U:
        return {"S": set(S_star), "size": len(S_star), "qvars": 0,
                "energy": None, "energy_target": None, "n_forced": len(S_star)}
    q = build_qubo(reduced, P)
    energy, x = solve_qubo_exact(q)
    Sp = q.recover_reasons(x)
    # The reduced QUBO's energy equals |S'|, the reduced-subproblem objective.
    return {"S": Sp | S_star, "size": len(Sp | S_star), "qvars": q.n_vars,
            "energy": energy, "energy_target": len(Sp), "n_forced": len(S_star)}


def stage5(inst, P=None):
    _, S_star, _ = beasley(inst)
    q = build_qubo(inst, P)
    r2i = {rk: k for k, rk in enumerate(inst.R)}
    fixed = {r2i[lbl]: 1 for lbl in S_star if lbl in r2i}
    if fixed:
        sub, keep, remap = fix_variables(q, fixed)
        energy, x = solve_qubo_exact(sub)
        remaining = {inst.R[o] for o in keep if o < q.n_reason and x[remap[o]] == 1}
        S = set(S_star) | remaining
        solved_vars = sub.n_vars
    else:
        energy, x = solve_qubo_exact(q)
        S = q.recover_reasons(x)
        solved_vars = q.n_vars
    return {"S": S, "size": len(S), "qvars": solved_vars, "qvars_full": q.n_vars,
            "energy": energy, "energy_target": len(S), "n_fixed": len(fixed)}

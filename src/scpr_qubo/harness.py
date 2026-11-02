"""Verification harness.

Checks every stage against two independent exact oracles (brute force and an
ILP), cross-checks the enumeration oracle against dimod, then runs the pipeline
on random instances and counts discrepancies.

Run:  python -m scpr_qubo.harness [--n 4000] [--seed 1]
"""

import argparse
import platform
import random

import numpy as np

from . import benchmarks
from .dimod_backend import cross_check_dimod
from .instance import SCPRInstance
from .oracles import brute_force_optimum, ilp_optimum
from .pipeline import stage1, stage2, stage3, stage4, stage5
from .qubo import build_qubo

STAGES = (("1 Brute-force", stage1), ("2 Beasley+BF", stage2),
          ("3 QUBO no-red", stage3), ("4 QUBO pre-Q", stage4),
          ("5 QUBO post-Q", stage5))


def _stage_ok(inst, res, bf):
    S = res["S"]
    if S is None or not inst.check_coverage(S):
        return "coverage"
    if res["size"] != bf:
        return "size"
    if res.get("energy") is not None and res.get("energy_target") is not None:
        if abs(res["energy"] - res["energy_target"]) > 1e-6:
            return "energy"
    return None


def verify_instance(inst, P=None, verbose=True):
    bf, _ = brute_force_optimum(inst)
    ilp, _ = ilp_optimum(inst)
    all_ok = bf == ilp
    if verbose:
        print(f"\n  {inst}   optimum |S| = {bf}   (brute force {bf}, ILP {ilp}: "
              f"{'agree' if bf == ilp else 'MISMATCH'})")
        print(f"  {'Stage':<16}{'|S|':>5}{'covers':>8}{'optimal':>9}{'QUBO vars':>11}{'energy':>9}")
    for name, fn in STAGES:
        res = fn(inst) if name.startswith(("1", "2")) else fn(inst, P)
        err = _stage_ok(inst, res, bf)
        all_ok &= err is None
        if verbose:
            covers = res["S"] is not None and inst.check_coverage(res["S"])
            en = "--" if res["energy"] is None else f"{res['energy']:.3f}"
            qv = "--" if res["qvars"] is None else str(res["qvars"])
            print(f"  {name:<16}{str(res['size']):>5}{('yes' if covers else 'NO'):>8}"
                  f"{('yes' if res['size'] == bf else 'NO'):>9}{qv:>11}{en:>9}"
                  f"{'' if err is None else '   FAIL: ' + err}")
    return all_ok


def random_instance(rng, max_u=5, max_r=5, max_pairs=7):
    nu = rng.randint(2, max_u)
    nr = rng.randint(2, max_r)
    U = list(range(nu))
    R = [f"r{i}" for i in range(nr)]
    E = []
    for _ in range(rng.randint(2, max_pairs)):
        A = set(rng.sample(U, rng.randint(1, nu)))
        Ri = set(rng.sample(R, rng.randint(1, min(4, nr))))
        E.append((A, Ri))
    inst = SCPRInstance(U, R, E, "rand")
    return inst if inst.check_coverage(set(R)) else None


def self_test(n=1000, seed=0, qvar_cap=20, verbose=True):
    """Return a dict of counts over n random feasible instances."""
    rng = random.Random(seed)
    c = {"tested": 0, "failures": 0, "oracle_mismatch": 0,
         "qubo_checked": 0, "qubo_skipped": 0, "duplicate_pairs": 0}
    draws = 0
    while c["tested"] < n and draws < n * 40:
        draws += 1
        inst = random_instance(rng)
        if inst is None:
            continue
        keys = [(frozenset(A), frozenset(Ri)) for A, Ri in inst.E]
        if len(keys) != len(set(keys)):
            c["duplicate_pairs"] += 1
        bf, _ = brute_force_optimum(inst)
        ilp, _ = ilp_optimum(inst)
        c["tested"] += 1
        if bf != ilp:
            c["oracle_mismatch"] += 1
            c["failures"] += 1
            continue
        reasons = []
        b = _stage_ok(inst, stage2(inst), bf)
        if b:
            reasons.append(f"S2:{b}")
        if build_qubo(inst).n_vars > qvar_cap:
            c["qubo_skipped"] += 1
        else:
            c["qubo_checked"] += 1
            for label, st in (("3", stage3), ("4", stage4), ("5", stage5)):
                b = _stage_ok(inst, st(inst), bf)
                if b:
                    reasons.append(f"S{label}:{b}")
        if reasons:
            c["failures"] += 1
            if verbose:
                print(f"  FAIL {reasons} BF={bf} E={[(sorted(a), sorted(r)) for a, r in inst.E]}")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4000, help="random instances")
    ap.add_argument("--seed", type=int, default=1, help="random-instance seed")
    ap.add_argument("--sa-seed", type=int, default=0, help="seed for simulated annealing")
    args = ap.parse_args(argv)

    import dimod
    import scipy
    print("SCPR -> QUBO verification harness")
    print(f"python {platform.python_version()}  numpy {np.__version__}  "
          f"scipy {scipy.__version__}  dimod {dimod.__version__}")
    print(f"random instances n={args.n} seed={args.seed}  SA seed={args.sa_seed}")

    print("\nBENCHMARK INSTANCES")
    overall = True
    for make in benchmarks.ALL:
        overall &= verify_instance(make())

    print("\nDIMOD CROSS-CHECK (enumeration oracle vs ExactSolver vs SA, 200 reads)")
    random.seed(args.sa_seed)
    np.random.seed(args.sa_seed)
    print(f"  {'Instance':<18}{'enum':>8}{'Exact':>8}{'SA':>8}{'enum=Exact':>12}{'SA optimal':>12}")
    for make in benchmarks.ALL:
        inst = make()
        r = cross_check_dimod(inst)
        overall &= r["fs==exact"] and r["sa_optimal"] and r["energy_is_opt"]
        print(f"  {inst.name:<18}{r['fs_energy']:>8.3f}{r['exact_energy']:>8.3f}{r['sa_energy']:>8.3f}"
              f"{('yes' if r['fs==exact'] else 'NO'):>12}{('yes' if r['sa_optimal'] else 'NO'):>12}")

    print("\nRANDOMISED SELF-TEST")
    c = self_test(n=args.n, seed=args.seed)
    print(f"  instances tested                    : {c['tested']}")
    print(f"  brute force vs ILP mismatches       : {c['oracle_mismatch']}")
    print(f"  instances with QUBO stages checked  : {c['qubo_checked']}")
    print(f"  instances with QUBO stages skipped  : {c['qubo_skipped']}  (more than 20 QUBO variables)")
    print(f"  instances with duplicate pairs      : {c['duplicate_pairs']}")
    print(f"  stage failures                      : {c['failures']}")
    ok = overall and c["failures"] == 0
    print(f"\nVERDICT: {'ALL VERIFIED' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

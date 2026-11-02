"""Run the SCPR QUBOs on an annealer and measure how often it finds the optimum.

Each benchmark yields up to three QUBO variants: the full encoding (Stage 3),
the encoding of the Beasley-reduced instance (Stage 4, pre-Q) and the full
encoding with the Rule 1 reasons fixed to 1 (Stage 5, post-Q). A variant
identical to the full encoding is skipped, and so is pre-Q when Beasley
resolves the instance outright.

Every variant is minor-embedded once, and the same embedding serves each chain
strength in the sweep, so differences between rows come from chain strength
alone. Per row the script records success probability (reads at the exact
minimum energy that decode to a valid cover), the fraction of reads that decode
to any valid cover, the mean chain-break fraction and, on the QPU, the QPU
access time.

Samplers:
  mock  Ocean's MockDWaveSampler on a Pegasus graph. Tests the embedding and
        bookkeeping offline. It returns local minima, not a QPU distribution,
        so its success rates say nothing about hardware.
  sa    Simulated annealing on the logical QUBO, no embedding. Classical
        baseline.
  qpu   DWaveSampler. Reads the API token from the Ocean config file or the
        DWAVE_API_TOKEN environment variable; never put a token in this file.
        Refuses to run without --yes.

Examples:
  python -m scpr_qubo.qpu_experiment --sampler mock --dry-run
  python -m scpr_qubo.qpu_experiment --sampler mock --reads 200
  python -m scpr_qubo.qpu_experiment --sampler qpu --reads 1000 --yes
"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import benchmarks
from .beasley import beasley
from .dimod_backend import to_bqm
from .qubo import QUBO, build_qubo, fix_variables, solve_qubo_exact

DEFAULT_SCALES = ["default", 0.25, 0.5, 1.0, 2.0]
TOL = 1e-6


@dataclass
class Problem:
    instance: str
    variant: str
    qubo: QUBO
    decode: Callable[[dict], set]
    target_energy: float
    optimum_size: int
    check_coverage: Callable[[set], bool]
    bqm: object = field(init=False)

    def __post_init__(self):
        self.bqm = to_bqm(self.qubo)

    @property
    def n_vars(self):
        return self.qubo.n_vars


def _all_instances():
    return {i.name: i for i in (benchmarks.small_4x4(), benchmarks.reduction_demo(),
                                benchmarks.thesis_1_5_2())}


def build_problems(instances, P=None):
    """Return the QUBO variants worth sending, each with a decoder that maps a
    sample back to a cover of the original instance."""
    problems = []
    for inst in instances:
        full = build_qubo(inst, P)
        e_full, x_full = solve_qubo_exact(full)
        opt = len(full.recover_reasons(x_full))

        def dec_full(s, q=full):
            return q.recover_reasons([s[i] for i in range(q.n_vars)])
        problems.append(Problem(inst.name, "full", full, dec_full, round(e_full, 9),
                                opt, inst.check_coverage))

        reduced, S_star, stats = beasley(inst)
        if (stats["rule1"] or stats["rule2"]) and reduced.U:
            pre = build_qubo(reduced, P)
            e_pre, _ = solve_qubo_exact(pre)

            def dec_pre(s, q=pre, forced=frozenset(S_star)):
                return q.recover_reasons([s[i] for i in range(q.n_vars)]) | forced
            problems.append(Problem(inst.name, "pre-q", pre, dec_pre, round(e_pre, 9),
                                    opt, inst.check_coverage))

        r2i = {rk: k for k, rk in enumerate(inst.R)}
        fixed = {r2i[lbl]: 1 for lbl in S_star if lbl in r2i}
        if fixed:
            sub, keep, remap = fix_variables(full, fixed)
            e_post, _ = solve_qubo_exact(sub)

            def dec_post(s, keep=keep, remap=remap, nr=full.n_reason,
                         R=list(inst.R), forced=frozenset(S_star)):
                chosen = {R[o] for o in keep if o < nr and s[remap[o]] == 1}
                return chosen | forced
            problems.append(Problem(inst.name, "post-q", sub, dec_post, round(e_post, 9),
                                    opt, inst.check_coverage))
    return problems


def _max_abs_bias(bqm):
    biases = [abs(v) for v in bqm.linear.values()] + [abs(v) for v in bqm.quadratic.values()]
    return max(biases) if biases else 1.0


def chain_strength_values(bqm, scales):
    """Map each scale to an absolute chain strength. "default" defers to Ocean's
    uniform torque compensation; a number multiplies the largest |bias|."""
    big = _max_abs_bias(bqm)
    out = []
    for s in scales:
        if s == "default":
            out.append(("default", None))
        else:
            out.append((float(s), float(s) * big))
    return out


def summarise(problem, sampleset):
    """Reduce a logical SampleSet to one result row."""
    rec = sampleset.record
    labels = list(sampleset.variables)
    occ = rec.num_occurrences
    reads = int(occ.sum())
    ok = feasible = 0
    best_E = math.inf
    best_size = None
    for row, n in zip(rec.sample, occ):
        s = dict(zip(labels, (int(v) for v in row)))
        E = problem.bqm.energy(s)
        S = problem.decode(s)
        valid = problem.check_coverage(S)
        best_E = min(best_E, E)
        if valid:
            feasible += int(n)
            best_size = len(S) if best_size is None else min(best_size, len(S))
            if abs(E - problem.target_energy) < TOL:
                ok += int(n)
    cbf = None
    if "chain_break_fraction" in rec.dtype.names:
        cbf = float((rec.chain_break_fraction * occ).sum() / reads)
    return {
        "instance": problem.instance, "variant": problem.variant,
        "logical_vars": problem.n_vars, "num_reads": reads,
        "success_prob": ok / reads, "feasible_frac": feasible / reads,
        "mean_chain_break_frac": cbf, "best_energy": best_E,
        "target_energy": problem.target_energy,
        "best_feasible_size": best_size, "optimum_size": problem.optimum_size,
    }


# ── samplers and embedding ──────────────────────────────────────────────────

def _mock_sampler(seed=None):
    from dwave.system.testing import MockDWaveSampler
    return MockDWaveSampler(topology_type="pegasus", topology_shape=[16])


def _qpu_sampler(solver=None):
    from dwave.system import DWaveSampler
    return DWaveSampler(solver=solver) if solver else DWaveSampler()


def embed(bqm, child, seed=None):
    """Find one minor embedding of bqm on the child's working graph. Variables
    with no couplings get a free qubit of their own."""
    import minorminer
    edges = list(bqm.quadratic)
    emb = minorminer.find_embedding(edges, child.edgelist, random_seed=seed) if edges else {}
    if edges and not emb:
        raise RuntimeError("minorminer found no embedding")
    used = {q for chain in emb.values() for q in chain}
    free = (q for q in child.nodelist if q not in used)
    for v in bqm.variables:
        if v not in emb:
            emb[v] = [next(free)]
    return {v: list(c) for v, c in emb.items()}


def _write(rows, out, meta):
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sampler", "solver", "instance", "variant", "logical_vars", "physical_qubits",
            "max_chain_length", "chain_scale", "chain_strength", "num_reads", "success_prob",
            "feasible_frac", "mean_chain_break_frac", "best_energy", "target_energy",
            "best_feasible_size", "optimum_size", "qpu_access_time_us"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def _plan(problems, scales, reads, sampler):
    n_chain = 1 if sampler == "sa" else len(scales)
    subs = len(problems) * n_chain
    print(f"{'instance':<16}{'variant':<9}{'vars':>5}{'target E':>10}")
    for p in problems:
        print(f"{p.instance:<16}{p.variant:<9}{p.n_vars:>5}{p.target_energy:>10.3f}")
    print(f"\n{subs} submissions x {reads} reads = {subs * reads} reads in total")
    return subs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sampler", choices=["mock", "sa", "qpu"], default="mock")
    ap.add_argument("--solver", help="QPU solver name; default lets Ocean choose")
    ap.add_argument("--instances", nargs="+", help="benchmark names; default all three")
    ap.add_argument("--reads", type=int, default=100)
    ap.add_argument("--chain-scales", nargs="+", default=DEFAULT_SCALES,
                    help='"default" and/or multiples of the largest |bias|')
    ap.add_argument("--annealing-time", type=float, help="microseconds, QPU only")
    ap.add_argument("--penalty", type=float, help="P; default |R| + 1 per instance")
    ap.add_argument("--seed", type=int, default=None, help="embedding seed")
    ap.add_argument("--out", type=Path, help="CSV path; a JSON with embeddings goes alongside")
    ap.add_argument("--dry-run", action="store_true", help="embed and plan, do not sample")
    ap.add_argument("--yes", action="store_true", help="confirm spending QPU time")
    a = ap.parse_args(argv)

    scales = [s if s == "default" else float(s) for s in a.chain_scales]
    pool = _all_instances()
    names = a.instances or list(pool)
    problems = build_problems([pool[n] for n in names], a.penalty)
    _plan(problems, scales, a.reads, a.sampler)

    if a.sampler == "qpu" and not a.yes and not a.dry_run:
        print("\nThis would spend QPU time. Re-run with --yes to submit, "
              "or --dry-run to embed without sampling.")
        return 2

    if a.sampler == "sa":
        from dimod.reference.samplers import SimulatedAnnealingSampler
        sampler, child = SimulatedAnnealingSampler(), None
    else:
        if a.sampler == "qpu" and not a.dry_run:
            child = _qpu_sampler(a.solver)
        else:
            child = _mock_sampler(a.seed)  # dry-run plans against Pegasus P16
        from dwave.system import FixedEmbeddingComposite

    rows, meta = [], {"sampler": a.sampler, "reads": a.reads, "seed": a.seed,
                      "penalty": a.penalty, "embeddings": {}}
    if child is not None:
        meta["solver"] = child.properties.get("chip_id", "mock")
    spent_us = 0.0
    for p in problems:
        key = f"{p.instance}/{p.variant}"
        if a.sampler == "sa":
            ss = sampler.sample(p.bqm, num_reads=a.reads)
            r = summarise(p, ss)
            r.update(sampler="sa", chain_scale="n/a")
            rows.append(r)
            continue
        emb = embed(p.bqm, child, a.seed)
        phys = sum(len(c) for c in emb.values())
        longest = max(len(c) for c in emb.values())
        meta["embeddings"][key] = emb
        print(f"{key:<24} {p.n_vars:>3} logical -> {phys:>4} qubits, longest chain {longest}")
        if a.dry_run:
            continue
        composite = FixedEmbeddingComposite(child, emb)
        for scale, cs in chain_strength_values(p.bqm, scales):
            kw = {"num_reads": a.reads, "label": f"scpr-qubo {key} cs={scale}",
                  "return_embedding": True}
            if cs is not None:
                kw["chain_strength"] = cs
            if a.annealing_time and a.sampler == "qpu":
                kw["annealing_time"] = a.annealing_time
            ss = composite.sample(p.bqm, **kw)
            r = summarise(p, ss)
            # The mock reports simulated timings; only a real QPU run spends time.
            t = ss.info.get("timing", {}).get("qpu_access_time") if a.sampler == "qpu" else None
            if t is not None:
                spent_us += t
            used_cs = cs if cs is not None else ss.info.get("embedding_context", {}).get("chain_strength")
            r.update(sampler=a.sampler, solver=meta["solver"], physical_qubits=phys,
                     max_chain_length=longest, chain_scale=scale, chain_strength=used_cs,
                     qpu_access_time_us=t)
            rows.append(r)
            print(f"   cs {str(scale):>7}: success {r['success_prob']:.3f}  "
                  f"valid {r['feasible_frac']:.3f}  breaks {r['mean_chain_break_frac']:.3f}"
                  + (f"  QPU so far {spent_us / 1e6:.3f} s" if t is not None else ""))

    if a.dry_run:
        return 0
    out = a.out or Path("results") / f"qpu_{a.sampler}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    meta["qpu_access_time_total_us"] = spent_us
    _write(rows, out, meta)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

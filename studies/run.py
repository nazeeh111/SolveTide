"""Reproducible chronological development and untouched confirmation studies."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import platform
from types import SimpleNamespace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from solvetide import Runtime, SolverConfig, assemble
from solvetide.policy import ConstantPlan, RankPolicy, ACTIONS, fit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "studies/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trajectories(split: str):
    protocol = PROTOCOL[split]
    for family in PROTOCOL["families"]:
        for i, seed in enumerate(protocol["seeds"]):
            rng = np.random.default_rng(seed)
            n = protocol["meshes"][i % len(protocol["meshes"])]
            boundary = protocol["boundaries"][i % len(protocol["boundaries"])]
            base = {"material": float(rng.uniform(.8, 1.5)), "source": 1.0,
                    "boundary_value": 0.0, "contrast": float(rng.uniform(1.1, 2.5))}
            cases = []
            for step in range(protocol["steps"]):
                p = dict(base)
                p["source"] *= 1 + .03 * step
                p["boundary_value"] = .002 * step
                p["contrast"] *= 1 + .04 * np.sin(step // 2)
                kind = "small_rhs" if step % 2 else "small_local_coefficient"
                if step == 4:
                    p["material"] *= 4.0
                    kind = "abrupt_global"
                if step == 6:
                    p["contrast"] = 1e4 if split == "confirmation" else 20.0
                    kind = "high_contrast"
                if step == 7:
                    p["contrast"] = .03 if split == "confirmation" else .5
                    kind = "contrast_reversal"
                cases.append({"family": family, "n": n + int(step == 9),
                              "boundary": "left" if step == 8 else boundary,
                              "parameters": p, "kind": "mapping_invalidation" if step >= 8 else kind})
            yield f"{split}-{family}-{seed}", cases


def run_sequence(cases, strategy, policy=None, *, action=None):
    sequence_start=perf_counter()
    reference_seconds=0.0
    config = SolverConfig(strategy='fixed' if strategy == 'separate' else 'learned' if strategy == 'balanced' else strategy, rtol=PROTOCOL["relative_tolerance"],
                          atol=PROTOCOL["absolute_tolerance"], memory_budget=PROTOCOL["memory_budget"],
                          maxiter=PROTOCOL["maxiter"], max_seconds=15)
    separate = strategy == 'separate'
    balanced = strategy == 'balanced'
    if separate:
        config = replace(config,strategy='fixed',memory_budget=6)
    elif balanced:
        config = replace(config,strategy='learned')
        action = 'balanced'
    runtime = Runtime(config, policy=ConstantPlan(action) if action is not None else policy)
    adjoint_runtime = Runtime(config) if separate else None
    rows = []
    setup_seconds=perf_counter()-sequence_start
    total_start=sequence_start
    for step, case in enumerate(cases):
        start = perf_counter()
        system = assemble(**{k: v for k, v in case.items() if k != "kind"})
        target = .12 if case["family"] == "heat" else .04
        forward = runtime.solve(system)
        if separate and forward.accepted:
            value, dj = system.objective(forward.x,target)
            adj = adjoint_runtime.solve(system,rhs=dj,role='adjoint')
            gradients = {k:float(adj.x @ (db-da@forward.x)) for k,(da,db) in system.derivatives.items()} if adj.accepted else None
            gradient = SimpleNamespace(status=adj.status if adj.accepted else 'adjoint_failed',objective=value if adj.accepted else None,gradient=gradients,adjoint=adj)
        else:
            gradient = runtime.gradient(system, forward, target=target)
        seconds = perf_counter() - start
        reference=None
        oracle_start=perf_counter()
        if strategy not in ['learned'] or action is None:
            xr=spsolve(system.A,system.b)
            jr,djr=system.objective(xr,target)
            lr=spsolve(system.A.T,djr)
            gr={k:float(lr@(db-da@xr)) for k,(da,db) in system.derivatives.items()}
            reference={'forward_relative_l2':float(np.linalg.norm(forward.x-xr)/max(np.linalg.norm(xr),1e-30)) if forward.accepted else None,
                       'objective_absolute_error':abs(gradient.objective-jr) if gradient.objective is not None else None,
                       'gradient_relative_l2':float(np.linalg.norm([gradient.gradient[k]-gr[k] for k in gr])/max(np.linalg.norm(list(gr.values())),1e-30)) if gradient.gradient else None}
        reference_seconds+=perf_counter()-oracle_start
        if step==0: seconds+=setup_seconds
        results = [forward] + ([gradient.adjoint] if gradient.adjoint else [])
        rows.append({"step": step, "kind": case["kind"], "n": system.n,
                     "seconds": seconds, "assembly_seconds": system.assembly_seconds,
                     "status": gradient.status, "objective": gradient.objective,
                     "gradient": gradient.gradient, "reference_errors": reference,
                     "solves": [r.summary() for r in results],
                     "retained_storage_bytes": runtime.state()["memory_bytes"] + (adjoint_runtime.state()["memory_bytes"] if separate else 0)})
    return {"seconds": perf_counter()-total_start-reference_seconds, "reference_check_seconds":reference_seconds, "setup_seconds":setup_seconds, "wall_including_reference_checks": perf_counter()-total_start, "steps": rows,
            "failures": sum(r["status"] != "converged" for r in rows), "state": runtime.state()}


def train():
    source_hashes={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in ["solvetide/runtime.py","solvetide/assembly.py","solvetide/policy.py","studies/run.py"]}
    records = []
    validation = defaultdict(list)
    specs = list(trajectories("development"))
    # The last entire development trajectory of each family is validation only.
    train_specs = [(name, cases) for name, cases in specs if not name.endswith("104")]
    val_specs = [(name, cases) for name, cases in specs if name.endswith("104")]
    started = perf_counter()
    for repeat in range(PROTOCOL["development"]["repeats"]):
        for name, cases in train_specs:
            for action in np.random.default_rng(700 + repeat).permutation(ACTIONS):
                data = run_sequence(cases, "learned", action=str(action))
                for step in data["steps"]:
                    records.append({"trajectory": name, "step": step["step"], "repeat": repeat,
                                    "action": str(action), "features": step["solves"][0]["policy_features"],
                                    "seconds": step["seconds"], "status": step["status"],
                                    "actual_actions": [r.get("policy_action") for r in step["solves"]],
                                    "used_ranks": [r["reused_rank"] for r in step["solves"]]})
    document = fit(records, {"split": "development-training-only", "trajectory_names": [n for n, _ in train_specs],
                             "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                             "source_sha256":source_hashes,"records": len(records), "data_origin": "original synthetic finite-element solves", "target": "complete forward-adjoint pair incl assembly, all overhead and failure penalty"})
    policy = RankPolicy(document, force=True)
    for name, cases in val_specs:
        for rep in range(3):
            methods=['warm','joint','learned']
            np.random.default_rng(601+rep).shuffle(methods)
            for method in methods:
                data = run_sequence(cases, method, policy)
                validation[method].append({"trajectory": name, "seconds": data["seconds"], "failures": data["failures"]})
    medians = {k: float(np.median([v["seconds"] for v in values])) for k, values in validation.items()}
    family_medians={family:{method:float(np.median([v['seconds'] for v in values if family in v['trajectory']])) for method,values in validation.items()} for family in PROTOCOL['families']}
    document["enabled"] = bool(all(v["learned"] < .95 * min(v["warm"],v["joint"]) for v in family_medians.values()) and
                                not any(v["failures"] for v in validation["learned"]))
    document["development_validation"] = {"median_seconds": medians, "family_median_seconds": family_medians, "runs": dict(validation),
                                            "gate": "at least 5% faster than warm and joint, zero failures"}
    document["training_seconds"] = perf_counter() - started
    write(ROOT / "experiments/training.json", {"schema_version": 1, "records": records, "validation": dict(validation)})
    write(ROOT / "solvetide/data/policy.json", document)
    print(json.dumps({"training_records": len(records), "enabled": document["enabled"], "validation": medians}), flush=True)


def confirm():
    policy_path = ROOT / "solvetide/data/policy.json"
    policy_hash = sha256(policy_path.read_bytes()).hexdigest()
    policy_document=json.loads(policy_path.read_text())
    if policy_document["training"]["protocol_sha256"]!=sha256(PROTOCOL_PATH.read_bytes()).hexdigest():raise ValueError("protocol changed after training")
    source_hashes={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in ["solvetide/runtime.py","solvetide/assembly.py","solvetide/policy.py","studies/run.py"]}
    # Force evaluates the learned proposal even if development disabled deployment.
    # This flag does not change the shipped gate or train on confirmation data.
    policy = RankPolicy.load(policy_path, force=True)
    runs = []
    for repeat in range(PROTOCOL["confirmation"]["repeats"]):
        methods = list(PROTOCOL["strategies"])
        np.random.default_rng(5000 + repeat).shuffle(methods)
        for name, cases in trajectories("confirmation"):
            for method in methods:
                data = run_sequence(cases, method, policy)
                runs.append({"trajectory": name, "family": cases[0]["family"], "strategy": method,
                             "repeat": repeat, **data})
            print(f"confirmation {repeat + 1}: {name}", flush=True)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["family"], r["strategy"])].append(r)
    summary = []
    for (family, method), values in groups.items():
        solves = [s for v in values for step in v["steps"] for s in step["solves"]]
        residuals = [s["relative_residual"] for s in solves if s["status"] == "converged"]
        summary.append({"family": family, "strategy": method, "runs": len(values),
                        "median_sequence_seconds": float(np.median([v["seconds"] for v in values])),
                        "total_seconds": sum(v["seconds"] for v in values),
                        "failures": sum(v["failures"] for v in values),
                        "iterations": sum(s["iterations"] for s in solves),
                        "max_retained_storage_bytes": max(step["retained_storage_bytes"] for v in values for step in v["steps"]),
                        "max_relative_residual_accepted": max(residuals, default=None),
                        "max_backward_error_accepted": max((s["backward_error"] for s in solves if s["status"] == "converged"), default=None)})
    assert sha256(policy_path.read_bytes()).hexdigest() == policy_hash
    import scipy, skfem
    write(ROOT / "experiments/confirmation.json", {"schema_version": 1, "protocol": PROTOCOL,
          "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()).hexdigest(), "policy_sha256": policy_hash,"source_sha256":source_hashes,
          "environment": {"python": platform.python_version(), "platform": platform.platform(),
                          "numpy": np.__version__, "scipy": scipy.__version__, "scikit_fem": skfem.__version__},
          "summary": summary, "runs": runs,
          "limits": "Exploratory local CPU wall times; three repeats, finite trajectories; rank budget equal for recycling methods, AMG and LU account own storage. No unexecuted systems ranked."})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "confirm"])
    args = parser.parse_args()
    train() if args.stage == "train" else confirm()

if __name__ == "__main__":
    main()

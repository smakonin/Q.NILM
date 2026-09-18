#!/usr/bin/env python3
"""Independent audit of every training call and matched-loss result.

Does not import the new loss module or runner. Exact ideal amplitudes and
all statistics are reconstructed independently from the two load models.
"""
import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import DensityMatrix, Statevector

ROOT = Path(__file__).resolve().parents[1]
POWERS = np.array([[a,b] for b in (0.,120.,280.) for a in (0.,400.,900.)])
INDICES = np.array([2**a+2**(3+b) for b in range(3) for a in range(3)])


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle: return json.load(handle)
    return json.loads(path.read_text())


def costs_scale(y):
    e = (y-POWERS.sum(axis=1))**2
    levels = (0.,400.,900.,0.,120.,280.)
    scale = max(1., sum(abs(x*x-2*y*x) for x in levels)
                + sum(abs(2*a*b) for a in (0.,400.,900.) for b in (0.,120.,280.)))
    return e, scale


def reference(y, angles):
    e, scale = costs_scale(y)
    v = np.exp(-1j*angles[0]*e/scale)/3
    c, s = np.cos(angles[1]), -1j*np.sin(angles[1])
    for stride in (1,3):
        for category in (0,1):
            for i in range(9):
                if i//stride % 3 == category:
                    left, right = v[i], v[i+stride]
                    v[i], v[i+stride] = c*left+s*right, s*left+c*right
    p = abs(v)**2
    return p/p.sum()


def loss(p, e, name, raw=None):
    if raw is not None:
        x = e[np.asarray(raw)]
        if name == "mean": return float(sum(x)/len(x))
        if name.startswith("cvar"):
            count = 64 if name == "cvar25" else 128
            return float(sum(sorted(x)[:count])/count)
        k = 4 if name == "best4" else 16
        return float(sum(min(x[i:i+k]) for i in range(0,256,k))/(256//k))
    if name == "mean": return float(sum(p*e))
    if name.startswith("cvar"):
        alpha = .25 if name == "cvar25" else .5
        remaining, total = alpha, 0.
        for i in np.argsort(e):
            mass = min(remaining, p[i]); total += mass*e[i]; remaining -= mass
            if remaining <= 0: break
        return float(total/alpha)
    k = 4 if name == "best4" else 16
    return float(sum(value*(sum(p[e>=value])**k-sum(p[e>value])**k) for value in np.unique(e)))


def check_metrics(y, true_powers, p, stored):
    e, _ = costs_scale(y)
    valid = float(np.clip(sum(p),0,1))
    optimum = float(sum(p[e==min(e)]))
    require(abs(stored["raw_valid_probability"]-valid)<1e-9, "Validity")
    require(abs(stored["raw_optimum_probability"]-optimum)<1e-9, "Optimal mass")
    mae = np.mean(abs(POWERS-np.array(true_powers)),axis=1)
    for k in (4,16,256):
        win = np.zeros(9)
        for i, value in enumerate(e):
            better = (e<value) | ((e==value)&(np.arange(9)<i))
            before = float(sum(p[better]))
            win[i] = np.clip(1-before,0,1)**k-np.clip(1-before-p[i],0,1)**k
        no_valid = max(0.,1-valid)**k
        require(abs(sum(win)-(1-no_valid))<1e-8, "Raw batch accounting")
        target = stored["budgets"][str(k)]
        expected = {"no_valid_probability": no_valid,
                    "optimum_hit_probability": 1-max(0.,1-optimum)**k,
                    "conditional_best_cost_w2": float(win@e/sum(win)) if sum(win)>1e-15 else None,
                    "conditional_selected_macro_mae_w": float(win@mae/sum(win)) if sum(win)>1e-15 else None}
        for key, value in expected.items():
            if value is None: require(target[key] is None, "Conditional empty metric")
            else: require(abs(target[key]-value)<(1e-6 if "cost" in key or "mae" in key else 1e-8), f"Metric mismatch: {key}")
    if valid:
        require(abs(stored["conditional_mean_cost_w2"]-p@e/valid)<1e-6, "Mean cost")


def load_circuit(path):
    with path.open("rb") as handle: return qpy.load(handle)[0]


def permute(p, mapping):
    out = np.zeros(len(p))
    for i, prob in enumerate(p):
        classical = sum(((i>>q)&1)<<bit for bit,q in enumerate(mapping))
        out[classical] += prob
    return out


def audit(folder):
    output = folder/"independent_audit.json"
    require(not output.exists(), "Do not overwrite audit")
    plan, result = read(folder/"plan.json"), read(folder/"summary.json")
    binding = sha(folder/"plan.json")
    require(result["plan_sha256"] == binding, "Plan binding")
    for group in ("source_sha256", "input_sha256"):
        for name, expected in plan[group].items(): require(sha(ROOT/name)==expected, "Frozen input/source changed")
    require(result["training_manifest_sha256"]==sha(folder/"training_manifest.json"), "Training manifest")
    require(read(folder/"training_manifest.json")["fits"]==result["fits"], "Fit summary")
    require(result["ideal_evaluation_sha256"]==sha(folder/"ideal_evaluation.json"), "Ideal file")
    require(read(folder/"ideal_evaluation.json")["rows"]==result["ideal_rows"], "Ideal summary")
    require(result["hardware_jobs"]==result["qpu_seconds"]==0, "Offline scope")
    expected_ids = {f"{config}_trial{trial}_{name}" for config in plan["configs"] for trial in range(1,6) for name in plan["losses"]}
    require(len(result["fits"])==75 and {f["id"] for f in result["fits"]}==expected_ids, "Training coverage")
    e, scale = costs_scale(680.)
    fit_by_id, calls, shots, probability_errors, loss_errors = {}, 0, 0, [], []
    for item in result["fits"]:
        require(sha(folder/item["file"])==item["sha256"], "Fit checksum")
        record = read(folder/item["file"])
        fit = record["fit"]; fit_by_id[item["id"]] = fit
        require(record["plan_sha256"]==binding, "Fit plan")
        config = plan["configs"][item["config"]]
        require(fit["seeds"]==plan["seed_pairs"][item["trial"]-1] and fit["loss"]==item["loss"], "Fit pairing")
        require(fit["sampled_training"]==config["sampled"] and fit["gamma_max"]==config["gamma_max"], "Fit configuration")
        require(fit["evaluations"]==1024 and len(fit["restarts"])==2, "Fit budget")
        for restart in fit["restarts"]:
            rng = np.random.default_rng(restart["seed"])
            expected_pool = np.vstack([np.zeros(2), rng.uniform([0,0],[config["gamma_max"],np.pi],size=(31,2))])
            np.testing.assert_array_equal(restart["initial_candidates"], expected_pool)
            np.testing.assert_array_equal([t["angles"] for t in restart["trace"][:32]], expected_pool)
            require(len(restart["trace"])==restart["evaluations"]==512, "Restart budget")
            measurement_rng = np.random.default_rng(restart["seed"]+1000000)
            for trace in restart["trace"]:
                angles = np.array(trace["angles"])
                require(np.all(angles>=0) and np.all(angles<=[config["gamma_max"],np.pi]), "Bounds")
                p = reference(680., angles)
                raw = trace["sample_indices"]
                if config["sampled"]:
                    require(len(raw)==256, "Shot count")
                    u = measurement_rng.random(256)
                    cdf = np.cumsum(p); cdf[-1]=1.
                    np.testing.assert_array_equal(raw, np.searchsorted(cdf,u,side="right"))
                    shots += 256
                else: require(raw is None, "Unexpected samples")
                error = abs(loss(p,e,item["loss"],raw)/scale-trace["loss"])
                require(error<1e-10, "Training loss recomputation")
                loss_errors.append(error); calls += 1
            j = int(np.argmin([t["loss"] for t in restart["trace"]]))
            require(j==restart["best_evaluation_index"] and restart["angles"]==restart["trace"][j]["angles"], "Candidate selection")
            require(restart["best_loss"]==restart["trace"][j]["loss"], "Selected loss")
            require(sum(r["calls"]+r["padding_calls"] for r in restart["refinements"])+32==512, "Refinement accounting")
        j = int(np.argmin([r["best_loss"] for r in fit["restarts"]]))
        require(j==fit["selected_restart_index"] and fit["angles"]==fit["restarts"][j]["angles"], "Restart selection")
        probability_errors.append(float(max(abs(reference(680., fit["angles"])-fit["probabilities"]))))
        require(item["angles"]==fit["angles"], "Evaluation angles changed")
        if len(fit_by_id)%10==0: print(f"Audited {len(fit_by_id)}/75 complete training traces", flush=True)
    require(calls==result["training_evaluations"]==76800 and shots==result["classically_sampled_training_shots"]==6553600, "Campaign training budget")
    cases = {c["id"]:c for c in plan["transfer_cases"]}
    expected_ideal = {(key,case) for key in expected_ids|{"uniform"} for case in cases}
    require(len(result["ideal_rows"])==684 and {(r["fit_id"],r["case_id"]) for r in result["ideal_rows"]}==expected_ideal, "Ideal coverage")
    for row in result["ideal_rows"]:
        case = cases[row["case_id"]]
        a,b = case["truth"]; truth = [(0,400,900)[a],(0,120,280)[b]]
        angles = [0.,0.] if row["fit_id"]=="uniform" else fit_by_id[row["fit_id"]]["angles"]
        p = reference(case["aggregate"], angles)
        probability_errors.append(float(max(abs(p-row["probabilities"]))))
        check_metrics(case["aggregate"], truth, p, row["metrics"])
    compiled_checks = 0
    for path in sorted(folder.glob("*_compiled.json")):
        metadata = read(path)
        require(metadata["plan_sha256"]==binding and sha(folder/metadata["qpy_file"])==metadata["qpy_sha256"], "Compiled provenance")
        circuit = load_circuit(folder/metadata["qpy_file"])
        mapping = {}
        for op in circuit.data:
            if op.operation.name=="measure": mapping[circuit.find_bit(op.clbits[0]).index] = circuit.find_bit(op.qubits[0]).index
        require([mapping[i] for i in range(6)]==metadata["logical_measurement_to_compact"], "Output map")
        actual = permute(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities(), [mapping[i] for i in range(6)])
        angles = [0.,0.] if metadata["fit_id"]=="uniform_w" else fit_by_id[metadata["fit_id"]]["angles"]
        expected = np.zeros(64); expected[INDICES] = reference(680.,angles)
        probability_errors.append(float(max(abs(actual-expected)))); compiled_checks += 1
    require(compiled_checks==52, "Compiled coverage")
    noise_ids = {f"{fit}_{placement}_{model}" for fit in {i for i in expected_ids if i.startswith("exact8_")}|{"uniform_w"}
                 for placement,model in (("suspect","calibration_combined"),("suspect","generic_combined"),("comparison","calibration_combined"))}
    require(len(result["noise_rows"])==78 and {r["id"] for r in result["noise_rows"]}==noise_ids, "Noise coverage")
    noise_errors = []
    for row in result["noise_rows"]:
        require(row==read(folder/f"{row['id']}.json"), "Noise checkpoint")
        for name, checksum in row["files_sha256"].items(): require(sha(folder/name)==checksum, "Noise artifact")
        circuit = load_circuit(folder/f"{row['id']}_noise.qpy")
        p = DensityMatrix.from_instruction(circuit).probabilities()
        matrix = np.array([[1.]])
        for p10,p01 in reversed(row["readout_errors"]): matrix = np.kron(matrix, [[1-p10,p01],[p10,1-p01]])
        p = permute(matrix@p, row["measurement_map"])
        saved = np.load(folder/f"{row['id']}.npy", allow_pickle=False)
        noise_errors.append(float(max(abs(p-saved))))
        check_metrics(680., [400,280], saved[INDICES], row["metrics"])
        timing = row["timing"]
        require(abs(timing["summed_idle_qubit_s"]+timing["summed_active_qubit_s"]-6*timing["gate_makespan_s"])<1e-12, "Timing reconciliation")
    require(max(probability_errors)<1e-9 and max(noise_errors)<1e-9, "Probability cross-check")
    receipt = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_calls_recomputed": calls, "sampled_training_shots_reproduced": shots,
        "ideal_metric_rows": 684, "compiled_ideal_checks": compiled_checks, "independent_density_checks": 78,
        "max_probability_error": max(probability_errors), "max_density_probability_error": max(noise_errors),
        "max_training_loss_error": max(loss_errors), "plan_sha256": binding,
        "summary_sha256": sha(folder/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
        "limitations": ["Models, not hardware physics, are verified", "Optimizer trajectories are audited, not independently rerun", "No real-data or hardware evidence"]}
    save = json.dumps(receipt,indent=2,allow_nan=False)+"\n"
    with output.open("x") as handle: handle.write(save)
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    audit(parser.parse_args().folder.resolve())

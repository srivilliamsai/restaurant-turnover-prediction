"""Read-only model evidence audit; writes only the V17 research evidence JSON.

Requires NumPy. No model packages, training, internet, or GPU allocation.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V16 = PROJECT / "improvement_v16"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    with (PROJECT / "Train_dataset.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (PROJECT / "Test_dataset.csv").open(newline="") as stream:
        test = list(csv.DictReader(stream))
    y = np.array([float(row["Annual Turnover"]) for row in rows])
    ids = [int(row["Registration Number"]) for row in rows]
    registry = read(V16 / "baseline_registry.json")
    frozen = read(V16 / "provenance/v14_source_frozen.json")
    inputs = {name: sha(PROJECT / name) for name in registry["inputs"]}
    assert inputs == registry["inputs"] == frozen["inputs"]
    assert digest(ids) == registry["row_ids_sha256"]
    for name, expected in frozen["sources"].items():
        assert sha(V16 / "provenance/v14_sources" / name) == expected

    portal = read(V16 / "reference/v14_portal_trial.json")
    assert sha(V16 / "reference/scored_v14.csv") == portal["csv_sha256"]
    assert sha(V16 / "reference/scored_v14.ipynb") == portal["notebook_sha256"]
    measured = list(csv.DictReader((V16 / "results/screening.csv").open()))
    results = {}
    verified = []
    for seed in (42, 71):
        record = registry["schemes"][str(seed)]
        assert sha(V16 / record["file"]) == record["sha256"]
        with np.load(V16 / record["file"], allow_pickle=False) as z:
            v9 = z["v9"].copy()
            assert np.max(np.abs(v9 - (.5 * z["v7"] + .5 * z["t9"]))) < .01
        shuffled = np.random.RandomState(seed).permutation(len(y))
        p35, tabm = np.zeros(len(y)), np.zeros(len(y))
        seen = np.zeros(len(y), int)
        for fold, indices in enumerate(np.array_split(shuffled, 5)):
            b = np.sort(indices)
            a = np.setdiff1d(np.arange(len(y)), b)
            seen[b] += 1
            for directory, name, output in (
                ("provenance/v14_cache", "P35_8", p35),
                ("results/cache", "TM32-PLE", tabm),
            ):
                path = V16 / directory / f"{name}_{seed}_fold{fold}.npz"
                meta = read(path.with_suffix(".json"))
                context = meta["context"]
                assert sha(path) == meta["sha256"]
                assert context["inputs"] == inputs
                assert context["name"] == name
                assert context["train_indices"] == digest(a.tolist())
                assert context["valid_indices"] == digest(b.tolist())
                assert context["training_targets"] == hashlib.sha256(y[a].tobytes()).hexdigest()
                if name == "P35_8":
                    assert context["sources"] == frozen["sources"]
                with np.load(path, allow_pickle=False) as z:
                    pred = z["prediction"]
                    assert pred.shape == b.shape and np.isfinite(pred).all()
                    output[b] = pred
                verified.append({"path": str(path.relative_to(PROJECT)), "sha256": meta["sha256"]})
        assert np.all(seen == 1)
        baseline = .25 * v9 + .75 * p35
        old = rmse(y, baseline)
        comparisons = []
        for weight in (.1, .25, .5, .75, 1.):
            prediction = (1 - weight) * baseline + weight * tabm
            new = rmse(y, prediction)
            saved = next(row for row in measured if float(row["weight"]) == weight)
            assert abs(new - float(saved[f"rmse_{seed}"])) < 1e-6
            comparisons.append({"weight": weight, "rmse_INR": new,
                                "gain_INR": old - new, "relative_gain": (old - new) / old})
        error = (y - baseline) ** 2
        top = np.argsort(error)[-5:][::-1]
        segments = {}
        for label, mask in (("at_most_40M", y <= 40e6),
                            ("40M_to_80M", (y > 40e6) & (y <= 80e6)),
                            ("above_80M", y > 80e6)):
            segments[label] = {"rows": int(mask.sum()), "rmse_INR": rmse(y[mask], baseline[mask]),
                               "squared_error_share": float(error[mask].sum() / error.sum())}
        results[str(seed)] = {
            "v14_rmse_INR": old,
            "tabm_alone_rmse_INR": rmse(y, tabm),
            "error_correlation": float(np.corrcoef(y - baseline, y - tabm)[0, 1]),
            "comparisons": comparisons,
            "segments": segments,
            "top_five_squared_error_share": float(error[top].sum() / error.sum()),
            "top_five_errors": [{"id": ids[i], "target_INR": float(y[i]),
                                 "v14_prediction_INR": float(baseline[i])} for i in top],
        }
    eligible = []
    for j, weight in enumerate((.1, .25, .5, .75, 1.)):
        gains = [results[str(s)]["comparisons"][j]["relative_gain"] for s in (42, 71)]
        if min(gains) > 0 and float(np.mean(gains)) >= .002:
            eligible.append(weight)
    assert not eligible and not read(V16 / "results/decision.json")["qualified"]
    best = max(range(5), key=lambda j: np.mean([
        results[str(s)]["comparisons"][j]["relative_gain"] for s in (42, 71)]))
    best_gain = float(np.mean([results[str(s)]["comparisons"][best]["relative_gain"] for s in (42, 71)]))
    columns = [k for k in rows[0] if k not in ("Registration Number", "Annual Turnover")]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(tuple(row[c].strip().lower() for c in columns), []).append(i)
    evidence = {
        "checked_at_IST": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "status": "PASS",
        "scope": "Existing development predictions reconstructed; no new model fitted or GPU allocated.",
        "input_hashes": inputs,
        "training_rows": len(rows), "test_rows": len(test),
        "original_predictors": len(columns),
        "target_max_INR": float(y.max()),
        "tail_above_80M_rows": int((y > 80e6).sum()),
        "exact_duplicate_predictor_groups": sum(len(g) > 1 for g in groups.values()),
        "verified_fold_outputs": verified,
        "v14_portal_rmse_INR": portal["actual_portal_rmse_INR"],
        "v14_csv_sha256": portal["csv_sha256"],
        "v14_notebook_sha256": portal["notebook_sha256"],
        "v16_best_relative_development_gain": best_gain,
        "v16_qualified": False,
        "development": results,
        "label_errors_established": False,
        "hidden_test_distribution_inferred": False,
        "script_sha256": sha(Path(__file__)),
    }
    (ROOT / "review_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "verified_fold_outputs": len(verified),
                      "v16_best_gain_percent": best_gain * 100,
                      "v14_portal_rmse_INR": portal["actual_portal_rmse_INR"],
                      "evidence": str(ROOT / "review_evidence.json")}, indent=2))


if __name__ == "__main__":
    main()

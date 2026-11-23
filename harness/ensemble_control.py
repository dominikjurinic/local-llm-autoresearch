#!/usr/bin/env python3
"""
Second positive control for the held-out gate: an ensemble of two seeds of the start file.

pc1 (longer training) made the model worse, so it could not show that the gate accepts a real gain. Averaging the
predictions of two independently trained models is a change known in advance to help, especially here where seed
noise is large. This script trains the unchanged start file with 10 fresh seeds (10..19, 180 s each), then forms:

  e001  single model    fresh seeds 10..14 as they are     -> a NULL control: no real change, the gate should reject it
  e002  2-seed ensemble  (10+11), (12+13), ..., (18+19)    -> the POSITIVE control: should gain, the gate should accept it

Each is compared with the start file's 8 seeds from b5 (the steadier baseline) and, as in h1/n1 as run, its first 5.
Gate: visible half > 1.0 SE, then hidden half > 0.5 SE; SE over seeds and validation images. Test read afterwards.
No LLM involved. About 32 minutes of GPU.

  uv run python ensemble_control.py pc2
Stopped or disconnected? Run the same command again: finished seeds are kept.
Writes /data/ar2/runs/<name>/ensemble_control.md and prints the same table.
Env options: FROM=b5  HOLDOUT=0.5  HOLDOUT_SEED=1  ACCEPT_SE=1.0  CONFIRM_K=0.5  (defaults = settings of h1 and n1)
"""
import argparse
import csv
import json
import math
import os
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
argv = sys.argv[1:]
sys.argv = sys.argv[:1]
import ar_loop as A  # noqa: E402
from det_boot import paired_image_se  # noqa: E402

ROOT = Path(os.environ.get("AR_ROOT", "/data/ar2"))
FRESH = list(range(10, 20))
GRACE = 30


def busy():
    for pat, what in ((r"ar_loop\.py", "an AutoResearch run"), ("llama-server", "the local LLM server")):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).stdout.strip():
            return what
    return None


def average_preds(a, b, out):
    """Mean of two preds.csv files per (det_id, class); a row missing in one file (score below its cut-off) counts as 0."""
    sums = {}
    for path in (a, b):
        for r in csv.DictReader(open(path)):
            k = (r["det_id"], int(r["cls"]))
            sums[k] = sums.get(k, 0.0) + float(r["score"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "cls", "score"])
        for (d, c), s in sums.items():
            w.writerow([d, c, f"{s / 2:.6f}"])


def gain(new, base, split):
    sn = [A.TASK.score(p, split)[0]["primary"] for p in new]
    sb = [A.TASK.score(p, split)[0]["primary"] for p in base]
    se_seed = math.sqrt(st.variance(sn) / len(sn) + st.variance(sb) / len(sb))
    se_img = paired_image_se([A.TASK.parts(p, split) for p in new], [A.TASK.parts(p, split) for p in base], 400)
    return st.mean(sn) - st.mean(sb), math.sqrt(se_seed ** 2 + se_img ** 2), st.mean(sb), st.mean(sn)


def main():
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    name = argv[0]
    src_name = os.environ.get("FROM", "b5")
    frac, hseed = float(os.environ.get("HOLDOUT", 0.5)), int(os.environ.get("HOLDOUT_SEED", 1))
    k_sel, k_conf = float(os.environ.get("ACCEPT_SE", 1.0)), float(os.environ.get("CONFIRM_K", 0.5))

    A.TASK = A.TASKS["det"]()
    P = A.Paths(str(ROOT), name, A.TASK)
    src = ROOT / "runs" / src_name
    try:
        base = json.loads((src / "experiments.json").read_text())[0]
        smeta = json.loads((src / "meta.json").read_text())
    except Exception as ex:
        sys.exit(f"cannot read the baseline of run {src_name} ({ex}). Set FROM=<run with the start-file measurements>.")
    seconds = float(smeta.get("seconds", 180))
    P.dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "e000", P.dir / "e000", dirs_exist_ok=True)
    base_ok = [r["seed"] for r in base["runs"] if r["status"] == "ok"]
    base8 = [P.dir / "e000" / f"s{s}" / "preds.csv" for s in base_ok[:8]]
    if len(base8) < 8 or not all(p.exists() for p in base8):
        sys.exit(f"run {src_name} does not hold the predictions of 8 start-file seeds. Set FROM=<other run>.")
    base5 = base8[:5]

    A.TASK.load(P)
    n_sel, n_conf = A.TASK.split_holdout(frac, hseed)
    (P.dir / "meta.json").write_text(json.dumps({
        "llm": "none (ensemble control)", "task": "det", "seconds": seconds, "accept_se": k_sel, "holdout": frac,
        "holdout_seed": hseed, "confirm_k": k_conf, "holdout_images": [n_sel, n_conf], "baseline_from": src_name,
        "fresh_seeds": FRESH}, indent=1))

    # 1. train the unchanged start file with fresh seeds
    fresh_dir = P.dir / "fresh"
    fresh_dir.mkdir(exist_ok=True)
    shutil.copyfile(P.dir / "e000" / "train.py", fresh_dir / "train.py")
    done = lambda s: (fresh_dir / f"s{s}" / "preds.csv").exists() and (fresh_dir / f"s{s}.ok").exists()
    todo = [s for s in FRESH if not done(s)]
    print(f"{name}: baseline from {src_name} ({seconds:.0f} s, 8 seeds), {n_sel} visible and {n_conf} hidden "
          f"validation images; {len(todo)} of {len(FRESH)} fresh training runs to do, about "
          f"{len(todo) * (seconds + 10) / 60:.0f} min", flush=True)
    if todo and busy():
        sys.exit(f"The GPU is in use by {busy()}. Wait until it has finished (or stop it), then run this again.")
    args = argparse.Namespace(seconds=seconds, grace=GRACE)
    for s in todo:
        for attempt in (1, 2):
            rec = A.run_seed(P, args, fresh_dir, s)
            if rec["status"] == "ok":
                (fresh_dir / f"s{s}.ok").write_text(json.dumps(rec))
                break
            print(f"  seed {s} failed ({rec['status']}):\n{rec['error'][-600:]}", flush=True)
        else:
            sys.exit(f"seed {s} failed twice; stopping. Log: {fresh_dir}/s{s}.log")

    # 2. the two candidates
    single = [fresh_dir / f"s{s}" / "preds.csv" for s in FRESH[:5]]
    ens = []
    for i in range(5):
        a, b = FRESH[2 * i], FRESH[2 * i + 1]
        out = P.dir / "ensemble" / f"s{a}_{b}" / "preds.csv"
        average_preds(fresh_dir / f"s{a}" / "preds.csv", fresh_dir / f"s{b}" / "preds.csv", out)
        ens.append(out)

    # 3. the gate, against 8 and against 5 baseline seeds, and the test set
    print("\nScoring on every split (a few minutes) ...", flush=True)
    yn = lambda x: "yes" if x else "no"
    rows, verdicts, result = [], [], {}
    for label, cand in (("single model, fresh seeds (null control)", single), ("2-seed ensemble (positive control)", ens)):
        for bname, bset in (("8 seeds", base8), ("5 seeds", base5)):
            g = {s: gain(cand, bset, s) for s in ("val", "confirm", "test")}
            p_sel, p_conf = g["val"][0] > k_sel * g["val"][1], g["confirm"][0] > k_conf * g["confirm"][1]
            acc = p_sel and p_conf
            result[f"{label} | {bname}"] = {s: {"gain": v[0], "se": v[1]} for s, v in g.items()} | {"accepted": acc}
            rows.append(f"| {label} | {bname} | {g['val'][0]:+.4f} ± {g['val'][1]:.4f} | {yn(p_sel)} | "
                        f"{g['confirm'][0]:+.4f} ± {g['confirm'][1]:.4f} | {yn(p_conf)} | "
                        f"{'ACCEPTED' if acc else 'rejected'} | {g['test'][0]:+.4f} ± {g['test'][1]:.4f} |")
            if bname == "8 seeds":
                real = g["test"][0] > 2 * g["test"][1]
                if "ensemble" in label:
                    v = ("PASS: really better on test, and the gate accepted it." if real and acc else
                         "MISS: really better on test, but the gate rejected it." if real else
                         "accepted, but the test gain is below 2 SE." if acc else
                         "no verdict: the ensemble is not clearly better on test.")
                else:
                    v = ("PASS: no real change, and the gate rejected it." if not acc else
                         "FALSE ACCEPT: the same file with other seeds was accepted.")
                verdicts.append(f"- {label}: {v}")
    (P.dir / "ensemble_control.json").write_text(json.dumps(result, indent=1))
    lines = [f"# Ensemble control {name}: does the held-out gate accept a real gain and reject a non-change?", "",
             f"Baseline: start file, {seconds:.0f} s, seeds from {src_name}. Candidates: the same file trained with fresh "
             f"seeds {FRESH[0]}..{FRESH[-1]}; single = seeds {FRESH[0]}..{FRESH[4]}; ensemble = mean scores of seed pairs "
             f"({FRESH[0]}+{FRESH[1]}) ... ({FRESH[-2]}+{FRESH[-1]}). Gate as in h1 and n1: visible half ({n_sel} images) "
             f"> {k_sel} SE, then hidden half ({n_conf} images) > {k_conf} SE. SE over seeds and images. "
             "Test scores read afterwards, for analysis only.", "",
             "| candidate | baseline | visible val gain ± SE | passes | hidden val gain ± SE | passes | gate | test gain ± SE |",
             "|---|---|---|---|---|---|---|---|"] + rows + ["", "Verdict (8-seed baseline):"] + verdicts
    (P.dir / "ensemble_control.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {P.dir / 'ensemble_control.md'}")


if __name__ == "__main__":
    main()

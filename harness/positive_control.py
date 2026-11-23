#!/usr/bin/env python3
"""
Positive control for the held-out confirmation gate (harness v3.2).

h1 and n1 accepted nothing. This checks that the gate does accept a change that is really better: the unchanged
start file, trained with a longer time budget (default 360 s and 720 s instead of 180 s). No LLM is involved.
Each control gets 5 seeds and goes through the same acceptance code as an agent's idea (beats() in ar_loop.py:
visible half > ACCEPT_SE standard errors, then hidden half > CONFIRM_K standard errors). Afterwards the test set is
read, for analysis only, to see whether the control really was better.

  uv run python positive_control.py pc1             budgets 360 s and 720 s  (about 1.5 h of GPU)
  uv run python positive_control.py pc1 360         one budget              (about 35 min)
Stopped or disconnected? Run the same command again: finished seeds are kept.

Env options: FROM=b5 (run whose start-file measurements are the baseline)  HOLDOUT=0.5  HOLDOUT_SEED=1
             ACCEPT_SE=1.0  CONFIRM_K=0.5   (defaults = the settings of h1 and n1)
Writes /data/ar2/runs/<name>/positive_control.md and prints the same table.
"""
import argparse
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

ROOT = Path(os.environ.get("AR_ROOT", "/data/ar2"))
SEEDS = 5
GRACE = 30
LABEL = {"val": "visible val", "confirm": "hidden val", "test": "test"}


def busy():
    for pat, what in ((r"ar_loop\.py", "an AutoResearch run"), ("llama-server", "the local LLM server")):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).stdout.strip():
            return what
    return None


def gains(P, e, base):
    """Gain of e over base on each split, with the same uncertainty as the harness (seeds and images)."""
    pe, pb = A.seed_preds(P.dir, e, SEEDS), A.seed_preds(P.dir, base, SEEDS)
    out = {}
    for s in ("val", "confirm", "test"):
        se_ = [A.TASK.score(p, s)[0]["primary"] for p in pe]
        sb = [A.TASK.score(p, s)[0]["primary"] for p in pb]
        se_seed = math.sqrt(st.variance(se_) / len(se_) + st.variance(sb) / len(sb))
        se_img = A.image_se(P.dir, e, base, s, n=SEEDS) or 0.0
        out[s] = (st.mean(se_) - st.mean(sb), math.sqrt(se_seed ** 2 + se_img ** 2), st.mean(sb), st.mean(se_))
    return out


def main():
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    name = argv[0]
    budgets = [float(x) for x in argv[1:]] or [360.0, 720.0]
    src_name = os.environ.get("FROM", "b5")
    frac, hseed = float(os.environ.get("HOLDOUT", 0.5)), int(os.environ.get("HOLDOUT_SEED", 1))
    k_sel, k_conf = float(os.environ.get("ACCEPT_SE", 1.0)), float(os.environ.get("CONFIRM_K", 0.5))

    A.TASK = A.TASKS["det"]()
    A.NSEEDS = SEEDS
    P = A.Paths(str(ROOT), name, A.TASK)
    src = ROOT / "runs" / src_name
    try:
        base = json.loads((src / "experiments.json").read_text())[0]
        smeta = json.loads((src / "meta.json").read_text())
    except Exception as ex:
        sys.exit(f"cannot read the baseline of run {src_name} ({ex}). Set FROM=<run with the start-file measurements>.")
    if base.get("outcome") != "baseline" or not (src / "e000" / "train.py").exists():
        sys.exit(f"run {src_name} has no usable start-file measurements (e000). Set FROM=<other run>.")
    base_s = float(smeta.get("seconds", 180))
    if any(b <= base_s for b in budgets):
        sys.exit(f"the control budgets must be longer than the baseline's {base_s:.0f} s")

    P.dir.mkdir(parents=True, exist_ok=True)
    state_file = P.dir / "experiments.json"
    old = {e["id"]: e for e in json.loads(state_file.read_text())} if state_file.exists() else {}
    shutil.copytree(src / "e000", P.dir / "e000", dirs_exist_ok=True)
    if len([p for p in A.seed_preds(P.dir, base, SEEDS) if p.exists()]) < SEEDS:
        sys.exit(f"run {src_name} does not hold the predictions of {SEEDS} start-file seeds. Set FROM=<other run>.")

    A.TASK.load(P)
    n_sel, n_conf = A.TASK.split_holdout(frac, hseed)
    for r in base["runs"]:   # the baseline as the gate sees it: scored on the visible half only
        if r["status"] == "ok":
            m, _ = A.TASK.score(P.dir / "e000" / f"s{r['seed']}" / "preds.csv", "val")
            r["val_f1"], r["per_class"] = m["primary"], m["per_class"]
    A.summarize_runs(base)
    base["reused_from"] = src_name
    A.BOOT = A.RUNDIR = P.dir
    A.HOLD = {"k": k_conf}
    meta = {"llm": "none (positive control)", "task": "det", "seeds": SEEDS, "seconds": base_s, "accept_se": k_sel,
            "accept_boot": True, "holdout": frac, "holdout_seed": hseed, "confirm_k": k_conf,
            "holdout_images": [n_sel, n_conf], "positive_control_budgets": budgets, "baseline_from": src_name}
    (P.dir / "meta.json").write_text(json.dumps(meta, indent=1))

    exps = [base]
    for i, b in enumerate(budgets, 1):
        eid = f"e{i:03d}"
        e = old.get(eid) or {}
        if e.get("budget") != b:
            e = {"id": eid, "base_id": "e000", "category": "control", "budget": b, "runs": [],
                 "hypothesis": f"positive control: the unchanged start file with a {b:.0f} s training budget "
                               f"instead of {base_s:.0f} s"}
        exps.append(e)
    save = lambda: state_file.write_text(json.dumps(exps, indent=1))
    save()

    todo = sum(SEEDS - len([r for r in e["runs"] if r["status"] == "ok"]) for e in exps[1:])
    need_min = sum((SEEDS - len([r for r in e["runs"] if r["status"] == "ok"])) * (e["budget"] + 10) for e in exps[1:]) / 60
    print(f"{name}: baseline from {src_name} ({base_s:.0f} s), {n_sel} visible and {n_conf} hidden validation images; "
          f"controls: {', '.join(f'{b:.0f} s' for b in budgets)}; {todo} training runs to do, about {need_min:.0f} min",
          flush=True)
    if todo and busy():
        sys.exit(f"The GPU is in use by {busy()}. Wait until it has finished (or stop it), then run this again.")

    for e in exps[1:]:
        exp_dir = P.dir / e["id"]
        exp_dir.mkdir(exist_ok=True)
        shutil.copyfile(P.dir / "e000" / "train.py", exp_dir / "train.py")
        (exp_dir / "diff.patch").write_text("")
        args = argparse.Namespace(seconds=e["budget"], grace=GRACE)
        e["runs"] = [r for r in e["runs"] if r["status"] == "ok"
                     and (exp_dir / f"s{r['seed']}" / "preds.csv").exists()]
        seed, fails = 0, 0
        while len(e["runs"]) < SEEDS:
            if any(r["seed"] == seed for r in e["runs"]):
                seed += 1
                continue
            rec = A.run_seed(P, args, exp_dir, seed)
            seed += 1
            if rec["status"] != "ok":
                fails += 1
                print(f"  {e['id']} seed {rec['seed']} failed ({rec['status']}):\n{rec['error'][-600:]}", flush=True)
                if fails >= 2:
                    sys.exit(f"{e['id']}: two failed runs; stopping. Log: {exp_dir}/s{rec['seed']}.log")
                continue
            e["runs"].append(rec)
            e["runs"].sort(key=lambda r: r["seed"])
            save()
        A.summarize_runs(e)
        print(f"{e['id']} ({e['budget']:.0f} s): visible-half mean {e['mean']:.4f} vs baseline {base['mean']:.4f}", flush=True)
        ok = A.beats(e, base, k_sel)
        e["outcome"] = "accepted" if ok else "not confirmed"
        save()

    print("\nScoring on every split (a few minutes) ...", flush=True)
    lines = [f"# Positive control {name}: does the held-out gate accept a real improvement?", "",
             f"Baseline: start file, {base_s:.0f} s budget ({SEEDS} seeds, from {src_name}). Controls: the same file, "
             f"unchanged, with a longer budget ({SEEDS} seeds each). Gate as in h1 and n1: gain on the visible half "
             f"({n_sel} images) > {k_sel} SE, then gain on the hidden half ({n_conf} images) > {k_conf} SE. "
             "SE over seeds and images. Test scores read afterwards, for analysis only.", "",
             "| control | visible val gain ± SE | passes | hidden val gain ± SE | passes | gate | test gain ± SE | real on test (> 2 SE) |",
             "|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for e in exps[1:]:
        g = gains(P, e, base)
        e["gains"] = {s: {"gain": v[0], "se": v[1], "base": v[2], "new": v[3]} for s, v in g.items()}
        p_sel, p_conf = g["val"][0] > k_sel * g["val"][1], g["confirm"][0] > k_conf * g["confirm"][1]
        accepted, real = e["outcome"] == "accepted", g["test"][0] > 2 * g["test"][1]
        yn = lambda x: "yes" if x else "no"
        lines.append(f"| {e['budget']:.0f} s | {g['val'][0]:+.4f} ± {g['val'][1]:.4f} | {yn(p_sel)} | "
                     f"{g['confirm'][0]:+.4f} ± {g['confirm'][1]:.4f} | {yn(p_conf)} | "
                     f"{'ACCEPTED' if accepted else 'rejected'} | {g['test'][0]:+.4f} ± {g['test'][1]:.4f} | {yn(real)} |")
        if real and accepted:
            v = "PASS: really better on test, and the gate accepted it."
        elif real:
            v = ("MISS: really better on test, but the gate rejected it "
                 f"(failed on the {'visible' if not p_sel else 'hidden'} half). The gate is too strict for a gain of this size.")
        elif accepted:
            v = "accepted by the gate, but the test gain is below 2 SE, so this control cannot show that the gain is real."
        else:
            v = "no verdict: not clearly better on test, so this control cannot test the gate. Use a longer budget."
        verdicts.append(f"- {e['budget']:.0f} s: {v}")
    save()
    lines += ["", "Test mAP@0.5: baseline " + f"{exps[1]['gains']['test']['base']:.4f}; "
              + "; ".join(f"{e['budget']:.0f} s {e['gains']['test']['new']:.4f}" for e in exps[1:]), "", "Verdict:"] + verdicts
    (P.dir / "positive_control.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {P.dir / 'positive_control.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
After a run with --holdout: how did the candidates that passed on the visible validation half do on the hidden half
and on the test set? Uses the predictions the candidates already saved (no training, a few minutes, CPU only).

This reads the test set for analysis only, after the run has finished; nothing here feeds back into any run.

  uv run python score_candidates.py h1                 every candidate that reached 5 seeds
  uv run python score_candidates.py h1 e007 e012       only these
  BASE_SEEDS=8 uv run python score_candidates.py h1    compare against all 8 start-file seeds instead of the first 5
                                                       (a steadier baseline; writes candidates_test_base8.md)
Writes /data/ar2/runs/<run>/candidates_test.md and prints the same table.
"""
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
args = sys.argv[1:]
sys.argv = sys.argv[:1]
import ar_loop as A  # noqa: E402

ROOT = Path(os.environ.get("AR_ROOT", "/data/ar2"))
BASE_SEEDS = int(os.environ.get("BASE_SEEDS", 0))   # 0 = the same seeds the run itself compared against


def main():
    if not args:
        sys.exit(__doc__)
    name, only = args[0], set(args[1:])
    A.TASK = A.TASKS["det"]()
    P = A.Paths(str(ROOT), name, A.TASK)
    meta = json.loads((P.dir / "meta.json").read_text())
    A.NSEEDS = meta.get("seeds", 5)
    A.TASK.load(P)
    if meta.get("holdout"):
        n_sel, n_conf = A.TASK.split_holdout(meta["holdout"], meta.get("holdout_seed", 1))
        print(f"{name}: {n_sel} visible and {n_conf} hidden validation images (split seed {meta.get('holdout_seed', 1)})")
    exps = {e["id"]: e for e in json.loads((P.dir / "experiments.json").read_text())}
    cands = [e for e in exps.values() if e["id"] != "e000" and e.get("accept")
             and (not only or e["id"] in only)]
    if not cands:
        sys.exit("no candidate with a 5-seed result")
    splits = ["val"] + (["confirm"] if meta.get("holdout") else []) + ["test"]
    label = {"val": "visible val", "confirm": "hidden val", "test": "test"}
    rows = []
    for e in cands:
        base = exps[e["accept"]["vs"]]
        pe, pb = A.seed_preds(P.dir, e, A.NSEEDS), A.seed_preds(P.dir, base, BASE_SEEDS or A.NSEEDS)
        if BASE_SEEDS and len(pb) < BASE_SEEDS:
            sys.exit(f"{base['id']} has only {len(pb)} finished seeds, not {BASE_SEEDS}")
        if len(pe) < A.NSEEDS or not all(p.exists() for p in pe + pb):
            print(f"  {e['id']}: predictions missing, skipped")
            continue
        out = {"id": e["id"], "vs": base["id"], "outcome": e["outcome"]}
        for s in splits:
            se_ = [A.TASK.score(p, s)[0]["primary"] for p in pe]
            sb = [A.TASK.score(p, s)[0]["primary"] for p in pb]
            gain = st.mean(se_) - st.mean(sb)
            se_seed = math.sqrt(st.variance(se_) / len(se_) + st.variance(sb) / len(sb))
            if BASE_SEEDS:   # different seed counts on the two sides: call the bootstrap directly
                from det_boot import paired_image_se
                se_img = paired_image_se([A.TASK.parts(p, s) for p in pe], [A.TASK.parts(p, s) for p in pb], 400)
            else:
                se_img = A.image_se(P.dir, e, base, s, n=A.NSEEDS) or 0.0
            out[s] = (gain, math.sqrt(se_seed ** 2 + se_img ** 2))
        rows.append(out)
        print(f"  {e['id']}: " + ", ".join(f"{label[s]} {out[s][0]:+.4f} ± {out[s][1]:.4f}" for s in splits), flush=True)
    head = "| id | vs | outcome | " + " | ".join(f"{label[s]} gain ± SE" for s in splits) + " |"
    title = f"# Candidates of {name}: gains on each split ({A.NSEEDS} seeds, SE over seeds and images)"
    if BASE_SEEDS:
        title += f"; baseline = first {BASE_SEEDS} seeds of the start file"
    out_name = f"candidates_test_base{BASE_SEEDS}.md" if BASE_SEEDS else "candidates_test.md"
    lines = [title, "",
             "Test scores read after the run, for analysis only.", "", head,
             "|" + "---|" * (3 + len(splits))]
    for r in rows:
        lines.append(f"| {r['id']} | {r['vs']} | {r['outcome']} | "
                     + " | ".join(f"{r[s][0]:+.4f} ± {r[s][1]:.4f}" for s in splits) + " |")
    (P.dir / out_name).write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {P.dir / out_name}")


if __name__ == "__main__":
    main()

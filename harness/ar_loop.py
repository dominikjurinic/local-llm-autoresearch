#!/usr/bin/env python3
"""
AutoResearch ("Mode B"): a local LLM edits train.py itself, not just settings.

Each experiment:
  1. the LLM sees the current best train.py, its scores, and the history of all experiments,
     and replies with one idea (hypothesis) plus SEARCH/REPLACE edits to the code
  2. the edits are applied and syntax-checked (on failure the LLM gets the error and 2 more tries)
  3. the new train.py runs in a sandbox (bubblewrap): no network, no access to anything except
     the label-hidden data folder and its own output folder; hard wall-clock limit
  4. the harness scores its predictions on validation crops (labels the agent never sees)
  5. ratchet: if seed 0 beats the best 3-seed mean, seeds 1 and 2 run too; only if the
     3-seed mean still beats the best does the new train.py become the best
After the loop: the starting file and the final best are re-run with extra seeds and scored on
the hidden test set (only here), and report.md is written.

Everything is saved in /data/ar/runs/<name>/ and the loop resumes after an interruption.

Usage:
  python ar_loop.py --check                                  # sandbox self-test
  python ar_loop.py --name ar-dry --max-exps 2 --extra-seeds 0   # short trial (~20-30 min)
  python ar_loop.py --name ar1 --llm qwen3.8:27b --until 07:30   # the night run
  python ar_loop.py --name ar1 --report-only
Phase 4 (two-stage detection, mAP@0.5):
  python ar_loop.py --task det --root /data/ar2 --start ar2_train.py --program ar2_program.md \
         --seconds 180 --confirm-seeds 4 --name ar2-1 --until 07:00
"""
import argparse
import csv
import difflib
import hashlib
import json
import re
import math
import os
import shlex
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import CLASSES, metrics  # noqa: E402  (the same fixed metric as before)

OLLAMA = "http://localhost:11434"
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True
    print("\nStop requested: finishing the current step, then writing the report.", flush=True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ paths and tasks
class Paths:
    def __init__(self, root, name, task):
        self.root = Path(root)
        self.data = self.root / "data"
        self.labels = self.root / "private" / task.labels_file
        self.hf = self.root / "hf-cache"
        self.dir = self.root / "runs" / name if name else None


class ClsTask:
    """Defect-type classification of given crops; metric: macro-F1 (evaluate.py)."""
    name, metric, index_file, labels_file = "cls", "macro-F1", "index.csv", "labels.csv"

    def load(self, P):
        self.labels = load_labels(P.labels)

    def refs_text(self, P):
        return ""

    def score(self, preds_path, split):
        m, problem = score(preds_path, self.labels, split)
        if problem:
            return None, problem
        return {"primary": m["macro_f1"], "per_class": {k: v["f1"] for k, v in m["per_class"].items()},
                "cm": m["confusion_matrix"]}, None


class DetTask:
    """Two-stage detection: predictions for detector boxes; metric: mAP@0.5 (det_metric.py, = YOLOv7)."""
    name, metric, index_file, labels_file = "det", "mAP@0.5", "dets.csv", "gt.csv"

    def load(self, P):
        from collections import defaultdict
        self.gts = defaultdict(list)
        for r in csv.DictReader(open(P.labels)):
            self.gts[r["image"]].append((int(r["cls"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
        self.img_split = {r["image"]: r["split"] for r in csv.DictReader(open(P.root / "private" / "images.csv"))}
        self.dets = {}
        self.dets_by_img = defaultdict(list)
        for r in csv.DictReader(open(P.data / "dets.csv")):
            if r["split"] == "eval":
                box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                self.dets[r["det_id"]] = (r["image"], box)
                self.dets_by_img[r["image"]].append((r["det_id"], box))
        self.diagnostics = False
        ref = P.root / "private" / "reference.json"
        self.ref = json.loads(ref.read_text()) if ref.exists() else {}

    def split_holdout(self, frac, seed):
        """v3.2: move a fraction of the validation images to a 'confirm' split the agent never sees. Stratified by the
        rarest class in each image, so the small pore class is split evenly. Deterministic for a given seed."""
        import random
        groups = {}
        for img, s in sorted(self.img_split.items()):
            if s == "val":
                cls = [c for c, _ in self.gts.get(img, [])]
                groups.setdefault(min(cls) if cls else 4, []).append(img)
        rng = random.Random(seed)
        for key in sorted(groups):
            imgs = groups[key]
            rng.shuffle(imgs)
            for img in imgs[:round(len(imgs) * frac)]:
                self.img_split[img] = "confirm"
        return (sum(1 for s in self.img_split.values() if s == "val"),
                sum(1 for s in self.img_split.values() if s == "confirm"))

    def refs_text(self, P):
        names = {"d1_val_map50": "one-step 4-class detector (the baseline to beat)",
                 "d2_class_agnostic_val_map50": "stage-1 boxes scored as a single class 'defect'",
                 "d2_boxes_perfect_classes_val_map50": "stage-1 boxes with perfect classes (ceiling for stage 2)"}
        return "\n".join(f"- {names.get(k, k)}: {v:.3f} mAP@0.5" for k, v in self.ref.items() if v is not None)

    def parts(self, preds_path, split):
        """Per-image matched rows of one prediction file (det_boot.Parts), for image-level uncertainty."""
        from det_boot import Parts
        key = (str(preds_path), split)
        cache = self.__dict__.setdefault("_parts", {})
        if key not in cache:
            preds = {}
            for r in csv.DictReader(open(preds_path)):
                img, box = self.dets[r["det_id"]]
                preds.setdefault(img, []).append((int(r["cls"]), float(r["score"]), box))
            G = {img: self.gts.get(img, []) for img, s in self.img_split.items() if s == split}
            cache[key] = Parts(preds, G)
        return cache[key]

    def score(self, preds_path, split):
        from det_metric import map50
        if not preds_path.exists():
            return None, "train.py finished but wrote no /work/out/preds.csv"
        preds, n, rows = {}, 0, {}
        try:
            for r in csv.DictReader(open(preds_path)):
                did, c, sc = r["det_id"], int(r["cls"]), float(r["score"])
                if did not in self.dets:
                    return None, f"preds.csv has det_id {did!r}, which is not an eval detection in dets.csv"
                if c not in (0, 1, 2, 3):
                    return None, f"preds.csv has class {c} (must be 0..3)"
                if not (sc == sc) or sc in (float("inf"), float("-inf")):
                    return None, f"preds.csv has a non-finite score for {did}"
                img, box = self.dets[did]
                preds.setdefault(img, []).append((c, sc, box))
                rows.setdefault(did, []).append((c, sc))
                n += 1
        except Exception as e:
            return None, f"preds.csv could not be read ({e}); expected columns det_id,cls,score"
        if n == 0:
            return None, "preds.csv has no prediction rows"
        G = {img: self.gts.get(img, []) for img, s in self.img_split.items() if s == split}
        m, aps = map50(preds, G)
        out = {"primary": m, "per_class": {CLASSES[c]: a for c, a in aps.items()}, "cm": None}
        if self.diagnostics and split == "val":
            try:
                out["diag"] = self.diagnose(rows, G)
            except Exception as e:  # diagnostics must never break scoring
                out["diag"] = {"error": str(e)}
        return out, None

    def diagnose(self, rows, G):
        """Where the error is (validation only): class decisions on detector boxes that overlap a true defect,
        accuracy by defect size, defects the detector missed, and mAP if every box had its correct class."""
        import numpy as np
        from det_metric import iou, map50
        cm = [[0] * 5 for _ in range(4)]   # columns: predicted class 0-3, 4 = no row written for the box
        missed, sized = 0, []
        oracle = {}
        for img, g in G.items():
            ds = self.dets_by_img.get(img, [])
            if ds:
                gt_of_det = [0] * len(ds)
                if g:
                    m = iou([b for _, b in ds], [b for _, b in g])          # dets x gts
                    for j in range(len(ds)):
                        if m[j].max() > 0.5:
                            gt_of_det[j] = g[int(m[j].argmax())][0]
                for j, (did, box) in enumerate(ds):   # "perfect classes, your scores": best row score per box
                    r = rows.get(did)
                    if r:
                        oracle.setdefault(img, []).append((gt_of_det[j], max(sc for _, sc in r), box))
            if not g:
                continue
            if not ds:
                missed += len(g)
                continue
            m = iou([b for _, b in g], [b for _, b in ds])                  # gts x dets
            for gi, (c, b) in enumerate(g):
                j = int(m[gi].argmax())
                if m[gi, j] <= 0.5:
                    missed += 1
                    continue
                r = rows.get(ds[j][0])
                if not r:
                    cm[c][4] += 1
                    continue
                pc = max(r, key=lambda t: t[1])[0]
                cm[c][pc] += 1
                sized.append(((b[2] - b[0]) * (b[3] - b[1]), pc == c))
        acc = {}
        if len(sized) >= 9:
            areas = np.array([a for a, _ in sized]); ok = np.array([k for _, k in sized], float)
            e1, e2 = np.percentile(areas, [33.3, 66.7])
            for name, sel in (("small", areas <= e1), ("medium", (areas > e1) & (areas <= e2)), ("large", areas > e2)):
                acc[name] = float(ok[sel].mean()) if sel.any() else None
        return {"cm": cm, "missed_by_detector": missed, "acc_by_size": acc,
                "map_if_classes_perfect": map50(oracle, G)[0] if oracle else None}


TASKS = {"cls": ClsTask, "det": DetTask}
TASK = None   # set in main()
NSEEDS = 3    # seeds per confirmed result (1 + --confirm-seeds)


def load_labels(path):
    lab = {}
    for r in csv.DictReader(open(path)):
        lab[r["id"]] = (r["split"], int(r["cls"]))
    return lab


def available_models(hf):
    hub = hf / "hub"
    names = sorted({p.name.split("--", 2)[2].split(".")[0] for p in hub.glob("models--timm--*")}) if hub.exists() else []
    return names


# ------------------------------------------------------------------ sandbox
def symlink_hops(path, bound):
    """Symlinks on the way from `path` to the real file that lie outside the bound folders;
    the sandbox has to recreate them so the venv's python can be found."""
    inside = lambda p: any(p == b or p.startswith(b.rstrip("/") + "/") for b in bound)
    hops, todo, steps = [], str(path), 0
    while steps < 40:
        steps += 1
        parts = Path(todo).parts
        cur, nxt_path = "/", None
        for i, part in enumerate(parts[1:], 1):
            nxt = os.path.join(cur, part)
            if os.path.islink(nxt):
                tgt = os.readlink(nxt)
                if not inside(nxt) and nxt not in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
                    hops.append((tgt, nxt))
                nxt_path = os.path.join(os.path.normpath(os.path.join(cur, tgt)), *parts[i + 1:])
                break
            cur = nxt
        if nxt_path is None:
            break
        todo = nxt_path
    return hops


def sandbox_cmd(P, train_py, out_dir):
    c = ["bwrap", "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid", "--clearenv"]
    for p in ("/usr", "/etc", "/sys"):
        c += ["--ro-bind", p, p]
    for p in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
        if os.path.islink(p):
            c += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            c += ["--ro-bind", p, p]
    c += ["--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    bound = ["/usr", "/etc", "/sys"]
    for p in {sys.base_prefix, sys.prefix}:  # the Python installation and the project's venv (read-only)
        if not p.startswith("/usr"):
            c += ["--ro-bind", p, p]
            bound.append(p)
    for target, link in symlink_hops(sys.executable, bound):  # e.g. uv's cpython-3.12 -> cpython-3.12.x link
        c += ["--symlink", target, link]
    c += ["--ro-bind", str(P.data), "/work/data", "--ro-bind", str(train_py), "/work/train.py",
          "--bind", str(out_dir), "/work/out", "--chdir", "/work"]
    if P.hf.exists():
        c += ["--bind", str(P.hf), str(P.hf), "--setenv", "HF_HOME", str(P.hf)]
    env = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TORCH_HOME": "/tmp/torch", "LANG": "C.UTF-8"}
    for k in ("CUDA_VISIBLE_DEVICES",):
        if k in os.environ:
            env[k] = os.environ[k]
    for k, v in env.items():
        c += ["--setenv", k, v]
    return c


def ensure_gpu_devices():
    """/dev/nvidia-uvm appears only after the first CUDA use since boot; the sandbox cannot create it."""
    if not os.path.exists("/dev/nvidia-uvm") and shutil.which("nvidia-modprobe"):
        subprocess.run(["nvidia-modprobe", "-u", "-c=0"], check=False)


def run_sandboxed(P, train_py, out_dir, log_path, argv, limit):
    """Run python <train_py> argv in the sandbox; returns (status, seconds)."""
    ensure_gpu_devices()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = sandbox_cmd(P, train_py, out_dir) + [sys.executable, "-u", "/work/train.py"] + argv
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=limit)
            status = "ok" if rc == 0 else "crashed"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            status = "timeout"
    return status, time.time() - t0


def tail(path, n=25):
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def score(preds_path, labels, split):
    """Macro-F1 etc. of preds.csv on one hidden split ('val' or 'test'). Returns (metrics, problem)."""
    if not preds_path.exists():
        return None, "train.py finished but wrote no /work/out/preds.csv"
    pred = {}
    try:
        for r in csv.DictReader(open(preds_path)):
            pred[r["id"]] = int(r["pred"])
    except Exception as e:
        return None, f"preds.csv could not be read ({e}); expected columns id,pred"
    ids = [i for i, (s, _) in labels.items() if s == split]
    missing = [i for i in ids if i not in pred]
    if missing:
        return None, f"preds.csv lacks {len(missing)} of the eval crops (e.g. {missing[0]})"
    bad = [i for i in ids if pred[i] not in (0, 1, 2, 3)]
    if bad:
        return None, f"preds.csv has predictions outside 0..3 (e.g. id {bad[0]}: {pred[bad[0]]})"
    return metrics([labels[i][1] for i in ids], [pred[i] for i in ids]), None


# ------------------------------------------------------------------ state
def sha256_file(path, whole_up_to=1 << 30):
    """sha256 of a file; for files over 1 GB (model weights) a fingerprint of size + first and last 16 MB."""
    p = Path(path)
    if not p.is_file():
        return None
    h, size = hashlib.sha256(), p.stat().st_size
    with open(p, "rb") as f:
        if size <= whole_up_to:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            return h.hexdigest()
        h.update(f.read(16 << 20)); f.seek(size - (16 << 20)); h.update(f.read(16 << 20)); h.update(str(size).encode())
    return "fingerprint:" + h.hexdigest()


def provenance(args, P):
    """Everything that determines a result, as hashes and versions (recorded once per run, shown in the report)."""
    from importlib import metadata
    info = {"python": sys.version.split()[0]}
    for k, f in (("start_file", args.start), ("brief", args.program), ("digest", args.digest),
                 ("ar_loop.py", __file__), ("det_metric.py", HERE / "det_metric.py"),
                 ("data/dets.csv", P.data / "dets.csv"), ("data/crops.csv", P.data / "crops.csv"),
                 ("data/index.csv", P.data / "index.csv"), ("hidden labels", P.labels)):
        if f and sha256_file(f):
            info[k] = sha256_file(f)
    if args.llm_server_cmd:
        a = shlex.split(args.llm_server_cmd)
        if "-m" in a:
            info["model_file"] = a[a.index("-m") + 1]
            info["model_file_sha"] = sha256_file(info["model_file"])
        repo = next((q for q in Path(a[0]).resolve().parents if (q / ".git").exists()), None)
        if repo:
            info["llama_cpp_commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                      capture_output=True, text=True).stdout.strip()
    pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])
    (P.dir / "packages.txt").write_text("\n".join(pk) + "\n")
    info["packages_sha"] = hashlib.sha256("\n".join(pk).encode()).hexdigest()
    info["key_packages"] = {n: v for n, v in (p.split("==") for p in pk)
                            if n.lower() in ("torch", "timm", "numpy", "pillow", "torchvision")}
    return info


class State:
    def __init__(self, d):
        self.dir = d
        self.path = d / "experiments.json"
        self.exps = json.loads(self.path.read_text()) if self.path.exists() else []
        self.events_path = d / "events.jsonl"
        self.events = [json.loads(l) for l in open(self.events_path)] if self.events_path.exists() else []

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.exps, indent=1))
        tmp.replace(self.path)

    def event(self, rec):
        self.events.append(rec)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def best(self):
        good = [e for e in self.exps if e["outcome"] in ("baseline", "new best") and e.get("mean") is not None]
        return good[-1] if good else None

    def get(self, eid):
        return next(e for e in self.exps if e["id"] == eid)

    def next_id(self):
        return f"e{len(self.exps):03d}"


def summarize_runs(e):
    ok = [r for r in e["runs"] if r["status"] == "ok"]
    s0 = next((r for r in e["runs"] if r["seed"] == 0), None)
    e["seed0"] = s0["val_f1"] if s0 and s0["status"] == "ok" else None
    if len(ok) >= NSEEDS:
        e["mean"] = st.mean(r["val_f1"] for r in ok[:NSEEDS])
        e["std"] = st.pstdev(r["val_f1"] for r in ok[:NSEEDS])
    else:
        e["mean"] = e["std"] = None


HOLD = None   # harness v3.2: {"k": ...} when part of validation is held out for confirmation (--holdout)
RUNDIR = None
BOOT = None   # harness v3.1: run folder whose seed predictions feed the image bootstrap (set in main with --accept-boot)


def seed_preds(run_dir, e, n):
    """Prediction files of the first n successful seed runs of experiment e."""
    ok = [r for r in e["runs"] if r["status"] == "ok"][:n]
    return [run_dir / e["id"] / f"s{r['seed']}" / "preds.csv" for r in ok]


def image_se(run_dir, e, best, split="val", n=None, n_boot=400):
    """Paired bootstrap over images of mean(e) - mean(best), using their seed runs (det task only)."""
    from det_boot import paired_image_se
    n = n or NSEEDS
    pe, pb = seed_preds(run_dir, e, n), seed_preds(run_dir, best, n)
    if not pe or not pb or not all(p.exists() for p in pe + pb):
        return None
    return paired_image_se([TASK.parts(p, split) for p in pe], [TASK.parts(p, split) for p in pb], n_boot)


def beats(e, best, k):
    """New best? k <= 0: any higher mean (the original rule). k > 0: the gain must exceed k standard errors --
    over seeds only (v2), or over seeds and validation images together when --accept-boot is on (v3.1)."""
    if e.get("mean") is None:
        return False
    if k <= 0:
        return e["mean"] > best["mean"]
    n = NSEEDS
    var = lambda x: (x["std"] or 0.0) ** 2 * n / max(n - 1, 1)  # pstdev -> sample variance
    se_seed = math.sqrt(var(e) / n + var(best) / n)
    gain = e["mean"] - best["mean"]
    se_img = None
    if BOOT is not None and TASK.name == "det":
        try:
            se_img = image_se(BOOT, e, best)
        except Exception as ex:   # never let the uncertainty estimate crash the night; fall back to seeds only
            print(f"  image bootstrap failed ({ex}); using the seed SE only", flush=True)
    se = math.sqrt(se_seed ** 2 + (se_img or 0.0) ** 2)
    e["accept"] = {"vs": best["id"], "gain": gain, "se_seed": se_seed, "se_img": se_img, "se": se, "k": k,
                   "needed": k * se}
    if se_img is not None:
        print(f"  gain {gain:+.4f}; needed > {k * se:.4f} (seeds ±{se_seed:.4f}, validation images ±{se_img:.4f})",
              flush=True)
    ok = gain > k * se
    if ok and HOLD is not None:
        try:
            ok = confirm_on_holdout(e, best)
        except Exception as ex:   # a failed check must not accept a change
            print(f"  held-out confirmation failed ({ex}); not accepted", flush=True)
            e["accept"]["confirm_ok"], ok = False, False
    return ok


def confirm_on_holdout(e, best):
    """v3.2: the same seed predictions, scored on the held-out validation images that the agent never sees and that
    played no part in selecting e. Accept only if the gain there exceeds HOLD['k'] x its uncertainty."""
    n = NSEEDS
    pe, pb = seed_preds(RUNDIR, e, n), seed_preds(RUNDIR, best, n)
    ce = [TASK.score(p, "confirm")[0]["primary"] for p in pe]
    cb = [TASK.score(p, "confirm")[0]["primary"] for p in pb]
    gain = st.mean(ce) - st.mean(cb)
    se_seed = math.sqrt(st.variance(ce) / len(ce) + st.variance(cb) / len(cb)) if min(len(ce), len(cb)) > 1 else 0.0
    se_img = image_se(RUNDIR, e, best, "confirm", n=n) or 0.0
    se = math.sqrt(se_seed ** 2 + se_img ** 2)
    ok = gain > HOLD["k"] * se
    e["accept"].update(confirm_gain=gain, confirm_se=se, confirm_needed=HOLD["k"] * se, confirm_ok=ok)
    print(f"  held-out images: gain {gain:+.4f}; needed > {HOLD['k'] * se:.4f} (±{se:.4f}) -> "
          + ("confirmed" if ok else "did not hold"), flush=True)
    return ok


def run_seed(P, args, exp_dir, seed):
    status, secs = run_sandboxed(P, exp_dir / "train.py", exp_dir / f"s{seed}", exp_dir / f"s{seed}.log",
                                 ["--data", "/work/data", "--out", "/work/out", "--seed", str(seed),
                                  "--seconds", str(args.seconds)], args.seconds + args.grace)
    rec = {"seed": seed, "status": status, "seconds": round(secs, 1), "time": now(),
           "val_f1": None, "per_class": None, "cm": None, "error": ""}
    if status == "ok":
        m, problem = TASK.score(exp_dir / f"s{seed}" / "preds.csv", "val")
        if problem:
            rec["status"], rec["error"] = "bad output", problem
        else:
            rec["val_f1"] = m["primary"]   # the task's main metric (macro-F1 or mAP@0.5)
            rec["per_class"] = m["per_class"]
            rec["cm"] = m["cm"]
            if m.get("diag"):
                rec["diag"] = m["diag"]
    elif status == "timeout":
        rec["error"] = (f"Killed after {args.seconds + args.grace:.0f} s (budget {args.seconds:.0f} s). "
                        f"Last log lines:\n{tail(exp_dir / f's{seed}.log', 15)}")
    else:
        rec["error"] = tail(exp_dir / f"s{seed}.log", 25)
    shown = f"{rec['val_f1']:.4f}" if rec["val_f1"] is not None else rec["status"].upper()
    print(f"  [{now()}] {exp_dir.name} seed {seed}: val {TASK.metric} {shown}  ({secs:.0f}s)", flush=True)
    return rec


# ------------------------------------------------------------------ edits
def parse_reply(text):
    base, category, hyp_lines, blocks = "best", "", [], []
    mode, search, replace, in_hyp = None, [], [], False
    for line in text.splitlines():
        s = line.strip()
        if mode is None:
            u = s.upper()
            if s.startswith("<<<<<<<"):
                mode, search, replace, in_hyp = "search", [], [], False
            elif u.startswith("BASE:"):
                base = "last" if "LAST" in u[5:] else "best"
                in_hyp = False
            elif u.startswith("CATEGORY:"):
                category = s[9:].strip().strip("*").strip().lower()
                in_hyp = False
            elif u.startswith("HYPOTHESIS:"):
                hyp_lines = [s[11:].strip()]
                in_hyp = True
            elif in_hyp and s and not s.startswith("```"):
                hyp_lines.append(s)
            continue
        if s.startswith("```"):  # code fences copied into a block
            continue
        if mode == "search" and s.startswith("=======") and set(s) == {"="}:
            mode = "replace"
        elif s.startswith(">>>>>>>"):
            blocks.append(("\n".join(search), "\n".join(replace)))
            mode = None
        elif mode == "search":
            search.append(line)
        else:
            replace.append(line)
    hyp = " ".join(h for h in hyp_lines if h).strip()
    return base, category[:40], hyp, blocks


def find_lines(src, search):
    """Tolerant match: ignore trailing spaces and blank lines at the edges of the SEARCH text."""
    sl = [l.rstrip() for l in search.splitlines()]
    while sl and not sl[0]:
        sl.pop(0)
    while sl and not sl[-1]:
        sl.pop()
    if not sl:
        return None, 0
    lines = src.splitlines(keepends=True)
    stripped = [l.rstrip() for l in lines]
    hits = [i for i in range(len(lines) - len(sl) + 1) if stripped[i:i + len(sl)] == sl]
    if len(hits) != 1:
        return None, len(hits)
    i = hits[0]
    a = sum(len(l) for l in lines[:i])
    b = a + sum(len(l) for l in lines[i:i + len(sl)])
    return (a, b), 1


def find_lines_dedent(src, search):
    """Last resort: match lines ignoring indentation; returns ((a, b), shift) if exactly one match.
    shift = spaces to add to every REPLACE line so it gets the file's indentation."""
    sl = [l.rstrip() for l in search.splitlines()]
    while sl and not sl[0]:
        sl.pop(0)
    while sl and not sl[-1]:
        sl.pop()
    if not sl:
        return None, 0
    lines = src.splitlines(keepends=True)
    key = [l.strip() for l in lines]
    want = [l.strip() for l in sl]
    hits = [i for i in range(len(lines) - len(sl) + 1) if key[i:i + len(sl)] == want]
    if len(hits) != 1:
        return None, 0
    i = hits[0]
    ind = lambda t: len(t) - len(t.lstrip(" "))
    f_src = next(l for l in lines[i:i + len(sl)] if l.strip())
    f_srch = next(l for l in sl if l.strip())
    shift = ind(f_src) - ind(f_srch)
    # all non-blank lines must be shifted by the same amount, otherwise it is not just an indentation slip
    if any(l.strip() and ind(a) - ind(l) != shift for a, l in zip(lines[i:i + len(sl)], sl)):
        return None, 0
    a = sum(len(l) for l in lines[:i])
    return (a, a + sum(len(l) for l in lines[i:i + len(sl)])), shift


def reindent(text, shift):
    out = []
    for l in text.splitlines(keepends=True):
        if not l.strip() or shift == 0:
            out.append(l)
        elif shift > 0:
            out.append(" " * shift + l)
        else:
            k = min(-shift, len(l) - len(l.lstrip(" ")))
            out.append(l[k:])
    return "".join(out)


def apply_edits(src, blocks):
    if not blocks:
        return None, "Your reply contained no SEARCH/REPLACE block."
    for i, (s, r) in enumerate(blocks, 1):
        if not s.strip():
            return None, f"Block {i}: the SEARCH part is empty. Put existing lines from the file there."
        n = src.count(s)
        if n == 1:
            src = src.replace(s, r, 1)
            continue
        if n > 1:
            return None, f"Block {i}: the SEARCH text appears {n} times; add surrounding lines to make it unique."
        pos, hits = find_lines(src, s)
        if pos is None and hits == 0:
            pos, shift = find_lines_dedent(src, s)
            if pos is not None:
                r = reindent(r, shift)
        if pos is None:
            first = next((l for l in s.splitlines() if l.strip()), "")
            why = f"appears {hits} times" if hits > 1 else "was not found"
            return None, (f"Block {i}: the SEARCH text {why} in the file (its first line: {first.strip()!r}). "
                          "Copy the lines exactly from the current file, including indentation.")
        a, b = pos
        rep = r if r.endswith("\n") or not r else r + "\n"
        src = src[:a] + rep + src[b:]
    return src, None


def udiff(a, b, name_a, name_b, limit=None):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), name_a, name_b, lineterm="", n=2))
    if limit and len(d) > limit:
        d = d[:limit] + [f"... ({len(d) - limit} more diff lines)"]
    return "\n".join(d)


# ------------------------------------------------------------------ LLM
def start_llm_server(args):
    """Start the LLM server for one call (so its GPU memory is free while training runs)."""
    log = open(Path(args.server_log), "a")
    p = subprocess.Popen(shlex.split(args.llm_server_cmd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = args.llm_url.rsplit("/v1", 1)[0]
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError(f"LLM server exited with code {p.returncode}; see {args.server_log}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return p
        except Exception:
            pass
        time.sleep(2)
    stop_llm_server(p)
    raise RuntimeError("LLM server did not become ready within 15 minutes")


def stop_llm_server(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


REMOTE_TIMEOUT = 900   # s without any data before a remote request counts as a dropped connection
DEADLINE = None        # end of the run (set in main): waiting for the network stops here


def network_up(host, port=443):
    import socket
    try:
        socket.create_connection((host, port), timeout=10).close()
        return True
    except OSError:
        return False


def wait_for_network(host, port=443):
    """Block until the API host is reachable again (phone unplugged, hotspot off, tunnel down). Training runs need no
    network, so nothing is lost: the same request is simply sent again. Gives up only on Ctrl+C / SIGINT or at the end
    of the run."""
    t0, said = time.time(), 0.0
    print(f"  [{now()}] no connection to {host}: waiting for the network (the run continues when it is back)", flush=True)
    while not network_up(host, port):
        if STOP:
            raise RuntimeError("stopped while waiting for the network")
        if DEADLINE is not None and datetime.now() > DEADLINE:
            raise RuntimeError("the run ended while waiting for the network")
        if time.time() - said > 600 and time.time() - t0 > 60:
            print(f"  [{now()}] still waiting for the network ({(time.time() - t0) / 60:.0f} min)", flush=True)
            said = time.time()
        time.sleep(30)
    if time.time() - t0 > 5:
        print(f"  [{now()}] network back after {(time.time() - t0) / 60:.1f} min, sending the request again", flush=True)


def post_json_retry(url, payload, headers, tries=6):
    """POST to a remote API. Rate limits and server errors: up to `tries` retries with backoff. Lost connection: wait
    for the network to come back, as long as it takes, then send the same request again (not counted as a try)."""
    u = urllib.parse.urlparse(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    delay, attempt, net_fail = 20, 0, 0
    while True:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=REMOTE_TIMEOUT) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            attempt += 1
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < tries:
                print(f"  API {e.code}, retrying in {delay}s: {body[:120]}", flush=True)
                time.sleep(delay); delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"API error {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if STOP:
                raise
            if not network_up(host, port):
                wait_for_network(host, port)
                net_fail = 0
                continue
            net_fail += 1          # host reachable but the request failed: a server-side problem, bounded retries
            if net_fail < tries:
                print(f"  network problem ({e}), retrying in {delay}s", flush=True)
                time.sleep(delay); delay = min(delay * 2, 300)
                continue
            raise


def openai_chat(args, messages):
    """OpenAI-compatible chat: a local llama-server started for this call (llm_server_cmd), or a remote API
    (no server command; key read from the environment variable named in api_key_env, e.g. NVIDIA_API_KEY)."""
    key_env = getattr(args, "api_key_env", None)
    server = start_llm_server(args) if args.llm_server_cmd else None
    try:
        payload = {"model": args.llm, "messages": messages, "temperature": getattr(args, "temperature", 0.6),
                   "max_tokens": args.max_tokens,
                   "stream": False}
        if args.reasoning_effort and not key_env:
            payload["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
        payload.update(getattr(args, "extra_body", None) or {})
        headers = {"Content-Type": "application/json"}
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"{key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
            d = post_json_retry(args.llm_url + "/chat/completions", payload, headers)
        else:
            req = urllib.request.Request(args.llm_url + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=3600) as r:
                d = json.loads(r.read())
    finally:
        stop_llm_server(server)
    ch = d["choices"][0]
    msg = ch.get("message", {})
    content, thinking = msg.get("content") or "", msg.get("reasoning_content") or ""
    if "</think>" in content:  # reasoning returned inline
        t, content = content.split("</think>", 1)
        thinking += t.replace("<think>", "")
    u = d.get("usage", {})
    raw = {"done_reason": "length" if ch.get("finish_reason") == "length" else "stop",
           "prompt_eval_count": u.get("prompt_tokens"), "eval_count": u.get("completion_tokens"),
           "timings": d.get("timings")}
    return content.strip(), thinking, raw


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def anthropic_chat(args, messages):
    """Claude via the Messages API (reference runs). Key from ANTHROPIC_API_KEY; adaptive thinking with summarized
    thinking text; effort from --reasoning-effort (default high). Retries network errors and overload, since the
    station is on a phone hotspot. The training sandbox never sees the key (bwrap --clearenv, no network)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    payload = {"model": args.llm, "max_tokens": args.max_tokens, "system": system, "messages": msgs,
               "thinking": {"type": "adaptive", "display": "summarized"},
               "output_config": {"effort": args.reasoning_effort or "high"}}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    delay = 20
    for attempt in range(6):
        req = urllib.request.Request(ANTHROPIC_URL, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 5:
                print(f"  API {e.code}, retrying in {delay}s: {body[:120]}", flush=True)
                time.sleep(delay); delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"API error {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < 5:
                print(f"  network problem ({e}), retrying in {delay}s", flush=True)
                time.sleep(delay); delay = min(delay * 2, 300)
                continue
            raise
    content = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    thinking = "\n".join(b.get("thinking", "") for b in d.get("content", []) if b.get("type") == "thinking")
    u = d.get("usage", {})
    raw = {"done_reason": "length" if d.get("stop_reason") == "max_tokens" else "stop",
           "prompt_eval_count": u.get("input_tokens"), "eval_count": u.get("output_tokens")}
    return content.strip(), thinking, raw


def llm_chat(args, messages):
    if args.backend == "openai":
        return openai_chat(args, messages)
    if args.backend == "anthropic":
        return anthropic_chat(args, messages)
    opts = {"temperature": 0.6, "num_ctx": args.ctx, "num_predict": args.max_tokens}
    payload = {"model": args.llm, "messages": messages, "stream": False, "keep_alive": 0, "options": opts}
    think = args.think
    if think == "auto":
        think = "medium" if args.llm.startswith("gpt-oss") else True
    if think not in ("off", False):
        payload["think"] = think
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if "think" in payload and e.code == 400:
            payload.pop("think")
            req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                d = json.loads(r.read())
        else:
            raise
    m = d.get("message", {})
    return m.get("content", "") or "", m.get("thinking", "") or "", d


def ollama_unload(model):
    try:
        req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass


def history_table(state, limit=40):
    rows = [f"id | base | category | hypothesis | seed-0 score | {NSEEDS}-seed mean | outcome"]
    exps = state.exps[1:]
    shown = exps[-limit:]
    if len(exps) > limit:
        rows.append(f"(... {len(exps) - limit} older experiments not shown)")
    for e in shown:
        s0 = f"{e['seed0']:.4f}" if e.get("seed0") is not None else "-"
        mn = f"{e['mean']:.4f}" if e.get("mean") is not None else "-"
        h = e["hypothesis"].replace("\n", " ")[:220]
        rows.append(f"{e['id']} | {e['base_id']} | {e['category'] or '-'} | {h} | {s0} | {mn} | {e['outcome']}")
    return "\n".join(rows)


def lessons_table(state):
    """Per-category summary of what was tried so far, so repeated dead ends are visible at a glance."""
    from collections import OrderedDict
    cats = OrderedDict()
    for e in state.exps[1:]:
        c = cats.setdefault(e.get("category") or "-", {"n": 0, "ran": 0, "best": 0, "close": 0, "fail": 0})
        c["n"] += 1
        c["ran"] += e.get("seed0") is not None
        c["best"] += e["outcome"] == "new best"
        c["close"] += e["outcome"] == "not confirmed"
        c["fail"] += e["outcome"] in ("crashed", "timeout", "bad output", "suspicious (likely bug)", "edit failed")
    rows = ["category | ideas | ran | new best | close but not confirmed | crashed/timeout/bug/edit failed | status"]
    for k, c in cats.items():
        status = "EXHAUSTED? (3+ ideas, no new best)" if c["n"] >= 3 and c["best"] == 0 else ""
        rows.append(f"{k} | {c['n']} | {c['ran']} | {c['best']} | {c['close']} | {c['fail']} | {status}")
    bugs = [e for e in state.exps[1:] if e["outcome"] == "suspicious (likely bug)"]
    if bugs:
        rows.append("Likely-bug results so far (do not repeat these without fixing the cause): "
                    + "; ".join(f"{e['id']}: {e['hypothesis'][:70]}" for e in bugs))
    return "\n".join(rows)


def diagnostics_text(best):
    """Average the diagnostics of the best version's seed runs into a short prompt block."""
    ds = [r["diag"] for r in best["runs"] if r["status"] == "ok" and r.get("diag") and "cm" in r["diag"]][:NSEEDS]
    if not ds:
        return ""
    n = len(ds)
    cm = [[sum(d["cm"][i][j] for d in ds) / n for j in range(5)] for i in range(4)]
    head = "".join(f"{c[:8]:>9}" for c in CLASSES) + "  no row"
    lines = [f"## Diagnostics of the current best (validation, mean of {n} seed runs, computed by the harness)",
             "Class chosen (highest-scoring row) on detector boxes that overlap a true defect by IoU > 0.5;",
             "rows = true class, columns = chosen class:",
             f"  {'':<18}" + head]
    for i, c in enumerate(CLASSES):
        tot = sum(cm[i][:5])
        right = f"({cm[i][i] / tot:.0%} right)" if tot else "(none)"
        lines.append(f"  true {c:<13}" + "".join(f"{v:>9.0f}" for v in cm[i]) + f"   {right}")
    acc = ds[0].get("acc_by_size") or {}
    if acc:
        avg = {k: st.mean(d["acc_by_size"][k] for d in ds if d.get("acc_by_size", {}).get(k) is not None) for k in acc}
        lines.append("Share of those boxes with the right class, by defect size (area thirds): "
                     + ", ".join(f"{k} {v:.0%}" for k, v in avg.items()))
    lines.append(f"True defects that no detector box overlaps by IoU > 0.5 (stage 1, cannot be fixed here): "
                 f"{st.mean(d['missed_by_detector'] for d in ds):.0f}")
    mp = [d["map_if_classes_perfect"] for d in ds if d.get("map_if_classes_perfect") is not None]
    if mp:
        lines.append(f"mAP@0.5 if every box had its correct class but kept your scores: {st.mean(mp):.4f} "
                     f"(yours: {best['mean']:.4f}). The difference is what wrong classes cost; the remaining gap to the "
                     "perfect-classes ceiling in the brief is what your scores cost compared with the detector's conf.")
    return "\n".join(lines)


def best_details(state, best):
    ok = [r for r in best["runs"] if r["status"] == "ok"][:NSEEDS]
    pc = ", ".join(f"{k} {st.mean(r['per_class'][k] for r in ok):.3f}" for k in CLASSES if k in ok[0]["per_class"])
    if not ok[0].get("cm"):
        return pc, ""
    cm = [[sum(r["cm"][i][j] for r in ok) for j in range(4)] for i in range(4)]
    cm_txt = "\n".join(f"  true {CLASSES[i]:<13} " + " ".join(f"{v:>5}" for v in cm[i]) for i in range(4))
    return pc, cm_txt


def build_prompt(state, P, args, program):
    best = state.best()
    src = (state.dir / best["id"] / "train.py").read_text()
    pc, cm_txt = best_details(state, best)
    unit = "F1" if TASK.name == "cls" else "AP@0.5"
    parts = [f"## Current best train.py ({best['id']}, {NSEEDS}-seed val {TASK.metric} {best['mean']:.4f}, "
             f"seed std {best['std']:.4f})", "```python", src.rstrip(), "```", "",
             f"Per-class val {unit} (mean of {NSEEDS} seeds): {pc}"]
    if cm_txt:
        parts += [f"Val confusion matrix, summed over {NSEEDS} seeds (rows = true class, columns = predicted "
                  f"{', '.join(CLASSES)}):", cm_txt]
    parts += ["",
             f"Log of its seed-0 run (last lines):\n{tail(state.dir / best['id'] / 's0.log', 8)}", "",
             f"## Experiment history ({len(state.exps) - 1} experiments so far)", history_table(state)]
    if getattr(args, "diagnostics", False):
        dt = diagnostics_text(best)
        if dt:
            parts += ["", dt]
    if getattr(args, "lessons", False) and len(state.exps) > 2:
        parts += ["", "## Lessons so far, by category (computed by the harness)", lessons_table(state)]
    last = state.exps[-1] if len(state.exps) > 1 else None
    if last:
        parts += ["", f"## Most recent experiment: {last['id']} ({last['outcome']})"]
        a = last.get("accept")
        if a and a.get("se_img") is not None:
            parts.append(f"Its {NSEEDS}-seed gain over {a['vs']}: {a['gain']:+.4f}. A new best needed more than "
                         f"{a['needed']:.4f} = {a['k']:g} x the uncertainty of the gain (seeds ±{a['se_seed']:.4f}, "
                         f"validation images ±{a['se_img']:.4f}). A gain carried by a few images or one class "
                         "does not count.")
        if a and a.get("confirm_ok") is False:
            parts.append("It passed on the validation images you see, but did not hold on a separate set of held-out "
                         "validation images that you never see, so it was not accepted. Gains that depend on the "
                         "particular images you are scored on are not real improvements.")
        if (state.dir / last["id"] / "diff.patch").exists():
            dl = (state.dir / last["id"] / "diff.patch").read_text().splitlines()
            parts += ["Its change, relative to its base " + last["base_id"] + ":",
                      "\n".join(dl[:120] + ([f"... ({len(dl) - 120} more diff lines)"] if len(dl) > 120 else []))]
        err = next((r["error"] for r in last["runs"] if r["error"]), "") or last.get("edit_error", "")
        if err:
            parts += ["What went wrong:", err]
        if (state.dir / last["id"] / "train.py").exists() and last["id"] != best["id"]:
            parts += ["", "If you want to build on (or fix) this most recent version instead of the best, "
                          "write BASE: last. Its full train.py differs from the best only by the change above."]
    parts += ["", "Propose the next experiment, in the reply format."]
    return [{"role": "system", "content": program}, {"role": "user", "content": "\n".join(parts)}]


PLANNER_ASK = """You are the research lead. Choose the next experiment, following the research method above.
Reply in this format and nothing else:
BASE: best            (or: last)
CATEGORY: one of data, augmentation, architecture, loss, sampling, optimisation, inference, scoring, other
HYPOTHESIS: two or three sentences: the change, why it should help, the expected gain, how it differs from earlier experiments.
PLAN: a concrete, numbered description of the code changes: which constants, functions, classes and loops of the
current train.py change and how, what the new code must compute, and the time-budget arithmetic.
Do NOT write code and do NOT write SEARCH/REPLACE blocks: an engineer will implement your plan exactly."""

IMPLEMENTER_ASK = """## The next experiment, chosen by the research lead
{plan}

Implement exactly this experiment in the current train.py, nothing more. Reply in the reply format: copy BASE,
CATEGORY and HYPOTHESIS from the lead, then the SEARCH/REPLACE blocks. If a detail of the plan cannot work as written
(time budget, tensor shapes, missing names), make the smallest fix that keeps the idea and say so at the end of the
HYPOTHESIS. Check every changed function end to end: what the dataset returns, what the loops unpack, what is used."""


CRITIC_ASK = """## Proposed next experiment (from the research lead)
{plan}

You are the reviewer. Judge this proposal against the history, the lessons table and the results of earlier runs
above. Check in particular:
1. Does it repeat an idea family that already failed three or more times, or is marked "no gain", without a
   fundamentally different mechanism?
2. Is the expected effect broad (several classes, many images), or would it be carried by one small class or a
   handful of images? Gains like that are not accepted and rarely hold on new data.
3. Does it fit the 180 s budget, prediction included? Estimate the cost of every added view, resolution or model.
4. What is most likely to break in the implementation?
Reply in this format and nothing else:
VERDICT: APPROVE  (or: VERDICT: REVISE)
REASONS: up to five short bullet points. For REVISE, say what to change or what to try instead."""

REVISE_ASK = """The reviewer asked for a revision:
{critique}

Revise the experiment, or choose a different one if the reviewer is right that it cannot work. Reply in the same
format as before (BASE, CATEGORY, HYPOTHESIS, PLAN), with no code."""


def has_role(args, role):
    """Is a planner / critic model configured, either local (server command) or remote (API URL)?"""
    return bool(getattr(args, f"{role}_server_cmd", None) or getattr(args, f"{role}_url", None))


def role_args(args, role):
    """Call settings for the planner or critic: a local llama-server, or a remote OpenAI-compatible API
    (e.g. NVIDIA NIM) with the key taken from an environment variable. The key itself is never stored."""
    r = argparse.Namespace(**vars(args))
    url = getattr(args, f"{role}_url", None)
    r.llm_server_cmd = None if url else getattr(args, f"{role}_server_cmd")
    r.llm_url = url.rstrip("/") if url else args.llm_url
    r.api_key_env = getattr(args, f"{role}_key_env") if url else None
    r.llm = getattr(args, f"{role}_llm")
    r.temperature = getattr(args, f"{role}_temperature")
    r.reasoning_effort = args.critic_effort if role == "critic" else None
    r.extra_body = json.loads(getattr(args, f"{role}_extra", None) or "{}")
    if getattr(args, f"{role}_max_tokens", None):
        r.max_tokens = getattr(args, f"{role}_max_tokens")
    return r


def chat_as(args, role, msgs, state, exp_dir, attempt=0):
    """One call to a role model (planner / critic). Returns content or None."""
    rargs = role_args(args, role)
    llm = rargs.llm
    t0 = time.time()
    try:
        content, thinking, raw = openai_chat(rargs, msgs)
    except Exception as e:
        state.event({"time": now(), "type": "llm_error", "role": role, "exp": exp_dir.name, "error": str(e)})
        print(f"  {role} call failed: {e}", flush=True)
        return None
    secs = time.time() - t0
    state.event({"time": now(), "type": "llm_call", "role": role, "exp": exp_dir.name, "attempt": attempt,
                 "seconds": round(secs, 1), "prompt_tokens": raw.get("prompt_eval_count"),
                 "output_tokens": raw.get("eval_count"), "ok": bool(content), "problem": ""})
    print(f"  {role} ({llm}): {secs:.0f}s", flush=True)
    return content.strip() if content else None


def review_plan(state, P, args, messages, plan, exp_dir, pm):
    """Trio mode: the critic reviews the lead's plan; on REVISE the lead revises once. Returns the final plan."""
    body = messages[1]["content"].rsplit("Propose the next experiment, in the reply format.", 1)[0]
    cm = [messages[0], {"role": "user", "content": body + CRITIC_ASK.format(plan=plan)}]
    critique = chat_as(args, "critic", cm, state, exp_dir)
    if not critique:
        return plan
    (exp_dir / "critique.txt").write_text(critique)
    verdict = "REVISE" if re.search(r"VERDICT:\s*REVISE", critique, re.I) else "APPROVE"
    state.event({"time": now(), "type": "review", "exp": exp_dir.name, "verdict": verdict})
    print(f"  critic verdict: {verdict}", flush=True)
    if verdict == "APPROVE":
        return plan
    rm = pm + [{"role": "assistant", "content": plan}, {"role": "user", "content": REVISE_ASK.format(critique=critique)}]
    revised = chat_as(args, "planner", rm, state, exp_dir, attempt=1)
    if revised and "HYPOTHESIS:" in revised.upper() and "PLAN:" in revised.upper():
        (exp_dir / "plan_revised.txt").write_text(revised)
        return revised
    return plan


def plan_with_lead(state, P, args, messages, exp_dir):
    """Pair mode: the planner model chooses the experiment and writes a plan (no code). Returns the plan text."""
    pm = [messages[0], {"role": "user", "content": messages[1]["content"].rsplit(
        "Propose the next experiment, in the reply format.", 1)[0] + PLANNER_ASK}]
    pargs = role_args(args, "planner")
    for attempt in range(2):
        t0 = time.time()
        try:
            content, thinking, raw = openai_chat(pargs, pm)
        except Exception as e:
            state.event({"time": now(), "type": "llm_error", "role": "planner", "exp": exp_dir.name, "error": str(e)})
            print(f"  planner call failed: {e}", flush=True)
            time.sleep(15)
            continue
        secs = time.time() - t0
        (exp_dir / "plan.txt").write_text(content)
        if thinking:
            (exp_dir / "plan_thinking.txt").write_text(thinking)
        ok = "HYPOTHESIS:" in content.upper() and "PLAN:" in content.upper()
        state.event({"time": now(), "type": "llm_call", "role": "planner", "exp": exp_dir.name, "attempt": attempt,
                     "seconds": round(secs, 1), "prompt_tokens": raw.get("prompt_eval_count"),
                     "output_tokens": raw.get("eval_count"), "ok": ok, "problem": "" if ok else "no plan"})
        print(f"  planner ({args.planner_llm}): {secs:.0f}s" + ("" if ok else ", reply has no HYPOTHESIS/PLAN"), flush=True)
        if ok:
            if has_role(args, "critic"):
                return review_plan(state, P, args, messages, content.strip(), exp_dir, pm)
            return content.strip()
    return None


def propose(state, P, args, program, exp_dir):
    """Ask the LLM for one idea + edits, apply them. Returns dict with src/base/category/hypothesis or error."""
    messages = build_prompt(state, P, args, program)
    if has_role(args, "planner"):
        plan = plan_with_lead(state, P, args, messages, exp_dir)
        if plan is None:
            return {"error": "the research lead gave no usable plan", "category": "", "hypothesis": "",
                    "base_id": state.best()["id"]}
        messages[1]["content"] = (messages[1]["content"].rsplit("Propose the next experiment, in the reply format.", 1)[0]
                                  + IMPLEMENTER_ASK.format(plan=plan))
    (exp_dir / "prompt.txt").write_text("\n\n".join(f"### {m['role']}\n{m['content']}" for m in messages))
    best = state.best()
    last = state.exps[-1]
    err, out = "", None
    for attempt in range(3):
        t0 = time.time()
        try:
            content, thinking, raw = llm_chat(args, messages)
        except Exception as e:
            secs = time.time() - t0
            state.event({"time": now(), "type": "llm_error", "exp": exp_dir.name, "seconds": round(secs, 1), "error": str(e)})
            print(f"  LLM call failed: {e}", flush=True)
            time.sleep(15)
            err = f"LLM call failed: {e}"
            continue
        secs = time.time() - t0
        with open(exp_dir / "reply.txt", "a") as f:
            f.write(f"===== attempt {attempt + 1} ({secs:.0f}s) =====\n{content}\n")
        if thinking:
            with open(exp_dir / "thinking.txt", "a") as f:
                f.write(f"===== attempt {attempt + 1} =====\n{thinking}\n")
        base, category, hyp, blocks = parse_reply(content)
        base_exp = last if base == "last" and (state.dir / last["id"] / "train.py").exists() else best
        src0 = (state.dir / base_exp["id"] / "train.py").read_text()
        new, err = apply_edits(src0, blocks)
        if new is not None and new == src0:
            new, err = None, "The edits do not change the file."
        if new is not None:
            try:
                compile(new, "train.py", "exec")
            except SyntaxError as e:
                new, err = None, f"After your edits, train.py has a syntax error: {e.msg} (line {e.lineno}): {e.text!r}"
        if new is not None and not hyp:
            new, err = None, "Your reply has no HYPOTHESIS line."
        if raw.get("done_reason") == "length" and new is None:
            err = "Your reply was cut off (too long). Keep the edits short and focused. " + (err or "")
        state.event({"time": now(), "type": "llm_call", "exp": exp_dir.name, "attempt": attempt, "seconds": round(secs, 1),
                     "prompt_tokens": raw.get("prompt_eval_count"), "output_tokens": raw.get("eval_count"),
                     "ok": new is not None, "problem": err or ""})
        print(f"  LLM reply {attempt + 1}: {secs:.0f}s, " + ("edits applied" if new is not None else f"problem: {err[:150]}"),
              flush=True)
        if new is not None:
            out = {"src": new, "base_id": base_exp["id"], "category": category, "hypothesis": hyp[:600],
                   "diff": udiff(src0, new, f"{base_exp['id']}/train.py", f"{exp_dir.name}/train.py")}
            break
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": err + " Reply again in the reply format, with corrected edits."}]
        out = {"error": err, "category": category, "hypothesis": hyp[:600], "base_id": base_exp["id"]}
    return out or {"error": err, "category": "", "hypothesis": "", "base_id": best["id"]}


def reproduce(state, P, args):
    """Independent re-run: check the stored files against their recorded hashes, then run the start file and the
    final best again with fresh seeds (100, 101, ...) in a separate folder and score both on the hidden test set."""
    base, best = state.exps[0], state.best()
    res = {}
    for e in [base] + ([best] if best["id"] != base["id"] else []):
        src = state.dir / e["id"] / "train.py"
        h = sha256_file(src)
        d = state.dir / "repro" / e["id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / "train.py")
        val, test = [], []
        for seed in range(100, 100 + args.reproduce):
            if STOP:
                break
            rec = run_seed(P, args, d, seed)
            if rec["status"] == "ok":
                m, _ = TASK.score(d / f"s{seed}" / "preds.csv", "test")
                if m:
                    val.append(rec["val_f1"]); test.append(m["primary"])
        res[e["id"]] = {"sha256": h, "matches_record": (e.get("sha256") in (None, h)), "val": val, "test": test}
    (state.dir / "repro.json").write_text(json.dumps(res, indent=1))
    orig = json.loads((state.dir / "final.json").read_text()) if (state.dir / "final.json").exists() else {}
    L = ["", "| version | file hash ok | original test (8 seeds) | reproduction test (fresh seeds) |", "|---|---|---|---|"]
    for eid, r in res.items():
        o = orig.get(eid, {}).get("test", [])
        fmt = lambda x: (f"{mean_se(x)[0]:.4f} ± {mean_se(x)[1]:.4f} (n={len(x)})" if len(x) > 1 else "-")
        L.append(f"| {eid} | {'yes' if r['matches_record'] else 'NO'} | {fmt(o)} | {fmt(r['test'])} |")
    if len(res) == 2 and all(len(r["test"]) > 1 for r in res.values()):
        (b, bt), (x, xt) = [(k, v["test"]) for k, v in res.items()]
        dlt = st.mean(xt) - st.mean(bt)
        se = math.sqrt(st.variance(xt) / len(xt) + st.variance(bt) / len(bt))
        L.append(f"\nReproduced test gain of {x} over the start file: {dlt:+.4f} ± {se:.4f} ({dlt / se:+.1f} SE)")
    (state.dir / "repro.md").write_text("# Reproduction of " + state.dir.name + "\n" + "\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def reuse_baseline(state, P, args):
    """Copy the start file's runs (validation seeds and, if present, final test seeds) from an earlier run with the
    same start file, budget and seed count, instead of measuring them again."""
    src = P.root / "runs" / args.reuse_baseline
    try:
        sexp = json.loads((src / "experiments.json").read_text())[0]
        smeta = json.loads((src / "meta.json").read_text())
    except Exception as e:
        print(f"Cannot reuse the baseline of {src}: {e}; measuring it instead.", flush=True)
        return False
    same = (sha256_file(src / "e000" / "train.py") == sha256_file(args.start) and smeta.get("seconds") == args.seconds
            and smeta.get("seeds") == NSEEDS and smeta.get("task") == args.task and sexp.get("outcome") == "baseline")
    if not same:
        print(f"Baseline of {src.name} differs (start file, budget, seeds or task); measuring it instead.", flush=True)
        return False
    shutil.copytree(src / "e000", P.dir / "e000", dirs_exist_ok=True)
    sexp["reused_from"] = src.name
    for r in sexp["runs"]:   # recompute diagnostics on the copied predictions (validation only)
        if r["status"] == "ok" and (HOLD is not None or not r.get("diag")):
            m, _ = TASK.score(P.dir / "e000" / f"s{r['seed']}" / "preds.csv", "val")
            if m and HOLD is not None:   # v3.2: scores on the selection images only
                r["val_f1"], r["per_class"] = m["primary"], m["per_class"]
            if m and m.get("diag"):
                r["diag"] = m["diag"]
    if HOLD is not None:
        summarize_runs(sexp)
    state.exps.append(sexp)
    state.save()
    print(f"Baseline reused from {src.name} ({len(sexp['runs'])} runs, no new training).", flush=True)
    return True


def rescore(args):
    """Re-judge every multi-seed candidate of earlier runs with seeds + validation-image uncertainty, and show the
    final best's test gain with its image uncertainty. Uses the saved predictions only (no training, no LLM)."""
    k = args.accept_se if args.accept_se > 0 else 1.0
    L = [f"# Rescoring with image bootstrap (k = {k:g})", "",
         "| run | exp | vs | val gain | needed, seeds only | needed, seeds + images | images SE | v2 verdict | "
         "v3.1 verdict | test gain (final best) |", "|---|---|---|---|---|---|---|---|---|---|"]
    finals = []
    for name in [x.strip() for x in args.rescore.split(",") if x.strip()]:
        P = Paths(args.root, name, TASK)
        state = State(P.dir)
        if not state.exps:
            L.append(f"| {name} | (no experiments.json) | | | | | | | | |")
            continue
        TASK.load(P)
        TASK.__dict__.pop("_parts", None)
        fj = P.dir / "final.json"
        res = json.loads(fj.read_text()) if fj.exists() else {}
        base = state.exps[0]
        for e in state.exps[1:]:
            if e.get("mean") is None or e["outcome"] not in ("new best", "not confirmed"):
                continue
            best = max((x for x in state.exps if x["outcome"] in ("baseline", "new best") and x["id"] < e["id"]),
                       key=lambda x: x["id"])
            n = NSEEDS
            var = lambda x: (x["std"] or 0.0) ** 2 * n / max(n - 1, 1)
            se_seed = math.sqrt(var(e) / n + var(best) / n)
            gain = e["mean"] - best["mean"]
            se_img = image_se(P.dir, e, best)
            se = math.sqrt(se_seed ** 2 + (se_img or 0) ** 2)
            tg = "-"
            if e["id"] in res and base["id"] in res:
                d = [b - a for a, b in zip(res[base["id"]]["test"], res[e["id"]]["test"])]
                tg = f"{st.mean(d):+.4f}"
            L.append(f"| {name} | {e['id']} | {best['id']} | {gain:+.4f} | {k * se_seed:.4f} | "
                     + (f"{k * se:.4f} | {se_img:.4f}" if se_img is not None else "- | -")
                     + f" | {'accept' if gain > k * se_seed else 'reject'} | "
                     + (('accept' if gain > k * se else 'reject') if se_img is not None else '-') + f" | {tg} |")
            print(L[-1], flush=True)
        best = state.best()
        if best and best["id"] != base["id"] and best["id"] in res:
            nf = min(len(res[base["id"]]["test"]), len(res[best["id"]]["test"]))
            sv = image_se(P.dir, best, base, "val", n=nf)
            stt = image_se(P.dir, best, base, "test", n=nf)
            d = [b - a for a, b in zip(res[base["id"]]["test"], res[best["id"]]["test"])]
            sd = st.stdev(d) / len(d) ** 0.5
            finals.append(f"| {name} | {best['id']} | {st.mean(d):+.4f} | {sd:.4f} | {stt:.4f} | "
                          f"{math.sqrt(sd ** 2 + stt ** 2):.4f} | {sv:.4f} |")
    if finals:
        L += ["", f"## Final best vs start file on the hidden test set ({NSEEDS}+ seeds, paired)", "",
              "| run | final best | test gain | SE seeds | SE test images | SE total | SE val images |",
              "|---|---|---|---|---|---|---|"] + finals
    L += ["", "v2 verdict: gain > k x seed SE (what the run used). v3.1 verdict: gain > k x sqrt(seed SE² + image SE²), "
              "image SE from a paired bootstrap over validation images (400 samples)."]
    out = Path(args.root) / "rescore.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L) + f"\n\nWritten to {out}", flush=True)


# ------------------------------------------------------------------ final evaluation and report
def final_eval(state, P, args):
    base, best = state.exps[0], state.best()
    cands = [base] + ([best] if best["id"] != base["id"] else [])
    res = {}
    for e in cands:
        d = state.dir / e["id"]
        have = {r["seed"] for r in e["runs"] if r["status"] == "ok"}
        for seed in range(NSEEDS, NSEEDS + args.extra_seeds):
            if seed in have or STOP:
                continue
            rec = run_seed(P, args, d, seed)
            rec["final"] = True
            e["runs"].append(rec)
            state.save()
        val, test, conf = [], [], []
        for r in e["runs"]:
            if r["status"] != "ok":
                continue
            m, _ = TASK.score(d / f"s{r['seed']}" / "preds.csv", "test")
            if m:
                val.append(r["val_f1"])
                test.append(m["primary"])
                if HOLD is not None:
                    conf.append(TASK.score(d / f"s{r['seed']}" / "preds.csv", "confirm")[0]["primary"])
        res[e["id"]] = {"val": val, "test": test}
        if HOLD is not None:
            res[e["id"]]["confirm"] = conf
    if len(cands) == 2 and TASK.name == "det":   # v3.1: image-level uncertainty of the final gain (val and test)
        try:
            n = min(len(res[base["id"]]["test"]), len(res[best["id"]]["test"]))
            res["image_se"] = {"val": image_se(state.dir, best, base, "val", n=n),
                               "test": image_se(state.dir, best, base, "test", n=n)}
            if HOLD is not None:
                res["image_se"]["confirm"] = image_se(state.dir, best, base, "confirm", n=n)
        except Exception as ex:
            print(f"image bootstrap of the final gain failed: {ex}", flush=True)
    (state.dir / "final.json").write_text(json.dumps(res, indent=1))
    return res


def mean_se(xs):
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan"))


def write_report(state, P, args):
    exps = state.exps
    if not exps:
        print("Nothing to report yet.")
        return
    base, best = exps[0], state.best()
    ideas = exps[1:]
    count = lambda o: sum(1 for e in ideas if e["outcome"] == o)
    calls = [e for e in state.events if e["type"] == "llm_call"]
    llm_s = sum(e["seconds"] for e in calls)
    run_s = sum(r["seconds"] for e in exps for r in e["runs"])
    meta = json.loads((state.dir / "meta.json").read_text())
    L = [f"# AutoResearch report: {state.dir.name}", "",
         f"- Task: {meta.get('task', 'cls')} ({TASK.metric}); LLM: **{meta['llm']}** (thinking: {meta['think']}), "
         f"training budget {meta['seconds']} s wall time per run, {NSEEDS} seeds per confirmed result",
         f"- Started {meta.get('started', '?')}, report written {now()}",
         f"- Harness: brief {meta.get('program', '?')}; acceptance "
         + (f"gain > {meta['accept_se']} SE" if meta.get('accept_se') else "any higher mean")
         + ("; second seed for close calls" if meta.get('second_look') else "")
         + ("; lessons table in prompt" if meta.get('lessons') else "")
         + ("; diagnostics in prompt" if meta.get('diagnostics') else "")
         + (f"; background digest {meta['digest']}" if meta.get('digest') else "")
         + (f"; start-file runs reused from {meta['reuse_baseline']}" if meta.get('reuse_baseline') else "")
         + ("; uncertainty includes validation images (bootstrap)" if meta.get('accept_boot') else "")
         + (f"; memory of earlier runs {meta['memory']}" if meta.get('memory') else "")
         + (f"; research guide {meta['guide']}" if meta.get('guide') else "")
         + (f"; PAIR MODE: {meta['planner']} plans each experiment, {meta['llm']} implements it"
            if meta.get('planner') else "")
         + (f"; TRIO: {meta['critic']} reviews each plan first" if meta.get('critic') else "")
         + (f"; HELD-OUT CONFIRMATION: {meta['holdout_images'][1]} of {sum(meta['holdout_images'])} validation images never "
            f"shown to the agent, a change must also gain > {meta['confirm_k']:g} SE on them (split seed {meta['holdout_seed']})"
            if meta.get('holdout') else ""),
         f"- Ideas tried: {len(ideas)} | edits unusable: {count('edit failed')} | crashed/timeout/bad output: "
         f"{sum(1 for e in ideas if e['outcome'] in ('crashed', 'timeout', 'bad output'))} | likely bugs (score far below best): "
         f"{count('suspicious (likely bug)')} | ran but worse: {count('rejected')} | "
         f"not confirmed: {count('not confirmed')} | **confirmed improvements: {count('new best')}**",
         f"- LLM time: {llm_s / 3600:.1f} h in {len(calls)} calls ({llm_s / max(len(calls), 1):.0f} s per call); "
         f"training time: {run_s / 3600:.1f} h"
         + (f"; API cost about ${sum((e.get('prompt_tokens') or 0) * 4 + (e.get('output_tokens') or 0) * 20 for e in calls) / 1e6:.2f}"
            f" (at $4 / $20 per million tokens)" if meta.get("backend") == "anthropic" else ""), ""]
    cats = {}
    for e in ideas:
        c = cats.setdefault(e["category"] or "-", {"n": 0, "ran": 0, "new best": 0})
        c["n"] += 1
        c["ran"] += e.get("seed0") is not None
        c["new best"] += e["outcome"] == "new best"
    if cats:
        L += ["| category | ideas | ran | confirmed |", "|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['ran']} | {v['new best']} |" for k, v in sorted(cats.items(), key=lambda x: -x[1]["n"])]
        L.append("")
    fj = state.dir / "final.json"
    if fj.exists():
        res = json.loads(fj.read_text())
        L += ["## Final evaluation (hidden test set, read only here)", "",
              f"| version | seeds | val {TASK.metric} | test {TASK.metric} | test gain vs start |", "|---|---|---|---|---|"]
        bt, bse = mean_se(res[base["id"]]["test"])
        img = res.pop("image_se", None)
        gse = None
        for eid, r in res.items():
            vm, vse = mean_se(r["val"])
            tm, tse = mean_se(r["test"])
            gain = "-"
            if eid != base["id"] and tm is not None and bt is not None:
                g, gse = tm - bt, (tse ** 2 + bse ** 2) ** 0.5
                gain = f"{g:+.4f} ± {gse:.4f} ({g / gse:+.1f} SE)" if gse > 0 else f"{g:+.4f}"
            label = "start file" if eid == base["id"] else f"final best ({eid})"
            L.append(f"| {label} | {len(r['test'])} | {vm:.4f} ± {vse:.4f} | {tm:.4f} ± {tse:.4f} | {gain} |")
        L += ["", "Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.", ""]
        if all("confirm" in res[k] for k in res if k != "image_se") and len([k for k in res if k != "image_se"]) == 2:
            cs = [mean_se(res[k]["confirm"]) for k in res if k != "image_se"]
            L += [f"Held-out validation half (never shown to the agent): start {cs[0][0]:.4f} ± {cs[0][1]:.4f}, final best "
                  f"{cs[1][0]:.4f} ± {cs[1][1]:.4f}, gain {cs[1][0] - cs[0][0]:+.4f}"
                  + (f" (images ±{img['confirm']:.4f})" if img and img.get("confirm") is not None else "") + ".", ""]
        if img and img.get("test") is not None:
            L += [f"Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±{img['val']:.4f}, "
                  f"test ±{img['test']:.4f}. Seeds and images together, the test gain's SE is "
                  f"{math.sqrt(gse ** 2 + img['test'] ** 2):.4f}." if gse is not None else
                  f"Uncertainty over images: validation ±{img['val']:.4f}, test ±{img['test']:.4f}.", ""]
    if best and best["id"] != base["id"]:
        for e, label in ((base, "start file"), (best, f"final best {best['id']}")):
            pc, _ = best_details(state, e)
            L.append(f"Per-class val score, {label}: {pc}")
        chain, e = [], best
        while e["id"] != base["id"]:
            chain.append(e["id"])
            e = state.get(e["base_id"])
        L += ["", f"Chain of accepted changes: {' -> '.join([base['id']] + chain[::-1])}", "",
              "## What changed from the start file to the final best", "", "```diff",
              udiff((state.dir / base["id"] / "train.py").read_text(), (state.dir / best["id"] / "train.py").read_text(),
                    "start/train.py", f"{best['id']}/train.py"), "```", ""]
    L += ["## All experiments", "", f"| id | base | category | hypothesis | seed-0 | {NSEEDS}-seed mean | outcome |",
          "|---|---|---|---|---|---|---|"]
    for e in exps:
        s0 = f"{e['seed0']:.4f}" if e.get("seed0") is not None else "-"
        mn = f"{e['mean']:.4f}" if e.get("mean") is not None else "-"
        h = e["hypothesis"].replace("|", "/").replace("\n", " ")[:200]
        L.append(f"| {e['id']} | {e['base_id']} | {e['category'] or '-'} | {h} | {s0} | {mn} | {e['outcome']} |")
    L += ["", f"Each experiment's folder ({state.dir}/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply "
              "and thinking, and the run logs."]
    pv = meta.get("provenance")
    if pv:
        L += ["", "## Provenance", "", "| item | value |", "|---|---|"]
        L += [f"| {k} | {v if not isinstance(v, dict) else ', '.join(f'{a} {b}' for a, b in v.items())} |"
              for k, v in pv.items()]
        if best:
            L.append(f"| train.py of start / final best | {base.get('sha256', '-')} / {best.get('sha256', '-')} |")
        if base.get("reused_from"):
            L.append(f"| start-file runs reused from | {base['reused_from']} |")
    (state.dir / "report.md").write_text("\n".join(L) + "\n")
    print(f"\nReport written to {state.dir / 'report.md'}", flush=True)


# ------------------------------------------------------------------ self-test
PROBE = r'''
import json, os, socket, sys
labels, orig, home = sys.argv[1:4]
index = sys.argv[5]
res = {}
def readable(p):
    try:
        open(p, "rb").read(1); return True
    except Exception:
        return False
res["can read the data index"] = readable(index)
try:
    open("/work/out/x.txt", "w").write("x"); res["can write /work/out"] = True
except Exception:
    res["can write /work/out"] = False
res["can read hidden labels"] = readable(labels)
res["can read original crop index (has labels)"] = readable(orig)
res["can read the project folder (evaluate.py)"] = readable(home)
def connect(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close(); return True
    except Exception:
        return False
res["can reach the internet"] = connect("1.1.1.1", 53) or connect("8.8.8.8", 53)
res["can reach Ollama on localhost"] = connect("127.0.0.1", 11434)
try:
    import torch
    res["torch sees the GPU"] = torch.cuda.is_available()
    if res["torch sees the GPU"]:
        x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
        res["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:
    res["torch sees the GPU"] = False; res["torch error"] = str(e)[:200]
try:
    import timm
    timm.create_model(sys.argv[4], pretrained=True, num_classes=4); res["pretrained weights load offline"] = True
except Exception as e:
    res["pretrained weights load offline"] = False; res["weights error"] = str(e)[:200]
print("RESULT " + json.dumps(res))
'''


def self_test(P, args):
    print(f"Sandbox self-test (data: {P.data}, hidden labels: {P.labels})")
    if not shutil.which("bwrap"):
        raise SystemExit("bwrap is not installed: sudo apt install bubblewrap")
    for p in (P.data / TASK.index_file, P.labels):
        if not p.exists():
            raise SystemExit(f"{p} is missing: run the data preparation first (ar_setup.sh / ar2_setup.sh)")
    models = available_models(P.hf)
    d = P.root / "runs" / "_selftest"
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe.py").write_text(PROBE)
    status, secs = run_sandboxed(P, d / "probe.py", d / "out", d / "probe.log",
                                 [str(P.labels), args.orig_index, str(HERE / "evaluate.py"), models[0] if models else "resnet18",
                                  f"/work/data/{TASK.index_file}"], 300)
    out = (d / "probe.log").read_text()
    line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
    if status != "ok" or not line:
        print(f"The sandbox did not start properly ({status}). Output:\n{out[-3000:]}")
        print("\nIf you see 'setting up uid map: Permission denied' or 'No permissions to create new namespace',"
              " Ubuntu's AppArmor blocks bubblewrap: run ar_setup.sh, which installs a small AppArmor profile for it.")
        raise SystemExit(1)
    res = json.loads(line[7:])
    want = {"can read the data index": True, "can write /work/out": True, "can read hidden labels": False,
            "can read original crop index (has labels)": False, "can read the project folder (evaluate.py)": False,
            "can reach the internet": False, "can reach Ollama on localhost": False,
            "torch sees the GPU": True, "pretrained weights load offline": True}
    bad = 0
    for k, v in want.items():
        ok = res.get(k) == v
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {k}: {res.get(k)} (should be {v})")
    for k in ("gpu", "torch error", "weights error"):
        if k in res:
            print(f"        {k}: {res[k]}")
    print(f"  pretrained models available offline: {', '.join(models) or 'none'}")
    print("All checks passed." if not bad else f"{bad} check(s) failed.")
    shutil.rmtree(d, ignore_errors=True)
    return bad == 0


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="name of this research run (folder under <root>/runs)")
    ap.add_argument("--task", choices=list(TASKS), default="cls", help="cls: crop classification; det: two-stage detection")
    ap.add_argument("--confirm-seeds", type=int, default=2, help="extra seeds to confirm a promising result (2 -> 3 seeds)")
    ap.add_argument("--suspicious", type=float, default=0.8,
                    help="a seed-0 score below this fraction of the best mean is flagged as a likely bug")
    ap.add_argument("--backend", choices=["ollama", "openai", "anthropic"], default="ollama",
                    help="openai: any OpenAI-compatible server, e.g. llama.cpp's llama-server")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080/v1", help="for --backend openai")
    ap.add_argument("--api-key-env", help="--backend openai with a remote API (no --llm-server-cmd): environment variable "
                                          "holding the key, e.g. NVIDIA_API_KEY (the key is never stored)")
    ap.add_argument("--llm-extra", help="JSON merged into every request of the main model, "
                                        "e.g. '{\"chat_template_kwargs\": {\"thinking\": true}}'")
    ap.add_argument("--llm-server-cmd", help="command that starts the LLM server; it is started for each call and "
                                             "stopped afterwards so the GPU is free for training")
    ap.add_argument("--reasoning-effort", help="sent as chat_template_kwargs.reasoning_effort (openai backend), "
                                               "e.g. low, medium, xhigh for Qwen3.8")
    ap.add_argument("--root", default="/data/ar")
    ap.add_argument("--start", default=str(HERE / "ar_train.py"), help="starting train.py")
    ap.add_argument("--program", default=str(HERE / "ar_program.md"))
    ap.add_argument("--llm", default="qwen3.8:27b")
    ap.add_argument("--think", default="auto", help="auto, off, low, medium, high (gpt-oss), or on")
    ap.add_argument("--ctx", type=int, default=24576, help="LLM context length (tokens)")
    ap.add_argument("--max-tokens", type=int, default=12000, help="max LLM output tokens incl. thinking")
    ap.add_argument("--seconds", type=float, default=120, help="wall-clock budget per training run")
    ap.add_argument("--grace", type=float, default=15, help="extra seconds before a run is killed")
    ap.add_argument("--until", help="stop time HH:MM (next occurrence)")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--max-exps", type=int, default=10_000, help="max number of ideas")
    ap.add_argument("--extra-seeds", type=int, default=5, help="extra seeds in the final evaluation (0 = skip it)")
    ap.add_argument("--orig-index", default="/data/datasets/lohi-weld/crops_index.csv")
    ap.add_argument("--accept-se", type=float, default=0.0,
                    help="new best only if the mean gain exceeds this many standard errors (0 = any increase)")
    ap.add_argument("--second-look", action="store_true",
                    help="a seed-0 score within one seed-std below the best gets a second seed before rejection")
    ap.add_argument("--lessons", action="store_true", help="add a per-category summary of past ideas to the prompt")
    ap.add_argument("--diagnostics", action="store_true",
                    help="det task: add class-confusion / size / error-split diagnostics of the best to the prompt")
    ap.add_argument("--digest", help="file with general background reading, appended to the brief")
    ap.add_argument("--reuse-baseline", help="name of an earlier run whose start-file measurements are copied")
    ap.add_argument("--memory", help="file with results of earlier runs on this task, appended to the brief")
    ap.add_argument("--guide", help="file with a research strategy for this run, appended to the brief")
    ap.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (openai backend)")
    ap.add_argument("--planner-server-cmd", help="pair mode: server command of a second model that chooses each "
                                                 "experiment and writes a plan; --llm-server-cmd then implements it")
    ap.add_argument("--planner-llm", default="planner", help="pair mode: name of the planner model (for the report)")
    ap.add_argument("--planner-temperature", type=float, default=1.0)
    ap.add_argument("--critic-server-cmd", help="trio mode: server command of a reviewer model that approves or "
                                                "sends back each plan before it is implemented (needs --planner-server-cmd)")
    ap.add_argument("--critic-llm", default="critic")
    ap.add_argument("--critic-temperature", type=float, default=1.0)
    ap.add_argument("--critic-effort", default="medium", help="reasoning effort passed to the critic's chat template")
    for role in ("planner", "critic"):
        ap.add_argument(f"--{role}-url", help=f"remote OpenAI-compatible API for the {role} instead of a local server, "
                                              "e.g. https://integrate.api.nvidia.com/v1")
        ap.add_argument(f"--{role}-key-env", default="NVIDIA_API_KEY",
                        help=f"environment variable holding the API key for --{role}-url (the key is never stored)")
        ap.add_argument(f"--{role}-max-tokens", type=int, help=f"max output tokens for the {role} (default --max-tokens)")
        ap.add_argument(f"--{role}-extra", help=f"JSON merged into the {role}'s request body, "
                                                "e.g. '{\"chat_template_kwargs\": {\"thinking\": true}}'")
    ap.add_argument("--max-usd", type=float, default=0, help="stop the loop when API spending reaches this (0 = off)")
    ap.add_argument("--usd-in", type=float, default=4.0, help="API price per million input tokens")
    ap.add_argument("--usd-out", type=float, default=20.0, help="API price per million output tokens (incl. thinking)")
    ap.add_argument("--accept-boot", action="store_true",
                    help="det task: the --accept-se uncertainty also includes a paired bootstrap over validation images")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="det task, v3.2: fraction of validation images held out from the agent; a change accepted on "
                         "the rest must also gain on them (0 = off)")
    ap.add_argument("--holdout-seed", type=int, default=1, help="seed of the held-out split (keep it fixed across runs)")
    ap.add_argument("--confirm-k", type=float, default=0.0,
                    help="v3.2: the gain on the held-out images must exceed this many of its standard errors (0 = any gain)")
    ap.add_argument("--rescore", help="comma-separated earlier runs: re-judge their candidates with the image "
                                      "bootstrap (no training), print a table, then exit")
    ap.add_argument("--reproduce", type=int, default=0,
                    help="re-run the start file and final best of --name with N fresh seeds, then exit")
    ap.add_argument("--check", action="store_true", help="sandbox self-test only")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.think == "on":
        args.think = True
    if args.think == "auto":
        args.think = "medium" if args.llm.startswith("gpt-oss") else True

    global TASK, NSEEDS
    TASK = TASKS[args.task]()
    NSEEDS = 1 + args.confirm_seeds
    P = Paths(args.root, args.name, TASK)
    if args.check:
        sys.exit(0 if self_test(P, args) else 1)
    args.extra_body = json.loads(args.llm_extra) if args.llm_extra else {}
    if args.rescore:
        rescore(args)
        return
    if not args.name:
        raise SystemExit("--name is required")
    P.dir.mkdir(parents=True, exist_ok=True)
    meta_p = P.dir / "meta.json"
    meta = {"llm": args.llm, "think": str(args.reasoning_effort or args.think), "seconds": args.seconds, "started": now(),
            "task": args.task, "seeds": NSEEDS, "backend": args.backend, "accept_se": args.accept_se,
            "second_look": args.second_look, "lessons": args.lessons, "program": Path(args.program).name,
            "llm_server_cmd": args.llm_server_cmd, "diagnostics": args.diagnostics,
            "digest": Path(args.digest).name if args.digest else None, "reuse_baseline": args.reuse_baseline,
            "memory": Path(args.memory).name if args.memory else None, "accept_boot": args.accept_boot,
            "guide": Path(args.guide).name if args.guide else None,
            "planner": args.planner_llm if has_role(args, "planner") else None,
            "planner_server_cmd": args.planner_server_cmd, "planner_url": args.planner_url,
            "planner_extra": args.planner_extra,
            "critic": args.critic_llm if has_role(args, "critic") else None,
            "critic_server_cmd": args.critic_server_cmd, "critic_url": args.critic_url,
            "holdout": args.holdout, "holdout_seed": args.holdout_seed, "confirm_k": args.confirm_k,
            "llm_url": args.llm_url if args.api_key_env else None, "llm_extra": args.llm_extra}
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        was = (old["llm"], old["seconds"], old.get("task", "cls"), old.get("seeds", 3))
        if was != (args.llm, args.seconds, args.task, NSEEDS):
            raise SystemExit(f"{P.dir} was run with llm/seconds/task/seeds = {was}; use the same settings or another --name.")
        if (old.get("holdout", 0), old.get("holdout_seed", 1)) != (args.holdout, args.holdout_seed):
            raise SystemExit(f"{P.dir} was run with holdout {old.get('holdout', 0)} (seed {old.get('holdout_seed', 1)}); "
                             "use the same settings or another --name.")
        meta["started"] = old.get("started", meta["started"])
        if old.get("provenance"):
            meta["provenance"] = old["provenance"]
        if args.reproduce:   # a reproduction check must not change the recorded settings of the run
            meta = old
    meta_p.write_text(json.dumps(meta, indent=1))
    state = State(P.dir)
    if args.report_only:
        write_report(state, P, args)
        return

    TASK.load(P)
    TASK.diagnostics = args.diagnostics
    global HOLD, RUNDIR
    RUNDIR = P.dir
    if args.holdout > 0:
        if args.task != "det":
            raise SystemExit("--holdout is implemented for the det task only")
        n_sel, n_conf = TASK.split_holdout(args.holdout, args.holdout_seed)
        HOLD = {"k": args.confirm_k}
        meta["holdout_images"] = [n_sel, n_conf]
        meta_p.write_text(json.dumps(meta, indent=1))
        print(f"Held-out confirmation: {n_conf} validation images hidden from the agent, {n_sel} used for selection "
              f"(split seed {args.holdout_seed}, confirm k = {args.confirm_k:g})", flush=True)
    args.server_log = str(P.dir / "llm_server.log")
    if args.reproduce:
        reproduce(state, P, args)
        return
    if "provenance" not in meta:
        meta["provenance"] = provenance(args, P)
        meta_p.write_text(json.dumps(meta, indent=1))
    program = (Path(args.program).read_text().replace("{MODELS}", ", ".join(available_models(P.hf)) or "(none)")
               .replace("{REFS}", TASK.refs_text(P)))
    if args.digest:
        program += ("\n\n## Background reading: general techniques (not specific to this dataset)\n\n"
                    + Path(args.digest).read_text())
    if args.memory:
        program += ("\n\n## Results of earlier research runs on this same task (validation scores only)\n\n"
                    + Path(args.memory).read_text())
        if HOLD is not None:
            program += ("\n\nNote: in this run your validation scores use only part of the validation images, so they "
                        "are a little different from the numbers above; compare ideas with each other, not with those numbers.")
    if args.guide:
        program += "\n\n## Research guide for this run\n\n" + Path(args.guide).read_text()
    global BOOT
    if args.accept_boot and args.task == "det":
        BOOT = P.dir
    deadline = datetime.max
    if args.until:
        h, m = map(int, args.until.split(":"))
        d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        deadline = d if d > datetime.now() else d + timedelta(days=1)
    if args.hours:
        deadline = min(deadline, datetime.now() + timedelta(hours=args.hours))
    global DEADLINE
    DEADLINE = deadline
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if args.backend == "ollama":
        ollama_unload(args.llm)
    print(f"[{now()}] AutoResearch '{args.name}' with {args.llm} -> {P.dir}"
          + (f", until {deadline:%Y-%m-%d %H:%M}" if deadline != datetime.max else ""), flush=True)

    def minutes_left():
        return (deadline - datetime.now()).total_seconds() / 60 if deadline != datetime.max else 1e9

    def run_min():
        s = [r["seconds"] for e in state.exps for r in e["runs"] if r["status"] == "ok"]
        return (st.median(s) if s else args.seconds) / 60

    def llm_min():   # LLM minutes per experiment (all calls of one experiment summed; pair mode makes two)
        per = {}
        for e in state.events:
            if e["type"] == "llm_call":
                per[e.get("exp")] = per.get(e.get("exp"), 0) + e["seconds"]
        return (st.median(per.values()) if per else 300) / 60

    final_need = 2 * args.extra_seeds * (args.seconds + 10) / 60

    # baseline: the starting file, NSEEDS seeds
    if not state.exps and args.reuse_baseline:
        reuse_baseline(state, P, args)
    if not state.exps:
        d = P.dir / "e000"
        d.mkdir(exist_ok=True)
        shutil.copy(args.start, d / "train.py")
        state.exps.append({"id": "e000", "base_id": "-", "category": "", "hypothesis": "starting file",
                           "outcome": "running", "runs": [], "sha256": sha256_file(d / "train.py")})
        state.save()
    base = state.exps[0]
    if base["outcome"] != "baseline":
        print(f"Baseline: the starting train.py, {NSEEDS} seeds", flush=True)
        for seed in range(NSEEDS):
            if any(r["seed"] == seed and r["status"] == "ok" for r in base["runs"]) or STOP:
                continue
            base["runs"] = [r for r in base["runs"] if r["seed"] != seed] + [run_seed(P, args, P.dir / "e000", seed)]
            state.save()
        summarize_runs(base)
        if base["mean"] is None:
            base["outcome"] = "failed"
            state.save()
            print(f"The starting file did not run {NSEEDS} times successfully; see the logs in", P.dir / "e000")
            write_report(state, P, args)
            return
        base["outcome"] = "baseline"
        state.save()
    print(f"Baseline val {TASK.metric}: {base['mean']:.4f} ± {base['std']:.4f}\n", flush=True)
    for e in state.exps[1:]:
        if e["outcome"] == "confirming":  # interrupted while confirming: finish it now
            best = max((x for x in state.exps if x["outcome"] in ("baseline", "new best") and x["id"] < e["id"]),
                       key=lambda x: x["id"])
            for seed in range(1, NSEEDS):
                if not any(r["seed"] == seed for r in e["runs"]):
                    e["runs"].append(run_seed(P, args, P.dir / e["id"], seed))
                    state.save()
            summarize_runs(e)
            e["outcome"] = ("failed on confirmation" if e["mean"] is None
                            else "new best" if beats(e, best, args.accept_se) else "not confirmed")
            state.save()

    def spent_usd():   # API reference runs: input / output price per million tokens (thinking counts as output)
        calls = [e for e in state.events if e["type"] == "llm_call"]
        return sum((e.get("prompt_tokens") or 0) * args.usd_in + (e.get("output_tokens") or 0) * args.usd_out
                   for e in calls) / 1e6

    while (not STOP and len(state.exps) - 1 < args.max_exps
           and minutes_left() > llm_min() + run_min() + final_need + 3):
        if args.max_usd and spent_usd() >= args.max_usd:
            print(f"Spending cap reached (${spent_usd():.2f} >= ${args.max_usd:.2f}); stopping the loop.", flush=True)
            break
        best = state.best()
        eid = state.next_id()
        d = P.dir / eid
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        print(f"[{now()}] {eid}: asking {args.llm} (current best {best['id']}, {best['mean']:.4f})", flush=True)
        prop = propose(state, P, args, program, d)
        e = {"id": eid, "base_id": prop["base_id"], "category": prop.get("category", ""),
             "hypothesis": prop.get("hypothesis", ""), "runs": [], "time": now()}
        if "error" in prop:
            e.update(outcome="edit failed", edit_error=prop["error"], seed0=None, mean=None, std=None)
            state.exps.append(e)
            state.save()
            print(f"  {eid}: no usable edit after 3 tries", flush=True)
            recent = state.exps[-3:]
            if len(recent) == 3 and all(x.get("edit_error", "").startswith("LLM call failed") for x in recent):
                print("The LLM failed to answer 3 experiments in a row (is Ollama running?). Stopping.", flush=True)
                break
            continue
        (d / "train.py").write_text(prop["src"])
        e["sha256"] = sha256_file(d / "train.py")
        (d / "diff.patch").write_text(prop["diff"] + "\n")
        print(f"  [{e['category'] or '-'}] {e['hypothesis'][:300]}", flush=True)
        if STOP:
            break
        rec = run_seed(P, args, d, 0)
        e["runs"].append(rec)
        summarize_runs(e)
        if rec["status"] != "ok":
            e["outcome"] = rec["status"]
        elif rec["val_f1"] < args.suspicious * best["mean"]:
            e["outcome"] = "suspicious (likely bug)"
            rec["error"] = (f"Score {rec['val_f1']:.4f} is far below the current best ({best['mean']:.4f}). Such a drop "
                            "almost always means a bug (wrong class order, wrong box format, broken normalisation, a model "
                            "that did not train), not a bad idea. Log tail:\n" + tail(d / "s0.log", 12))
        elif rec["val_f1"] <= best["mean"] and not (
                args.second_look and rec["val_f1"] >= best["mean"] - best["std"]
                and minutes_left() > NSEEDS * run_min() + final_need + 1):
            e["outcome"] = "rejected"
        elif STOP or minutes_left() < (NSEEDS - 1) * run_min() + final_need + 1:
            e["outcome"] = "unconfirmed (out of time)"
        else:
            if rec["val_f1"] <= best["mean"]:  # within one seed-std below the best: one more seed before deciding
                print(f"  close ({rec['val_f1']:.4f} vs best {best['mean']:.4f}): second seed", flush=True)
                e["outcome"] = "confirming"
                state.exps.append(e)
                state.save()
                r1 = run_seed(P, args, d, 1)
                e["runs"].append(r1)
                state.save()
                two = [r["val_f1"] for r in e["runs"] if r["status"] == "ok"]
                e["second_look"] = True
                if r1["status"] != "ok" or st.mean(two) <= best["mean"]:
                    e["outcome"] = "rejected"
                    print(f"  2-seed mean {st.mean(two):.4f} -> rejected", flush=True)
                    state.save()
                    continue
                state.exps.pop()
            print(f"  promising ({rec['val_f1']:.4f} vs best {best['mean']:.4f}): confirming with seeds 1-{NSEEDS - 1}", flush=True)
            e["outcome"] = "confirming"
            state.exps.append(e)
            state.save()
            for seed in range(1, NSEEDS):
                if not any(r["seed"] == seed for r in e["runs"]):
                    e["runs"].append(run_seed(P, args, d, seed))
                    state.save()
            summarize_runs(e)
            if e["mean"] is None:
                e["outcome"] = "failed on confirmation"
            else:
                e["outcome"] = "new best" if beats(e, best, args.accept_se) else "not confirmed"
                print(f"  {NSEEDS}-seed mean {e['mean']:.4f} vs best {best['mean']:.4f} -> {e['outcome']}", flush=True)
            state.save()
            continue
        state.exps.append(e)
        state.save()

    print(f"\n[{now()}] Loop finished.", flush=True)
    if args.backend == "ollama":
        ollama_unload(args.llm)
    if args.extra_seeds > 0 and not STOP:
        print("Final evaluation: start file and final best with extra seeds, then the hidden test set", flush=True)
        final_eval(state, P, args)
    write_report(state, P, args)


if __name__ == "__main__":
    main()

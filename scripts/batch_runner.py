#!/usr/bin/env python3
"""Batch runner for 20 targets × 3 methods × 7 k-values.

Subcommands:
    prepare   Generate gage_id_cache files for every (target, method, k) combo.
    run       Scheduler loop: train then eval, fill GPU slots, write state.json.
    status    Print state summary.
    retry     Reset failed (or filtered) tasks back to pending.

Designed to run inside tmux/nohup; safe to Ctrl+C (SIGTERM is forwarded to children).
On restart, in-flight tasks are reset to pending unless --reclaim is given.
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from path_config import get_paths  # noqa: E402
from find_similar_basins import get_similar_basins  # noqa: E402


METHOD_CONFIG = {
    "method1": {
        "train_script": "experiments/train_with_camels_17attributes.py",
        "eval_script": "experiments/evaluate_with_camels_17attributes.py",
        "dataset": "StreamflowDataset",
        "sim_method": "method1",
        "needs_embedding_csv": False,
    },
    "method2": {
        "train_script": "experiments/train_with_us_aef.py",
        "eval_script": "experiments/evaluate_with_us_aef.py",
        "dataset": "AlphaEarthDataset",
        "sim_method": "method2",
        "needs_embedding_csv": True,
    },
    "method3": {
        "train_script": "experiments/train_with_camels_17attributes.py",
        "eval_script": "experiments/evaluate_with_camels_17attributes.py",
        "dataset": "StreamflowDataset",
        "sim_method": "method3",
        "needs_embedding_csv": False,
    },
}
K_VALUES = [100, 200, 300, 400, 500, 600, 670]
SCALER = "DapengScaler"
DIM = 128
DR = 0.4
PYTHON = sys.executable


# ---------- helpers ----------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_id(x) -> str:
    s = str(x).strip()
    return s.zfill(8)


def task_id(target: str, method: str, k: int) -> str:
    return f"T{target}_M{method[-1]}_K{k}"


def train_hru(target: str, method: str, k: int) -> str:
    return f"{target}_method{method[-1]}_k{k}"


def eval_hru(target: str, method: str, k: int) -> str:
    return f"{train_hru(target, method, k)}_test"


def model_dir_name(hru_delete: str) -> str:
    return f"simplelstm_{SCALER}_{DIM}_{DR}_ens_{hru_delete}"


def train_done_marker(paths: dict, hru_delete: str) -> Path:
    return Path(paths["project_dir"]) / "results" / "camels" / model_dir_name(hru_delete) / "best_model.pth"


def eval_done_marker(paths: dict, eval_hru_delete: str) -> Path:
    return Path(paths["eval_project_dir"]) / "results" / "camels_test" / model_dir_name(eval_hru_delete)


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


# ---------- targets ----------

def load_targets(paths: dict) -> list[str]:
    csv_path = paths.get("target_basins_csv")
    if not csv_path:
        raise KeyError("target_basins_csv missing in paths.yaml")
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            raise ValueError(f"{csv_path}: empty file")
        for row in rdr:
            if not row or not row[0].strip():
                continue
            out.append(fmt_id(row[0]))
    if not out:
        raise ValueError(f"{csv_path}: no targets found")
    return sorted(set(out))


# ---------- task list ----------

def all_tasks(targets: list[str]) -> list[dict]:
    out = []
    for t in targets:
        for m in ("method1", "method2", "method3"):
            for k in K_VALUES:
                if m == "method3" and k == 670:
                    continue  # dedup: same gage list & dataset as method1 k=670
                out.append({
                    "id": task_id(t, m, k),
                    "target": t,
                    "method": m,
                    "k": k,
                    "train_status": "pending",
                    "eval_status": "pending",
                    "train_pid": None,
                    "eval_pid": None,
                    "train_log": None,
                    "eval_log": None,
                    "train_started_at": None,
                    "train_ended_at": None,
                    "eval_started_at": None,
                    "eval_ended_at": None,
                    "gpu": None,
                    "error": None,
                })
    return out


# ---------- prepare ----------

def gage_cache_dir(paths: dict, target: str) -> Path:
    base = paths.get("gage_id_cache_dir") or str(REPO_ROOT / "gage_id_cache")
    return Path(base) / target


def train_gage_file(paths: dict, target: str, method: str, k: int) -> Path:
    return gage_cache_dir(paths, target) / f"{method}_k{k}.txt"


def eval_gage_file(paths: dict, target: str) -> Path:
    return gage_cache_dir(paths, target) / "eval_target.txt"


def cmd_prepare(args, paths) -> int:
    targets = load_targets(paths)
    if args.target:
        wanted = {fmt_id(t) for t in args.target.split(",") if t.strip()}
        targets = [t for t in targets if t in wanted]
    if not targets:
        print("no targets to prepare", file=sys.stderr)
        return 2
    n_written = 0
    n_skipped = 0
    for t in targets:
        gdir = gage_cache_dir(paths, t)
        gdir.mkdir(parents=True, exist_ok=True)
        ev = eval_gage_file(paths, t)
        if not ev.exists() or args.force:
            ev.write_text(t + "\n", encoding="utf-8")
            n_written += 1
        for m in ("method1", "method2", "method3"):
            for k in K_VALUES:
                if m == "method3" and k == 670:
                    continue
                f = train_gage_file(paths, t, m, k)
                if f.exists() and not args.force:
                    n_skipped += 1
                    continue
                ids = get_similar_basins(t, METHOD_CONFIG[m]["sim_method"], k)
                f.write_text("\n".join(ids) + "\n", encoding="utf-8")
                n_written += 1
                print(f"[{t} {m} k={k}] -> {len(ids)} ids -> {f}")
    print(f"prepare done: {n_written} written, {n_skipped} skipped (use --force to regenerate)")
    return 0


# ---------- slots / GPU pick ----------

def parse_nvidia_smi() -> list[dict]:
    """Return list of {index, free_mib, util_pct} via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"warn: nvidia-smi unavailable ({e}); auto mode falls back to manual gpus list", file=sys.stderr)
        return []
    rows = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "free_mib": int(parts[1]),
                "util_pct": int(parts[2]),
            })
        except ValueError:
            continue
    return rows


def available_gpus(slot_cfg: dict, running_per_gpu: dict) -> list[int]:
    """Return GPU ids that have at least one free slot under the policy."""
    gpus_filter = set(slot_cfg.get("gpus") or [])
    max_per_gpu = int(slot_cfg.get("max_per_gpu", 1))
    if slot_cfg.get("mode") == "auto":
        smi = parse_nvidia_smi()
        if not smi:
            # fall back to manual gpus list
            candidates = sorted(gpus_filter)
        else:
            min_free = int(slot_cfg.get("min_free_mib", 0))
            max_util = int(slot_cfg.get("max_utilization_pct", 100))
            candidates = []
            for r in smi:
                if gpus_filter and r["index"] not in gpus_filter:
                    continue
                if r["free_mib"] < min_free or r["util_pct"] > max_util:
                    continue
                candidates.append(r["index"])
    else:
        candidates = sorted(gpus_filter)
    return [g for g in candidates if running_per_gpu.get(g, 0) < max_per_gpu]


def pick_task_for_gpu(gpu: int, queue: list[dict], running_on_gpu: list[dict],
                      slot_cfg: dict) -> dict | None:
    if not queue:
        return None
    if not slot_cfg.get("pair_high_low"):
        return queue[0]
    thr = int(slot_cfg.get("high_k_threshold", 400))
    have_high = any(t["k"] >= thr for t in running_on_gpu)
    have_low = any(t["k"] < thr for t in running_on_gpu)
    if have_high and not have_low:
        for t in queue:
            if t["k"] < thr:
                return t
    elif have_low and not have_high:
        for t in queue:
            if t["k"] >= thr:
                return t
    return queue[0]


# ---------- subprocess builders ----------

def build_train_cmd(task: dict, paths: dict) -> list[str]:
    mc = METHOD_CONFIG[task["method"]]
    return [
        PYTHON,
        str(REPO_ROOT / mc["train_script"]),
        "--gage-id-file", str(train_gage_file(paths, task["target"], task["method"], task["k"])),
        "--hru-delete", train_hru(task["target"], task["method"], task["k"]),
        "--dataset", mc["dataset"],
        "--ctx", "0",
        "--camels-dir", str(paths["camels_dir"]),
        "--project-dir", str(paths["project_dir"]),
    ]


def build_eval_cmd(task: dict, paths: dict) -> list[str]:
    mc = METHOD_CONFIG[task["method"]]
    return [
        PYTHON,
        str(REPO_ROOT / mc["eval_script"]),
        "--gage-id-file", str(eval_gage_file(paths, task["target"])),
        "--train-hru-delete", train_hru(task["target"], task["method"], task["k"]),
        "--eval-hru-delete", eval_hru(task["target"], task["method"], task["k"]),
        "--dataset", mc["dataset"],
        "--ctx", "0",
        "--camels-dir", str(paths["camels_dir"]),
        "--project-dir", str(paths["eval_project_dir"]),
        "--training-project-dir", str(paths["project_dir"]),
    ]


def make_env(task: dict, gpu: int, paths: dict) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if METHOD_CONFIG[task["method"]]["needs_embedding_csv"]:
        ec = paths.get("embedding_csv")
        if ec:
            env["TORCHHYDRO_EMBEDDING_CSV"] = ec
    return env


# ---------- scheduler ----------

class Scheduler:
    def __init__(self, args, paths: dict, slot_cfg: dict):
        self.args = args
        self.paths = paths
        self.slot_cfg = slot_cfg
        self.server = args.server
        self.state_path = REPO_ROOT / "batch_state" / args.server / "state.json"
        self.logs_root = Path(paths.get("logs_dir") or (REPO_ROOT / "logs")) / args.server
        self.poll_sec = float(args.poll)
        self.running: dict[str, dict] = {}  # task_id -> {proc, phase, gpu}
        self.stop = False
        self.state = self._load_or_init_state()

    def _load_or_init_state(self):
        existing = load_json(self.state_path)
        targets = load_targets(self.paths)
        # filter by --target if given
        if self.args.target:
            wanted = {fmt_id(t) for t in self.args.target.split(",") if t.strip()}
            targets = [t for t in targets if t in wanted]
        fresh = all_tasks(targets)
        if existing is None:
            return {"server": self.server, "started_at": now_iso(), "tasks": fresh}
        # merge: keep existing tasks, add any new ones
        idx = {t["id"]: t for t in existing.get("tasks", [])}
        merged = []
        for t in fresh:
            if t["id"] in idx:
                merged.append(idx[t["id"]])
            else:
                merged.append(t)
        existing["tasks"] = merged
        return existing

    def save(self):
        atomic_write_json(self.state_path, self.state)

    def reconcile_orphans(self):
        """On startup, decide what to do with tasks that were running before."""
        for t in self.state["tasks"]:
            for phase in ("train", "eval"):
                if t[f"{phase}_status"] == "running":
                    pid = t.get(f"{phase}_pid")
                    if pid_alive(pid) and self.args.reclaim:
                        # leave as running; we lack the Popen handle so we'll poll PIDs
                        # by registering a synthetic record.
                        self.running[f"{t['id']}::{phase}"] = {"pid": pid, "phase": phase, "task": t, "synthetic": True}
                        continue
                    t[f"{phase}_status"] = "pending"
                    t[f"{phase}_pid"] = None
                    t["error"] = (t.get("error") or "") + f"; {phase} interrupted on prior run"
        self.save()

    def install_signals(self):
        def handler(signum, frame):
            self.stop = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def shutdown(self):
        print("\nshutdown: terminating children...", file=sys.stderr)
        for key, rec in list(self.running.items()):
            proc = rec.get("proc")
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 15
        for key, rec in list(self.running.items()):
            proc = rec.get("proc")
            if not proc:
                continue
            remaining = max(0.5, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            t = rec["task"]
            phase = rec["phase"]
            t[f"{phase}_status"] = "pending"
            t[f"{phase}_pid"] = None
        self.save()

    # ----- task selection -----

    def pending_tasks(self) -> list[dict]:
        out = []
        for t in self.state["tasks"]:
            if t["train_status"] == "pending" or (
                t["train_status"] == "done" and t["eval_status"] == "pending"
            ):
                out.append(t)
        # Order: prefer eval-ready (cheap) before train; then high-k first to free pairing room
        out.sort(key=lambda t: (
            0 if (t["train_status"] == "done" and t["eval_status"] == "pending") else 1,
            -t["k"],
        ))
        return out

    def running_per_gpu(self) -> dict[int, int]:
        m: dict[int, int] = {}
        for rec in self.running.values():
            gpu = rec["gpu"] if "gpu" in rec else rec["task"].get("gpu")
            if gpu is None:
                continue
            m[int(gpu)] = m.get(int(gpu), 0) + 1
        return m

    def running_on_gpu_tasks(self, gpu: int) -> list[dict]:
        out = []
        for rec in self.running.values():
            g = rec.get("gpu", rec["task"].get("gpu"))
            if g is not None and int(g) == gpu:
                out.append(rec["task"])
        return out

    def launch(self, task: dict, phase: str, gpu: int) -> bool:
        log_dir = self.logs_root / task["id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{phase}.log"
        cmd = build_train_cmd(task, self.paths) if phase == "train" else build_eval_cmd(task, self.paths)
        env = make_env(task, gpu, self.paths)
        with open(log_path, "ab", buffering=0) as lf:
            header = f"\n=== {phase} start {now_iso()} gpu={gpu} ===\n{' '.join(cmd)}\n".encode()
            lf.write(header)
            try:
                proc = subprocess.Popen(
                    cmd, env=env, cwd=str(REPO_ROOT),
                    stdout=lf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                task[f"{phase}_status"] = "failed"
                task["error"] = f"{phase} spawn failed: {e}"
                self.save()
                print(f"[{task['id']} {phase}] spawn failed: {e}", file=sys.stderr)
                return False
        task[f"{phase}_status"] = "running"
        task[f"{phase}_pid"] = proc.pid
        task[f"{phase}_started_at"] = now_iso()
        task[f"{phase}_log"] = str(log_path)
        task["gpu"] = gpu
        self.running[f"{task['id']}::{phase}"] = {
            "proc": proc, "phase": phase, "task": task, "gpu": gpu,
        }
        self.save()
        print(f"[{task['id']} {phase}] launched on gpu {gpu} pid={proc.pid} log={log_path}")
        return True

    def fill_slots(self):
        queue = self.pending_tasks()
        if not queue:
            return
        running_per_gpu = self.running_per_gpu()
        gpus = available_gpus(self.slot_cfg, running_per_gpu)
        for gpu in gpus:
            if not queue:
                break
            phase_pref = []
            for t in queue:
                if t["train_status"] == "done" and t["eval_status"] == "pending":
                    phase_pref.append((t, "eval"))
                elif t["train_status"] == "pending":
                    phase_pref.append((t, "train"))
            if not phase_pref:
                break
            running_here = self.running_on_gpu_tasks(gpu)
            picked_task = pick_task_for_gpu(gpu, [t for t, _ in phase_pref], running_here, self.slot_cfg)
            if not picked_task:
                continue
            phase = next(p for tt, p in phase_pref if tt is picked_task)
            ok = self.launch(picked_task, phase, gpu)
            if ok:
                queue.remove(picked_task)
                running_per_gpu[gpu] = running_per_gpu.get(gpu, 0) + 1
                # if pair_high_low, the same gpu may have another slot free this tick
                # so loop again over remaining gpus list. Easier: re-add gpu if capacity.
                if running_per_gpu[gpu] < int(self.slot_cfg.get("max_per_gpu", 1)):
                    gpus.append(gpu)

    def reap(self):
        finished = []
        for key, rec in self.running.items():
            t = rec["task"]
            phase = rec["phase"]
            proc = rec.get("proc")
            if rec.get("synthetic"):
                if not pid_alive(rec["pid"]):
                    finished.append((key, None))
                continue
            rc = proc.poll()
            if rc is None:
                continue
            finished.append((key, rc))
        for key, rc in finished:
            rec = self.running.pop(key)
            t = rec["task"]
            phase = rec["phase"]
            t[f"{phase}_pid"] = None
            t[f"{phase}_ended_at"] = now_iso()
            success = self._phase_success(t, phase, rc)
            if success:
                t[f"{phase}_status"] = "done"
                print(f"[{t['id']} {phase}] done (rc={rc})")
            else:
                t[f"{phase}_status"] = "failed"
                t["error"] = f"{phase} failed (rc={rc}); see {t.get(f'{phase}_log')}"
                print(f"[{t['id']} {phase}] FAILED (rc={rc})", file=sys.stderr)
        if finished:
            self.save()

    def _phase_success(self, task: dict, phase: str, rc) -> bool:
        if rc is not None and rc != 0:
            return False
        if phase == "train":
            return train_done_marker(self.paths, train_hru(task["target"], task["method"], task["k"])).exists()
        else:
            d = eval_done_marker(self.paths, eval_hru(task["target"], task["method"], task["k"]))
            return d.exists() and any(d.iterdir())

    def has_outstanding(self) -> bool:
        if self.running:
            return True
        for t in self.state["tasks"]:
            if t["train_status"] in ("pending", "running"):
                return True
            if t["train_status"] == "done" and t["eval_status"] in ("pending", "running"):
                return True
        return False

    def loop(self) -> int:
        self.install_signals()
        self.reconcile_orphans()
        last_status = 0.0
        while not self.stop and self.has_outstanding():
            self.reap()
            if self.stop:
                break
            self.fill_slots()
            now = time.time()
            if now - last_status > 60:
                self._print_status_line()
                last_status = now
            time.sleep(self.poll_sec)
        if self.stop:
            self.shutdown()
            return 130
        # drain any in-flight (shouldn't happen because has_outstanding checks running)
        self.reap()
        self.save()
        self._print_status_line()
        print("all tasks complete.")
        return 0

    def _print_status_line(self):
        n = len(self.state["tasks"])
        c = sum(1 for t in self.state["tasks"] if t["eval_status"] == "done")
        f = sum(1 for t in self.state["tasks"] if "failed" in (t["train_status"], t["eval_status"]))
        r = len(self.running)
        print(f"[{now_iso()}] progress: {c}/{n} done, {r} running, {f} failed")


# ---------- run / status / retry commands ----------

def load_slot_cfg(server: str) -> dict:
    candidates = [
        REPO_ROOT / "configs" / f"{server}_slots.local.yaml",
        REPO_ROOT / "configs" / f"{server}_slots.example.yaml",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["_path"] = str(c)
            return cfg
    raise FileNotFoundError(f"no slot config for {server} (tried {candidates})")


def cmd_run(args, paths) -> int:
    slot_cfg = load_slot_cfg(args.server)
    print(f"slot config: {slot_cfg.get('_path')}")
    sched = Scheduler(args, paths, slot_cfg)
    return sched.loop()


def cmd_status(args, paths) -> int:
    state_path = REPO_ROOT / "batch_state" / args.server / "state.json"
    state = load_json(state_path)
    if state is None:
        print(f"no state at {state_path}")
        return 1
    tasks = state["tasks"]
    n = len(tasks)
    done = sum(1 for t in tasks if t["eval_status"] == "done")
    train_only = sum(1 for t in tasks if t["train_status"] == "done" and t["eval_status"] != "done")
    pend = sum(1 for t in tasks if t["train_status"] == "pending")
    run_t = sum(1 for t in tasks if t["train_status"] == "running")
    run_e = sum(1 for t in tasks if t["eval_status"] == "running")
    failed_train = [t for t in tasks if t["train_status"] == "failed"]
    failed_eval = [t for t in tasks if t["eval_status"] == "failed"]
    print(f"server: {args.server}    state: {state_path}")
    print(f"total: {n}    done: {done}    train-only: {train_only}    "
          f"pending-train: {pend}    running-train: {run_t}    running-eval: {run_e}")
    print(f"failed-train: {len(failed_train)}    failed-eval: {len(failed_eval)}")
    if args.verbose:
        print("\nfailed tasks:")
        for t in failed_train + failed_eval:
            print(f"  {t['id']}    train={t['train_status']}    eval={t['eval_status']}")
            if t.get("error"):
                print(f"    error: {t['error']}")
            if t.get("train_log"):
                print(f"    train log: {t['train_log']}")
            if t.get("eval_log"):
                print(f"    eval log: {t['eval_log']}")
    return 0


def cmd_retry(args, paths) -> int:
    state_path = REPO_ROOT / "batch_state" / args.server / "state.json"
    state = load_json(state_path)
    if state is None:
        print(f"no state at {state_path}", file=sys.stderr)
        return 1
    targets = {fmt_id(t) for t in args.target.split(",")} if args.target else None
    methods = set(args.method.split(",")) if args.method else None
    ks = {int(x) for x in args.k.split(",")} if args.k else None
    n = 0
    for t in state["tasks"]:
        if targets and t["target"] not in targets:
            continue
        if methods and t["method"] not in methods:
            continue
        if ks and t["k"] not in ks:
            continue
        any_failed = "failed" in (t["train_status"], t["eval_status"])
        if args.only_failed and not any_failed:
            continue
        for phase in ("train", "eval"):
            if t[f"{phase}_status"] == "failed" or args.reset_all:
                t[f"{phase}_status"] = "pending"
                t[f"{phase}_pid"] = None
                t[f"{phase}_started_at"] = None
                t[f"{phase}_ended_at"] = None
        # If train was reset, eval must also be pending
        if t["train_status"] == "pending":
            t["eval_status"] = "pending"
        t["error"] = None
        n += 1
    atomic_write_json(state_path, state)
    print(f"reset {n} task(s) -> pending")
    return 0


# ---------- main ----------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="generate gage_id_cache files")
    pp.add_argument("--target", default=None, help="comma-separated targets (default: all from CSV)")
    pp.add_argument("--force", action="store_true", help="regenerate even if files exist")

    pr = sub.add_parser("run", help="scheduler loop")
    pr.add_argument("--server", required=True, help="e.g., server1, server2")
    pr.add_argument("--target", default=None, help="restrict to comma-separated targets")
    pr.add_argument("--poll", type=float, default=5.0, help="scheduler poll interval seconds")
    pr.add_argument("--reclaim", action="store_true",
                    help="on startup, attach to live PIDs from a previous run instead of resetting them to pending")

    ps = sub.add_parser("status", help="print summary")
    ps.add_argument("--server", required=True)
    ps.add_argument("--verbose", "-v", action="store_true")

    pt = sub.add_parser("retry", help="reset failed tasks to pending")
    pt.add_argument("--server", required=True)
    pt.add_argument("--target", default=None)
    pt.add_argument("--method", default=None, help="comma-separated subset of method1,method2,method3")
    pt.add_argument("--k", default=None, help="comma-separated k values")
    pt.add_argument("--only-failed", action="store_true", default=True,
                    help="reset only failed (default)")
    pt.add_argument("--reset-all", action="store_true",
                    help="also reset done/running tasks (dangerous)")

    args = p.parse_args(argv)
    paths = get_paths()
    if args.cmd == "prepare":
        return cmd_prepare(args, paths)
    if args.cmd == "run":
        return cmd_run(args, paths)
    if args.cmd == "status":
        return cmd_status(args, paths)
    if args.cmd == "retry":
        return cmd_retry(args, paths)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

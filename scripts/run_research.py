"""Run isolated, recorded GAS-MARL experiments; no cloud or external service."""
from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ETA = {"lublin_256": "0.002", "Cirne": "0.008", "Jann": "0.003"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def execute_step(name, command, cwd, log_dir, timeout, expected=(), env=None, on_update=None):
    """A failed process or missing/empty artifact never counts as completed."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_dir / (name + ".stdout.log"), log_dir / (name + ".stderr.log")
    result = {"name": name, "command": list(map(str, command)), "cwd": str(cwd),
              "status": "running", "started_at": now(), "timeout_seconds": timeout,
              "stdout": str(stdout_path), "stderr": str(stderr_path)}
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err)
        result["pid"] = child.pid
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                result.update(elapsed_seconds=round(elapsed, 3), heartbeat_at=now())
                if on_update:
                    on_update(dict(result))
                if elapsed >= timeout:
                    child.kill()
                    child.wait()
                    result["status"] = "timed_out"
                    break
                time.sleep(min(2, max(0.01, timeout - elapsed)))
        except BaseException:
            child.kill()
            child.wait()
            raise
        result["returncode"] = child.wait()
    if result["status"] != "timed_out":
        result["status"] = "completed" if result["returncode"] == 0 else "failed"
    missing = [str(p) for p in map(Path, expected) if not p.is_file() or p.stat().st_size == 0]
    if missing:
        result.update(status="failed" if result["status"] == "completed" else result["status"],
                      error="Missing or empty expected artifacts: " + ", ".join(missing))
    result.update(ended_at=now(), elapsed_seconds=round(time.monotonic() - started, 3))
    return result


@contextmanager
def experiment_config(workload, copy_to):
    path = ROOT / "configFile" / "config.ini"
    original = path.read_bytes()
    parser = configparser.ConfigParser()
    parser.read_string(original.decode("utf-8-sig"))
    parser["GAS-MARL setting"]["eta"] = ETA[workload]
    parser["GAS-MARL setting"]["MAX_QUEUE_SIZE"] = "256"
    buffer = io.StringIO()
    parser.write(buffer)
    written = buffer.getvalue().encode("utf-8")
    try:
        path.write_bytes(written)
        Path(copy_to).write_bytes(written)
        yield
    finally:
        if path.read_bytes() == written:
            path.write_bytes(original)
        else:
            raise RuntimeError("config.ini was edited during this run; preserved the newer edit")


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def dataset_manifest():
    return [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted((ROOT / "data").iterdir()) if p.is_file()]


def capture_source(run_dir):
    """Preserve the actual executed source, including uncommitted additions."""
    run_dir = Path(run_dir)
    manifest = {}
    patterns = ["*.py", "requirements*.txt", "configFile/*.ini", "scripts/*.py", "tests/*.py"]
    for pattern in patterns:
        for source in sorted(ROOT.glob(pattern)):
            relative = source.relative_to(ROOT)
            data = source.read_bytes()
            target = run_dir / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest[str(relative)] = hashlib.sha256(data).hexdigest()
    write_json(run_dir / "source_manifest.json", manifest)
    (run_dir / "source.patch").write_text(git_output("diff", "--binary", "HEAD"), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "baseline", "all"], default="smoke")
    parser.add_argument("--workloads", nargs="+", choices=list(ETA), default=["lublin_256"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    if args.num_threads < 1 or args.timeout_hours <= 0:
        parser.error("num-threads and timeout-hours must be positive")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit = git_output("rev-parse", "HEAD")
    dirty_status = git_output("status", "--short")
    lock = ROOT / ".research.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error("Another run owns .research.lock; inspect its PID before removing a stale lock")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "run_dir": str(run_dir), "started_at": now()}, handle)
    state = {"status": "running", "started_at": now(), "run_dir": str(run_dir),
             "pid": os.getpid(), "arguments": vars(args), "git_commit": commit,
             "git_status": dirty_status, "steps": [],
             "interpretation": "Smoke runs validate execution only; no reproduction success is inferred."}
    latest_path = ROOT / "research" / "LATEST_RUN.json"

    def publish(current=None):
        if current is not None:
            state["current_step"] = current
        write_json(run_dir / "state.json", state)
        write_json(latest_path, {"status": state["status"], "run_dir": str(run_dir),
                                "state_file": str(run_dir / "state.json"), "updated_at": now(),
                                "current_step": state.get("current_step")})

    try:
        capture_source(run_dir)
        write_json(run_dir / "datasets.json", dataset_manifest())
        publish()
        stages = ["smoke"] if args.stage == "smoke" else ["smoke", "baseline"]
        for stage in stages:
            for workload in args.workloads:
                for seed in args.seeds:
                    target = run_dir / stage / workload / ("seed_" + str(seed))
                    target.mkdir(parents=True)
                    with experiment_config(workload, target / "config.ini"):
                        model_root = target / "models"
                        common = ["--workload", workload, "--backfill", "1", "--seed", str(seed),
                                  "--device", args.device, "--num-threads", str(args.num_threads)]
                        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1",
                                   GREEN_RL_DELAY_SEMANTICS="author",
                                   OMP_NUM_THREADS=str(args.num_threads), MKL_NUM_THREADS=str(args.num_threads))
                        for script, directory, actor, critic in [
                            ("MARL.py", "MARL", "_actor.pkl", "_critic.pkl"),
                            ("MaskablePPO.py", "MaskablePPO", "actor.pkl", "critic.pkl")]:
                            output = model_root / workload / directory
                            command = [sys.executable, "-X", "utf8", str(ROOT / script), *common,
                                       "--epochs", "1" if stage == "smoke" else "300",
                                       "--traj-num", "2" if stage == "smoke" else "100",
                                       "--output-dir", str(output)]
                            name = f"{stage}-{workload}-{seed}-{directory}-train"
                            record = execute_step(name, command, ROOT, target / "logs",
                                                  args.timeout_hours * 3600,
                                                  [output / actor, output / critic], env, publish)
                            state["steps"].append(record)
                            publish(record)
                            if record["status"] != "completed":
                                raise RuntimeError(name + " " + record["status"])
                        for semantics in ["author", "paper"]:
                            output = target / "evaluation" / semantics
                            env["GREEN_RL_DELAY_SEMANTICS"] = semantics
                            command = [sys.executable, "-X", "utf8", str(ROOT / "compare.py"), *common,
                                       "--len", "64" if stage == "smoke" else "1024",
                                       "--iter", "1" if stage == "smoke" else "10",
                                       "--model-root", str(model_root), "--output-dir", str(output),
                                       "--algorithms", "FCFS", "F2", "PPO", "MARL"]
                            name = f"{stage}-{workload}-{seed}-{semantics}-evaluate"
                            record = execute_step(name, command, ROOT, target / "logs",
                                                  args.timeout_hours * 3600, [output / "result.csv"], env, publish)
                            record["evaluation_semantics"] = semantics
                            record["training_semantics"] = "author"
                            state["steps"].append(record)
                            publish(record)
                            if record["status"] != "completed":
                                raise RuntimeError(name + " " + record["status"])
        state.update(status="completed", ended_at=now())
        publish()
        print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        state.update(status="failed", ended_at=now(), error=str(exc))
        publish()
        print(json.dumps({"status": "failed", "error": str(exc), "run_dir": str(run_dir)}, ensure_ascii=False))
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

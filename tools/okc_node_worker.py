#!/usr/bin/env python3
"""Runs inside the node container. For each variant: copy the image's SGLang source, apply the patch, start the
server with the judge's flags, apply the load, record, stop. Standard library only; the load comes from
okc_node_loadgen.py next to this file.

session.json:
  {"variants": [{"name": str, "patch": file name next to this script or null, "flags": {name: value or null},
                 "loads": [...], "reps": [...]}],            # loads/reps optional per variant
   "replica_flags": {name: value}, "loads": [...], "reps": [...], "seed": 1, "out": dir, "models": dir,
   "deadline_s": seconds this worker may run, "preflight": bool}

A flag value of null adds a bare flag; "<drop>" removes one of the judge's.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SGLANG = Path(os.environ.get("OKC_SGLANG_ROOT", "/sgl-workspace/sglang"))
PORT = 30000
URL = f"http://127.0.0.1:{PORT}"
JUDGE_ENV = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800"}
# The launch command from the serving rules ("Running it yourself"), with local model paths as in the judge's log.
JUDGE_FLAGS = [
    ("model-path", "{models}/Muse-Spark-Open-816B-A42B-NVFP4"),
    ("speculative-draft-model-path", "{models}/Muse-Spark-Open-DFlash2"),
    ("tp-size", "8"), ("ep-size", "8"), ("quantization", "modelopt_fp4"), ("moe-runner-backend", "marlin"),
    ("mem-fraction-static", "0.85"), ("speculative-algorithm", "DFLASH"), ("speculative-num-draft-tokens", "8"),
    ("speculative-draft-attention-backend", "fa4"), ("mm-feature-transport", "cpu"),
    ("enable-hierarchical-cache", None), ("hicache-size", "160"), ("chunked-prefill-size", "16384"),
    ("context-length", "131072"), ("enable-cache-report", None), ("incremental-streaming-output", None),
    ("random-seed", "1234"), ("max-running-requests", "32"), ("cuda-graph-max-bs-decode", "32"),
]


def sh(*cmd, **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def http(path: str, body=None, timeout: float = 600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def launch_flags(models: str, *overrides: dict) -> list:
    flags = {name: (value.format(models=models) if value else value) for name, value in JUDGE_FLAGS}
    for extra in overrides:
        for name, value in (extra or {}).items():
            name = name.lstrip("-")
            if value == "<drop>":
                flags.pop(name, None)
            else:
                flags[name] = None if value is None else str(value)
    out = []
    for name, value in flags.items():
        out += [f"--{name}"] if value is None else [f"--{name}", value]
    return out


def prepare_tree(work: Path, variant: dict) -> dict:
    """Copy python/ from the image's checkout and apply the variant's patch. Returns what happened."""
    tree = work / f"src_{variant['name']}"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SGLANG / "python", tree / "python", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    info = {"tree": str(tree), "patch": variant.get("patch"), "applied": None}
    if variant.get("patch"):
        patch = HERE / variant["patch"]
        info["files"] = re.findall(r"^\+\+\+ b/(\S+)", patch.read_text(), flags=re.M)
        done = sh("git", "apply", "--whitespace=nowarn", "-p1", str(patch), cwd=tree)
        how = "git apply"
        if done.returncode != 0:
            first = (done.stderr or done.stdout)[-600:]
            done = sh("patch", "-p1", "--no-backup-if-mismatch", "-i", str(patch), cwd=tree)
            how = "patch"
            if done.returncode != 0:
                raise RuntimeError(f"patch {patch.name} did not apply: {first} | {(done.stderr or done.stdout)[-600:]}")
        info["applied"] = how
    env = os.environ | {"PYTHONPATH": str(tree / "python")}
    where = sh(sys.executable, "-c", "import importlib.util as u; print(u.find_spec('sglang').origin)", env=env)
    info["import_path"] = where.stdout.strip() or where.stderr[-400:]
    if not info["import_path"].startswith(str(tree)):
        raise RuntimeError(f"variant tree is not the one imported: {info['import_path']}")
    return info


class GpuMemory(threading.Thread):
    """Peak memory.used per GPU, sampled every few seconds while a server runs."""

    def __init__(self, period: float = 5.0):
        super().__init__(daemon=True)
        self.period, self.peak, self.stop_event = period, [], threading.Event()

    @staticmethod
    def read() -> list:
        out = sh("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").stdout.split()
        return [int(x) for x in out if x.isdigit()]

    def run(self):
        while not self.stop_event.wait(self.period):
            now = self.read()
            self.peak = [max(a, b) for a, b in zip(self.peak or now, now)]


def stop_server(proc) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(30)
    sh("pkill", "-9", "-f", "sglang.launch_server")
    for _ in range(60):
        used = GpuMemory.read()
        if not used or max(used) < 2000:
            return
        time.sleep(2)


# Compiled kernels are rebuilt on every fresh machine (about five minutes of an 8-GPU node). Keep them as tar
# files on the results volume: restored before the first server start, saved once a server is up. The build key
# covers sources and toolchain, so a stale entry is rebuilt, never reused.
BUILD_CACHES = {"jit": Path("~/.cache/sglang/jit").expanduser(), "triton": Path("~/.triton/cache").expanduser()}


def restore_build_caches(store: Path) -> dict:
    got = {}
    for name, target in BUILD_CACHES.items():
        tar = store / f"{name}.tar"
        if tar.exists() and not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            done = sh("tar", "-xf", str(tar), "-C", str(target))
            got[name] = "restored" if done.returncode == 0 else f"tar exit {done.returncode}"
    return got


def save_build_caches(store: Path) -> dict:
    got = {}
    store.mkdir(parents=True, exist_ok=True)
    for name, source in BUILD_CACHES.items():
        if source.is_dir() and any(source.iterdir()):
            tmp = store / f"{name}.tar.tmp"
            done = sh("tar", "-cf", str(tmp), "-C", str(source), ".")
            if done.returncode == 0:
                tmp.replace(store / f"{name}.tar")
                got[name] = f"{(store / f'{name}.tar').stat().st_size >> 20} MiB"
            else:
                got[name] = f"tar exit {done.returncode}"
    return got


def start_server(variant: dict, tree: Path, flags: list, log_path: Path, wait_s: float):
    env = os.environ | JUDGE_ENV | {"PYTHONPATH": str(tree / "python")} | (variant.get("env") or {})
    log = log_path.open("w")
    cmd = [sys.executable, "-m", "sglang.launch_server", *flags, "--host", "127.0.0.1", "--port", str(PORT)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd="/tmp", start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}: {log_path.read_text(errors='replace')[-3000:]}")
        try:
            if http("/health", timeout=5)[0] == 200:
                return proc, time.time() - t0
        except Exception:
            time.sleep(3)
    stop_server(proc)
    raise RuntimeError(f"server not ready after {wait_s:.0f} s: {log_path.read_text(errors='replace')[-3000:]}")


def judge_warmup(out: Path) -> None:
    """The requests the judge sends before its load: model info, one short chat, freeze_gc, a health generation."""
    http("/model_info")
    model = json.loads(http("/v1/models")[1])["data"][0]["id"]
    http("/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 8,
                                  "messages": [{"role": "user", "content": "Reply with the single word: ready. " * 20}]})
    http("/freeze_gc", {})
    http("/health_generate")
    info = json.loads(http("/get_server_info")[1])
    keep = {k: info[k] for k in ("internal_states", "max_total_num_tokens", "version") if k in info}
    (out / "server_info.json").write_text(json.dumps(keep, indent=1, default=str))


def run_variant(session: dict, variant: dict, work: Path, out: Path, deadline: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    record = {"name": variant["name"], "flags": variant.get("flags") or {}}
    record["tree"] = prepare_tree(work, variant)
    flags = launch_flags(session["models"], session.get("replica_flags"), variant.get("flags"))
    record["launch"] = flags
    proc, record["startup_s"] = start_server(variant, Path(record["tree"]["tree"]), flags, out / "server.log",
                                             min(2400, deadline - time.time() - 120))
    print(json.dumps({"name": variant["name"], "startup_s": round(record["startup_s"], 1)}), flush=True)
    if session.get("build_cache") and not session.get("_build_cache_saved"):
        record["build_cache_saved"] = save_build_caches(Path(session["build_cache"]))
        session["_build_cache_saved"] = True
    sampler = GpuMemory()
    sampler.start()
    try:
        judge_warmup(out)
        loads = variant.get("loads") or session["loads"]
        reps = variant.get("reps") or session["reps"]
        cmd = [sys.executable, str(HERE / "okc_node_loadgen.py"), "run", "--url", URL,
               "--tokenizer", f"{session['models']}/Muse-Spark-Open-816B-A42B-NVFP4", "--corpus", str(SGLANG / "python"),
               "--pool-cache", str(work / "pool.json"), "--loads", ",".join(map(str, loads)),
               "--reps", ",".join(map(str, reps)), "--seed", str(session.get("seed", 1)),
               "--out", str(out / "load.json"), *session.get("loadgen_args", [])]
        t0 = time.time()
        done = subprocess.run(cmd, timeout=max(60, deadline - time.time() - 90))
        record["load_s"] = time.time() - t0
        record["load_exit"] = done.returncode
    finally:
        sampler.stop_event.set()
        record["gpu_mem_peak_mib"] = sampler.peak
        stop_server(proc)
    if (out / "load.json").exists():
        record["rungs"] = [{k: v for k, v in rung.items() if k != "wall_s"}
                           for rung in json.loads((out / "load.json").read_text())["rungs"]]
    return record


def facts() -> dict:
    return {"gpus": sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader").stdout.strip().splitlines(),
            "nproc": os.cpu_count(), "mem": sh("free", "-g").stdout.splitlines()[1:2],
            "git": sh("git", "--version").stdout.strip(), "patch": sh("patch", "--version").stdout[:40].strip(),
            "sglang_head": sh("git", "-C", str(SGLANG), "rev-parse", "HEAD").stdout.strip(),
            "python": sys.version.split()[0]}


def preflight(session: dict, work: Path) -> dict:
    """No GPU needed: patches apply, patched modules import, and the load generator runs against its mock."""
    report = {"facts": facts(), "variants": []}
    for variant in session["variants"]:
        item = {"name": variant["name"]}
        try:
            item["tree"] = prepare_tree(work, variant)
            modules = [f[len("python/"):-3].replace("/", ".") for f in item["tree"].get("files", []) if f.endswith(".py")]
            code = "import importlib,sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\nprint('ok')"
            got = sh(sys.executable, "-c", code, *modules, env=os.environ | {"PYTHONPATH": item["tree"]["tree"] + "/python"})
            item["imports"] = got.stdout.strip() or got.stderr[-800:]
        except Exception as exc:  # report every variant, do not stop at the first
            item["error"] = str(exc)[-1200:]
        report["variants"].append(item)
    gen = str(HERE / "okc_node_loadgen.py")
    mock = subprocess.Popen([sys.executable, gen, "mock", "--port", "30917", "--prefill-tok-s", "2000000", "--step-ms", "0.1"])
    try:
        time.sleep(3)
        got = sh(sys.executable, gen, "run", "--url", "http://127.0.0.1:30917", "--tokenizer", "mock",
                 "--corpus", str(SGLANG / "python"), "--loads", "1,2", "--reps", "1,1", "--out", str(work / "mock.json"))
        report["loadgen"] = got.stdout[-700:] or got.stderr[-700:]
    finally:
        mock.kill()
    return report


def main():
    session = json.loads(Path(sys.argv[1]).read_text())
    work = HERE
    out = Path(session["out"])
    out.mkdir(parents=True, exist_ok=True)
    if session.get("preflight"):
        report = preflight(session, work)
        (out / "preflight.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return
    deadline = time.time() + session["deadline_s"]
    summary = {"facts": facts(), "session": {k: session[k] for k in ("loads", "reps", "seed", "replica_flags") if k in session},
               "variants": []}
    for name in ("Muse-Spark-Open-816B-A42B-NVFP4", "Muse-Spark-Open-DFlash2"):
        if not (Path(session["models"]) / name / "config.json").exists():
            raise SystemExit(f"weights missing: {session['models']}/{name}")
    if session.get("build_cache"):
        summary["build_cache_restored"] = restore_build_caches(Path(session["build_cache"]))
        print(json.dumps({"build_cache_restored": summary["build_cache_restored"]}), flush=True)
    for variant in session["variants"]:
        if deadline - time.time() < session.get("min_variant_s", 900):
            summary["variants"].append({"name": variant["name"], "skipped": "not enough time left"})
            continue
        try:
            record = run_variant(session, variant, work, out / variant["name"], deadline)
        except Exception as exc:
            record = {"name": variant["name"], "error": str(exc)[-4000:]}
        summary["variants"].append(record)
        (out / "session.json").write_text(json.dumps(summary, indent=1))
        subprocess.run(["sync"])
        print(json.dumps({k: v for k, v in record.items() if k in ("name", "startup_s", "load_s", "error")}), flush=True)
    print("SESSION DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Drive the private 8xH200 copy of the serving judge on Modal.

  modal_node.py preflight NAME[=PATCH] ...     CPU only, judge image: patches apply, modules import, load runs on a mock
  modal_node.py download [--secret NAME]       CPU only: fetch both model repos into the weights Volume
  modal_node.py bench NAME[=PATCH][:FLAG=VALUE,...] ... [--loads 1,4,16,32] [--reps 1,1,1,1] [--minutes 60]

A variant with no patch is unpatched SGLang. Every run first reserves its worst-case cost against
tools/modal_paid_budget.json, has a hard sandbox timeout, and is terminated in a finally block. The Hugging Face token
lives only in the Modal secret; this script never reads or prints it.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import modal_paid_budget as budget  # noqa: E402

IMAGE = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
WEIGHTS_VOLUME, RESULTS_VOLUME = "okc-muse-weights", "okc-node-results"
REPOS = [("RadixArk/Muse-Spark-Open-DFlash2", "Muse-Spark-Open-DFlash2"),
         ("meta-models/Muse-Spark-Open-816B-A42B-NVFP4", "Muse-Spark-Open-816B-A42B-NVFP4")]
DOWNLOAD = '''
import os, sys, threading, time
from pathlib import Path
from huggingface_hub import snapshot_download

def size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9

def report(path, stop):
    while not stop.wait(120):
        print(f"  {path}: {size(path):.1f} GB so far", flush=True)

for repo, name in REPOS:
    target = f"/models/{name}"
    stop, start = threading.Event(), time.time()
    threading.Thread(target=report, args=(target, stop), daemon=True).start()
    try:
        snapshot_download(repo, local_dir=target, max_workers=16)
    finally:
        stop.set()
    print(f"{repo}: {size(target):.1f} GB in {time.time() - start:.0f} s", flush=True)
os.sync()
print("DOWNLOAD DONE", flush=True)
'''


def parse_variant(spec: str) -> dict:
    head, _, flags = spec.partition(":")
    name, _, path = head.partition("=")
    variant = {"name": name, "patch": None, "flags": {}}
    if path:
        data = Path(path).read_bytes()
        variant |= {"patch": f"{name}.diff", "patch_text": data.decode(), "patch_sha256": hashlib.sha256(data).hexdigest()}
    for item in filter(None, flags.split(",")):
        key, eq, value = item.partition("=")
        if key in ("loads", "reps"):  # per-variant override, e.g. loads=1+16
            variant[key] = [int(x) for x in value.split("+")]
        else:
            variant["flags"][key] = value if eq else None
    return variant


def run_sandbox(run: str, dest: Path, *, image, command: list, files: dict, timeout: int, gpu=None, cpu: float = 4,
                memory_gib: int = 16, block_network: bool = True, volumes=None, secrets=None, fetch=(),
                capacity_retries: int = 0) -> int:
    """Create one sandbox, write `files` under /tmp/w, run `command`, copy `fetch` paths back. Returns the exit code."""
    import modal
    reservation = budget.reserve(run, timeout, int(gpu.split(":")[1]) if gpu else 0, cpu, memory_gib,
                                 wait_s=1500 * capacity_retries)
    (dest / "reservation.json").write_text(json.dumps(reservation, indent=1))
    print("reserved", reservation["usd"], "USD; left after:", reservation["left_after_usd"], flush=True)
    app = modal.App(f"okc-node-{run.split('-')[-1]}")
    sandbox, code, t0 = None, -1, time.time()
    try:
        with app.run(environment_name=reservation["environment"]):
            try:
                for attempt in range(capacity_retries + 1):
                    try:
                        sandbox = modal.Sandbox.create(
                            "sleep", str(timeout), app=app, image=image, gpu=gpu, cpu=(cpu, cpu),
                            memory=(memory_gib * 1024, memory_gib * 1024), timeout=timeout, block_network=block_network,
                            volumes=volumes or {}, secrets=secrets or [])
                        break
                    except modal.exception.ResourceExhaustedError:  # no machine of this size; waiting is not billed
                        print(f"no capacity after {time.time() - t0:.0f} s (attempt {attempt + 1})", flush=True)
                        if attempt == capacity_retries:
                            raise
                sandbox.exec("mkdir", "-p", "/tmp/w").wait()
                print(f"sandbox up after {time.time() - t0:.0f} s: {sandbox.object_id}", flush=True)
                (dest / "sandbox_id.txt").write_text(sandbox.object_id + "\n")
                for name, text in files.items():
                    sandbox.filesystem.write_text(text, f"/tmp/w/{name}")
                proc = sandbox.exec(*command, timeout=timeout - 45)
                with (dest / "worker.log").open("w") as log:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                code = proc.wait()
                (dest / "stderr.log").write_text(proc.stderr.read()[-200_000:])
                for remote in fetch:
                    try:
                        local = dest / remote.strip("/").replace("/", "__")
                        local.write_text(sandbox.filesystem.read_text(remote))
                    except Exception as exc:
                        print(f"could not fetch {remote}: {type(exc).__name__}", flush=True)
            finally:
                if sandbox is not None:
                    sandbox.terminate()
    finally:
        budget.close(run, f"exit {code}, {time.time() - t0:.0f} s")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("preflight", "download", "bench"))
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--secret", default="huggingface")
    ap.add_argument("--loads", default="1,4,16,32")
    ap.add_argument("--reps", default="1,1,1,1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--cpu", type=float, default=32)
    ap.add_argument("--memory-gib", type=int, default=512)
    ap.add_argument("--hicache-size", default="24", help="GB of host memory per rank; the judge's 160 needs a 1.9 TiB host")
    ap.add_argument("--loadgen-arg", action="append", default=[])
    ap.add_argument("--capacity-retries", type=int, default=0, help="bench: ask again when Modal has no such machine")
    ap.add_argument("--min-variant-s", type=int, default=900, help="bench: skip a variant when less than this is left")
    a = ap.parse_args()
    import modal
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = f"{stamp}-{a.action}"
    dest = ROOT / "experiments" / "modal_node" / run
    dest.mkdir(parents=True)
    variants = [parse_variant(v) for v in a.variants]
    files = {"okc_node_worker.py": (ROOT / "tools" / "okc_node_worker.py").read_text(),
             "okc_node_loadgen.py": (ROOT / "tools" / "okc_node_loadgen.py").read_text()}
    for v in variants:
        if v["patch"]:
            files[v["patch"]] = v.pop("patch_text")
    session = {"variants": variants, "loads": [int(x) for x in a.loads.split(",")], "reps": [int(x) for x in a.reps.split(",")],
               "seed": a.seed, "models": "/models", "replica_flags": {"hicache-size": a.hicache_size},
               "loadgen_args": a.loadgen_arg, "out": f"/results/{run}", "deadline_s": a.minutes * 60 - 120,
               "build_cache": "/results/_build_cache", "min_variant_s": a.min_variant_s}
    manifest = {"run": run, "action": a.action, "image": IMAGE, "session": session,
                "tools_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()}}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    judge_image = modal.Image.from_registry(IMAGE)
    if a.action == "preflight":
        session |= {"preflight": True, "out": "/tmp/w/out"}
        files["session.json"] = json.dumps(session)
        code = run_sandbox(run, dest, image=judge_image, timeout=600, files=files, cpu=4, memory_gib=16,
                           command=["python3", "/tmp/w/okc_node_worker.py", "/tmp/w/session.json"],
                           fetch=["/tmp/w/out/preflight.json"])
    elif a.action == "download":
        image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34")
        weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True, version=2)
        script = f"REPOS = {REPOS!r}\n" + DOWNLOAD
        code = run_sandbox(run, dest, image=image, timeout=a.minutes * 60, files={"download.py": script}, cpu=8,
                           memory_gib=32, block_network=False, volumes={"/models": weights},
                           secrets=[modal.Secret.from_name(a.secret)],
                           command=["env", "HF_HUB_DISABLE_PROGRESS_BARS=1", "HF_HOME=/tmp/hf", "python3", "/tmp/w/download.py"])
    else:
        files["session.json"] = json.dumps(session)
        weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2).read_only()
        results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
        fetch = [f"/results/{run}/session.json"]
        for v in variants:
            fetch += [f"/results/{run}/{v['name']}/{f}" for f in ("load.json", "server_info.json", "server.log")]
        code = run_sandbox(run, dest, image=judge_image, timeout=a.minutes * 60, files=files, gpu=f"H200:{a.gpus}",
                           cpu=a.cpu, memory_gib=a.memory_gib, volumes={"/models": weights, "/results": results},
                           command=["python3", "/tmp/w/okc_node_worker.py", "/tmp/w/session.json"], fetch=fetch,
                           capacity_retries=a.capacity_retries)
    print(f"exit {code}; results in {dest}")
    sys.exit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()

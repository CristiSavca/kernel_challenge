#!/usr/bin/env python3
"""Run tools/serving_moe_ab_worker.py on one Modal H200 in the serving judge's image.

Credit policy (from Nov 12 2026, at the user's request, "we still have $25.35 left on modal"): allocate only when
    grant - headroom - max(live provider spend, saved high-water) - this run's reservation >= 0
with live spend read from `modal billing summary` just before allocation, no other Modal app running, one H200,
a hard sandbox timeout, and termination in a finally block. The Nov 4 reservation journal and lock are not used,
changed or deleted. Each run writes its record under experiments/modal_serving/<stamp>/.

Usage: modal_serving_ab.py name=path [name=path ...] [--timeout 900] [--dry-run]
Each path is a patched muse_spark_fp4_moe.py compared with the image's stock module.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from modal_credit_budget import (  # noqa: E402
    CreditGuardError, modal_json, provider_spend, require_stopped_apps, reservation_usd,
)

IMAGE = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
GRANT = json.loads((ROOT / "tools" / "modal_credit_grant.json").read_text(), parse_float=Decimal)
HIGHWATER = Decimal("4.83274092")  # the Nov 4 saved provider-spend high-water mark
WORKER = ROOT / "tools" / "serving_moe_ab_worker.py"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def preflight(timeout: int) -> dict:
    profile = GRANT["profile"]
    summary = modal_json("billing", "summary", "--for", datetime.now(timezone.utc).strftime("%Y-%m"),
                         "--json", "--profile", profile)
    spend = provider_spend(summary)
    require_stopped_apps(modal_json("app", "list", "--json", "--profile", profile))
    reserve = reservation_usd(timeout) + Decimal("0.30")  # 8 CPUs instead of the 4 the base estimate assumes
    used = max(spend, HIGHWATER)
    allowance = Decimal(GRANT["grant_usd"]) - Decimal(GRANT["headroom_usd"]) - used - reserve
    if allowance < 0:
        raise CreditGuardError(f"Insufficient free credit: allowance {allowance}")
    return {"live_spend_usd": str(spend), "used_usd": str(used), "reservation_usd": str(reserve),
            "allowance_after_usd": str(allowance), "grant_usd": str(GRANT["grant_usd"]),
            "headroom_usd": str(GRANT["headroom_usd"]), "profile": profile, "environment": GRANT["environment"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tokens", default="")
    ap.add_argument("--import-check", action="append", default=[],
                    help="module to import after installing files; the run then stops (smoke test)")
    ap.add_argument("--gpu", default="H200", choices=("H200", "H100"),
                    help="H100 is the same architecture at a lower rate; use it when no H200 is available")
    ap.add_argument("--install", action="append", default=[],
                    help="<path under the sglang package>=<local file>: a new source file the variants need")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    variants = {v.split("=", 1)[0]: Path(v.split("=", 1)[1]).read_bytes() for v in a.variants}
    # kern:name=spec.json uploads the spec plus the fp4 file it names (as /tmp/w/variant_<name>_fp4.py)
    kernels = {}
    for k in [v for v in variants if v.startswith("kern:")]:
        spec = json.loads(variants.pop(k))
        fp4 = Path(spec["fp4"]).read_bytes()
        name = k[5:]
        variants[f"{name}_fp4"] = fp4
        spec["fp4"] = f"/tmp/w/variant_{name}_fp4.py"
        kernels[name] = json.dumps(spec).encode()
    worker = WORKER.read_bytes()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ROOT / "experiments" / "modal_serving" / stamp
    dest.mkdir(parents=True)
    manifest = {"image": IMAGE, "timeout_s": a.timeout, "gpu": a.gpu, "worker_sha256": sha(worker),
                "variants": {v.split("=", 1)[0]: {"path": v.split("=", 1)[1], "sha256": sha(Path(v.split("=", 1)[1]).read_bytes())}
                             for v in a.variants}}
    manifest["install"] = {i.split("=", 1)[0]: sha(Path(i.split("=", 1)[1]).read_bytes()) for i in a.install}
    manifest["credit"] = preflight(a.timeout)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    if a.dry_run:
        return
    import modal
    app = modal.App("okc-serving-moe-ab")
    snap = ROOT / "experiments" / "modal_serving" / "jit_snapshot_image_id.txt"
    image = modal.Image.from_id(snap.read_text().strip()) if snap.exists() else modal.Image.from_registry(IMAGE)
    sandbox = None
    t0 = time.time()
    with app.run(environment_name=manifest["credit"]["environment"]):
        run_sandbox(app, image, a, worker, variants, kernels, dest, t0)
    print((dest / "result.json").read_text() if (dest / "result.json").exists() else (dest / "worker.log").read_text()[-4000:])


def run_sandbox(app, image, a, worker, variants, kernels, dest, t0):
    import modal
    sandbox = None
    try:
        sandbox = modal.Sandbox.create("sleep", str(a.timeout), app=app, image=image, gpu=a.gpu,
                                       cpu=(8, 8), memory=(32768, 32768), timeout=a.timeout, block_network=True)
        sandbox.exec("mkdir", "-p", "/tmp/w").wait()
        sandbox.filesystem.write_text(worker.decode(), "/tmp/w/worker.py")
        for name, data in variants.items():
            sandbox.filesystem.write_text(data.decode(), f"/tmp/w/variant_{name}.py")
        for name, data in kernels.items():
            sandbox.filesystem.write_text(data.decode(), f"/tmp/w/kernel_{name}.json")
        args = [f"{n}=/tmp/w/variant_{n}.py" for n in variants if not n.endswith("_fp4")]
        args += [f"kern:{n}=/tmp/w/kernel_{n}.json" for n in kernels]
        if a.tokens:
            args.append(f"tokens:{a.tokens}")
        args += [f"import:{m}" for m in a.import_check]
        for i, item in enumerate(a.install):
            rel, local = item.split("=", 1)
            sandbox.filesystem.write_text(Path(local).read_text(), f"/tmp/w/install_{i}")
            args.append(f"install:{rel}=/tmp/w/install_{i}")
        proc = sandbox.exec("python3", "/tmp/w/worker.py", "/tmp/w/result.json", *args, timeout=max(60, a.timeout - 60))
        log = proc.stdout.read()[-20000:] + "\n--- stderr ---\n" + proc.stderr.read()[-20000:]
        proc.wait()
        (dest / "worker.log").write_text(log)
        try:
            (dest / "result.json").write_text(sandbox.filesystem.read_text("/tmp/w/result.json"))
        except Exception as ex:
            (dest / "result_error.txt").write_text(repr(ex))
        # Keep the JIT builds for later runs: snapshot the filesystem once, from the pinned image only.
        snap = ROOT / "experiments" / "modal_serving" / "jit_snapshot_image_id.txt"
        if not snap.exists() and (dest / "result.json").exists():
            try:
                snap.write_text(sandbox.snapshot_filesystem(timeout=240).object_id)
            except Exception as ex:
                (dest / "snapshot_error.txt").write_text(repr(ex))
    finally:
        if sandbox is not None:
            sandbox.terminate()
        (dest / "timing.json").write_text(json.dumps({"wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()

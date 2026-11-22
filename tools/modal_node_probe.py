#!/usr/bin/env python3
"""Ask Modal for one machine with N H200s, record what it gives us, and release it.

Uses the same free-credit guard as modal_serving_ab.py with the reservation scaled by the GPU count. The sandbox
has a hard timeout and is terminated in a finally block. No model, weights or network access are involved.

Usage: modal_node_probe.py [--gpus 8] [--timeout 180] [--dry-run]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import modal_serving_ab as ab  # noqa: E402
from modal_credit_budget import CreditGuardError, modal_json, provider_spend, require_stopped_apps, reservation_usd  # noqa: E402

CHECKS = {
    "gpus": "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    "topology": "nvidia-smi topo -m 2>/dev/null | head -14",
    "cpu": "nproc; free -g | sed -n 2p",
    "disk": "df -h / /tmp 2>/dev/null",
}


def preflight(timeout: int, gpus: int) -> dict:
    profile = ab.GRANT["profile"]
    summary = modal_json("billing", "summary", "--for", datetime.now(timezone.utc).strftime("%Y-%m"),
                         "--json", "--profile", profile)
    spend = provider_spend(summary)
    require_stopped_apps(modal_json("app", "list", "--json", "--profile", profile))
    reserve = gpus * reservation_usd(timeout)
    used = max(spend, ab.HIGHWATER)
    allowance = Decimal(ab.GRANT["grant_usd"]) - Decimal(ab.GRANT["headroom_usd"]) - used - reserve
    if allowance < 0:
        raise CreditGuardError(f"Insufficient free credit: allowance {allowance}")
    return {"live_spend_usd": str(spend), "used_usd": str(used), "reservation_usd": str(reserve),
            "allowance_after_usd": str(allowance), "environment": ab.GRANT["environment"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ROOT / "experiments" / "modal_serving" / f"{stamp}-node-probe"
    dest.mkdir(parents=True)
    out = {"gpu": f"H200:{a.gpus}", "timeout_s": a.timeout, "credit": preflight(a.timeout, a.gpus)}
    (dest / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["credit"]))
    if a.dry_run:
        return
    import modal
    app = modal.App("okc-serving-node-probe")
    image = modal.Image.from_registry(ab.IMAGE)
    sandbox = None
    t0 = time.time()
    try:
        with app.run(environment_name=out["credit"]["environment"]):
            try:
                sandbox = modal.Sandbox.create("sleep", str(a.timeout), app=app, image=image, gpu=out["gpu"],
                                               cpu=(8, 8), memory=(32768, 32768), timeout=a.timeout, block_network=True)
                out["checks"] = {}
                for name, cmd in CHECKS.items():
                    p = sandbox.exec("bash", "-lc", cmd, timeout=60)
                    out["checks"][name] = p.stdout.read().strip()
                    p.wait()
                    if "started_after_s" not in out:
                        out["started_after_s"] = round(time.time() - t0, 1)
                out["status"] = "ok"
            finally:
                if sandbox is not None:
                    sandbox.terminate()
    except Exception as e:  # capacity errors are the answer we are after
        out["status"] = f"{type(e).__name__}: {str(e)[:400]}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    (dest / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

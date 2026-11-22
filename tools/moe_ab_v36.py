#!/usr/bin/env python3
"""Generate a private-check-only MoE A/B candidate from the exact V35 decoder source.

The output appends a prepare-time paired A/B over compile-time MoE variants. It uses synthetic inputs only, never
kernel_fn data, and timed kernels stay the V35 builds. Results are read from the check's log. Never score these files:
a scored candidate is produced separately, from a source edit proven here.

Usage: moe_ab_v36.py <experiment-json> <out.py>
The JSON holds {"source_patches": [[old, new], ...], "cases": {"T": [[name, [define, ...], base_key], ...]}}.
A name's defines are appended to the base build named by base_key (tiny, tiny2, mid, b4, main).
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/candidates/muse_decode_layer/dd57f4575e7c99689a9ce9f519c6df3913ff229d922d1dba69d2ab844cdc4bb0.py"
BASE_SHA = "dd57f4575e7c99689a9ce9f519c6df3913ff229d922d1dba69d2ab844cdc4bb0"
LIMIT = 1_000_000

HARNESS = r'''

# ---------------------------------------------------------------------------
# Private-check experiment only (team CristiSavca): paired prepare-time MoE A/B on synthetic inputs.
# ---------------------------------------------------------------------------
_AB_CASES = __CASES__
_AB_BUDGET_S = 270.0
_AB_SEEDS = 3
_AB_REPS = 16
_AB_WIDTH = 3.0
_AB_DONE = set()
for _ab_name, _ab_val in (("OKC_WGSFCI", "4"), ("OKC_WGSGS", "2"), ("OKC_GATHB", "8"), ("OKC_WGSDFIN", "2"), ("OKC_WGSPR", "1"),
                          ("OKC_WGSREORD", "1"), ("OKC_WGSU", "160"), ("OKC_WGSUB", "140"), ("OKC_WGSNX", "3"),
                          ("OKC_WGSA", "112, 135, 140, 158, 173, 185, 200, 207"), ("OKC_WGSB", "100, 104, 115, 123, 132, 144, 156, 174")):
    _ab_old = f"#define {_ab_name} {_ab_val}"
    assert MOE_CUDA_SRC.count(_ab_old) == 1, _ab_name
    MOE_CUDA_SRC = MOE_CUDA_SRC.replace(_ab_old, f"#ifndef {_ab_name}\n{_ab_old}\n#endif")
MFAB_LAYER = False
_ab_prior_prepare = prepare


def _ab_base_defines(key):
    return {"tiny": MOE_TINY_DEFINES, "tiny2": MOE_TINY2_DEFINES, "mid": MOE_MID_DEFINES,
            "b4": tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",), "main": tuple(MOE_DEFINES)}[key]


def _ab_defines(key, extra, weights):
    sfx = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    base = tuple(_ab_base_defines(key))
    sc6 = ("-DOKC_SC6W=1",) if ("-DOKC_WGS=1" in base and SC6_WGS and sfx and _sc6_ok(weights["w13_sf"], weights["w2_sf"])) else ()
    return base + tuple(extra) + sc6 + tuple(sfx)


def _ab_start_builds(weights):
    jobs = []
    for rows in _AB_CASES.values():
        for name, extra, key in rows:
            if extra:
                jobs.append(_ab_defines(key, extra, weights))

    def run(defs):
        try:
            _ma_ext(defs)
            _AB_DONE.add(defs)
        except Exception as ex:
            print(f"[ab] build failed {defs[-3:]}: {str(ex)[-300:]}", flush=True)
    ths = [threading.Thread(target=run, args=(d,), daemon=True) for d in dict.fromkeys(jobs)]
    for t in ths:
        t.start()
    return ths


def _ab_input(T, seed, dev):
    """Layer-like routing: about 1.6 local routes per token (2.6 at T = 1) over skewed local experts, -1 elsewhere."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((T, 4096), generator=gen, device=dev).to(torch.bfloat16)
    mean = 2.6 if T == 1 else 1.6
    p = torch.exp(-torch.arange(32, device=dev, dtype=torch.float32) / _AB_WIDTH)
    ids = torch.full((T, 8), -1, dtype=torch.int32, device=dev)
    for t in range(T):
        n = int((torch.rand(8, generator=gen, device=dev) < mean / 8).sum())
        n = max(n, 1) if T == 1 else n
        if n:
            ids[t, :n] = torch.multinomial(p, n, replacement=False, generator=gen).to(torch.int32)
    tw = torch.rand((T, 8), generator=gen, device=dev).float()
    tw = tw / tw.sum(-1, keepdim=True)
    return x, ids.contiguous(), tw.contiguous()


def _ab_run(st, weights, t0):
    dev = torch.device("cuda")
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    for Ts, rows in _AB_CASES.items():
        T = int(Ts)
        base = _moe(T, st)
        exts = {"prod": base}
        for name, extra, key in rows:
            defs = _ab_defines(key, extra, weights)
            if not extra or defs in _AB_DONE:
                exts[name] = _ma_ext(defs)
            else:
                print(f"[ab] T={T} {name}: build unavailable", flush=True)
        names = list(exts)
        ratios = {n: [] for n in names if n != "prod"}
        meds = {n: [] for n in names}
        maxd = {n: 0.0 for n in names if n != "prod"}
        routes = 0
        for seed in range(_AB_SEEDS):
            if _rq_time.time() - t0 > _AB_BUDGET_S:
                print(f"[ab] T={T} stopped on budget", flush=True)
                break
            x, ids, tw = _ab_input(T, 4242 + 97 * seed + T, dev)
            routes = int((ids >= 0).sum())
            outs = {n: torch.empty((T, 4096), dtype=torch.bfloat16, device=dev) for n in names}
            graphs = {}
            for n in names:
                ext = exts[n]

                def fn(n=n, ext=ext):
                    ext.okc_run(st["ma_w"], st["ma_scratch"], x, ids, tw, outs[n], 0, 32, st["nsm"])
                if MOE_HAS_PDL:
                    ext.okc_set_pdl(0)
                try:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        fn()
                    graphs[n] = g
                except Exception as ex:
                    print(f"[ab] T={T} {n} capture failed: {str(ex)[-160:]}", flush=True)
            torch.cuda.synchronize()
            names = [n for n in names if n in graphs]
            if "prod" not in graphs:
                print(f"[ab] T={T} prod capture failed", flush=True)
                break
            ts = {n: [] for n in names}
            for r in range(_AB_REPS):
                order = names if r % 2 == 0 else names[::-1]
                for n in order:
                    _FLUSH_SINK.copy_(flush.sum())
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[n].replay()
                    b.record()
                    torch.cuda.synchronize()
                    ts[n].append(a.elapsed_time(b) * 1000.0)
            for n in names:
                meds[n].append(sorted(ts[n])[len(ts[n]) // 2])
            for n in names:
                if n == "prod":
                    continue
                pr = sorted(u / v for u, v in zip(ts["prod"], ts[n]))
                ratios[n].append(pr[len(pr) // 2])
                maxd[n] = max(maxd[n], float((outs[n].float() - outs["prod"].float()).abs().max()))
        pm = sorted(meds["prod"])[len(meds["prod"]) // 2] if meds["prod"] else -1
        print(f"[ab] T={T} routes={routes} prod {pm:.1f} us", flush=True)
        for n in ratios:
            if ratios[n]:
                print(f"[ab]   {n}: x{sorted(ratios[n])[len(ratios[n]) // 2]:.4f} ({' '.join(f'{v:.3f}' for v in ratios[n])}) "
                      f"{sorted(meds[n])[len(meds[n]) // 2]:.1f} us maxd {maxd[n]:.3g}", flush=True)


def prepare(weights):
    t0 = _rq_time.time()
    ths = _ab_start_builds(weights)
    st = _ab_prior_prepare(weights)
    print(f"[ab] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    for th in ths:
        th.join(timeout=max(1.0, 225.0 - (_rq_time.time() - t0)))
    print(f"[ab] {sum(th.is_alive() for th in ths)} of {len(ths)} builds unfinished at {_rq_time.time() - t0:.0f}s", flush=True)
    try:
        _ab_run(st, weights, t0)
    except Exception as ex:
        print(f"[ab] failed: {str(ex)[-500:]}", flush=True)
    print(f"[ab] done at {_rq_time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    return st
'''


def main(spec_path, out_path):
    src = BASE.read_text()
    if hashlib.sha256(src.encode()).hexdigest() != BASE_SHA:
        raise SystemExit("base source hash mismatch")
    spec = json.loads(Path(spec_path).read_text())
    for old, new in spec.get("source_patches", []):
        if src.count(old) != 1:
            raise SystemExit(f"patch target not unique ({src.count(old)}): {old[:80]}")
        src = src.replace(old, new)
    out = src + HARNESS.replace("__CASES__", repr(spec["cases"]))
    ast.parse(out)
    size = len(out.encode())
    if size > LIMIT:
        raise SystemExit(f"too large: {size}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(out)
    print(size, hashlib.sha256(out.encode()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Finite source-bound schedule audit for the single v21 unroll-hint experiment."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
CANDIDATE_SHA = "4fcf108fc860865b0284031075c9e35461b761eb11f5cd4681917e44165871da"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_steady_lg_v21.py"
OLD = "#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {"
NEW = "#pragma unroll (FIRST ? 4 : 1)\n    for (int lg = 0; lg < 4; ++lg) {"
OLD_NAME = 'name=f"okc_moe_kperm3_v9_{tag}_sm{arch}"'
NEW_NAME = 'name=f"okc_moe_steady_lg_v21_{tag}_sm{arch}"'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def native(source):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))


def controls(cuda, candidate):
    start = cuda.index("template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(")
    end = cuda.index("template <bool PB>\n__device__ __forceinline__ void wgs_phase(", start)
    seg = cuda[start:end]
    expected = NEW if candidate else OLD
    assert seg.count(expected) == 1
    # Bind every modeled scheduling decision to actual source; full restoration
    # below binds all unmodeled arithmetic, packing and producer behavior.
    for text in (
        "constexpr int GS = OKC_WGSGS;", "uint32_t af[2][GS][4];",
        "const int ws = seq % WGS_NS, xs = seq % WGS_NX;",
        "(uint32_t)(seq / WGS_NS) & 1u", "for (int h = 0; h < 4 / GS; ++h)",
        "const int ci = lg * (4 / GS) + h;", "for (int jj = 0; jj < GS; ++jj)",
        "const int js = h * GS + jj;", "af[ci & 1][jj][0] = bmul2l(r0, sgp)",
        "if (lg < 3)", "((lg + 1) * 4 + wp)", "else if (lane == 0)",
        "mbar_arrive1(S.bars + BW_E + ws)", "if constexpr(NTT == 1)",
        "if (ci & 1)", "FIRST && ci == 1 && jj == 0", "FIRST && ci == 0 && jj == 0",
        "wg_fence();", "wg_commit();", "wg_wait<1>();", "if (ci == 0)",
        "if (prev >= 0)", "mbar_arrive1(S.bars + BX_E + prev)", "prev = -1;",
        "FIRST && ci == OKC_WGSFCI && pd.kind", "prev = xs;", "++seq;",
        "stage(std::true_type{});", "#pragma unroll 1\n  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});",
        "wg_wait<0>();", "d[0][q]+=d_odd[0][q]",
    ):
        assert text in seg, text
    assert seg.index("wg_commit();") < seg.index("wg_wait<1>();") < seg.index("if (ci == 0)") < seg.index("FIRST && ci == OKC_WGSFCI")
    gs = int(re.search(r"#define OKC_WGSGS (\d+)", cuda)[1])
    ns = int(re.search(r"constexpr int WGS_NS = (\d+)", cuda)[1])
    nx = int(re.search(r"#define OKC_WGSNX (\d+)", cuda)[1])
    fci = int(re.search(r"#define OKC_WGSFCI (\d+)", cuda)[1])
    assert (gs, ns, nx, fci) == (2, 3, 3, 3)
    return {"gs": gs, "ns": ns, "nx": nx, "fci": fci,
            "unroll": (4, 1) if candidate else (4, 4), "segment_sha256": sha(seg)}


def trace(c, ntt, pb, ranges, seq, pending_kind=2, mutant=None):
    events = []
    live_x = set()
    pending = []
    gs = c["gs"]
    for s0, s1 in ranges:
        assert 0 <= s0 <= s1 <= 16
        if s0 == s1:
            continue  # outer phase does not invoke wgs_seg for an empty range
        prev = -1
        initialized = set()
        for st in range(s0, s1):
            first = st == s0
            ws, xs = seq % c["ns"], seq % c["nx"]
            assert xs not in live_x
            live_x.add(xs)
            events.extend([("wait_full", seq, ws, (seq // c["ns"]) & 1),
                           ("code_load", seq, 0), ("scale_load_and_four_pairs", seq)])
            factor = c["unroll"][0 if first else 1]
            # Expand a hinted group of four or execute four single iterations.
            # This is C++ loop semantics, not proof nvcc honors either hint.
            for group in range(0, 4, factor):
                for lg in range(group, min(group + factor, 4)):
                    if lg < 3:
                        events.append(("code_load", seq, lg + 1))
                    else:
                        events.append(("release_weight", seq, ws))
                    for h in range(4 // gs):
                        ci = lg * (4 // gs) + h
                        bank = ci & 1 if mutant != "one_bank" else 0
                        assert bank == (h & 1)
                        assert all(g[2] != bank for g in pending), "AF bank overwritten before retirement"
                        for jj in range(gs):
                            js = h * gs + jj
                            events.append(("decode_and_AF", seq, lg, js, bank, jj))
                        events.append(("wg_fence", seq, ci))
                        for jj in range(gs):
                            acc = ci & 1 if ntt == 1 else 0
                            zero = first and jj == 0 and (ci == 0 or (ntt == 1 and ci == 1))
                            if not zero:
                                assert acc in initialized
                            initialized.add(acc)
                            events.append(("mma0" if zero else "mma", seq, lg * 4 + h * gs + jj, ntt, acc))
                        events.append(("commit", seq, ci))
                        pending.append((seq, ci, bank))
                        if mutant == "release_before_wait" and ci == 0 and prev >= 0:
                            assert all(g[0] % c["nx"] != prev for g in pending), "X released before retirement"
                        pending = pending[-1:]
                        events.append(("wait1", seq, ci))
                        if ci == 0 and prev >= 0:
                            assert all(g[0] % c["nx"] != prev for g in pending)
                            events.append(("release_x", prev))
                            live_x.remove(prev)
                            prev = -1
                        if first and ci == c["fci"] and pending_kind:
                            events.append(("flush", pb, pending_kind, seq, ci))
                            pending_kind = 0
            prev = xs
            seq += 1
        pending = []
        events.append(("wait0",))
        if ntt == 1:
            events.append(("add_odd_accumulator",))
        assert prev >= 0
        events.append(("release_x", prev)); live_x.remove(prev)
        events.append(("inherited_epilogue", pb, ntt, s0, s1))
        pending_kind = 1 if (pb == 0 and s1 - s0 == 16) else 2
    assert not pending and not live_x
    return events


def run():
    parent = (ROOT / f"experiments/candidates/muse_moe_w4a16/{PARENT_SHA}.py").read_text()
    source = CANDIDATE.read_text()
    assert sha(parent) == PARENT_SHA and sha(source) == CANDIDATE_SHA
    assert source.count(NEW) == source.count(NEW_NAME) == 1
    assert source.replace(NEW, OLD).replace(NEW_NAME, OLD_NAME) == parent
    pc, cc = controls(native(parent), False), controls(native(source), True)
    cases = events = 0
    digest = hashlib.sha256()
    for ntt in range(1, 9):
        for pb in (0, 1):
            for seq in range(6):  # every depth-3 slot and both barrier generations
                for s0 in range(16):
                    for s1 in range(s0 + 1, 17):
                        a = trace(pc, ntt, pb, [(s0, s1)], seq)
                        b = trace(cc, ntt, pb, [(s0, s1)], seq)
                        assert a == b
                        events += len(a); cases += 1
                        digest.update(repr(a).encode())
                for kind in range(4):
                    ranges = [(0, 0), (7, 8), (3, 16), (0, 16), (0, 1), (1, 3)]
                    a = trace(pc, ntt, pb, ranges, seq, kind)
                    assert a == trace(cc, ntt, pb, ranges, seq, kind)
                    events += len(a); cases += 1
                    digest.update(repr(a).encode())
    negatives = []
    for mutant in ("one_bank", "release_before_wait"):
        try:
            trace(cc, 1, 0, [(0, 2)], 2, mutant=mutant)
        except AssertionError:
            negatives.append(mutant)
        else:
            raise AssertionError("mutant escaped: " + mutant)
    report = {"candidate_sha256": CANDIDATE_SHA, "parent_sha256": PARENT_SHA,
              "native_sha256": sha(native(source)), "verifier_sha256": sha(Path(__file__).read_text()),
              "full_parent_restoration": True, "hint_first_steady": list(cc["unroll"]),
              "cases": cases, "matched_model_events": events, "event_sha256": digest.hexdigest(),
              "negative_controls_detected": negatives,
              "limits": ["Source-only pragma/namespace delta; all executable statements and non-WGS source preserved.",
                         "Finite model follows bound source schedule and conservative wait1 retirement, not GPU warp timing or full grid progress.",
                         "Empty local ranges are skipped; no new global-zero-job support claim.",
                         "Compiler pragma compliance, physical registers, WGMMA serialization, numerical GPU output and performance remain untested."]}
    return report


if __name__ == "__main__":
    result = run()
    path = ROOT / "experiments/iterations/20261104-wgs-steady-lg-cpu.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

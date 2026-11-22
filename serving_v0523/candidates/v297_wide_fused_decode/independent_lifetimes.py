"""Check physical register lifetimes and stage-release order in five actual SM90a gate/up images."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


INST = re.compile(r"/\*([0-9a-f]+)\*/\s+(?:(@!?U?P\d+)\s+)?([A-Z][.A-Za-z0-9_]+)\s*([^;\n]*);")
GPR = re.compile(r"\bR(\d+)\b")


def insns(sass):
    return [(int(m[1], 16), m[2], m[3], m[4].strip()) for m in INST.finditer(sass)]


def code_order(src, wide):
    loop = src.split("for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters;", 1)[1]
    release = "if (warp_iter_id + 1 == Ctx::kWarpIters) consumer.arrive(stage_id)"
    assert loop.index("mma.run(stage_id, warp_iter_id)") < loop.index(release)
    if "mma.template run_as<Op>" in loop:
        assert loop.index("mma.template run_as<Op>(stage_id, warp_iter_id)") < loop.index(release)
    assert loop.count("consumer.arrive(stage_id)") == 1
    assert "#define MUSE_PAIR_LAST_K_SLICES" in src
    if wide:
        assert loop.index(release) < loop.index("s2r_pipe.load_stage_iter(")
        kernel = src.rsplit("template<class SliceOp> CUDA_INLINE void run_as", 1)[1]
        terms = ["if constexpr (bands == 4)", "dequant<ElementB", "arith.may_apply_bs_and_zp_on_b", "wgmma_fence()", "SliceOp::fma(", "if (bands == 4 || iteration % 2 == 1)", "wgmma_commit()", "wgmma_wait<0>()", "warpgroup_fence_operand(values[i])"]
        p = 0
        for term in terms:
            p = kernel.index(term, p) + len(term)
        assert "for (uint32_t i = 0; i < bands * 4; ++i)" in kernel
        assert "kWarpIters == 4" in src


def verify(code, wide):
    # After each issue the HGMMA source is four 32-bit words, and each
    # output band of 64xN accumulates 64*N/128 FP32 registers per thread.
    live_a, live_d = set(), set()
    writes_since_fence = set()
    group_sizes, issues, checks, deps = Counter(), Counter(), Counter(), Counter()
    group, released = [], []
    issues_seen = 0
    branches, live_at, extra_checks = [], {}, set()
    jumps = {int(args, 16) for _, _, op, args in code if op == "BRA"}
    by_pc = {x[0]:x for x in code}
    for pc, pred, op, args in code:
        mn = op.split(".")[0]
        if group:
            live_at[pc] = (frozenset(live_a), frozenset(live_d))
        if wide and (group or op.startswith("HGMMA.")):
            assert pc not in jumps, (hex(pc), "branch target inside HGMMA issue group")
        if op.startswith("WARPGROUP.ARRIVE"):
            assert pred is None
            writes_since_fence.clear()
            continue
        if op.startswith("HGMMA."):
            fmt = re.fullmatch(r"HGMMA.64x(\d+)x16.F32.BF16", op)
            args0 = re.fullmatch(r"R(\d+), R(\d+), gdesc\[UR\d+\], R(\d+)(?:, gsb0)?", args)
            assert fmt and args0 and not pred
            m = int(fmt[1]); d, a, cin = map(int, args0.groups())
            src = set(range(a, a + 4)); out = set(range(d, d + m//2))
            assert d == cin and src.isdisjoint(out | live_d) and out.isdisjoint(live_a)
            if wide:
                assert src.isdisjoint(writes_since_fence), (hex(pc), "ordinary RF write after arrival on source", src & writes_since_fence)
            live_a.update(src); live_d.update(out)
            group.append(m); issues[m] += 1; issues_seen += 1
            continue
        if op.startswith("WARPGROUP.DEPBAR"):
            assert args == "gsb0, 0x0" and not pred
            if group:
                assert len(set(group)) == 1
                group_sizes[len(group)] += 1
            group.clear(); live_a.clear(); live_d.clear()
            continue
        if op.startswith("SYNCS.ARRIVE"):
            assert not group, (hex(pc), "barrier arrival before full drain")
            if issues_seen:
                released.append(issues_seen)
            continue
        # Disallow leaving the basic block with an in-flight group.
        if group and mn in {"BRA", "JMP", "BRX", "BSSY", "BSYNC", "CALL", "RET", "EXIT"}:
            assert not wide and op == "BRA", (hex(pc), "control flow before full drain", op)
            branches.append((int(args,16), *live_at[pc]))

        # Bound range expansion here to opcodes observed inside an outstanding
        # group; a future data-moving opcode requires its own operand analysis.
        dest = re.match(r"R(\d+)\b", args)
        refs = set(map(int, GPR.findall(args)))
        if group:
            assert refs.isdisjoint(live_a | live_d), (hex(pc), "ordinary access to live weight or accumulator", op, args, refs & (live_a | live_d))
            checks["ordinary_instructions_in_flight"] += 1
            if dest:
                assert mn in {"LDS","IMAD","PRMT","LOP3","SHF","HMUL2","VIADD","SEL","MOV","S2R","IADD3","LEA","ISETP","SYNCS","SHFL","UF2FP","F2FP"}, (op,args)
        if dest and mn not in {"STG","STS","STL","STM","RED","BAR","BRA"}:
            rw = 4 if ".128" in op else 2 if ".64" in op else 1
            written = set(range(int(dest[1]), int(dest[1]) + rw))
            if group:
                assert written.isdisjoint(live_a | live_d), (hex(pc), "ordinary RF write before full drain", op, args, written & (live_a | live_d))
                checks["register_writes_in_flight"] += len(written)
                deps[mn] += 1
            writes_since_fence.update(written)
    # Narrow images can branch to out-of-line mbarrier polling with a pending
    # pair. Walk taken/fall-through edges until they rejoin the same live set.
    while branches:
        pc, a, d = branches.pop()
        if (pc,a,d) in extra_checks or live_at.get(pc) == (a,d):
            continue
        extra_checks.add((pc,a,d))
        _, pred, op, args = by_pc[pc]
        assert (a | d).isdisjoint(map(int,GPR.findall(args)))
        if op == "BRA":
            branches.append((int(args,16),a,d))
            if not pred:
                continue
        else:
            assert op == "SYNCS.PHASECHK.TRANS64.TRYWAIT", (hex(pc),op,args)
        branches.append((pc+16,a,d))
    assert issues and not group
    assert set(group_sizes) == {4}
    bands = 4 if wide else 2
    expected = []
    for i in range(len(issues)):
        expected.extend(4*bands*(4*i+s) for s in (1,2,3,4,4))
    assert released == expected, (released,expected)
    return dict(out_of_line_poll_states=len(extra_checks),issued_by_width=issues, drain_sizes=group_sizes, release_positions=released,
                checks=checks, in_flight_writer_opcodes=deps)


def rejected(fn):
    try:
        fn()
    except AssertionError as err:
        return str(err)
    raise RuntimeError("negative control falsely admitted")


def main(folder):
    report={}
    for path in sorted(folder.glob("w13_*.sass")):
        wide = "n512" in path.stem
        get = lambda s: path.with_suffix(s).read_bytes()
        src = get(".cu").decode()
        code_order(src, wide)
        code = insns(path.read_text())
        facts = verify(code, wide)
        if wide:
            i = next(i for i,x in enumerate(code) if x[2].startswith("HGMMA."))
            a = int(code[i][3].split(", ")[1][1:])
            corrupt = code[:i+1]+[(code[i][0],None,"MOV",f"R{a+3}, RZ")]+code[i+1:]
            read = code[:i+1]+[(code[i][0],None,"R2UR",f"UR10, R{a+1}")]+code[i+1:]
            early = code[:i+1]+[(code[i][0],None,"SYNCS.ARRIVE.TRANS64.A1T0","RZ, [UR4], RZ")]+code[i+1:]
            unfenced = code[:i]+[(code[i][0],None,"MOV",f"R{a+1}, RZ")]+code[i:]
            facts["mutants_rejected"] = dict(overwrite_after_issue=rejected(lambda:verify(corrupt,wide)), read_after_issue=rejected(lambda:verify(read,wide)), producer_reuse=rejected(lambda:verify(early,wide)), rewrite_after_arrival=rejected(lambda:verify(unfenced,wide)))
        facts["sha256"]={s:sha256(get("."+s)).hexdigest() for s in ["cu","sass","cubin"]}
        machine=json.loads((Path(__file__).resolve().with_name("w13_sm90a.json")).read_bytes())[path.stem]
        assert machine["cubin_sha256"] == facts["sha256"]["cubin"] and machine["sass_sha256"] == facts["sha256"]["sass"]
        report[path.stem]=facts
    assert len(report)==5
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main(Path(sys.argv[1]))

"""CPU provenance/role proof for V11; no CUDA compilation or GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import native_q8_role_pipeline_v10_verify as v10

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_q8_role_pipeline_v11 import PARENT, OUTPUT, PARENT_SHA, NEW_DISPATCH, build
from build_decoder_attention_screen_v10 import build as build_wrapper


def definition(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise AssertionError("unclosed definition")


def role_proof(parent, child):
    for marker in ("__device__ __forceinline__ void q8_role_qk_v10(",
                   "__device__ __forceinline__ void q8_role_pv_v10(",
                   "auto run_segments = [&](auto role)", "struct Q8RoleMetaV10 {"):
        assert definition(parent, marker) == definition(child, marker), marker
    old_body = definition(parent, "__device__ __forceinline__ void q8_role_body_v10(")
    new_body = definition(child, "__device__ __forceinline__ void q8_role_body_v10(")
    sync = r"(?:csync<[^;]+;|__syncthreads\(\);|mbar_(?:init_n|fence_init)\([^;]+;)"
    assert re.findall(sync, old_body) == re.findall(sync, new_body)
    assert new_body.count("csync<true, 256>();") == 3
    assert new_body.count("__syncthreads();") == 1
    assert "if constexpr (ROLE == 3) q8_role_qk_v10" in new_body
    assert 'static_assert(ROLE == 3 || ROLE == 4' in new_body
    assert "if ((ROLE == 2 || ROLE == 3) && tid == 0)" in child
    # Both role calls remain below their matching setmax and inside that branch.
    assert NEW_DISPATCH in child
    for limit, role in ((96, 3), (112, 4)):
        assert f'''setmaxnreg.inc.sync.aligned.u32 {limit};\\n" ::: "memory");
#endif
            run_segments(std::integral_constant<int, {role}>());''' in NEW_DISPATCH
    # The shared finalizer remains outside the role dispatcher, unchanged.
    finalizer = '''      if constexpr (L == 8 || OKC_FLATCOOP1C) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }'''
    assert parent.count(finalizer) == child.count(finalizer) == 1
    assert NEW_DISPATCH + finalizer in child
    # Exhaustive whole-CTA role mapping, including compile/runtime fallback.
    for enabled in (False, True):
        for runtime in (False, True):
            roles = [1 if tid >= 256 else (3 if tid < 128 else 4) if enabled and runtime else 2
                     for tid in range(384)]
            assert all(len(set(roles[first:first + 128])) == 1 for first in (0, 128, 256))
            assert sum(role == 1 for role in roles) == 128
            assert sum(role != 1 for role in roles) == 256
            assert [(tid, role) for tid, role in enumerate(roles) if role in (2, 3) and tid == 0] == [(0, 3 if enabled and runtime else 2)]
            counts = {role: roles.count(role) for role in set(roles)}
            assert counts == ({1: 128, 3: 128, 4: 128} if enabled and runtime else {1: 128, 2: 256})
            # Every consumer executes three256-thread body barriers and oneCTA drain.
            if enabled and runtime:
                assert sum(role in (3, 4) for role in roles) == 256
                pool = sum({1: 32, 3: 96, 4: 112}[role] for role in roles)
                assert pool == 384 * 80 == 30720
    return {"whole_cta_role_maps": 4, "consumer_threads": 256, "ring_initializer_threads": 1,
            "body_consumer_barriers": 3, "body_cta_drains": 1,
            "common_finalizer_unchanged": True, "qk_pv_math_and_segment_scheduler_byte_exact": True}


def main():
    parent = PARENT.read_text()
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    child, proof = build()
    assert OUTPUT.read_text() == child
    result = role_proof(parent, child)
    wrapper, wrapper_proof = build_wrapper(child, proof["candidate_sha256"], "decoder_q8_role_v11")
    assert (ROOT / "muse_attention_decode/candidates/decoder_q8_role_v11.py").read_text() == wrapper
    result.update(v10.layout_proof())
    result["schedule_events"] = sum(v10.schedule_proof(seed, [1, 2, 3, 4, 5, 9, 37]) for seed in range(128))
    moves = [v10.rescale_proof(seed) for seed in range(16)]
    result["rescale_moves"] = sum(x[0] for x in moves)
    result["rescale_fast_halves"] = sum(x[1] for x in moves)
    fragment = child.split("// Q8V10_BEGIN_HELPERS\n", 1)[1].split("// Q8V10_END_HELPERS", 1)[0]
    for role in (3, 4):
        fragment += f"\ntemplate void q8_role_body_v10<{role}>(const Params&, Smem<8,true>&, int, int, int, int, int, int, float);\n"
    official = ROOT / "experiments/candidates/muse_decode_layer" / (v10.PARENT_SHA + ".py")
    v10.host_syntax_and_layout(official.read_text(), child, fragment)
    result.update(candidate_sha256=proof["candidate_sha256"], wrapper_sha256=wrapper_proof["wrapper_sha256"],
                  exact_reverse_proof=proof["exact_reverse_proof"],
                  host_syntax_and_exact_smem_layout="pass, both role templates instantiated; CUDA/PTX not compiled",
                  gpu_tested=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

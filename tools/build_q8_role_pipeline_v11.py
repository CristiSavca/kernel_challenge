"""One causal allocator-visibility revision of frozen V10; CPU only."""
import ast
import hashlib
import json
from pathlib import Path

from build_q8_role_pipeline_v10 import BODY_INSERT, REG_NEW, replace_once
from build_decoder_attention_screen_v10 import build as build_wrapper

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "6eb318df58c3ef5576c4ea456b4b494e1ef4548090fa7707c84e247f960c2222"
PARENT = ROOT / "muse_decode_layer/candidates/native_q8_role_pipeline_v10.py"
OUTPUT = ROOT / "muse_decode_layer/candidates/native_q8_role_pipeline_v11.py"

OLD_DISPATCH = '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n' + REG_NEW + '''#endif
#if OKC_DYN
      run_dynamic(std::integral_constant<int, 2>());
#else
      run_segments(std::integral_constant<int, 2>());
#endif
'''

NEW_DISPATCH = '''      // Q8V11: keep each register budget directly above its compile-time role.
      if constexpr (Q8RoleEnabledV10<L, FLAT>::value) {
        if (q8_role_runtime_v10(p)) {
          if (threadIdx.x < 128) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\\n" ::: "memory");
#endif
            run_segments(std::integral_constant<int, 3>());
          } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\\n" ::: "memory");
#endif
            run_segments(std::integral_constant<int, 4>());
          }
        } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
          asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
#endif
          run_segments(std::integral_constant<int, 2>());
        }
      } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
#endif
#if OKC_DYN
        run_dynamic(std::integral_constant<int, 2>());
#else
        run_segments(std::integral_constant<int, 2>());
#endif
      }
'''

NEW_BODY = '''  // Q8V11: roles3/4 exist only inside the outer guarded ownership branches.
  if constexpr (Q8RoleEnabledV10<L, FLAT>::value && (ROLE == 3 || ROLE == 4)) {
    q8_role_body_v10<ROLE>(p, S, b, pair, start, end, seq, copy, sl2);
    return true;
  }
'''


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build():
    parent = PARENT.read_text()
    assert digest(parent) == PARENT_SHA
    edits = [
        ("__device__ __forceinline__ void q8_role_body_v10(",
         "template <int ROLE>\n__device__ __forceinline__ void q8_role_body_v10("),
        ("  if (tid < 128) q8_role_qk_v10(p, S, M, start, end, seq, ssc);\n"
         "  else q8_role_pv_v10(p, S, M, b, pair, copy, (end - start + 63) / 64);",
         "  static_assert(ROLE == 3 || ROLE == 4, \"specialized QK/PV role required\");\n"
         "  if constexpr (ROLE == 3) q8_role_qk_v10(p, S, M, start, end, seq, ssc);\n"
         "  else q8_role_pv_v10(p, S, M, b, pair, copy, (end - start + 63) / 64);"),
        ("if (ROLE == 2 && tid == 0)", "if ((ROLE == 2 || ROLE == 3) && tid == 0)"),
        (BODY_INSERT, NEW_BODY),
        (OLD_DISPATCH, NEW_DISPATCH),
        ("muse_att_q8role_v10_", "muse_att_q8role_v11_"),
        ("okc_attn_q8role_v10{tag}_", "okc_attn_q8role_v11{tag}_"),
    ]
    child = parent
    for before, after in edits:
        child = replace_once(child, before, after)
    restored = child
    for before, after in reversed(edits):
        restored = replace_once(restored, after, before)
    assert restored == parent, "unlisted V11 delta"
    ast.parse(child)
    return child, {"schema": 1, "parent_sha256": PARENT_SHA, "candidate_sha256": digest(child),
                   "exact_reverse_proof": True, "changes": ["compile-time body role", "compile-time QK/PV call",
                   "ROLE3 initializes ring", "specialized segment dispatch", "register budget encloses each role",
                   "two unique extension namespace renames"],
                   "register_budgets": {"producer": 32, "qk": 96, "pv": 112},
                   "gpu_tested": False}


if __name__ == "__main__":
    source, proof = build()
    OUTPUT.write_text(source)
    wrapper, wrapper_proof = build_wrapper(source, proof["candidate_sha256"], "decoder_q8_role_v11")
    wrapper_path = ROOT / "muse_attention_decode/candidates/decoder_q8_role_v11.py"
    wrapper_path.write_text(wrapper)
    proof["wrapper"] = wrapper_proof
    proof["wrapper"]["path"] = str(wrapper_path.relative_to(ROOT))
    (ROOT / "experiments/attention_q8_role_v11_provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))

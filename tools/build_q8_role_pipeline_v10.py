"""Reproducibly embed the isolated Q8 prototype; CPU only, no submission."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
PARENT = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")
BASE = ROOT / "muse_decode_layer/candidates/native_q8_role_pipeline_v10"

UNION_OLD = """  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
"""
UNION_NEW = """  // Q8V10_BEGIN_UNION
  union {
    struct {
      __nv_bfloat16 kvst[L * 4 * 64];
      __nv_bfloat16 gst[L * 4 * 64];
    };
    // Same byte count/alignment as the old fields, for every L and F.
    uint8_t q8_role_scratch_v10[L * 4 * 64 * 4];
  };
  // Q8V10_END_UNION
"""
BODY_ANCHOR = "  // ---- consumer state ----\n"
BODY_INSERT = """  // Q8V10_BEGIN_BODY
  if constexpr (Q8RoleEnabledV10<L, FLAT>::value) {
    if (q8_role_runtime_v10(p)) {
      q8_role_body_v10(p, S, b, pair, start, end, seq, copy, sl2);
      return true;
    }
  }
  // Q8V10_END_BODY
"""
REG_OLD = """      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
"""
REG_NEW = """      // Q8V10_BEGIN_REG
      if constexpr (Q8RoleEnabledV10<L, FLAT>::value && OKC_SETMAXNREG) {
        if (q8_role_runtime_v10(p)) {
          if (threadIdx.x < 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\\n" ::: "memory");
          else asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\\n" ::: "memory");
        } else asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
      } else if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
      // Q8V10_END_REG
"""
HELPER_ANCHOR = "template <int L, bool FLAT, int ROLE, int MODE = 0>\n"
MODULE_NAMES = (("muse_att_v298w1_", "muse_att_q8role_v10_"), ("okc_attn_v298{tag}_", "okc_attn_q8role_v10{tag}_"))


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def build():
    parent = PARENT.read_text()
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    fragment = BASE.with_suffix(".cuh").read_text()
    helpers = "// Q8V10_BEGIN_HELPERS\n" + fragment + "// Q8V10_END_HELPERS\n\n"
    child = replace_once(parent, UNION_OLD, UNION_NEW)
    child = replace_once(child, HELPER_ANCHOR, helpers + HELPER_ANCHOR)
    child = replace_once(child, BODY_ANCHOR, BODY_INSERT + BODY_ANCHOR)
    child = replace_once(child, REG_OLD, REG_NEW)
    for old, new in MODULE_NAMES:
        child = replace_once(child, old, new)
    ast.parse(child)
    restored = child
    for old, new in reversed(MODULE_NAMES):
        restored = replace_once(restored, new, old)
    restored = replace_once(restored, REG_NEW, REG_OLD)
    restored = replace_once(restored, BODY_INSERT, "")
    restored = replace_once(restored, helpers, "")
    restored = replace_once(restored, UNION_NEW, UNION_OLD)
    assert restored == parent, "unlisted parent delta"
    # Separately prove Python outside ATT7's native string and module names.
    pa, ca = ast.parse(parent), ast.parse(child)
    for tree in (pa, ca):
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("\n#include <torch/extension.h>") and "q8_role" in node.value:
                    node.value = next(n.value for n in ast.walk(pa) if isinstance(n, ast.Constant)
                                      and isinstance(n.value, str) and "struct Params" in n.value and "attn_kernel" in n.value)
                for old, new in MODULE_NAMES:
                    node.value = node.value.replace(new.split("{")[0], old.split("{")[0])
    assert ast.dump(pa, include_attributes=False) == ast.dump(ca, include_attributes=False)
    return child


if __name__ == "__main__":
    text = build()
    BASE.with_suffix(".py").write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())

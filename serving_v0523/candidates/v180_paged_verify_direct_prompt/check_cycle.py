"""Exercise the measured V68 and combined backend on one prompt followed by verification."""

from collections import Counter
from copy import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
fixture = HERE.parent / "v178b_live_direct_prompt/check_replay.py"
spec = importlib.util.spec_from_file_location("direct_replay", fixture)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
shared = replay.shared
shared.V64 = "f9966bf0dfca287d58d8d24ea2d055b3c2162702"
shared.src.cache_clear()
stats = Counter()
native_make = shared.mk


def make_backend(*args):
    backend, pool, calls, writes = native_make(*args)
    backend._hopper_verify_sms = 132
    backend._disable_scheduler_metadata_precompute = False
    original = backend.forward_extend
    namespace = original.__func__.__globals__
    ordinary = namespace["flash_attn_with_kvcache"]

    def prompt_then_verify(q, k, v, layer, batch, save_kv_cache, **kw):
        result = original(q, k, v, layer, batch, save_kv_cache, **kw)
        original_metadata = backend.forward_metadata
        verify_md = copy(original_metadata)
        verify_md.cu_seqlens_q = torch.tensor([0, 8], dtype=torch.int32)
        verify_md.max_seq_len_q = 8
        batch = copy(batch)
        batch.forward_mode = S(is_target_verify=lambda: True)
        batch.spec_algorithm = S(is_dflash=lambda: True)
        batch.spec_info = S(ragged_verify_layout=None)
        batch._attn_output = None
        layer = copy(layer)
        layer.logit_cap = 0.
        table = verify_md.swa_page_table if layer.sliding_window_size >= 0 else verify_md.page_table
        physical = pool.get_kv_buffer(layer.layer_id)
        trace = []
        def check(q_, key, value, page, lens):
            assert page is table and lens is verify_md.cache_seqlens_int32
            assert key.data_ptr() == physical[0].data_ptr() and value.data_ptr() == physical[1].data_ptr()
            assert q_.shape[0] == 8
            return q_.bfloat16()
        def paged(q_, key, value, page, lens, cu, **opts):
            assert opts["num_sms"] == 132 and opts["max_q"] == 8 and cu is verify_md.cu_seqlens_q
            trace.append("paged")
            output = check(q_, key, value, page, lens)
            return output if accept else None
        def native_fa(**opts):
            trace.append("FA3")
            return check(opts["q"], opts["k_cache"], opts["v_cache"], opts["page_table"], opts["cache_seqlens"])
        try:
            backend.forward_metadata = verify_md
            namespace["paged_verify_attention"] = paged
            namespace["flash_attn_with_kvcache"] = native_fa
            for accept in (True, False):
                trace.clear()
                output = original(q[:8], None, None, layer, batch, False)
                assert trace == (["paged"] if accept else ["paged", "FA3"])
                assert output.shape[0] == 8
                stats["paged_verify_success" if accept else "paged_verify_decline_to_FA3"] += 1
        finally:
            backend.forward_metadata = original_metadata
            namespace["flash_attn_with_kvcache"] = ordinary
        stats["prompt_verify_cycles"] += 1
        return result

    backend.forward_extend = prompt_then_verify
    return backend, pool, calls, writes


if __name__ == "__main__":
    cases = []
    with patch.object(shared, "add", side_effect=replay.inject), patch.object(shared, "mk", side_effect=make_backend):
        for window in (-1, 2047):
            for prefix in (0, 5017):
                for label, zeros, dense in (("warmup", True, False), ("full", False, False),
                    ("eager_dummy", True, False), ("eager", False, True), ("live", False, True), ("wrong_raw", False, False)):
                    cases.append(replay.test(label, 4127, prefix, window, 97, zeros, dense))
    stats["physical_pool_bytes_compared"] = sum(c["all_physical_pool_bytes"] for c in cases)
    stats["matched_V68_V180_prompt_pairs"] = len(cases)
    stats["direct_dense_V180_calls"] = sum(c["dense"] for c in cases)
    print(json.dumps(dict(reference="H200-passing V68 source", torch=torch.__version__, counts=stats,
        backend_scope="Real Radix eager callback, write, cache metadata and FA3 selection; host substitutes for device math",
        physical_H200=False), indent=2))

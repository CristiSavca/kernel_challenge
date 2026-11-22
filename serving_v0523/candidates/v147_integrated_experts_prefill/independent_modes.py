"""Run prefill and verify host oracles on the merged backend, including stale certificates."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace


def check(stage):
    dirs=Path(__file__).resolve().parents[1]
    ext=runpy.run_path(str(dirs/"v146_zero_copy_window_prefill/independent_dispatch.py"))["run"](stage)
    verify=runpy.run_path(str(dirs/"v136_fused_verify_qk_rope_cache/independent_operands.py"))["check"]
    env=verify.__globals__
    raw_extract=env["extract"]
    class StaleCertificate:
        def view(self,*args):
            raise AssertionError("target verification consumed a stale prefill certificate")
    counter=[]
    def namespace(**kwargs):
        if "max_seq_len_q" in kwargs and "cu_seqlens_k" in kwargs:
            kwargs.update(prefill_kv_span=StaleCertificate(),window_kv_span=StaleCertificate())
            counter.append(1)
        return SimpleNamespace(**kwargs)
    # Install only host-fixture inputs. Both actual forward definitions still come from the target tree.
    env["NS"]=namespace
    env["extract"]=lambda *args:raw_extract(*args)+"\nclass ForwardMode: EXTEND=object()\n"
    result=verify(stage)
    assert len(counter)==1 and result["passed_layers"]==62
    return dict(prefill_wrapper_pairs=ext["wrapper_pairs"],prefill_physical_bytes=ext["physical_kv_bytes"],
                numerical_reference_components=ext["fp32_reference_outputs"],
                inherited_full_prefill_cases=ext["inherited_full_and_fallback"]["pair_cases"],
                stale_certificates_ignored_in_verify=result,backend_sha256=ext["backend_sha256"],
                kv_span_sha256=ext["source_sha256"],physical_gpu=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path)
    print(json.dumps(check(p.parse_args().stage),indent=2))

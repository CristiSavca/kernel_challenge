#!/usr/bin/env python3
"""Bounded SS-A geometry/lifetime/scope arithmetic; no candidate or GPU execution."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    parent = ROOT / "experiments/candidates/muse_moe_w4a16" / (PARENT_SHA + ".py")
    assert digest(parent) == PARENT_SHA
    source = parent.read_text()
    tree = ast.parse(source)
    cuda = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets)))
    for text in ("constexpr int WGS_NS = 3;", "constexpr int WGS_PIECE = 9216;", "constexpr int WGS_XS = WGS_NT * 4096;",
                 "uint32_t af[2][GS][4];", "wg_wait<1>();", "constexpr int MMAX = 384"):
        assert text in cuda, text
    # Full byte bijection from each RS register fragment to the proposed SS-A descriptor.
    seen = set(); stores = 0
    for bank in range(2):
        for wg in range(4):
            for jj in range(2):
                for wp in range(4):
                    for reg in range(4):
                        banks = []
                        for lane in range(32):
                            m = wp*16 + (lane>>2) + 8*(reg&1)
                            k = 2*(lane&3) + 8*(reg>>1)
                            off = (m>>3)*256 + (k>>3)*128 + (m&7)*16 + (k&7)*2
                            assert off % 4 == 0
                            banks.append((off//4)%32)
                            addr = bank*16384 + wg*4096 + jj*2048 + off
                            for half in (0,1):
                                assert addr+2*half not in seen
                                seen.add(addr+2*half)
                            stores += 1
                        assert len(set(banks)) == 32
    assert seen == set(range(0,32768,2))
    # Serial per-WG event model. wait1 retires all but newest group; the next
    # pre-overwrite WG barrier is required to make this fact collective.
    lifetimes = 0
    for ntt in range(1,9):
        for start_seq in range(6):
            for stage_count in range(1,17):
                pending=[];owners={};releases=[];trace=[]
                for stage in range(stage_count):
                    seq=start_seq+stage
                    for ci in range(8):
                        bank=ci&1
                        assert all(g[2]!=bank for g in pending), (stage,ci,pending)
                        owners[bank]=(stage,ci)
                        trace.append(("preoverwrite_barrier",bank))
                        trace.append(("stores_proxy_fence_publication_barrier",bank))
                        pending.append((seq,ci,bank))
                        if len(pending)>1: pending.pop(0)
                        if ci==0 and stage>0:
                            assert all(g[0]!=seq-1 for g in pending)
                            releases.append(seq-1)
                    lifetimes += 1
                pending.clear()  # unchanged terminal wait0 before final x release/epilogue
                releases.append(start_seq+stage_count-1)
                assert releases == list(range(start_seq,start_seq+stage_count))
    # Too few banks under the same wait1 schedule must be rejected.
    pending=[];caught=False
    for ci in range(3):
        if any(g==0 for g in pending):caught=True;break
        pending.append(0)
        if len(pending)>1:pending.pop(0)
    assert caught
    metadata=22272;raw_stage=36864;x_stage=32768;a_banks=32768
    old_dynamic=metadata+3*(raw_stage+x_stage)
    new_dynamic=metadata+2*(raw_stage+x_stage)+a_banks
    assert (old_dynamic,new_dynamic)==(231168,194304)
    stages=48*16;groups=stages*8
    per_job={"weight_bytes":27*1024**2,"raw_weight_TMA_copies":stages,"activation_TMA_copies":stages,
             "CTA_stage_events":stages,"CTA_commit_group_events":groups,"WG_commit_groups":groups*4,
             "WG_WGMMA_instructions":groups*4*2,"new_A_shared_store_bytes":groups*16384,
             "new_A_logical_tensor_read_bytes":groups*16384,"new_preoverwrite_WG_barriers":groups*4,
             "new_publication_WG_barriers":groups*4}
    assert per_job["new_A_shared_store_bytes"]==96*1024**2
    routes_path=ROOT/"experiments/modal/20261104T043758Z-24108d21/result.json"
    routes=json.loads(routes_path.read_text())
    calls=[]
    for c in routes["cases"]:
        jobs=c["routing"]["J"]
        calls.append({"M":c["M"],"input_seed":c["input_seed"],"jobs":jobs,
                      "additional_A_shared_store_plus_logical_read_bytes":2*96*1024**2*jobs,
                      "additional_WG_barriers":2*groups*4*jobs})
    roi_path=ROOT/"experiments/iterations/20261104-next-dense-attention-roi-gate.json"
    profile_path=ROOT/"experiments/candidates/muse_decode_layer/profiles/97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2.7a8c4941741645ee.json"
    cases=json.loads(roi_path.read_text())["cases"];profiles=json.loads(profile_path.read_text())["profiles"]
    durations={}
    for cap in profiles:
        rows=[float(m.group(1)) for line in cap["report"].splitlines()
              if "moe_fused<" in line and (m:=re.search(r"([\d.]+) µs",line))]
        assert len(rows)==1
        durations[cap["index"]]=rows[0]
    def scenario(reduction):
        loads={}
        for c in cases:
            before=c["normalized_us"]*c["weight"]
            fraction=durations[c["index"]]/c["profile_us"] if c["tokens"]>=32 else 0
            pair=loads.setdefault(c["load"],[0.,0.]);pair[0]+=before;pair[1]+=before*(1-reduction*fraction)
        return 1-math.prod(p[1]/p[0] for p in loads.values())**(1/len(loads))
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if scenario(mid)<.05:lo=mid
        else:hi=mid
    return {"parent_sha256":PARENT_SHA,"parent_source":str(parent.relative_to(ROOT)),
        "compiler_evidence":"Parent C7511/C7512 status UNKNOWN; only failed WAIT2/REG112 variants positively serialized.",
        "geometry":{"rows":256,"K_per_stage":256,"FT":64,"threads":576,"weight_ring":2,"X_ring":2,
                    "decoded_A_banks":2,"K_per_A_bank":32,"metadata_bytes":metadata,"old_dynamic":old_dynamic,
                    "new_dynamic":new_dynamic,"A_bank_bytes":16384,"A_descriptor_LBO":128,"A_descriptor_SBO":256},
        "coordinate_proof":{"BF16_locations":len(seen),"uint32_stores":stores,"per_warp_store_conflict_free":True,
                            "memory_layout":"K-major no-swizzle core8x8; separate2048B m64k16 tile perjj/WG"},
        "lifetime_model":{"stage_checks":lifetimes,"NTT":list(range(1,9)),"initial_seq":list(range(6)),
                          "stage_lengths":list(range(1,17)),"single_bank_negative_detected":True,
                          "scope":"abstract perWG retirement + explicit collective pre/post barriers; no CUDA memory-order proof"},
        "per_job":per_job,"standalone_route_examples":calls,"routes_source_sha256":digest(routes_path),
        "profile_sha256":digest(profile_path),"whole_decoder_profile_scenarios_not_predictions":{
            "MAIN_reduction20pct":scenario(.2),"MAIN_reduction25pct":scenario(.25),"MAIN_reduction50pct":scenario(.5),
            "MAIN_reduction_needed_for5pct_whole":hi},
        "decision":"Concrete fixed plan is source/math/resource feasible for1CTA; new compiled resources/correctness/performance UNKNOWN.",
        "candidate_written":False,"GPU_or_official_calls":False}


if __name__=="__main__":
    result=run();path=ROOT/"experiments/iterations/20261104-wgs-shared-a-feasibility.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"saved":str(path.relative_to(ROOT)),"geometry":result["geometry"],
                      "ROI":result["whole_decoder_profile_scenarios_not_predictions"]}))

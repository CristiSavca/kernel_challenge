"""Record terminal dependencies, fixed-row linear work and historical source-based traffic."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
URL="https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/resolve/46c90f412549c472941a68382d91a99e5e6372a0/config.json"


def main(stage):
    m=json.loads((HERE/"validation.json").read_bytes())
    contract=json.loads(subprocess.check_output(["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
    raw=subprocess.check_output(["curl","-fsSL",URL]);cfg=json.loads(raw)
    assert cfg["dflash_config"]["target_layer_ids"]==[1,16,30,44,59] and cfg["num_target_layers"]==62
    path="python/sglang/srt/configs/muse_spark.py"
    files={name:subprocess.check_output(["git","-C",stage,"show",m["tree"]+":"+name]).decode() for name in [
        path,"python/sglang/srt/models/muse_spark.py","python/sglang/srt/speculative/dflash_utils.py",
        "python/sglang/srt/layers/logits_processor.py","python/sglang/srt/layers/muse_spark/muse_spark_layers.py"]}
    cls=next(n for n in ast.parse(files[path]).body if getattr(n,"name",None)=="MuseSparkTextConfig")
    init=next(n for n in cls.body if getattr(n,"name",None)=="__init__")
    defaults={a.arg:ast.literal_eval(v) for a,v in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)}
    h,dim,q,kv,eh,inter,top,experts,layers=[defaults[k] for k in ["hidden_size","head_dim","num_attention_heads","num_key_value_heads","moe_hidden_size","moe_intermediate_size","num_experts_per_tok","num_local_experts","num_hidden_layers"]]
    per_token=dict(qkv_gate=2*h*dim*(2*q+2*kv), output=2*h*dim*q, router_two_factors=4*h*experts,
                   pool_and_unpool=4*h*eh, expert_gate_up_down=2*3*top*eh*inter)
    fixed=sum(per_token.values());kv_only=4*h*dim*kv
    trace=json.loads((HERE.parent/"v251_hicache_residency/trace.json").read_bytes())
    hist=json.loads((HERE.parent/"v197_grouped_prompt_gate_up/independent_reach.json").read_bytes())
    archive=[]
    for x,y in zip(trace["v41"]["windows"], hist["archive"]):
        assert x["concurrency"]==y["users"] and x["rep"]==y["rep"]
        n=y["eligible_logical_rows"]; saved=n*h*24//8
        archive.append(dict(users=x["concurrency"],rep=x["rep"],historical_grouped_rows=n,
            logical_stream_bytes_saved_per_rank=saved,seconds_at_peak_HBM=saved/4.8e12,
            equivalent_fraction_of_repetition_at_peak=saved/4.8e12/x["wall_s"]))
    print(json.dumps(dict(patch_sha256=m["patch_sha256"],published_draft_config=URL,
        checkpoint_config_sha256=sha256(raw).hexdigest(),configured_residual_taps=cfg["dflash_config"]["target_layer_ids"],
        last_layer=layers-1,scope="Linear work models 62 identical sparse layers at the pinned target dimensions. Attention-score work, softmax, normalization, activation, and uneven expert loads are omitted.",
        source_sha256={k:sha256(v.encode()).hexdigest() for k,v in files.items()},api=contract["rules"]["summary"],
        nominal_FLOPs_per_token_per_layer=per_token,necessary_last_KV_projection_FLOPs=kv_only,
        hypothetical_linear_fraction_if_last_layer_kept_only_KV=(fixed-kv_only)/(fixed*layers),
        hypothetical_linear_fraction_if_last_FFN_skipped=(per_token["router_two_factors"]+per_token["pool_and_unpool"]+per_token["expert_gate_up_down"])/(fixed*layers),
        hypothetical_16k_single_endpoint_expert_routes_omitted=(16384-1)*top,
        retained_rows_in_V257="all",parent_Torch_logical_stream_bytes_per_element=34,candidate_logical_stream_bytes_per_element=10,
        cold16k_owner_rows=2048,rank_16k_logical_bytes_saved=2048*h*24,estimated_weighted_traffic=archive),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))

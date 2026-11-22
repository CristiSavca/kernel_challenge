"""Inspect genuine NVCC production call wrappers, backedges, bulk copies and register resources."""
import gzip,hashlib,importlib.util,json,re,sys
from pathlib import Path
c=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("v153_aot",c/"v153_cooperative_expert_channels/compile_check.py")
parser=importlib.util.module_from_spec(sp);sp.loader.exec_module(parser)
prior={}
for parent in ("v161_tma_expert_channels","v153_cooperative_expert_channels"):
    prior.update(parser.functions(gzip.open(c/parent/"sm90a.sass.gz","rt").read()))
out=Path(sys.argv[1]);results={}
for arch in ("sm_90","sm_90a"):
    sections=parser.functions((out/(arch+".sass")).read_text());log=(out/(arch+".log")).read_text()
    kernels={}
    for name,body in sections.items():
        klass=next((s for s in ("streamed_projection","tensor_projection","projection_channels") if s in name),None)
        if not klass: continue
        projection="w13" if "ILb1" in name else "w2"
        k=parser.kernel(body,log,name)
        if klass=="streamed_projection":
            assert k["steady_backedge"]["histogram"]["BAR"]==k["steady_backedge"]["histogram"]["DEPBAR"]==0
            assert k["steady_backedge"]["histogram"]["HMMA"]==(16 if projection=="w13" else 8)
            assert "SYNCS.PHASECHK" in body and "UTMALDG" in body
        if name in prior and arch=="sm_90a":
            previous = prior[name]
            instructions = lambda data: re.findall(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*", data, re.M)
            old, new = instructions(previous), instructions(body)
            assert len(old) == len(new)
            changes = [(a,b) for a,b in zip(old,new) if a != b]
            def instruction_shape(op):
                return re.sub(r"\bR(?:10|11|12|14|16|18)\b", "R?", op)
            assert all(a[0] == b[0] and instruction_shape(a[1]) == instruction_shape(b[1]) for a,b in changes)
            assert all(a[1].split()[0] in ("CS2R", "IADD3", "IMAD.X") for a,b in changes)
            k["prior_binary_identical"] = parser.machine_code(body)==parser.machine_code(previous)
            k["prior_reordered_zero_init_or_renumbered_address_register_sites"] = len(changes)
            if klass == "tensor_projection": assert k["prior_binary_identical"]
        k["machine_words_sha256"]=hashlib.sha256(parser.machine_code(body)).hexdigest()
        kernels[klass+"_"+projection]=k
    assert len(kernels)==6
    results[arch]=dict(sha256=hashlib.sha256((out/(arch+".o")).read_bytes()).hexdigest(),kernels=kernels)
print(json.dumps(dict(nvcc="12.9.86",targets=results,v161_tensor_fallback_machine_code_exact_sm90a=True, v153_vector_fallback_same_instruction_order_except_reordered_zero_inits_and_address_register_allocation=True,
    compile_command="../v161_tma_expert_channels/compile_sm90.sh using assembled V165 source",hardware=False),indent=2))

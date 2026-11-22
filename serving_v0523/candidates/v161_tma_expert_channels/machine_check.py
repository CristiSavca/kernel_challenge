"""Compare the compiled TMA specialization with the frozen V153 instruction stream."""
import gzip, hashlib, importlib.util, json, re, sys
from collections import Counter
from pathlib import Path
here=Path(__file__).resolve().parent
prior=here.parent/"v153_cooperative_expert_channels"
sp=importlib.util.spec_from_file_location("v153_compile",prior/"compile_check.py")
ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
old=ref.functions(gzip.open(prior/"sm90a.sass.gz","rt").read())
out=Path(sys.argv[1]); data={}
for arch in ("sm_90","sm_90a"):
    section=ref.functions((out/(arch+".sass")).read_text())
    log=(out/(arch+".log")).read_text()
    ops={}
    for name, text in section.items():
        if "tensor_projection" not in name and "projection_channels" not in name:continue
        mode="tma" if "tensor_projection" in name else "vector"
        proj="w13" if "ILb1" in name else "w2"
        k=ref.kernel(text,log,name)
        if mode=="vector" and arch=="sm_90a":
            assert ref.machine_code(text)==ref.machine_code(old[name])
        a,b=(int(k["steady_backedge"][s],16) for s in ("begin","end"))
        loop=[op.split()[int(op.startswith("@"))].split(".")[0] for pc,op in
              re.findall(r"^\s*/\*([\da-f]+)\*/\s+(.*?)\s*;\s*/\*",text,re.M) if a<=int(pc,16)<=b]
        counts=Counter(loop)
        k["steady_backedge"]["histogram"]["UTMALDG"]=counts["UTMALDG"]
        assert counts["UTMALDG"]==((4 if proj=="w13" else 2) if mode=="tma" else 0)
        k["machine_words_sha256"]=hashlib.sha256(ref.machine_code(text)).hexdigest()
        ops[mode+"_"+proj]=k
    assert len(ops)==4
    data[arch]=dict(kernels=ops,object_sha256=hashlib.sha256((out/(arch+".o")).read_bytes()).hexdigest())
print(json.dumps(dict(nvcc="12.9.86",targets=data,vector_fallback_matches_prior_sm90a_machine_words=True,
    static_sites_are_not_cycles=True),indent=2))

"""Actual V33 packing/decoder/stream proof on CPU; no CUDA performance claims."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
b = load("v33_builder", HERE / "build.py")
v29 = load("v33_pack_adapter", HERE.parent / "decoder_exact_patch_v29/test_cpu.py")
Tensor, raw = v29.Tensor, v29.raw


def cases(env):
    out = []
    for base in range(0,128,8):
        x=np.array([(s<<15)|(e<<8)|lo for s in range(2) for e in range(base,base+8) for lo in range(256)],np.uint16).reshape(1,-1,512)
        out.append(("normal_allbits_"+str(base),x))
    # Two copies preserve every bit pattern even where forcing an outlier replaces a value.
    x=np.full((1,512,512),0x3f80,np.uint16)
    x[0,:128]=np.arange(65536,dtype=np.uint16).reshape(128,512)
    x[0,128:256]=x[0,:128]
    x[0,:128,0]=0; x[0,128:256,511]=0
    out.append(("raw_allbits_nonzero_indices",x))
    x=np.full((1,4352,512),0x3f80,np.uint16)
    special=[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc1,0x7f81]
    for i in range(4096): x[0,i,i%512]=special[i//512]
    out.append(("every_position_extremes",x))
    out.append(("zero_raw",np.full((1,32,512),0x3f80,np.uint16)))
    packed=[]
    for name,x in out:
        c=env["bfp_sk2_compress"](v29.from_units(x)); assert c is not None,name
        d=raw(c["descriptors"]).view(np.uint32); mask=(d>>31)!=0
        assert int(mask.sum())==c["stats"][2]
        assert np.all(d[~mask]==0)
        assert np.array_equal(d[mask]&0x7fffffff,np.arange(int(mask.sum()),dtype=np.uint32))
        assert raw(c["raw"]).shape==(max(1,int(mask.sum())),256)
        if name=="zero_raw": assert not mask.any()
        if name=="raw_allbits_nonzero_indices": assert mask[0,:256].all()
        packed.append((name,x,c))
    # Exact economic cutoff:62/64 is profitable;63/64 costs exactly raw BF16.
    for nraw in (62,63,64):
        x=np.full((1,64,512),0x3f80,np.uint16); x[0,:nraw,0]=0
        c=env["bfp_sk2_compress"](v29.from_units(x))
        assert (c is None)==(nraw>=63)
    entropy=np.arange(65536,dtype=np.uint16).reshape(512,128).T.reshape(1,128,512)
    assert env["bfp_sk2_compress"](v29.from_units(entropy)) is None
    return packed


def source_check():
    p=(HERE/"parent.py").read_text(); s=(HERE/"submission.py").read_text()
    built,edits=b.build(); assert built==s
    for e in reversed(edits): s=s.replace(e["new"],e["old"])
    assert s==p
    s=(HERE/"submission.py").read_text(); ps=v29.literal(ast.parse(p),"bfp_SK2_SRC"); cs=v29.literal(ast.parse(s),"bfp_SK2_SRC")
    # Every scalar MMA operand-selection and accumulation statement is inherited from V32.
    def region(x,a,z):return x[x.index(a):x.index(z,x.index(a))]
    assert region(ps,"      // 3) tensor-core work",'  if (pdl) asm volatile("griddepcontrol.launch_dependents;"')==region(cs,"      // 3) tensor-core work",'  if (pdl) asm volatile("griddepcontrol.launch_dependents;"')
    assert region(ps,"          const uint32_t a0 = wb", "      uint32_t a[4][MT][4];")==region(cs,"          const uint32_t a0 = wb", "      uint32_t a[4][MT][4];")
    for text in ["if (UPW == 1 && uu == 0)","if (UPW > 1)","if (lane == 0) mbar_arrive(empty0 + 8 * slot);",
                 "constexpr int NESC = COMP ? 0 : 1;","constexpr int WBY = WPAY + (COMP ? ((NW * UPW * NT * 4 + 15) & ~15) : 0);"]:
        assert ps.count(text)==cs.count(text)>0,text
    con=cs[cs.index("      uint4 lw[NT];"):cs.index("      // 3) tensor-core work")]
    assert con.index('"=r"(lw[nt].x)')<con.index('"=r"(repair[nt])')<con.index("mbar_arrive(empty0")<con.index("const uint2* hp =")
    assert "sk2_patch_word" not in cs and "sk2_decode_patch" not in cs
    return s,cs,{"parent_sha256":b.sha(p),"candidate_sha256":b.sha(s),"full_restoration":True,
                 "MMA_and_nonCOMP_loads_identical":True,"ring_handbacks_identical":True,
                 "primary_loads_independent_of_header":True,"immutable_sidecar_after_early_handback":True}


def stream_check(env):
    emitted=env["bfp__sk2_src"](True)
    vals=re.findall(r"launch_sk2<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (true|false), (true|false), (\d+), (\d+),",emitted)
    rows=[]; count=0
    for val in sorted(set(vals)):
        mt,tokr,nt,nwr,wsk,upw,ns=map(int,val[:7]); comp=val[7]=="true"; nw=nwr*wsk; ups=wsk*upw; r=nwr*nt*8
        tail=nw*upw*nt*4 if comp else 0; assert tail%16==0
        rows.append({"template":list(val),"header_bytes":tail,"padding_bytes":0})
        if not comp:continue
        shape=(r//8*3,ups*3); ids=np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)
        headers=ids.copy();headers[(ids&1)!=0]|=0x80000000
        primary=np.broadcast_to((ids&255).astype(np.uint8)[:,:,None],shape+(768,)).copy()
        c={"packed":Tensor(primary),"descriptors":Tensor(headers.view(np.int32))}
        st=raw(env["bfp_sk2_patch_stream_v29"](c,r*3,r,nt,nwr,wsk,upw));wp=nw*upw*nt*768
        for stage in range(3):
            for cta in range(3):
                for w in range(nw):
                    for u in range(upw):
                        for t in range(nt):
                            tile=cta*(r//8)+(w//wsk)*nt+t; k=stage*ups+(w%wsk)*upw+u;j=(w*upw+u)*nt+t
                            assert np.array_equal(st[stage,cta,j*768:(j+1)*768],primary[tile,k])
                            assert int(np.frombuffer(st[stage,cta,wp+j*4:wp+j*4+4].tobytes(),np.uint32)[0])==int(headers[tile,k]);count+=1
    assert len(rows)==17
    return {"geometries":rows,"unit_mappings":count,"all_actual_header_tails_aligned_no_extra_padding":True}


def native(cs, packed):
    normal=v29.block(cs,cs.index("__device__ __forceinline__ uint4 sk2_decode_value_v32("))
    complete=v29.block(cs,cs.index("__device__ __forceinline__ uint4 sk2_decode_raw_v33("))
    start=cs.index("          if (repair[nt] & 0x80000000u) {")
    mid=v29.block(cs,start); endstart=cs.index(" else {",start+len(mid))
    # Include both actual source branches; the only intrinsic substitution is the checked read.
    branch=cs[start:endstart]+v29.block(cs,endstart).replace("__ldg(hp)","checked(hp)")
    branch=branch.replace("__ldg(hp)","checked(hp)")
    body=r'''
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
struct uint4{uint32_t x,y,z,w;};struct uint2{uint32_t x,y;};
uint4 make_uint4(uint32_t x,uint32_t y,uint32_t z,uint32_t w){return{x,y,z,w};}
void ck(bool x){if(!x)std::exit(2);}
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c){return(a&~c)|(b&c);}
uint32_t prmt(uint32_t a,uint32_t b,unsigned s){uint64_t z=uint64_t(a)|(uint64_t(b)<<32);uint32_t r=0;for(int i=0;i<4;i++)r|=uint32_t((z>>(8*((s>>(4*i))&7)))&255)<<(8*i);return r;}
const uint2* first;const uint2* last;uint2 checked(const uint2*p){ck(p>=first&&p<last);return*p;}
'''+normal+'\n'+complete+r'''
int main(int argc,char**argv){std::ifstream f(argv[1],std::ios::binary);uint32_t nc;f.read((char*)&nc,4);uint64_t count=0,units=0,nraw=0;
for(unsigned ci=0;ci<nc;ci++){uint32_t meta[3];f.read((char*)meta,12);unsigned n=meta[0],nr=meta[1];uint32_t base4=meta[2];
std::vector<uint8_t> primary(n*768);std::vector<uint32_t> descriptors(n);std::vector<uint2> side(nr*32);std::vector<uint16_t> expected(n*512);
f.read((char*)primary.data(),primary.size());f.read((char*)descriptors.data(),n*4);f.read((char*)side.data(),side.size()*8);f.read((char*)expected.data(),expected.size()*2);ck(f.good());
first=side.data();last=first+side.size();const void*ptab=first;
for(unsigned u=0;u<n;u++){units++;nraw+=bool(descriptors[u]&0x80000000u);
for(int lane=0;lane<32;lane++){const int nt=0;uint32_t repair[1]={descriptors[u]};uint4 lw[1],bv0[1],bv1[1];uint2 cw[1];
std::memcpy(lw,primary.data()+u*768+lane*16,16);std::memcpy(cw,primary.data()+u*768+512+lane*8,8);
'''+branch+r'''
uint32_t a[8]={bv0[0].x,bv0[0].y,bv0[0].z,bv0[0].w,bv1[0].x,bv1[0].y,bv1[0].z,bv1[0].w};
for(int j=0;j<8;j++){ck(uint16_t(a[j])==expected[u*512+lane*16+2*j]);ck(uint16_t(a[j]>>16)==expected[u*512+lane*16+2*j+1]);count+=2;}
}}}
// Encoding/index arithmetic boundaries, with no impossible giant allocation.
for(uint32_t ix:{0u,1u,0x7ffffffeu,0x7fffffffu}){uint32_t d=0x80000000u|ix;ck((d&0x7fffffffu)==ix);ck((size_t)(d&0x7fffffffu)*32+31==size_t(ix)*32+31);}
std::cout<<"{\"units\":"<<units<<",\"raw_units\":"<<nraw<<",\"BF16_bit_comparisons\":"<<count<<"}";
}
'''
    mutants={"wrong_high_half":("(hw & 0xf0f0f0f0u)","(hw & 0x0f0f0f0fu)"),
             "wrong_raw_stride":("* 32 + lane","* 31 + lane"),
             "wrong_lane":("* 32 + lane","* 32 + (lane ^ 1)"),
             "wrong_second_plane":("cw[nt].y, h.y","cw[nt].y, h.x")}
    with tempfile.TemporaryDirectory(prefix="v33-codec-") as td:
        td=Path(td); data=td/"cases.bin"
        with data.open("wb") as f:
            f.write(np.uint32(len(packed)).tobytes())
            for name,x,c in packed:
                r=raw(c["raw"]);f.write(np.array([x.size//512,r.shape[0],c["base4"]],np.uint32).tobytes())
                f.write(raw(c["packed"]).tobytes());f.write(raw(c["descriptors"]).tobytes());f.write(r.tobytes());f.write(x.tobytes())
        def execute(src,name):
            exe=td/name;r=subprocess.run(["clang++","-std=c++17","-O2","-x","c++","-o",str(exe),"-"],input=src,text=True,capture_output=True)
            assert r.returncode==0,r.stderr
            return subprocess.run([str(exe),str(data)],text=True,capture_output=True)
        r=execute(body,"actual");assert r.returncode==0,r.stderr; result=json.loads(r.stdout)
        for name,(old,new) in mutants.items():
            assert body.count(old)==1,(name,body.count(old));r=execute(body.replace(old,new),name);assert r.returncode!=0,name
    return dict(result,native_source_sha256=b.sha(body),negative_controls=list(mutants),raw_index31_boundaries_checked=True)


def run():
    s,cs,identity=source_check();env=v29.helpers(ast.parse(s));packed=cases(env)
    result={"status":"CPU_PASS","identity":identity,"native":native(cs,packed),"stream":stream_check(env),
            "numpy_version":np.__version__,"zero_raw_and_economic_cutoff":True,"full_entropy_fallback":True,
            "limits":["Actual Python helpers execute through NumPy adapter, not Torch/CUDA", "CPU PRMT/LOP3 models do not prove CUDA code generation",
                      "No GPU, register allocation, achieved occupancy or speed claim", "Unchanged MMA and ring source inherit prior proof, not a new device-order proof"]}
    assert b.sha((HERE/"submission.py").read_bytes())==identity["candidate_sha256"]
    return result


if __name__=="__main__":
    text=json.dumps(run(),indent=2)+'\n';(HERE/"cpu_result.json").write_text(text);print(text)

"""Execute the actual pack/stream helpers and extracted decode/repair on a CPU.

No Torch/CUDA import, device compilation, GPU ordering or floating-point MMA proof.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import operator
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from test_dense_escape_global_cpu import ring_model


def raw(x): return x.a if isinstance(x, Tensor) else x


class Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    @property
    def shape(self): return self.a.shape
    @property
    def device(self): return "cpu"
    def __int__(self): return int(self.a)
    def numel(self): return self.a.size
    def long(self): return self.to(np.int64)
    def to(self, dtype): return Tensor(self.a.astype(dtype))
    def contiguous(self): return Tensor(np.ascontiguousarray(self.a))
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], type):
            assert self.a.flags.c_contiguous
            return Tensor(self.a.view(shape[0]))
        return Tensor(self.a.reshape(*shape))
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))
    def permute(self, *axes): return Tensor(self.a.transpose(*axes))
    def flatten(self): return Tensor(self.a.flatten())
    def clamp(self, lo, hi): return Tensor(self.a.clip(lo, hi))
    def sum(self, dim=None): return Tensor(self.a.sum(axis=dim))
    def argmax(self, dim=None): return Tensor(self.a.argmax(axis=dim))
    def unsqueeze(self, dim): return Tensor(np.expand_dims(self.a, dim))
    def squeeze(self, dim): return Tensor(np.squeeze(self.a, axis=dim))
    def gather(self, dim, idx): return Tensor(np.take_along_axis(self.a, raw(idx), axis=dim))
    def __getitem__(self, key):
        if isinstance(key, tuple): key = tuple(raw(x) for x in key)
        return Tensor(self.a[raw(key)])
    def __setitem__(self, key, value):
        if isinstance(key, tuple): key = tuple(raw(x) for x in key)
        self.a[raw(key)] = raw(value)


for name, op in {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                 "floordiv": operator.floordiv, "mod": operator.mod, "and": operator.and_,
                 "or": operator.or_, "lshift": operator.lshift, "rshift": operator.rshift,
                 "lt": operator.lt, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq}.items():
    setattr(Tensor, "__"+name+"__", lambda self, x, op=op: Tensor(op(self.a, raw(x))))
    setattr(Tensor, "__r"+name+"__", lambda self, x, op=op: Tensor(op(raw(x), self.a)))


class Torch:
    int16, int32, int64, uint8 = np.int16, np.int32, np.int64, np.uint8
    @staticmethod
    def arange(*args, dtype=np.int64, device=None): return Tensor(np.arange(*args, dtype=dtype))
    @staticmethod
    def zeros(shape, dtype, device=None): return Tensor(np.zeros(shape, dtype))
    @staticmethod
    def bincount(x, minlength): return Tensor(np.bincount(raw(x), minlength=minlength))
    @staticmethod
    def stack(xs): return Tensor(np.stack([raw(x) for x in xs]))
    @staticmethod
    def cat(xs, dim=-1): return Tensor(np.concatenate([raw(x) for x in xs], axis=dim))


def key(n):
    if isinstance(n, ast.FunctionDef): return n.name
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name): return n.targets[0].id


def literal(tree, name): return ast.literal_eval(next(n.value for n in tree.body if key(n) == name))


def block(src, start):
    end = src.index("{", start) + 1
    depth = 1
    while depth:
        depth += (src[end] == "{") - (src[end] == "}")
        end += 1
    return src[start:end]


def helpers(tree):
    names = {"bfp_SK2_SRC", "bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K", "bfp__sk2_geom",
             "bfp_sk2_tileunits", "bfp_sk2_compress", "bfp_sk2_stream", "bfp_sk2_patch_stream_v29",
             "bfp_sk2_table", "bfp__sk2_src", "_bfp__sk2_src_unfused",
             "SK2B", "SK2B_UN_TOKR", "SK2B1", "SK2B_O_TOKR"}
    env = {"torch": Torch}
    exec(compile(ast.Module(body=[n for n in tree.body if key(n) in names], type_ignores=[]), "<actual-pack-helpers>", "exec"), env)
    return env


def from_units(a):
    # Independent inverse of the documented B-fragment permutation.
    nt, nu, _ = a.shape
    out = np.empty((nt*8, nu*64), np.uint16)
    for lane in range(32):
        row, t = lane // 4, lane % 4
        for v in range(16):
            j, h, p = v//4, (v//2)%2, v%2
            out[row::8, 16*j+8*h+2*t+p::64] = a[:, :, lane*16+v]
    return Tensor(out)


def make_cases(env):
    cases = []
    # All 65,536 BF16 bit patterns in primary/no-repair mode, grouped by an
    # exact eight-value exponent-high window; includes signed zeros and NaNs.
    for base in range(0, 128, 8):
        vals = np.array([(s<<15)|(e<<8)|lo for s in range(2) for e in range(base,base+8) for lo in range(256)], np.uint16)
        units = vals.reshape(1, -1, 512)
        cases.append((f"allbits_window_{base}", units))
    # Every position is repaired, alternating extreme bit patterns. Majority
    # normal units make the matrix-wide window fixed without a codec assumption.
    special = [0, 0x8000, 0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0x7f81]
    a = np.full((1, 4096, 512), 0x3f80, np.uint16)
    for i in range(4096): a[0, i, i%512] = special[i//512]
    cases.append(("every_position_extremes", a))
    # All 65,536 replacement payloads, with positions varied across all512.
    a = np.full((1, 65536, 512), 0x3f80, np.uint16)
    a[0, np.arange(65536), (np.arange(65536)*37)%512] = np.arange(65536,dtype=np.uint16)
    cases.append(("all_repair_bits", a))
    # Raw units include every bit pattern; enough zero/single units to keep
    # the exact mixed format below whole-matrix raw requested bytes.
    a = np.full((1, 1024, 512), 0x3f80, np.uint16)
    a[0, :128] = np.arange(65536,dtype=np.uint16).reshape(128,512)
    a[0, 128, 0:2] = (0,0x7f80)
    a[0, 129, -2:] = (0x8000,0xffff)
    cases.append(("raw_all_bits_and_edges",a))
    packed = []
    for name, u in cases:
        comp = env["bfp_sk2_compress"](from_units(u))
        assert comp is not None, name
        assert np.array_equal(raw(env["bfp_sk2_tileunits"](from_units(u))).reshape(u.shape),u)
        desc = raw(comp["descriptors"]).view(np.uint32)
        assert np.all((desc>>30)<=2)
        assert comp["stats"][2] == int(np.sum((desc>>30)==2))
        packed.append((name,u,comp))
    # No speculative finite/outlier assumption: high entropy selects the
    # existing whole-matrix fallback rather than overflowing any encoding.
    entropy = np.arange(65536,dtype=np.uint16).reshape(1,128,512)
    assert env["bfp_sk2_compress"](from_units(entropy)) is None
    return packed


def native_codec(cs, all_source, cases):
    dec = block(all_source, all_source.index("__device__ __forceinline__ void decode8("))
    patch = block(cs, cs.index("__device__ __forceinline__ void sk2_patch_v29("))
    dec = dec.replace("__device__ __forceinline__", "inline")
    patch = patch.replace("__device__ __forceinline__", "inline")
    # The raw load expression is the exact device expression, with only the
    # CUDA type/load intrinsic replaced by CPU checked reads.
    rawstart = cs.index("            const uint4* rp = reinterpret_cast<const uint4*>(ptab)")
    rawend = cs.index("\n          } else {",rawstart)
    rawcode = cs[rawstart:rawend].replace("__ldg(rp)","checked(rp)").replace("__ldg(rp + 1)","checked(rp + 1)")
    body = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
struct uint4{uint32_t x,y,z,w;}; struct uint2{uint32_t x,y;};
const uint4* rawbegin;const uint4* rawend;
uint4 checked(const uint4*p){assert(p>=rawbegin&&p<rawend);return *p;}
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c){return(a&~c)|(b&c);}
uint32_t prmt(uint32_t a,uint32_t b,unsigned s){uint64_t z=uint64_t(a)|(uint64_t(b)<<32);uint32_t r=0;for(int i=0;i<4;i++)r|=uint32_t((z>>(8*((s>>(4*i))&7)))&255)<<(8*i);return r;}
'''+dec+'\n'+patch+r'''
int main(int argc,char**argv){std::ifstream f(argv[1],std::ios::binary);uint32_t nc;f.read((char*)&nc,4);uint64_t units=0,words=0,mode[3]={0,0,0};
for(uint32_t ci=0;ci<nc;ci++){uint32_t meta[3];f.read((char*)meta,12);uint32_t n=meta[0],nr=meta[1],base4=meta[2];
std::vector<uint8_t> packed(n*768);std::vector<uint32_t>d(n);std::vector<uint4>raw(nr*64);std::vector<uint16_t>expected(n*512);
f.read((char*)packed.data(),packed.size());f.read((char*)d.data(),d.size()*4);f.read((char*)raw.data(),raw.size()*16);f.read((char*)expected.data(),expected.size()*2);assert(f.good());
rawbegin=raw.data();rawend=raw.data()+raw.size();const void*ptab=raw.data();
for(uint32_t u=0;u<n;u++){mode[d[u]>>30]++;for(int lane=0;lane<32;lane++){uint32_t b[1][8];const int nt=0;uint32_t repair[1]={d[u]};
if((repair[nt]>>30)==2u){
'''+rawcode+r'''
}else{uint4 lw;uint2 cw;std::memcpy(&lw,packed.data()+u*768+lane*16,16);std::memcpy(&cw,packed.data()+u*768+512+lane*8,8);
decode8(lw.x,lw.y,cw.x,base4,&b[0][0]);decode8(lw.z,lw.w,cw.y,base4,&b[0][4]);sk2_patch_v29(repair[0],lane,b[0]);}
for(int j=0;j<8;j++){assert(uint16_t(b[0][j])==expected[u*512+lane*16+j*2]);assert(uint16_t(b[0][j]>>16)==expected[u*512+lane*16+j*2+1]);words+=2;}}
units++;}}
std::cout<<"{\"units\":"<<units<<",\"bf16_bit_comparisons\":"<<words<<",\"modes\":["<<mode[0]<<","<<mode[1]<<","<<mode[2]<<"]}";
}
'''
    with tempfile.TemporaryDirectory(prefix="v29-codec-") as td:
        td=Path(td); dat=td/"cases.bin"
        with dat.open("wb") as f:
            f.write(np.uint32(len(cases)).tobytes())
            for _,u,c in cases:
                p=raw(c["packed"]);r=raw(c["raw"])
                f.write(np.array([u.size//512,r.shape[0],c["base4"]],np.uint32).tobytes())
                f.write(p.tobytes());f.write(raw(c["descriptors"]).tobytes());f.write(r.tobytes());f.write(u.tobytes())
        def execute(src,label):
            exe=td/label
            p=subprocess.run(["clang++","-std=c++17","-O2","-x","c++","-o",str(exe),"-"],input=src,text=True,capture_output=True)
            assert p.returncode==0,p.stderr
            return subprocess.run([str(exe),str(dat)],capture_output=True,text=True)
        p=execute(body,"actual");assert p.returncode==0,p.stderr
        result=json.loads(p.stdout)
        mutations={"wrong_owner":("(d >> 20) & 31u","(d >> 21) & 31u"),
                   "wrong_half":("(v & 1u) * 16u","((v ^ 1u) & 1u) * 16u"),
                   "wrong_raw_index":("* 64 + lane * 2","* 63 + lane * 2")}
        for label,(a,b) in mutations.items():
            assert a in body
            assert execute(body.replace(a,b),label).returncode!=0,label
        result["source_mutants_rejected"]=list(mutations)
        result["native_translation_sha256"]=hashlib.sha256(body.encode()).hexdigest()
    return result


def geometry_and_stream(env, ps, cs):
    emitted=env["bfp__sk2_src"](True)
    templates=re.findall(r"launch_sk2<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (true|false), (true|false), (\d+), (\d+),",emitted)
    assert templates
    rows=[];mapped=0;ringcases=0
    for vals in sorted(set(templates)):
        mt,tokr,nt,nwr,wsk,upw,ns=map(int,vals[:7]);comp=vals[7]=="true";hl=vals[8]=="true";cl=int(vals[10])
        nw=nwr*wsk;r=nwr*nt*8;ups=wsk*upw;toks=16*mt;tu=768 if comp else 2048 if hl else 1024
        wpay=nw*upw*nt*tu;tail=nw*upw*nt*4 if comp else 0
        assert tail%16==0
        wby=wpay+tail;xby=ups*tokr*128;sst=(xby+wby+1023)//1024*1024
        ne=0 if comp else 1;rp=r+(40-r%32)%32
        dyn=1024+ns*sst+16*ns+32+ne*((toks*(r+1)+3)//4*4)*4+(cl-1)*wsk*toks*rp*4+128
        assert dyn+2048<=232448
        if comp:
            assert cl==1
            # Actual helper with unique unit fingerprints, three CTAs and
            # three stages. Includes every warp/UPW/tile, not just rowedges.
            ntile=(r//8)*3;nu=ups*3
            ids=np.arange(ntile*nu,dtype=np.int32).reshape(ntile,nu)
            primary=np.repeat(ids[:,:,None].view(np.uint8),192,axis=2)
            # Fill all bytes by the unit's low 8 bits; compare fullprimary to
            # source indexed unit independently rather than same streamhelper.
            primary=np.broadcast_to((ids&255).astype(np.uint8)[:,:,None],(ntile,nu,768)).copy()
            c={"packed":Tensor(primary),"descriptors":Tensor(ids)}
            stream=raw(env["bfp_sk2_patch_stream_v29"](c,r*3,r,nt,nwr,wsk,upw))
            assert stream.shape==(3,3,wby)
            for st in range(3):
                for cta in range(3):
                    for w in range(nw):
                        for uu in range(upw):
                            for t in range(nt):
                                tile=cta*(r//8)+(w//wsk)*nt+t;unit=st*ups+(w%wsk)*upw+uu
                                j=(w*upw+uu)*nt+t
                                assert np.array_equal(stream[st,cta,j*768:(j+1)*768],primary[tile,unit])
                                got=np.frombuffer(stream[st,cta,wpay+j*4:wpay+j*4+4].tobytes(),np.int32)[0]
                                assert got==ids[tile,unit]
                                mapped+=1
            for stages in (1,ns,ns+1,32,36):
                for seed in range(16):
                    assert ring_model(nw,ns,stages,seed)
                    ringcases+=1
        rows.append(dict(template=list(vals),threads=(nw+1+ne)*32,primary_stage_bytes=wpay,
                         descriptor_stage_bytes=tail,ring_stage_bytes=sst,dynamic_bytes=dyn))
    assert any(not ring_model(4,2,12,s,-1) for s in range(32))
    assert not ring_model(4,2,12,7,1)
    # Bind the event model to the actual source expressions/handback ordering.
    assert cs.count("NW + (COMP ? NESC : 0)")==ps.count("NW + (COMP ? NESC : 0)")==4
    for text in ("if (lane == 0) mbar_arrive(empty0 + 8 * slot);",):
        # One additional parent occurrence was the removed escape-warp loop.
        assert cs.count(text)==4 and ps.count(text)>=cs.count(text)
    assert "if constexpr (COMP && NESC > 0) mbar_wait(cbar, 0);" in cs
    assert "constexpr int NESC = COMP ? 0 : 1;" in cs
    assert cs.count("constexpr int WBY = WPAY + (COMP ? ((NW * UPW * NT * 4 + 15) & ~15) : 0);")==2
    assert cs.index('if (pdl) asm volatile("griddepcontrol.wait;"') < cs.index("const uint4* rp =")
    # Ring reuse after all participating regular warps loaded primary/header
    # and fallback operands; unchanged positions of early/late handbacks.
    con=cs[cs.index("      uint2 cw[NT];"):cs.index("      uint2 cw[NT];")+7000]
    assert con.index("const uint4* rp =")<con.index("mbar_arrive(empty0")
    return dict(emitted_template_geometries=rows,stage_unit_mappings=mapped,
                ring_interleavings=ringcases,ring_negative_controls=2,
                emitted_source_sha256=hashlib.sha256(emitted.encode()).hexdigest())


def run(expected_sha=None):
    spec=importlib.util.spec_from_file_location("v29_build",HERE/"build.py");b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    actual=(HERE/"submission.py").read_text();sha=hashlib.sha256(actual.encode()).hexdigest()
    assert expected_sha is None or sha==expected_sha
    built,edits=b.build();assert built==actual
    parent=b.PARENT.read_text();pt,ct=ast.parse(parent),ast.parse(actual)
    changed={"bfp_SK2_SRC","bfp__get_ext","bfp_sk2_compress","bfp_sk2_table","bfp_sk2_prepare"}
    added={"bfp_sk2_patch_stream_v29"}
    dump=lambda n:ast.dump(n,include_attributes=False)
    assert [dump(n)for n in pt.body if key(n)not in changed]==[dump(n)for n in ct.body if key(n)not in changed|added]
    ps,cs=literal(pt,"bfp_SK2_SRC"),literal(ct,"bfp_SK2_SRC")
    # Actual source generator remains identical; all UV/BND/ordinary/FUSE
    # dispatches receive the same geometries and runtime arguments.
    env=helpers(ct);cases=make_cases(env)
    native=native_codec(cs,actual,cases)
    geom=geometry_and_stream(env,ps,cs)
    # Entire epilogue and all tensor-MMA arithmetic remain unchanged; NESC=0
    # compile-time removes the old separate correction contribution.
    pstart=ps.index("  constexpr int R4 = R / 4;")
    cstart=cs.index("  constexpr int R4 = R / 4;")
    pend=ps.index("template <int MT, int TOKR",pstart)
    cend=cs.index("template <int MT, int TOKR",cstart)
    assert ps[pstart:pend]==cs[cstart:cend]
    report={"status":"pass","parent_sha256":b.PARENT_SHA,"candidate_sha256":sha,
            "test_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"numpy_version":np.__version__,
            "full_parent_restoration":True,"changed_python_nodes":sorted(changed),"added_python_nodes":sorted(added),
            "native_codec":native,"geometry":geom,"whole_raw_entropy_fallback":True,
            "gpu_compilation_or_numerical_correctness":False,
            "limits":["NumPy adapter executes actual packing expressions, not Torch/CUDA kernels.",
                      "Exact BF16 operand bits are proven on CPU; tensor MMA and FP32 correction grouping differ and require official numerical tests.",
                      "Ring model checks regular-warp ownership and inherited handback order, not CUDA memory ordering or compiler allocation.",
                      "Actual numRegs/local/occupancy and selection remain unknown; all eight affected profiles must show compressed V29 activation."]}
    assert hashlib.sha256((HERE/"submission.py").read_bytes()).hexdigest()==sha
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--expected-sha");ap.add_argument("--output",type=Path);args=ap.parse_args()
    result=run(args.expected_sha);out=json.dumps(result,indent=2)+"\n"
    if args.output: args.output.write_text(out)
    print(out)

"""Focused V25 source/packing/boundary checks. No CUDA or candidate import."""
from __future__ import annotations
import ast, contextlib, hashlib, importlib.util, io, json, re, subprocess, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import test_wgs_kperm3_cpu as packing
from test_wgs_wait2_cpu import preprocess
spec=importlib.util.spec_from_file_location('v25_build',HERE/'build.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def sha(x):return hashlib.sha256(x).hexdigest()
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def execute(nodes,env):exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),'<actual-v25-source>','exec'),env)
def body(src,needle):
    i=src.index(needle)
    while True:
        a=src.index('{',i)
        if ';' not in src[i:a]:break
        i=src.index(needle,i+len(needle))
    level=1;j=a+1
    while level:level+=(src[j]=='{')-(src[j]=='}');j+=1
    return src[i:j]
def bf(x):
    a=np.asarray(x,np.float32);u=a.view(np.uint32)
    u=(u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)
    return u.view(np.float32)

def scope():
    source=(HERE/'component.py').read_text();tree=ast.parse(source)
    parent=(HERE/'parent.py').read_text();pt=ast.parse(parent)
    assert sha(parent.encode())==b.PARENT_SHA
    writes={}
    with patch.object(Path,'write_text',lambda p,s,*a,**kw:writes.__setitem__(str(p),s)),contextlib.redirect_stdout(io.StringIO()):
        b.build();first=list(b.EDITS);b.build();assert first==b.EDITS
    assert writes[str(HERE/'component.py')]==source
    cs,ps=b.literal(tree,'CUDA_SRC'),b.literal(pt,'CUDA_SRC')
    # Builder wraps its exact CUDA payload in one new newline on either side.
    assert cs.startswith('\n') and cs.endswith('\n')
    restored=cs[1:-1]
    for old,new in reversed(first):
        assert new and restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored==ps
    names=json.loads((HERE/'manifest.json').read_text())['functions_retained_exact']
    for name in names:assert ast.dump(fn(tree,name))==ast.dump(fn(pt,name))
    assert body(cs,'void dq4(')==body(ps,'void dq4(')
    assert 'red.global.add.v4.f32' not in cs and 'add.rn.f32 %0, %1, %2;' in cs
    assert 'mul.rn.ftz.f32 %0, %1, %2;' in cs
    assert '--use_fast_math' not in ast.get_source_segment(source,fn(tree,'_ext'))
    assert not re.search(r'asm\s+volatile\("griddepcontrol\.',cs)
    kernels=[]
    for prmt in (0,1):
        pp=preprocess(cs,['-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_MULDEC=1','-DOKC_DQ7=1',
                          '-DOKC_NS=3','-DOKC_NSF=3','-DOKC_FNST=4',f'-DOKC_PRMTSC={prmt}'])
        active=[body(pp,n) for n in ('native_serving_v25(Params p)', 'void route(', 'void early_issue(', 'void wgs_epi_tile(',
                                     'void wgs_gather(', 'void wgs_wproducer(', 'void wgs_xproducer(', 'void wgs_seg(', 'void wgs_flush(')]
        for text in active:
            for dead in ('p.ss','p.ss1','p.ybuf','p.inter','p.normw','p.wAu','p.wBu','p.wAf','p.wBf','p.wAn','p.wBn','p.gsA','p.gsB'):
                assert dead not in text,(prmt,dead,text[:70])
        assert 'scale_pairs[j]' in active[-2]
        kernels.append(sha(pp.encode()))
    return source,tree,cs,{'complete_cuda_restoration':True,'exact_packing_helpers':names,'dq4_unchanged':True,
        'two_preprocessed_prmt_builds_sha256':kernels,'reachable_old_weight_norm_inter_pointers_absent':True,'formal_certificate':'none: composition outside existing strict Lean source binder'}

def packing_check(tree):
    names={'_stored_pos_mul','_pack_codes_nat','_wgs_kperm_index','_wgs_kperm_codes','_wgs_kperm_rows','_wgs_copy'}
    ns={'torch':packing.CpuTorch}
    execute([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],ns)
    ns['_POS_MUL']=ns['_stored_pos_mul']();ns['_POS']=ns['_POS_MUL']
    T=packing.Tensor
    row=np.arange(8192,dtype=np.uint32)[:,None];col=np.arange(2048,dtype=np.uint32)[None,:]
    codes=((row*37+col*11+((row^col)>>3))&255).astype(np.uint8)
    scales=((row*13+np.arange(256,dtype=np.uint32)[None,:]*7)%127).astype(np.uint8)
    gate=np.arange(4096).reshape(-1,8)
    rowsA=ns['_wgs_kperm_rows'](T(np.stack((gate,gate+4096),1).reshape(-1)))
    rowsB=T(np.arange(4096))
    # Execute the actual slab-packing loop from prepare_native, not a new packer.
    loop=next(n for n in fn(tree,'prepare_native').body if isinstance(n,ast.For) and ast.unparse(n.target)=='(code_name, scale_name, rows)')
    T.copy_=lambda self,x:self.a.__setitem__(slice(None),packing.raw(x))
    packing.CpuTorch.empty=staticmethod(lambda shape,dtype=None,device=None:T(np.empty(shape,dtype=dtype)))
    ns.update(device='cpu',local_count=1,start=0,rowsA=rowsA,rowsB=rowsB,banks=[],weights={
        'w13':T(codes[None]),'w13_sf':T(scales[None]),'w2':T(codes[:4096][None]),'w2_sf':T(scales[:4096][None])})
    execute([loop],ns)
    records=[]
    for i,(phase,rows) in enumerate((('gate_up',rowsA),('down',rowsB))):
        full=ns['_wgs_copy'](T(codes),T(scales),rows,ns['_POS_MUL']).a
        assert np.array_equal(ns['banks'][i].a,full),phase
        # Existing independent inverse reconstructs original code/scale coords.
        packing.check_actual_bytes(ns,codes,scales,rows.a,ns['_POS_MUL'],phase)
        records.append({'phase':phase,'packed_bytes':full.size,'inverse_coordinate_assertions':'PASS'})
    return {'actual_slab_loop_matches_complete_pack':True,'one_expert_checks':records,
            'retained_full32_bank_bytes':sum(x.a.size for x in ns['banks'])*32,'temporary_slab_rows':2048}

class V:
    def __init__(self,a):self.a=np.asarray(a)
    def to(self,d):return V(bf(self.a) if d=='bf16' else self.a.astype(d))
    def view(self,d):
        if d=='fp8':
            u=self.a.astype(np.uint8);ex=(u>>3)&15;mt=u&7
            v=np.where(ex==0,mt.astype(np.float32)*2**-9,(1+mt.astype(np.float32)/8)*np.exp2(ex.astype(np.float32)-7))
            v=np.where((ex==15)&(mt==7),np.nan,v)*np.where(u&128,-1,1)
            return V(v.astype(np.float32))
        return V(self.a.view(d))
    def contiguous(self):return self
for name,op in [('mul',np.multiply),('and',np.bitwise_and),('or',np.bitwise_or),('lshift',np.left_shift),('rshift',np.right_shift)]:
    setattr(V,'__'+name+'__',lambda self,other,op=op:V(op(self.a,other.a if isinstance(other,V) else other)))

def arithmetic(tree,cs):
    fake=type('Torch',(),dict(int32=np.int32,uint8=np.uint8,int16=np.int16,bfloat16='bf16',float16=np.float16,float8_e4m3fn='fp8',
         arange=staticmethod(lambda *a,device=None,dtype=None:V(np.arange(*a,dtype=dtype)))))
    env={'torch':fake};execute([fn(tree,'_marlin_scale_lut')],env)
    with np.errstate(all='ignore'):lut=env['_marlin_scale_lut']('cpu').a.astype(np.uint32)
    assert np.all((lut&65535)==(lut>>16))
    # Independent explicit binary16 source transformation for valid FP8 domain.
    mismatch=[]
    for byte in range(127):
        ex,mt=byte>>3,byte&7
        sf=mt*2**-9 if ex==0 else (1+mt/8)*2**(ex-7)
        h=np.float16(np.float16(sf)*np.float16(128))
        b2=((int(h.view(np.uint16))<<1)&65535)>>8
        bits=((b2&128)<<7)|((b2&127)<<4)
        assert int(lut[byte]&65535)==bits
        if byte>=8:assert bits==byte*16+0x3f80
        direct=int(bf(np.float32(sf*128)).view(np.uint32))>>16
        if bits!=direct:mismatch.append(byte)
    assert mismatch==list(range(1,8))
    # Low/high byte pair assembly used by the actual source, exhaustive127^2.
    l,h=np.indices((127,127),dtype=np.uint32)
    words=((lut[l]&65535)|(lut[h]<<16)).astype(np.uint32)
    assert np.array_equal(words&65535,lut[l]&65535) and np.array_equal(words>>16,lut[h]&65535)
    assert '(low & 0xffffu) | (high << 16)' in cs
    a=np.float32((1+2**-8)*2**-119);gs=np.float32((1+2**-7)*2**119)
    two=bf(bf(a)*gs);one=bf(a*gs)
    assert two.view(np.uint32)!=one.view(np.uint32)
    tiny=np.array(0x00400000,np.uint32).view(np.float32)
    assert np.float32(tiny+np.float32(0)).view(np.uint32)==0x00400000
    assert bf(tiny)*np.float32(2**119)==np.float32(2**-8)
    # Final fast activation multiply must flush its subnormal operands/results.
    sig=np.float32(16);up=np.array(0x00080000,np.uint32).view(np.float32)
    ordinary=bf(sig*up);flushed=bf(np.float32(0))
    assert ordinary.view(np.uint32)!=flushed.view(np.uint32)
    return {'scale_bytes':127,'normal_prmt_entries':119,'scale_pairs':127**2,'sf_times128_counterexample_bytes':mismatch,
            'two_round_bf16_bits':int(two.view(np.uint32)>>16),'incorrect_one_round_bits':int(one.view(np.uint32)>>16),
            'cas_add_subnormal_bits':0x00400000,'cas_result_after_prepared_global':float(bf(tiny)*np.float32(2**119)),
            'activation_ftz_counterexample':{'ordinary_bf16_bits':int(ordinary.view(np.uint32)>>16),'stock_ftz_bf16_bits':0},
            'limit':'NumPy IEEE model; not CUDA WGMMA, exp/div, atomic interleavings or GPU numerical validation.'}

def routing_output_checks(cs):
    epi=body(cs,'void wgs_epi_tile(')
    assert 'const int entry = S.ent[jst + tk];' in epi
    assert 'p.out + (size_t)entry * HD + n0' in epi and 'p.out + (size_t)entry * HD + n0 + 8' in epi
    assert 'if (tk < cnt)' in epi
    # All64 possible job sizes; source geometry makes each live route/column unique.
    coordinate_count=0
    for cnt in range(1,65):
        ub,wg,wp,lane,nt,h,half=np.indices((16,4,4,32,(cnt+7)//8,2,2))
        tk=nt*8+2*(lane&3)+h;col=(ub*16+wg*4+wp)*16+(lane>>2)+half*8
        index=(tk*4096+col)[tk<cnt]
        assert np.array_equal(np.sort(index),np.arange(cnt*4096))
        coordinate_count+=index.size
    ranges=body(cs,'__device__ __noinline__ void wgs_ranges(')
    # Compile actual range entry with adversarial helpers: any zero-job range
    # call fails, while every grid CTA must explicitly receive bothzero ranges.
    program=r'''
#include <cassert>
#define __device__
#define __noinline__
#define OKC_DEBUG 0
#define OKC_WGSFIT 0
struct Params{};struct Smem{int njF=0,njS=0,wgr[4]={99,99,99,99};};struct{int x;}blockIdx,gridDim;
int prefixes=0,calls=0;
void wgs_prefix(Smem&,int n){assert(n>=0);prefixes++;}
void wgs_range(Smem&,int n,int,int,int,int&a,int&b){assert(n>0);calls++;a=0;b=n;}
'''+ranges+r'''
int main(){Params p;gridDim.x=132;
for(int i=0;i<132;i++){Smem s;blockIdx.x=i;wgs_ranges(p,s);for(int x:s.wgr)assert(x==0);}
assert(prefixes==132&&calls==0);Smem s;s.njS=1;wgs_ranges(p,s);assert(calls==2);}
'''
    with tempfile.TemporaryDirectory(prefix='v25-zero-') as td:
        p=Path(td);(p/'test.cpp').write_text(program)
        c=subprocess.run(['/usr/bin/clang++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True,timeout=30)
        assert c.returncode==0,c.stderr
        assert subprocess.run([str(p/'test')],timeout=10).returncode==0
    assert 'if (blockIdx.x == 0)' in body(cs,'void __launch_bounds__(NTL, 1) native_serving_v25(')
    assert 'p.local_out[i] = (int)S.eid[i]' in cs
    rank_checks=0
    for lo in range(0,256,32):
        ids=np.arange(-1,256);local=np.where((ids>=lo)&(ids<lo+32),ids-lo,-1)
        assert np.count_nonzero(local>=0)==32 and local[0]==-1;rank_checks+=len(ids)
    return {'unique_live_route_output_coordinates':coordinate_count,'job_sizes':64,'zerojob_actual_range_entry_ctas':132,
            'rank_mapping_values':rank_checks,'zerojob_limit':'Source participation unchanged; no GPU barrier/counter reuse execution.'}

def memory_and_host(tree,cs):
    host=(HERE/'host.cpp.inc').read_text()
    assert host.count('cudaLaunchCooperativeKernel(')==1
    assert 'cudaLaunchKernelEx' not in host and '<<<' not in host
    for evidence in ('cudaDevAttrCooperativeLaunch','cudaOccupancyMaxActiveBlocksPerMultiprocessor','grid < okc::WGS_SLOTS','grid <= sms * resident',
                     'at::cuda::getCurrentCUDAStream()', 'v25_cuda(cudaLaunchCooperativeKernel', 'M <= okc::MMAX'):
        assert evidence in host
    init=body(host,'int64_t okc_init(')
    assert 'for (int big = 0; big < 2; ++big)' in init
    assert 'native_serving_v25<true> : okc::native_serving_v25<false>' in host
    emax=384*8;jmax=32+emax//64+8;fixf=2*160*4*4096+2*160*4+136*16;xrows=emax+8*jmax
    parts={'counters':(8+1024)*4,'fixup':fixf*4,'xA_xB':2*xrows*4096*2,'expert_out':384*8*4096*2,'local_ids':384*8*4}
    assert sum(parts.values())==108033568
    assert 8+512+jmax<=8+1024 and 2*160*4*4096+2*160*4+jmax*16<=fixf
    # Fixed geometry occupancy boundaries: actual runtime APIs must establish cap.
    checks=0
    for coop,resident,sms,grid in ((1,1,132,132),(0,1,132,132),(1,0,132,132),(1,1,132,160),(1,1,132,133),(1,2,132,159)):
        accepted=bool(coop and resident>=1 and 0<grid<160 and grid<=sms*resident)
        assert accepted==((coop,resident,sms,grid) in ((1,1,132,132),(1,2,132,159)));checks+=1
    return {'workspace_parts_bytes':parts,'workspace_total_bytes':sum(parts.values()),'full_weight_copy_bytes':905969664,
            'grid_policy_boundaries':checks,'host_API_status_checked':True,'limits':'Host source and policy boundaries, not CUDA runtime API execution.'}

def run():
    source,tree,cs,conformance=scope()
    result={'status':'PASS','component_sha256':sha(source.encode()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes()),
            'conformance':conformance,'packing':packing_check(tree),'arithmetic':arithmetic(tree,cs),
            'routing_and_output':routing_output_checks(cs),'memory_and_host':memory_and_host(tree,cs),
            'limits':['No CUDA compilation or GPU correctness/performance evidence.','Original stock finalizer image/source binding is a deployment/evaluator gate, not established by local module import.',
                      'Reusable workspace is serial in this component; overlapping reuse or shared capture pools are unsupported without owner proof.',
                      'No full-serving buffer replacement, model-memory fit, prefill/TTFT or eight-rank claim.']}
    return result
if __name__=='__main__':
    r=run();(HERE/'cpu_result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

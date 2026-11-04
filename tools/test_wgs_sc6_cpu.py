#!/usr/bin/env python3
"""Focused source/packing/dispatch proof; no candidate import, CUDA or GPU use."""
import argparse
import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import operator
from pathlib import Path
import re
import tempfile

import numpy as np
import build_wgs_sc6_candidate as builder
import test_wgs_kperm3_cpu as pack
import test_wgs_shared_a_cpu as common
from test_wgs_wait2_cpu import literal, segment, preprocess

ROOT=builder.ROOT
OUT=ROOT/'experiments/iterations/20261104-wgs-sc6-cpu.json'
Tensor,CpuTorch=pack.Tensor,pack.CpuTorch
for name,op in [('lt',operator.lt),('gt',operator.gt)]:
    setattr(Tensor,'__'+name+'__',lambda self,x,op=op:Tensor(op(self.a,pack.raw(x))))
Tensor.any=lambda self:bool(self.a.any())
Tensor.float=lambda self:Tensor(self.a.astype(np.float32))
Tensor.dtype=property(lambda self:self.a.dtype)
Tensor.data_ptr=lambda self:self.a.ctypes.data
Tensor.clone=lambda self:Tensor(self.a.copy())
old_view=Tensor.view
def view(self,*args):
    if len(args)==1 and args[0] in (np.int16,np.uint16):return Tensor(self.a.view(args[0]))
    return old_view(self,*args)
Tensor.view=view
CpuTorch.int16=np.int16
CpuTorch.float32=np.float32
CpuTorch.float8_e4m3fn=np.uint8 # LUT arithmetic is mocked; its source is preserved exactly.
CpuTorch.bfloat16=np.uint16
CpuTorch.arange=staticmethod(lambda *args,device=None,dtype=None:Tensor(np.arange(*args,dtype=np.int64 if dtype is None else dtype)))
CpuTorch.empty=staticmethod(lambda shape,dtype=None,device=None:Tensor(np.empty(shape,dtype=dtype)))

def sha(x):return hashlib.sha256(x).hexdigest()
def fn(tree,name):return pack.function(tree,name)
def dump(x):return ast.dump(x,include_attributes=False)
def source(text,name):return ast.get_source_segment(text,fn(ast.parse(text),name))
def execute_functions(tree,names,ns):
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual candidate AST>','exec'),ns)

def namespaces(tree):
    ns=pack.packing_namespace(tree)
    execute_functions(tree,['_sc6_scales_exact','_sc6_defines','_sc6_wgs_pack','_scales_prmt_exact'],ns)
    return ns

def conformance(parent,candidate):
    pt,ct=ast.parse(parent),ast.parse(candidate)
    def key(n):return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn={key(n):n for n in pt.body if key(n)};cn={key(n):n for n in ct.body if key(n)}
    assert cn.keys()-pn.keys()=={'_sc6_scales_exact','_sc6_defines','_sc6_wgs_pack'}
    modified={'CUDA_SRC','_ext','_wgs_copy','_prepare_layout','kernel_fn'}
    for k in pn:
        if k not in modified:assert dump(pn[k])==dump(cn[k]),k
    ext=source(candidate,'_ext').replace('okc_moe_wgs_sc6_v16_','okc_moe_kperm3_v9_').replace('verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence','verbose=False,')
    assert ext==source(parent,'_ext')
    w=source(candidate,'_wgs_copy').replace(', sc6=False):','):').replace('[8960 B SC6 / 9216 B ordinary].','[9216 B].').replace('Both code-bit layouts are preserved; qualified SC6 packs exact scales into six bits.','This format retains 8-bit scales and both existing code-bit positions mappings.').replace('    if sc6:\n        s = _sc6_wgs_pack(s)\n','')
    assert w==source(parent,'_wgs_copy')
    pre=source(candidate,'_prepare_layout')
    prmt='    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])\n'
    pre=pre.replace(prmt,'').replace('    sc6 = bool(use_mul and prmt and _sc6_scales_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt]))\n','').replace('    state["sc6_wgs"] = sc6\n','').replace(', sc6=sc6)',')').replace('_ext(_sc6_defines(d, sc6), use_mul=use_mul)','_ext(d, use_mul=use_mul)').replace('    if use_mul:\n        print(f"SC6W enabled={int(sc6)} piece={8960 if sc6 else 9216} stage={35840 if sc6 else 36864}", flush=True)\n','')
    pre=pre.replace('    state["sfx"] =',prmt+'    state["sfx"] =')
    assert pre==source(parent,'_prepare_layout'),'prepare scope'
    ker=source(candidate,'kernel_fn').replace('_sc6_defines(tuple(_main_defines(sfx)) + sfx, state["sc6_wgs"])','tuple(_main_defines(sfx)) + sfx')
    assert ker==source(parent,'kernel_fn')
    ps,cs=literal(pt,'CUDA_SRC'),literal(ct,'CUDA_SRC')
    norm=cs.replace(builder.HOST+'\n','').replace('#if OKC_WGS\n    okc_sc6_resources(sm, e0, e1);\n#endif\n','')
    norm=norm.replace('#ifndef OKC_SC6W\n#define OKC_SC6W 0\n#endif\n\n','').replace('static_assert(!OKC_SC6W || OKC_PRMTSC, "SC6 requires the qualified PRMT scale domain");\n','')
    norm=norm.replace('constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // unchanged codes, lossless W/H scale planes\nconstexpr int WGS_STG = 4 * WGS_PIECE;           // global payload; shared slots keep UNIT_BYTES','constexpr int WGS_PIECE = 9216;                 // one warpgroup\'s weights in a stage')
    norm=norm.replace('WGS_STG','UNIT_BYTES')
    norm=norm.replace('#if OKC_SC6W\n  const uint32_t k16c = 4u + p.zero;  // exact shifted six-bit planes, uint32 IMAD wrap\n#else\n','').replace('  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)\n#endif','  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)')
    a=norm.index(builder.SC6_DECODE);b=norm.index('#pragma unroll\n    for (int lg',a)
    norm=norm[:a]+norm[a:b].replace(builder.SC6_DECODE,'').removesuffix('#endif\n')+norm[b:]
    assert norm==ps,'full CUDA restoration'
    # No kernel-local consumer ordering, arithmetic, wrappers or other device functions change.
    oldseg,newseg=segment(ps),segment(cs)
    assert newseg[newseg.index('#pragma unroll\n    for (int lg'):]==oldseg[oldseg.index('#pragma unroll\n    for (int lg'):]
    builds=0
    for name in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        for mul in (0,1):
            for prmt in (0,1):
                defs=[*literal(pt,name),f'-DOKC_MULDEC={mul}',f'-DOKC_PRMTSC={prmt}','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
                assert preprocess(ps,defs)==preprocess(cs,defs),(name,mul,prmt)
                builds+=1
    assert '#define OKC_WGMIN 9' in cs
    assert all(not any('WGMIN' in f for f in literal(ct,n)) for n in ('MAIN_DEFINES','TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'))
    assert 'except' not in source(candidate,'prepare')+source(candidate,'kernel_fn')
    return dict(restored_parent_cuda_sha256=sha(norm.encode()),unchanged_python_nodes=len(pn)-len(modified),non_wgs_preprocessed_builds=builds,all_post_scale_decode_operand_order_preserved=True,natural_fat_aliases_inactive_WGMIN9=True),ps,cs

def inverse_scales(packed):
    lead=packed.shape[:-1]
    w=packed[...,:512].reshape(*lead,128,4)
    h=packed[...,512:].reshape(*lead,128,2)
    six=np.concatenate((w,h),axis=-1)
    out=np.empty((*lead,128,8),np.uint8)
    out[...,:6]=six>>2
    out[...,6]=(six[...,0]&3)|((six[...,1]&3)<<2)|((six[...,2]&3)<<4)
    out[...,7]=(six[...,3]&3)|((six[...,4]&3)<<2)|((six[...,5]&3)<<4)
    return (out+0x40).reshape(*lead,1024)

def actual_packing(pt,ct):
    ns=namespaces(ct);pns=pack.packing_namespace(pt)
    row=np.arange(8192,dtype=np.uint32)[:,None];col=np.arange(2048,dtype=np.uint32)[None,:]
    codes=((row*37+col*11+((row^col)>>3))&255).astype(np.uint8)
    scales=(0x40+(row*13+np.arange(256,dtype=np.uint32)[None,:]*7)%63).astype(np.uint8)
    gate=np.arange(4096).reshape(-1,8)
    arows=ns['_wgs_kperm_rows'](Tensor(np.stack((gate,gate+4096),1).reshape(-1))).a
    cases=[]
    for layout in ('_POS','_POS_MUL'):
        for phase,rows in [('W13',arows),('W2',np.arange(4096))]:
            args=(Tensor(codes),Tensor(scales),Tensor(rows),ns[layout])
            old=pns['_wgs_copy'](*args).a.reshape(-1,16,4,9216)
            fallback=ns['_wgs_copy'](*args,sc6=False).a.reshape(old.shape)
            assert np.array_equal(old,fallback)
            new=ns['_wgs_copy'](*args,sc6=True).a.reshape(-1,16,4,8960)
            assert np.array_equal(old[...,:8192],new[...,:8192])
            assert np.array_equal(old[...,8192:],inverse_scales(new[...,8192:]))
            cases.append(dict(phase=phase,layout=layout,codes_checked=old[...,:8192].size*2,scale_bytes_checked=old[...,8192:].size,packed_bytes=new.size,parent_bytes=old.size))
    # Arbitrary 8-bit scales retain the byte-identical fallback, not merely qualified values.
    scales=((row*13+np.arange(256,dtype=np.uint32)[None,:]*7)&255).astype(np.uint8)
    for layout in ('_POS','_POS_MUL'):
        args=(Tensor(codes[:256]),Tensor(scales[:256]),Tensor(np.arange(256)),ns[layout])
        assert np.array_equal(pns['_wgs_copy'](*args).a,ns['_wgs_copy'](*args,sc6=False).a)
    return dict(cases=cases,all_byte_values_fallback_layouts=2),ns

def decode_body(cs):
    seg=preprocess(segment(cs),['-DOKC_SC6W=1','-DOKC_PRMTSC=1','-DOKC_WGSGS=2'])
    start=seg.index('    uint32_t scale_pairs[4];') if '    uint32_t scale_pairs[4];' in seg else seg.index('uint32_t scale_pairs[4];')
    stop=seg.index('#pragma unroll',start)
    k=re.search(r'const uint32_t k16c =[^;]*;',seg)[0]
    return k+'\n'+seg[start:stop]

def native_scales(cs,ns,mutate=None):
    # Every qualified pair in each of four positions; backgrounds exercise cross-field carry/wrap.
    records=[]
    a,b=np.indices((63,63));pairs=np.stack((a.ravel(),b.ravel()),-1).astype(np.uint8)
    for j in range(4):
        for background in (0,1,21,62):
            v=np.full((len(pairs),8),background,np.uint8);v[:,2*j:2*j+2]=pairs;records.append(v+0x40)
    qualified=len(records)*len(pairs)
    # Algebraic superset: all64x64 values in both cross-packed fields, including
    # 0x7f solely in this pack/decode test (actual qualification rejects it).
    a,b=np.indices((64,64));superset=np.full((4096,8),0x7e,np.uint8)
    superset[:,6]=a.ravel()+0x40;superset[:,7]=b.ravel()+0x40
    records.append(superset)
    orig=np.concatenate(records);n=len(orig)
    orig=np.concatenate((orig,np.full(((-n)%128,8),0x40,np.uint8))).reshape(-1,1024)
    compressed=ns['_sc6_wgs_pack'](Tensor(orig)).a
    assert np.array_equal(inverse_scales(compressed),orig)
    fixture=np.concatenate((compressed,orig),axis=1)
    body=decode_body(cs)
    if mutate:body=mutate(body)
    code=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t c){uint64_t x=uint64_t(a)|(uint64_t(b)<<32);uint32_t r=0;for(int i=0;i<4;i++)r|=uint32_t((x>>(8*((c>>(4*i))&7)))&255)<<(8*i);return r;}
struct {uint32_t zero=0;}p;
void decode(const unsigned char*pc,int wp,int lane,uint32_t*out){
'''+body+r'''
for(int j=0;j<4;j++)out[j]=scale_pairs[j];}
int main(int argc,char**argv){assert(argc==2);FILE*f=fopen(argv[1],"rb");assert(f);alignas(128)unsigned char pc[8960]={};unsigned char original[1024];uint64_t pairs=0;
while(fread(pc+8192,1,768,f)==768){assert(fread(original,1,1024,f)==1024);for(int wp=0;wp<4;wp++)for(int lane=0;lane<32;lane++){
uint32_t got[4];decode(pc,wp,lane,got);for(int j=0;j<4;j++){const unsigned char*s=original+(wp*32+lane)*8+j*2;uint32_t want=(uint32_t(s[0])*16+0x5c00)|((uint32_t(s[1])*16+0x5c00)<<16);assert(got[j]==want);pairs++;}}}
assert(feof(f));fclose(f);printf("%llu\n",(unsigned long long)pairs);}
'''
    with tempfile.TemporaryDirectory(prefix='okc-sc6-fixture-') as td:
        path=Path(td)/'scales.bin';fixture.tofile(path);out,_=common.compile_run(code,[path])
    return dict(actual_packed_records=n,qualified_records=qualified,cross_field_64x64_algebraic_superset=4096,padded_native_pair_comparisons=int(out),qualified_pair_values=63*63,positions=4,backgrounds=[0,1,21,62],native_uint32_wrap=True)

def address_proof(cs):
    producer=common.function(cs,'wgs_wproducer');seg=segment(cs)
    assert producer.count('"r"(WGS_STG)')==2
    assert '* (size_t)WGS_STG' in producer and 'ring + ws * UNIT_BYTES' in producer
    assert 'wg * WGS_PIECE' in seg
    assert 'WGS_NS * UNIT_BYTES' in cs
    assert 'constexpr int WGS_NS = 3;' in cs and '#define OKC_WGSNX 3' in cs
    assert 'WGS_PIECE = OKC_SC6W ? 8960 : 9216' in cs and 'WGS_STG = 4 * WGS_PIECE' in cs
    count=0
    for units in (32,16):
        for expert in range(32):
            for unit in range(units):
                for stage in range(16):
                    base=((expert*units+unit)*16+stage)*35840
                    assert base%128==0
                    for wg in range(4):
                        p=wg*8960
                        assert p%128==0 and p+8192+512+127*2+1<p+8960
                        assert p+8960<=35840
                        count+=1
    for seq in range(96):
        slot=seq%3;dst=slot*36864
        assert dst+35840<=(slot+1)*36864<=3*36864
    return dict(expert_stage_piece_bounds=count,ring_wrap_stages=96,global_stage_bytes=35840,shared_slot_bytes=36864,unused_shared_tail_bytes=1024,shared_bytes=231168,weight_payload_reduction_fraction=1/36,unchanged_ring_wait_arrival_participants=True)

def actual_state_dispatch(ct):
    ns=namespaces(ct)
    constants=['PRMT_SCALES','MAIN_DEFINES','TINY_DEFINES','TINY_MAX_M','TINY2_DEFINES','TINY2_MAX_M','MID_DEFINES','MID_MAX_M','MAIN_FALLBACK_DEFINES','MMAX']
    ns.update({n:literal(ct,n) for n in constants})
    # Run complete actual prepare/kernel functions. Heavy unchanged weight-pack/LUT arithmetic is
    # mocked here; actual compact packing is verified separately at full expert dimensions.
    events=[];runs=[];pack_events=[]
    class Ext:
        def __init__(self,defs,mul):self.defs,self.mul=defs,mul
        def okc_run(self,w,scratch,x,ids,tw,out,*rest):
            compact=bool(w[12].a[0]);iswgs='-DOKC_WGS=1' in self.defs
            assert ('-DOKC_SC6W=1' in self.defs)==(compact and iswgs)
            if iswgs:assert self.mul
            runs.append((tuple(self.defs),self.mul,x.shape[0],compact))
    def ext(defs,use_mul=True):events.append((tuple(defs),use_mul));return Ext(tuple(defs),use_mul)
    ns['_ext']=ext;ns['_make_scratch']=lambda dev:[]
    for n in ('_pack_codes','_pack_scales','_fat_order','_unit_order'):ns[n]=lambda *args:Tensor(np.zeros(1,np.uint8))
    def wgs(*args,sc6=False):pack_events.append(sc6);return Tensor(np.array([int(sc6)],np.uint8))
    ns['_wgs_copy']=wgs
    CpuTorch.cuda=type('Cuda',(),{'synchronize':staticmethod(lambda:None),'get_device_properties':staticmethod(lambda dev:type('Props',(),{'multi_processor_count':132})())})
    execute_functions(ct,['_wgs_on','_main_defines','_prepare_layout','prepare','kernel_fn'],ns)
    qualifications=0
    for byte in range(256):
        for bank in (0,1):
            scales=[Tensor(np.array([0x40],np.uint8)),Tensor(np.array([0x7e],np.uint8))]
            scales[bank]=Tensor(np.array([byte],np.uint8))
            assert ns['_sc6_scales_exact'](*scales)==(0x40<=byte<=0x7e)
            assert ns['_scales_prmt_exact'](*scales)==(8<=byte<=0x7e)
            qualifications+=1
    def weights(byte=0x40,badbank='w13_sf'):
        w={'w13':Tensor(np.zeros((32,8192,1),np.uint8)),'w2':Tensor(np.zeros((32,4096,1),np.uint8)),
           'w13_sf':Tensor(np.full((32,1,1),0x40,np.uint8)),'w2_sf':Tensor(np.full((32,1,1),0x7e,np.uint8)),
           'w13_gs':Tensor(np.ones(32,np.float32)),'w2_gs':Tensor(np.ones(32,np.float32)),'post_norm_w':Tensor(np.ones(4096,np.uint16))}
        w[badbank].a[31,0,0]=byte;return w
    states=[];log=io.StringIO()
    with contextlib.redirect_stdout(log),np.errstate(invalid='ignore'):
        for byte,bank in [(0x40,'w13_sf'),(0x3f,'w13_sf'),(0x7e,'w2_sf'),(0x7f,'w2_sf'),(0,'w13_sf')]:
            events.clear();pack_events.clear();state=ns['prepare'](weights(byte,bank));eligible=0x40<=byte<=0x7e
            assert state['larger']['sc6_wgs']==eligible and not state['tiny']['sc6_wgs']
            assert pack_events==[False]*64+[eligible]*64
            for defs,mul in events:assert ('-DOKC_SC6W=1' in defs)==(eligible and mul and '-DOKC_WGS=1' in defs)
            states.append(state)
    for i in [0,1,0,3,2,4,1,2]:
        for m in (1,4,8,16,17,32,128,256,385):
            data=dict(hidden_states=Tensor(np.zeros((m,1),np.uint16)),topk_ids=Tensor(np.zeros((m,8),np.int32)),topk_weights=Tensor(np.zeros((m,8),np.float32)),config=dict(local_start=0,local_count=32))
            out=ns['kernel_fn'](data,states[i]);assert out.shape==(m,4096)
    assert len(set(d for d,mul,n,c in runs if '-DOKC_WGS=1' in d))==2
    assert len(log.getvalue().splitlines())==5
    return dict(qualification_bank_byte_cases=qualifications,complete_actual_prepare_calls=10,alternating_prepared_states=8,actual_kernel_calls=72,actual_extension_launches=len(runs),partial_final_batch_M385=True,prepared_WGS_only_defines=True,heavy_unchanged_prepare_ops_mocked=True,unqualified_PRMT_and_nonPRMT_fallback=True)

def host_mock(cs):
    # Reuse the reviewed native helper/error/retry check, changing only name, label and fixed size.
    code=inspect.getsource(common.host_mock).replace('def host_mock(cs):','def adapted(cs):').replace('okc_shared_a_resources','okc_sc6_resources').replace('194304','231168').replace('SSAR ','SC6R ').replace('#include <cstdint>','#include <cstdint>\n#define OKC_SC6W 1')
    ns=dict(common.__dict__);exec(code,ns);encoded=ns['adapted'](cs)
    exec(code.replace('#define OKC_SC6W 1','#define OKC_SC6W 0'),ns);ordinary=ns['adapted'](cs)
    return dict(sc6_enabled=encoded,ordinary_WGS_fallback=ordinary)

def run(candidate=builder.CANDIDATE,expected_sha=None):
    candidate=Path(candidate);raw=candidate.read_bytes();digest=sha(raw)
    if expected_sha:assert digest==expected_sha
    parent=builder.PARENT.read_bytes();assert sha(parent)==builder.PARENT_SHA
    assert builder.build().encode()==raw
    scope,ps,cs=conformance(parent.decode(),raw.decode())
    pt,ct=ast.parse(parent),ast.parse(raw)
    packed,ns=actual_packing(pt,ct)
    native=native_scales(cs,ns)
    negatives=[]
    for label,mutation in [('wrong_SC6_multiplier',lambda s:s.replace('4u + p.zero','16u + p.zero')),('old_scale_bias',lambda s:s.replace('0x60006000u','0x5c005c00u')),('lost_p6_field',lambda s:s.replace('0x00030303u','0x00030003u'))]:
        try:native_scales(cs,ns,mutation)
        except common.NativeCheckFailed:negatives.append(label)
        else:raise AssertionError('accepted negative '+label)
    for label,mutation in [('old_TMA_payload',lambda s:s.replace('"r"(WGS_STG)','"r"(UNIT_BYTES)')),('old_piece_stride',lambda s:s.replace('wg * WGS_PIECE','wg * 9216'))]:
        try:address_proof(mutation(cs))
        except AssertionError:negatives.append(label)
        else:raise AssertionError('accepted negative '+label)
    try:actual_state_dispatch(ast.parse(raw.decode().replace('if enabled and "-DOKC_WGS=1" in defines','if True and "-DOKC_WGS=1" in defines')))
    except AssertionError:negatives.append('global_SC6_mode_leaks_between_states')
    else:raise AssertionError('accepted global mode negative')
    result=dict(status='PASS',source_sha256=digest,parent_sha256=builder.PARENT_SHA,cuda_sha256=sha(cs.encode()),conformance=scope,packing=packed,native_decode=native,address_and_payload=address_proof(cs),actual_state_dispatch=actual_state_dispatch(ct),host_resources=host_mock(cs),negative_controls=negatives,
        limits='CPU source restoration, actual integer packing/native decode and mocked host/prepare/dispatch execution; not CUDA compilation, tensor-core arithmetic, GPU memory ordering or performance. Unchanged LUT and ordinary prepare packing are mocked only in the state integration check; actual WGS copies are independently exercised at complete expert dimensions.')
    assert sha(candidate.read_bytes())==digest
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=builder.CANDIDATE);ap.add_argument('--expected-sha');ap.add_argument('--output',type=Path,default=OUT)
    a=ap.parse_args();r=run(a.candidate,a.expected_sha);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(status=r['status'],source_sha256=r['source_sha256'],report_sha256=sha(a.output.read_bytes()),output=str(a.output)),indent=2))

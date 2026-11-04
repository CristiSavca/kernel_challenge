"""Bounded V24 source/API checks; no CUDA/Triton compilation or device execution."""
from __future__ import annotations
import ast, copy, enum, hashlib, importlib.util, json, subprocess, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('v24_build', HERE/'build.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
C,P,F,L,M=b.FILES

def sha(x):return hashlib.sha256(x).hexdigest()
def source(role,path):return (HERE/role/path).read_text()
def compile_ast(nodes,env):
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),'<actual-v24-source>','exec'),env)
def getclass(text,name):return next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name==name)
def block(text,start):
    i=text.index(start);a=text.index('{',i);depth=1;j=a+1
    while depth:
        depth+=(text[j]=='{')-(text[j]=='}');j+=1
    return text[i:j]

class Tensor:
    def __init__(self,shape,dtype='i32',device='cuda',contig=True,label='tensor'):
        self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.is_cuda=device=='cuda';self.contig=contig;self.label=label;self.filled=False
    def is_contiguous(self):return self.contig
    def dim(self):return len(self.shape)
    def float(self):return self if self.dtype=='f32' else Tensor(self.shape,'f32',self.device,label=self.label)
    def contiguous(self):return self
    def to(self,dtype):return self if self.dtype==dtype else Tensor(self.shape,dtype,self.device,label=self.label)
    def __getitem__(self,index):
        if isinstance(index,Tensor):return Tensor(index.shape,self.dtype,self.device,label="mapped")
        dims=tuple(len(range(*s.indices(n))) for s,n in zip(index,self.shape))
        return Tensor(dims,self.dtype,self.device,label=self.label+'_view')

class Module:
    def __init__(self):pass
    def __call__(self,*a,**kw):return self.forward(*a,**kw)
class KB(enum.Enum):
    TORCH='torch';JIT='jit';TRITON='triton';TORCH_COMPILE='torch_compile';AOT='aot'

def dispatch_checks():
    allocations=[];calls=[]
    def empty(shape,dtype,device):
        x=Tensor(shape,dtype,device,label='allocated');allocations.append(x);return x
    torch=types.SimpleNamespace(Tensor=Tensor,int32='i32',float32='f32',empty=empty,compile=lambda f:f)
    env=dict(torch=torch,nn=types.SimpleNamespace(Module=Module),ABC=object,abstractmethod=lambda f:f,
             FormatSignature=lambda:None,DEFAULT_PRIORITY=(KB.JIT,KB.TRITON),KernelBackend=KB,CapabilityRequirement=types.SimpleNamespace(CUDA='cuda',HIP='hip'),
             BACKEND_METHODS={x:'forward_'+('native' if x==KB.TORCH else x.value) for x in KB},
             _METHOD_BACKEND_LABELS={},_ALWAYS_AVAILABLE={KB.TORCH,KB.TORCH_COMPILE},
             _PLATFORM_METHODS={'cuda':('forward_cuda',)},_trace_enabled=False,
             _platform=lambda:'cuda',_platform_key=lambda:'cuda',_oot_dispatch_key=lambda:None,
             capabilities_satisfied=lambda caps,p:True,debug_kernel_api=lambda f:f,
             _warn_forced_backend_unavailable=lambda **kw:None)
    sentinel=object();env['_UNRESOLVED']=sentinel;env['_forced_backend']=None
    env['get_fused_op_backend']=lambda:env['_forced_backend']
    base=sourcefile=(ROOT/'serving/sglang/python/sglang/kernels/fused_op.py').read_text()
    assert sha(base.encode())==sha(subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',b.REV+':python/sglang/kernels/fused_op.py']))
    compile_ast([getclass(base,'BaseFusedOp')],env)
    cls=getclass(source('candidate',F),'FusedMuseSparkRouting')
    # Retain actual fallback guards, replace only untouched numerical bodies.
    # Full byte restoration below separately proves those bodies are stock.
    for meth in cls.body:
        if isinstance(meth,ast.FunctionDef) and meth.name in ('forward_native','forward_triton','forward_cpu'):
            guards=[x for x in meth.body if isinstance(x,ast.If) and ast.unparse(x.test)=='marlin_expert_map is not None']
            assert len(guards)==1
            meth.body=guards+ast.parse(f"return _old_backend('{meth.name}', want_packed)").body
    def old_backend(name,packed):
        calls.append(name);r=(Tensor((8,8),'f32',label='scores'),Tensor((8,8),label='global'))
        return (*r,Tensor((8,8),label='packed')) if packed else r
    env['_old_backend']=old_backend
    compile_ast([cls],env)
    routing=types.ModuleType('sglang.kernels.ops.muse_spark_v12.routing')
    def jit(gating,bias,topk,scale,**kw):
        calls.append(('jit',kw.copy()))
        local=kw.get('marlin_topk_ids')
        if local is not None:local.filled=True
        return Tensor((gating.shape[0],topk),'f32',label='scores'),Tensor((gating.shape[0],topk),label='global'),kw['packed_topk_ids']
    routing.fused_muse_spark_routing=jit
    cls=env['FusedMuseSparkRouting'];gate=Tensor((8,256),'f32');emap=Tensor((256,))
    checked=[]
    def fresh():
        env['_forced_backend']=None;env['_oot_dispatch_key']=lambda:None
        return cls(Tensor((256,),'f32'),1.0)
    def opt(o,m=emap):return o.forward_with_local_ids(None,gate,8,True,m)
    with patch.dict(sys.modules,{routing.__name__:routing}):
        o=fresh()
        for want,n in ((False,2),(True,3)):
            out=o(None,gate,8,True,want_packed=want);assert len(out)==n
            assert calls[-1][1]['marlin_topk_ids'] is None
            checked.append('default_'+str(want))
        out=opt(o);assert len(out)==4 and out[2] is None and out[3].filled
        assert calls[-1][1]['local_expert_mapping'] is None
        checked.append('selected_jit')
        # Fresh per-call maps: both calls must pass the actual input object.
        replacement=Tensor((256,),label='new-map');opt(o,replacement)
        assert calls[-1][1]['marlin_expert_map'] is replacement;checked.append('map_replacement')
        for forced in (KB.TRITON,KB.TORCH,KB.TORCH_COMPILE,KB.AOT):
            o=fresh();env['_forced_backend']=forced;out=opt(o)
            assert len(out)==4 and out[3] is None and out[2] is not None
            checked.append('forced_'+forced.value)
        o=fresh();env['_forced_backend']=KB.JIT;assert opt(o)[3].filled;checked.append('forced_jit_actual')
        for state in ('local_expert_mapping','num_valid_tokens','sol_topk_out'):
            o=fresh();setattr(o,state,Tensor((256,) if state=='local_expert_mapping' else ((8,8) if state=='sol_topk_out' else (1,))))
            out=opt(o);assert out[3] is None and out[2] is not None
            checked.append('state_'+state)
        o=fresh();o.enter_torch_compile(8);out=opt(o)
        assert out[3] is None and calls[-1]=='forward_native'
        o.leave_torch_compile();assert opt(o)[3].filled;checked.append('enter_leave_compile')
        # Registered OOT function accepts exactly the old API, no new keyword.
        def oot(self,hidden,gating,topk,renormalize,want_packed=False):
            calls.append('oot');return old_backend('oot',want_packed)
        o=fresh();env['_oot_dispatch_key']=lambda:'test-oot'
        env['BaseFusedOp']._oot_forward_registry={'test-oot':{cls:oot}}
        out=opt(o);assert out[3] is None and calls[-1]=='oot';checked.append('oot_old_signature')
        o=fresh();o._forward_method=types.MethodType(oot,o)
        assert opt(o)[3] is None;checked.append('custom_cached_old_signature')
        for bad in (None,Tensor((256,),device='cpu'),Tensor((255,)),Tensor((256,),dtype='f32'),Tensor((256,),contig=False)):
            o=fresh();out=opt(o,bad);assert out[3] is None;checked.append('map_metadata_fallback')
        # Per-call JIT eligibility rejects topk33; actual dynamic fallback stays old.
        o=fresh();out=o.forward_with_local_ids(None,gate,33,True,emap)
        assert out[3] is None and calls[-1]=='forward_triton';checked.append('dynamic_ineligible')
    return {'cases':len(checked),'names':checked,'base_fused_op_sha256':sha(base.encode()),'limits':'Actual dispatch/guards; backend numerical bodies and CUDA are stubbed.'}

def low_wrapper_checks():
    text=source('candidate',P);fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='fused_muse_spark_routing')
    calls=[]
    mod=types.SimpleNamespace(fused_muse_spark_routing=lambda *args:calls.append(args))
    env={'torch':types.SimpleNamespace(float32='f32',int32='i32',empty=lambda shape,dtype,device:Tensor(shape,dtype,device)),
         '_NORMALIZATION_MODES':{'sigmoid':0,'softmax':1},'_jit_module':lambda *a:mod,'_experts_per_lane':lambda e:(e+31)//32,'_warps_per_block':lambda t:1}
    compile_ast([fn],env);f=env[fn.name];g=Tensor((8,256),'f32');bias=Tensor((256,),'f32')
    r=f(g,bias,8,1.0);assert len(r)==3 and calls[-1][-2:]==(None,None)
    emap=Tensor((256,));ids=Tensor((8,8));r=f(g,bias,8,1.0,marlin_expert_map=emap,marlin_topk_ids=ids)
    assert len(r)==3 and calls[-1][-2:]==(emap,ids)
    for kw in ({'marlin_expert_map':emap},{'marlin_topk_ids':ids},{'marlin_expert_map':emap,'marlin_topk_ids':Tensor((9,8))}):
        before=len(calls)
        try:f(g,bias,8,1.0,**kw)
        except AssertionError:pass
        else:raise AssertionError('bad wrapper args accepted')
        assert len(calls)==before
    return {'cases':5,'default_return_length':3,'ffi_arity':len(calls[-1])}

def boundary_checks():
    import contextlib
    layer=next(n for n in ast.parse(source('candidate',L)).body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_marlin_expert_outputs' for m in n.body))
    meth=next(n for n in layer.body if isinstance(n,ast.FunctionDef) and n.name=='_marlin_expert_outputs')
    branch=next(n for n in meth.body if isinstance(n,ast.If) and ast.unparse(n.test)=='local_topk_ids is None')
    torch=types.SimpleNamespace(int32='i32')
    ids=Tensor((8,8));emap=Tensor((256,));cases=0
    for explicit in (None,Tensor((8,8))):
        env=dict(torch=torch,topk_ids=ids,expert_map=emap,local_topk_ids=explicit)
        compile_ast([copy.deepcopy(branch)],env)
        assert env['local_topk_ids'].label=='mapped' if explicit is None else env['local_topk_ids'] is explicit
        cases+=1
    for bad in (Tensor((8,8),'f32'),Tensor((8,8),device='cpu'),Tensor((7,8)),Tensor((8,8),contig=False)):
        env=dict(torch=torch,topk_ids=ids,expert_map=emap,local_topk_ids=bad)
        try:compile_ast([copy.deepcopy(branch)],env)
        except ValueError:pass
        else:raise AssertionError('bad Marlin metadata accepted')
        cases+=1
    model=ast.parse(source('candidate',M))
    fn=next(n for n in ast.walk(model) if isinstance(n,ast.FunctionDef) and n.name=='_forward_experts')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='marlin_expert_map' for t in n.targets))
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.NamedExpr))
    nodes=fn.body[start:end]
    assert len(nodes)==5 # assignments, routing with, main join, stream ownership
    class Tracked(Tensor):
        def record_stream(self,stream):events.append((self.label,stream))
    class Stream:
        def wait_stream(self,stream):events.append(('wait',self,stream))
    main,side=Stream(),Stream()
    scores,global_ids,packed,local=[Tracked((8,8),label=n) for n in ('scores','global','packed','local')]
    map_tensor=Tracked((256,),label='map')
    class Router:
        def __call__(self,*args,**kw):events.append('old_call');return scores,global_ids,packed
        def forward_with_local_ids(self,*args):
            assert args[-1] is map_tensor;events.append('opt_call')
            return scores,global_ids,(None if available else packed),(local if available else None)
    for nvfp4,marlin,sp,capture,available in ((True,True,False,False,True),(True,True,False,True,True),(True,True,False,False,False),(True,True,True,False,True),(False,True,False,False,True),(True,False,False,False,True)):
        events=[];router=Router();gating=Tracked((8,256),'f32',label='logits')
        obj=types.SimpleNamespace(experts=types.SimpleNamespace(_expert_map=map_tensor,use_marlin=marlin),nvfp4_checkpoint=nvfp4,top_k=8,gate=lambda x:gating,_routing_op=lambda:router)
        env=dict(self=obj,sp_active=sp,use_side_stream=True,side_stream=side,main_stream=main,use_split2=False,hidden_states=None,router_input=None,
                 contextlib=contextlib,get_forward=lambda:types.SimpleNamespace(sp_active=sp),torch=types.SimpleNamespace(cuda=types.SimpleNamespace(stream=lambda s:contextlib.nullcontext(),is_current_stream_capturing=lambda:capture)))
        compile_ast(copy.deepcopy(nodes),env)
        eligible=nvfp4 and marlin and not sp
        assert ('opt_call' in events)==eligible and ('old_call' in events)==(not eligible)
        assert env['topk_ids'] is global_ids
        assert (env['local_topk_ids'] is local)==(eligible and available)
        assert (('local',main) in events)==(eligible and available and not capture)
        assert (('map',side) in events)==(eligible and available and not capture)
        assert ('wait',main,side) in events
        cases+=1
    return {'cases':cases,'scope':'Actual Marlin mapping/metadata branch and model routing/join/record_stream statements; tensor/stream execution mocked.'}

def native_checks():
    s=source('candidate',C);old=source('baseline',C)
    params=block(s,'struct RoutingParams')+';'
    tail=s[s.index('  score_sum += params.division_eps;'):s.index('\n}\n\n}  // namespace muse_spark')]
    oldtail=old[old.index('  score_sum += params.division_eps;'):old.index('\n}\n\n}  // namespace muse_spark')]
    sent=block(s,'  if (params.num_valid_tokens != nullptr')
    guard=s[s.index('  RuntimeCheck(marlin_expert_map.has_value()'):s.index('  if (num_tokens == 0) return;',s.index('  RuntimeCheck(marlin_expert_map.has_value()'))]
    pre=r'''
#include <array>
#include <vector>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
using fp32_t=float;
namespace tvm { namespace ffi {
struct TensorView {void* p; int64_t r,c; void* data_ptr() const{return p;} int64_t stride(int d)const{return d==0?r:c;} };
template<class T> using Optional=std::optional<T>;
}}
struct TensorMatcher {TensorMatcher(std::initializer_list<int>){} template<class T>TensorMatcher& with_dtype(){return *this;} TensorMatcher& with_device(int){return *this;} void verify(tvm::ffi::TensorView){} };
template<class...T>void RuntimeCheck(bool good,T...){if(!good)throw std::runtime_error("rejected");}
float div_full(float a,float b){return a/b;}
int32_t pack_id_weight(int32_t i,float x){uint32_t u;memcpy(&u,&x,4);return (uint32_t(i)<<16)|(u>>16);}
'''
    src=pre+params+'''
void tail(RoutingParams params,int token,int lane,int out_id,float out_score,float score_sum,int kTopK){
'''+tail+'''
}
void oldtail(RoutingParams params,int token,int lane,int out_id,float out_score,float score_sum,int kTopK){
'''+oldtail+'''
}
void sentinel(RoutingParams params,int token,int lane,int kTopK){
'''+sent+'''
}
void guard(tvm::ffi::TensorView gating_output,tvm::ffi::TensorView bias,tvm::ffi::TensorView topk_weights,tvm::ffi::TensorView topk_ids,
 std::optional<tvm::ffi::TensorView> packed_topk_ids,std::optional<tvm::ffi::TensorView> local_expert_mapping,
 std::optional<tvm::ffi::TensorView> num_valid_tokens,std::optional<tvm::ffi::TensorView> marlin_expert_map,
 std::optional<tvm::ffi::TensorView> marlin_topk_ids){
int T=8,E=256,K=8,device_=0,num_tokens=T,num_experts=E,kTopK=K; RoutingParams params{};
'''+guard+'''
}
int main(){
std::array<int32_t,256> map,prior;std::array<int32_t,32> ids,oldids,local,packed,oldpacked;std::array<float,32>w,oldw;
RoutingParams p{};p.topk_ids=ids.data();p.topk_weights=w.data();p.packed_topk_ids=packed.data();p.marlin_topk_ids=local.data();p.marlin_expert_map=map.data();p.stride_im=p.stride_wm=p.stride_pm=p.stride_mm=32;p.route_scale=1.7f;p.division_eps=1e-8f;
RoutingParams q=p;q.topk_ids=oldids.data();q.topk_weights=oldw.data();q.packed_topk_ids=oldpacked.data();
int terms=0;
for(int rank=0;rank<8;rank++){
 for(int i=0;i<256;i++){map[i]=(i/32==rank)?i%32:-1;prior[i]=255-i;}
 for(int oldmap=0;oldmap<2;oldmap++){
  p.local_expert_mapping=q.local_expert_mapping=oldmap?prior.data():nullptr;
  for(int k:{1,8,32})for(int id=0;id<256;id++)for(int lane=0;lane<k;lane++){
   tail(p,0,lane,id,float(id+1)/256,3.5f,k);oldtail(q,0,lane,id,float(id+1)/256,3.5f,k);
   assert(ids[lane]==oldids[lane]&&packed[lane]==oldpacked[lane]);
   assert(memcmp(&w[lane],&oldw[lane],4)==0&&local[lane]==map[id]);terms++;
  }
 }
}
int valid=0;p.num_valid_tokens=&valid;std::fill(w.begin(),w.end(),123.f);
for(int lane=0;lane<32;lane++){sentinel(p,0,lane,32);assert(ids[lane]==-1&&local[lane]==-1&&packed[lane]==int32_t(0xffff0000)&&w[lane]==123.f);}
// Real allocated spans with offset aliases; metadata matchers above are stubs.
std::vector<int32_t> arena(8192);
auto t=[&](int offset,int stride){return tvm::ffi::TensorView{arena.data()+offset,stride,1};};
auto g=t(0,256),b=t(2048,1),ww=t(2304,8),ii=t(2368,8),pk=t(2432,8),mm=t(2496,1),ll=t(2752,8);
guard(g,b,ww,ii,pk,std::nullopt,std::nullopt,mm,ll);
int negatives=0;
for(auto bad:{g,b,ww,ii,pk,mm,t(2369,8),t(2497,8)}){
 try{guard(g,b,ww,ii,pk,std::nullopt,std::nullopt,mm,bad);}catch(std::runtime_error&){negatives++;continue;}return 2;
}
for(auto badmap:{ww,ii,pk,t(2369,1),t(2433,1)}){
 try{guard(g,b,ww,ii,pk,std::nullopt,std::nullopt,badmap,ll);}catch(std::runtime_error&){negatives++;continue;}return 3;
}
try{guard(g,b,ww,ii,pk,std::nullopt,std::nullopt,mm,std::nullopt);return 4;}catch(std::runtime_error&){negatives++;}
std::cout<<"{\\"terms\\":"<<terms<<",\\"alias_or_pair_negatives\\":"<<negatives<<",\\"sentinel_slots\\":32}\\n";
}
'''
    # Correct C++ JSON literal escapes after Python concatenation.
    src=src.replace('\\\\"','\\"').replace('}\\\\n','}\\n')
    with tempfile.TemporaryDirectory(prefix='v24-native-') as td:
        path=Path(td);(path/'test.cpp').write_text(src)
        p=subprocess.run(['/usr/bin/clang++','-std=c++17','-O2',str(path/'test.cpp'),'-o',str(path/'test')],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        p=subprocess.run([str(path/'test')],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,(p.returncode,p.stderr,p.stdout)
    result=json.loads(p.stdout);result['extracted_program_sha256']=sha(src.encode())
    result['limits']='Host execution of actual extracted store/sentinel/span expressions; CUDA/TVM metadata checks and GPU floating point are not emulated.'
    return result

def run():
    manifest=json.loads((HERE/'manifest.json').read_text());assert manifest['parent_revision']==b.REV
    for row in manifest['files']:
        path=row['path'];old=source('baseline',path);new=source('candidate',path)
        assert sha(old.encode())==row['baseline_sha256'] and sha(new.encode())==row['candidate_sha256']
        assert old.encode()==subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',f'{b.REV}:{path}'])
        restored=new
        for a,z in reversed(b.EDITS[path]):restored=b.replace(restored,z,a)
        assert restored==old
        if path.endswith('.py'):ast.parse(new)
    assert sha((HERE/'router_local_ids_v24.patch').read_bytes())==manifest['patch_sha256']
    # Guard source binding is strict: same original active arithmetic/PDL body
    # follows from complete restoration, and only new integer store sites differ.
    code=source('candidate',C)
    assert code.count('params.marlin_expert_map[out_id]')==1
    assert 'params.marlin_expert_map[id]' not in code
    model=source('candidate',M)
    assert 'capture_topk_ids = topk_ids' in model
    assert 'else (topk_scores, topk_ids, packed_topk_ids, local_topk_ids)' in model
    assert 'marlin_expert_map.record_stream(side_stream)' in model
    assert 'if self.nvfp4_checkpoint and not sp_active and getattr(self.experts, "use_marlin", False)' in model
    assert source('candidate',L).count('local_topk_ids = expert_map[local_topk_ids]')==1
    result={'status':'PASS','manifest_sha256':sha((HERE/'manifest.json').read_bytes()),'patch_sha256':manifest['patch_sha256'],
            'complete_file_restorations':5,'dispatch':dispatch_checks(),'low_wrapper':low_wrapper_checks(),'call_boundaries':boundary_checks(),'native':native_checks(),
            'limitations':['No CUDA compile/run, score-bit, resource, graph/capture or timing claim.','Untouched numerical backend bodies are stubbed only in dispatch tests.','Full serving and GPU component validation remain pending.']}
    return result
if __name__=='__main__':
    result=run();(HERE/'cpu_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""Execute actual V27 Python hooks with CPU tensor/module mocks; no CUDA claim."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v27_builder', HERE / 'build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
EVENTS = []


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_checks():
    before = (HERE / 'native_serving_v27.patch').read_bytes()
    manifest = builder.build()
    assert before == (HERE / 'native_serving_v27.patch').read_bytes()
    manifest2 = builder.build()
    assert manifest == manifest2
    for path in (builder.QUANT, builder.EXPERT, builder.MODEL):
        text = (HERE / 'candidate' / path).read_text()
        for edit in reversed(json.loads((HERE / 'edits.json').read_text())):
            if edit['path'] == path:
                assert text.count(edit['new']) == 1
                text = text.replace(edit['new'], edit['old'], 1)
        assert text == (HERE / 'baseline' / path).read_text()
    parent = (HERE.parent / 'native_serving_v26/component.py').read_text()
    candidate = (HERE / 'candidate' / builder.MODULE).read_text()
    p_ast, c_ast = ast.parse(parent), ast.parse(candidate)
    def nodes(tree):
        return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    pn,cn=nodes(p_ast),nodes(c_ast)
    kept = []
    for name in pn.keys() - {'prepare','kernel_fn'}:
        actual = ast.get_source_segment(candidate,cn[name])
        expected = ast.get_source_segment(parent,pn[name])
        if name == 'prepare_native':
            actual = actual.replace('raise NativeUnsupported(', 'raise ValueError(')
        if name == '_ext':
            actual = actual.replace('okc_serving_wgs_v27_d','okc_serving_wgs_v26_d')
        assert actual == expected, name
        kept.append(name)
    def literal(tree,name):
        return next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    cuda=literal(c_ast,'CUDA_SRC')
    for old,new in reversed(json.loads((HERE/'formatting.json').read_text())):
        assert cuda.count(new)==1
        cuda=cuda.replace(new,old)
    assert cuda == literal(p_ast,'CUDA_SRC')
    assert literal(c_ast,'CPP_SRC') == literal(p_ast,'CPP_SRC')
    # Patch applies to a bounded synthetic base, not the dirty production checkout.
    with tempfile.TemporaryDirectory(prefix='v27-patch-') as tmp:
        root=Path(tmp)
        for path in (builder.QUANT,builder.EXPERT,builder.MODEL):
            p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((HERE/'baseline'/path).read_bytes())
        subprocess.run(['git','apply','--check',str(HERE/'native_serving_v27.patch')],cwd=root,check=True,capture_output=True)
        subprocess.run(['git','apply',str(HERE/'native_serving_v27.patch')],cwd=root,check=True,capture_output=True)
        for path in manifest['files']:
            assert (root/path).read_bytes()==(HERE/'candidate'/path).read_bytes()
    assert not any('model_loader/' in p or 'finalizer' in p for p in manifest['files'])
    return {'restoration':'all three stock files exact','core':'CUDA literal exact after nine whitespace reversals; CPP exact',
            'unchanged_python_functions':sorted(kept),'patch_applies':True,'patch_bytes':manifest['patch_bytes']}


class Device:
    def __init__(self,value='cuda:0'):
        text=str(value);self.type=text.split(':')[0]
        self.index=int(text.split(':')[1]) if ':' in text else None
    def __str__(self):return self.type+(':'+str(self.index) if self.index is not None else '')
    def __eq__(self,other):return isinstance(other,Device) and str(self)==str(other)


class Tensor:
    def __init__(self,shape,dtype='bf16',device='cuda:0',tag='',data=None,param=False):
        self.shape=tuple(shape);self.dtype=dtype;self.device=Device(device)
        self.tag=tag;self.data=data;self.param=param
    def dim(self):return len(self.shape)
    def contiguous(self):return self
    def detach(self):return self
    def to(self,dtype):
        return Tensor(self.shape,dtype,self.device,self.tag,self.data)
    def data_ptr(self):return 16
    def clone(self):return Tensor(self.shape,self.dtype,self.device,self.tag,self.data)
    def new_empty(self,shape):return Tensor(shape,self.dtype,self.device,'empty')
    def __getitem__(self,key):
        if not isinstance(key,slice):raise AssertionError(key)
        a,b,c=key.indices(self.shape[0]);n=len(range(a,b,c))
        data=self.data[key] if self.data is not None else None
        return Tensor((n,*self.shape[1:]),self.dtype,self.device,self.tag+f'[{a}:{b}]',data)
    def __setitem__(self,key,value):
        assert self.data is not None
        self.data[key]=value.data
    def fill_(self,value):EVENTS.append(('fill',self.shape,value));return self
    def copy_(self,value):EVENTS.append(('copy',self.shape,value.shape));return self


class Torch:
    bfloat16='bf16';float32='f32';int32='i32';int64='i64';uint8='u8';float8_e4m3fn='e4m3'
    Tensor=Tensor
    def __init__(self):
        self.capturing=False
        self.cuda=NS(get_device_capability=lambda device:(9,0),
                     is_current_stream_capturing=lambda:self.capturing,
                     synchronize=lambda device:EVENTS.append(('sync',str(device))))
    device=staticmethod(Device)
    @staticmethod
    def empty(shape,*,dtype,device):
        if isinstance(shape,int):shape=(shape,)
        return Tensor(shape,dtype,device)
    zeros=empty
    @staticmethod
    def ones(shape,*,dtype,device):
        if isinstance(shape,int):shape=(shape,)
        EVENTS.append(('ones',shape,str(device)))
        return Tensor(shape,dtype,device,'ones')
    @staticmethod
    def empty_like(t):return Tensor(t.shape,t.dtype,t.device,'result')
    @staticmethod
    def full(shape,value,*,dtype,device):
        return Tensor(shape,dtype,device,data=[value]*shape[0])
    @staticmethod
    def arange(n,*,dtype,device):return Tensor((n,),dtype,device,data=list(range(n)))
    @staticmethod
    def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.data==b.data


class Layer:
    def __init__(self,owner,lo=0,n=32):
        object.__setattr__(self,'_parameters',{})
        object.__setattr__(self,'_buffers',{})
        object.__setattr__(self,'_non_persistent_buffers_set',set())
        self._native_v27_owner=owner
        self.use_marlin=True;self.expert_tp=False;self.use_megamoe=False
        self.params_dtype='bf16';self.hidden_size=4096
        self.intermediate_size_per_partition=4096;self.top_k=8
        self.global_num_experts=256;self.num_local_experts=n;self.local_expert_offset=lo
        self.quant_config=NS(group_size=16)
        self.post_expert_norm=NormInput(Tensor((4096,),tag='norm'),eps=1e-6)
        self._post_expert_norm=None;self._post_expert_norm_weight=None
        self._fused_finalize_op=None
        self.fail_register=0;self.register_calls=0;self.fail_delete=None
        names=('gate_up_proj','down_proj','gate_up_proj_weight_scale','down_proj_weight_scale',
               'gate_up_proj_weight_scale_2','down_proj_weight_scale_2')
        for name in names:self._parameters[name]=Tensor((n,),tag=name,param=True)
        data=[-1]*256;data[lo:lo+n]=list(range(n))
        self.register_buffer('_expert_map',Tensor((256,),'i32',data=data),False)
        self.register_calls=0
    def __getattr__(self,key):
        for name in ('_parameters','_buffers'):
            registry=object.__getattribute__(self,name)
            if key in registry:return registry[key]
        raise AttributeError(key)
    def __setattr__(self,key,value):
        if isinstance(value,Tensor) and value.param:
            self._parameters[key]=value
        elif key in self._buffers:self._buffers[key]=value
        else:object.__setattr__(self,key,value)
    def __delattr__(self,key):
        if key==self.fail_delete:raise RuntimeError('injected delete failure')
        if key in self._parameters:del self._parameters[key]
        elif key in self._buffers:del self._buffers[key]
        else:object.__delattr__(self,key)
    def register_buffer(self,name,tensor,persistent=False):
        self.register_calls+=1
        if self.fail_register==self.register_calls:raise RuntimeError('injected register failure')
        assert name not in self._buffers and name not in self._parameters
        self._buffers[name]=tensor
        if not persistent:self._non_persistent_buffers_set.add(name)


class Norm:
    def __init__(self,weight,eps):self.weight=weight;self.eps=eps
    def _get_effective_weight(self):return self.weight


class NormInput(Norm):pass


def env():
    torch=Torch()
    execution=NS(overlap=NS(enable_two_batch_overlap=False,enable_single_batch_overlap=False),
                 offload=NS(cpu_offload_gb=0,offload_group_size=-1),
                 features=NS(enable_memory_saver=False),moe=NS(enable_eplb=False),
                 graph=NS(enable_torch_compile=False,cuda_graph_config=NS(
                     decode=NS(backend='full'),prefill=NS(backend='breakable'))))
    disagg=NS(enable_pdmux=False);parallel=NS(enable_layernorm_sp=False)
    mod=ModuleType('sglang.srt.runtime_context')
    mod.get_exec=lambda:execution;mod.get_disagg=lambda:disagg;mod.get_parallel=lambda:parallel
    sys.modules[mod.__name__]=mod
    source=(HERE/'candidate'/builder.MODULE).read_text();tree=ast.parse(source)
    wanted={'Workspace','expert_outputs','NativeUnsupported','_serial_policy','NativeModelOwner',
            '_layer_eligible','try_install_native','native_forward'}
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    selected += [n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in ('MMAX','EMAX','JMAX','WGS_XROWS','WGS_FIXF','_V27_RAW','_V27_BUFFERS') for t in n.targets)]
    selected.sort(key=lambda n:n.lineno)
    ns={'torch':torch};exec(compile(ast.Module(selected,type_ignores=[]),str(HERE/'candidate'/builder.MODULE),'exec'),ns)
    def core(*args):EVENTS.append(('core',args[2].shape[0],args[-3],args[-2]))
    prepared=[]
    def prepare(weights,**kwargs):
        prepared.append((weights,kwargs));dev=weights['w13'].device
        return dict(w=[Tensor((1,),device=dev,tag='bank'+str(i)) for i in range(5)],
                    lo=kwargs['local_start'],cnt=kwargs['local_count'],device=dev,mod=NS(okc_run=core),grid=132,prmt=True)
    ns['prepare_native']=prepare
    def finalize(y,scores,**kwargs):
        EVENTS.append(('finalize',y.shape[0],scores.shape,kwargs))
        return Tensor((y.shape[0],4096),device=y.device,tag='finalized')
    ns['fused_post_expert_norm_weighted_sum']=finalize
    # Execute the exact modified stock setter, including None-weight handling.
    expert=ast.parse((HERE/'candidate'/builder.EXPERT).read_text())
    cls=next(n for n in expert.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xFp4Experts')
    setter=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_post_expert_norm')
    setter.decorator_list=[]
    normmod=ModuleType('sglang.srt.layers.muse_spark_layers')
    normmod.MuseSpark1xRMSNormWithInputScale=NormInput;sys.modules[normmod.__name__]=normmod
    space={'torch':torch,'Optional':__import__('typing').Optional,'nn':NS(Module=object)}
    exec(compile(ast.Module([setter],type_ignores=[]),str(HERE/'candidate'/builder.EXPERT),'exec'),space)
    Layer.set_post_expert_norm=space['set_post_expert_norm']
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_marlin')
    space['fused_post_expert_norm_weighted_sum']=finalize
    exec(compile(ast.Module([forward],type_ignores=[]),str(HERE/'candidate'/builder.EXPERT),'exec'),space)
    Layer._forward_marlin=space['_forward_marlin']
    # Real low-level quant hook with a fake parent Marlin repacker.
    class Parent:
        def process_weights_after_loading(self,layer):
            EVENTS.append(('stock_prepare',))
            for alias,_ in self._MARLIN_PARAM_ALIASES:
                setattr(layer,alias,Tensor((1,),tag='stock',param=True))
    quant=ast.parse((HERE/'candidate'/builder.QUANT).read_text())
    qcls=next(n for n in quant.body if isinstance(n,ast.ClassDef))
    body=[n for n in qcls.body if (isinstance(n,ast.FunctionDef) and n.name=='_process_weights_for_marlin')
          or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MARLIN_PARAM_ALIASES' for t in n.targets))]
    hook=ast.ClassDef(name='Hook',bases=[ast.Name('Parent',ast.Load())],keywords=[],body=body,decorator_list=[])
    hspace={'Parent':Parent};exec(compile(ast.fix_missing_locations(ast.Module([hook],type_ignores=[])),'actual_quant_hook','exec'),hspace)
    apimod=ModuleType('sglang.kernels.ops.muse_spark_v12.native_serving_v27')
    apimod.try_install_native=ns['try_install_native']
    apimod.native_forward=ns['native_forward']
    apimod.NativeModelOwner=ns['NativeModelOwner']
    sys.modules[apimod.__name__]=apimod
    return ns,torch,execution,disagg,parallel,prepared,hspace['Hook']


def expect_error(fn,kind=RuntimeError):
    try:fn()
    except kind:return
    raise AssertionError('negative control did not raise')


def run_mocks():
    ns,torch,ex,di,pa,prepared,Hook=env();owner=ns['NativeModelOwner']();checks=[]
    # Actual install and quant hook on every local EP offset.
    for rank in range(8):
        layer=Layer(owner,rank*32)
        originals=dict(layer._parameters)
        Hook()._process_weights_for_marlin(layer)
        assert layer._native_v27_bank['lo']==rank*32
        assert prepared[-1][1]==dict(local_start=rank*32,local_count=32,source_is_local=True,globals_prepared=False)
        assert not any(name in layer._parameters for name in originals)
        assert all(name in layer._buffers for name in ns['_V27_BUFFERS'])
        assert not any(name.startswith('w13') or name.startswith('w2') for name in layer._parameters)
        expect_error(lambda:ns['try_install_native'](layer))
    checks.append('eight rank offsets; actual quant hook installs without stock repack; double install rejected')
    assert len(owner._workspaces)==1
    assert owner.workspace('cuda:0') is owner.workspace('cuda:0')
    assert owner.workspace('cuda:0',lane='other') is not owner.workspace('cuda:0')
    assert ns['NativeModelOwner']().workspace('cuda:0') is not owner.workspace('cuda:0')
    torch.capturing=True
    assert owner.workspace('cuda:0') is owner.workspace('cuda:0')
    expect_error(lambda:owner.workspace('cuda:0',lane='new_capture'))
    torch.capturing=False
    checks.append('shared serial owner; distinct models/lanes; capture forbids first allocation')
    # Execute the exact new construction/attachment statements from the real
    # model initializer without pretending to instantiate the rest of SGLang.
    modeltree=ast.parse((HERE/'candidate'/builder.MODEL).read_text())
    modelcls=next(n for n in modeltree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xModel')
    init=next(n for n in modelcls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    ownerstmt=next(n for n in init.body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Attribute) and t.attr=='_native_v27_owner' for t in n.targets))
    loop=next(n for n in init.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='layer_idx')
    attach=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='is_moe_layer')
    model=NS();context={'self':model,'NativeModelOwner':ns['NativeModelOwner'],'side_stream':object()}
    exec(compile(ast.Module([ownerstmt],type_ignores=[]),'actual_model_owner','exec'),context)
    modules=[]
    for native in (True,True,False):
        layer_obj=NS(is_moe_layer=True,mlp=NS(nvfp4_checkpoint=native,experts=NS()))
        context['layer']=layer_obj;exec(compile(ast.Module([attach],type_ignores=[]),'actual_model_attach','exec'),context)
        modules.append(layer_obj)
    assert modules[0].mlp.experts._native_v27_owner is model._native_v27_owner
    assert modules[1].mlp.experts._native_v27_owner is model._native_v27_owner
    assert not hasattr(modules[2].mlp.experts,'_native_v27_owner')
    checks.append('actual model owner/attachment statements share only NVFP4 expert layers')
    policies=[(di,'enable_pdmux',True),(ex.overlap,'enable_two_batch_overlap',True),
              (ex.overlap,'enable_single_batch_overlap',True),(ex.offload,'cpu_offload_gb',1),
              (ex.offload,'offload_group_size',1),(ex.features,'enable_memory_saver',True),
              (ex.graph,'enable_torch_compile',True),(pa,'enable_layernorm_sp',True),
              (ex.moe,'enable_eplb',True),(ex.graph.cuda_graph_config.prefill,'backend','tc_piecewise'),
              (ex.graph.cuda_graph_config.decode,'backend','tc_piecewise'),(ex.graph,'cuda_graph_config',None)]
    for obj,key,value in policies:
        old=getattr(obj,key);setattr(obj,key,value)
        layer=Layer(owner);saved=dict(layer._parameters);count=len(prepared)
        assert ns['try_install_native'](layer) is False
        assert layer._parameters==saved and len(prepared)==count
        setattr(obj,key,old)
    for mutation in ('map','device','norm','dtype','width','owner'):
        layer=Layer(owner)
        if mutation=='map':layer._expert_map.data[0]=7
        elif mutation=='device':layer.gate_up_proj.device=Device('cpu')
        elif mutation=='norm':layer.post_expert_norm=None
        elif mutation=='dtype':layer.params_dtype='fp16'
        elif mutation=='width':layer.hidden_size=2048
        else:layer._native_v27_owner=None
        saved=dict(layer._parameters)
        assert ns['try_install_native'](layer) is False and layer._parameters==saved
    layer=Layer(None);Hook()._process_weights_for_marlin(layer)
    assert all(alias in layer._parameters for alias,_ in Hook._MARLIN_PARAM_ALIASES)
    assert all(native not in layer._parameters for _,native in Hook._MARLIN_PARAM_ALIASES)
    checks.append('12 unsupported policies +6 metadata declines; actual fallback invokes stock hook')
    prepare=ns['prepare_native']
    for failure in ('unsupported','compile','workspace','register','delete'):
        layer=Layer(ns['NativeModelOwner']());saved=dict(layer._parameters);savedb=dict(layer._buffers)
        if failure in ('unsupported','compile'):
            kind=ns['NativeUnsupported'] if failure=='unsupported' else RuntimeError
            def fail(*a,**kw):raise kind('injected')
            ns['prepare_native']=fail
        elif failure=='workspace':
            layer._native_v27_owner.workspace=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('allocation'))
        elif failure=='register':layer.fail_register=3
        else:layer.fail_delete='gate_up_proj_weight_scale'
        if failure=='unsupported':assert ns['try_install_native'](layer) is False
        else:expect_error(lambda:ns['try_install_native'](layer))
        assert layer._parameters==saved and layer._buffers==savedb
        assert getattr(layer,'_native_v27_bank',None) is None
        ns['prepare_native']=prepare
    checks.append('format decline; compile/allocation/register/delete failures preserve original registries')
    layer=Layer(owner);ns['try_install_native'](layer)
    for norm in (NormInput(Tensor((4096,),tag='weight'),3e-5),Norm(None,7e-7)):
        for total in (0,1,8,16,31,32,383,384,385,768,769,16384):
            EVENTS.clear();pre_calls=[]
            def pre(x):pre_calls.append(x);return Tensor(x.shape,x.dtype,x.device,'prenorm')
            output=ns['native_forward'](layer,Tensor((total,4096)),Tensor((total,8),'f32'),
                                        Tensor((total,8),'i64'),norm,pre)
            assert output.shape==(total,4096) and len(pre_calls)==1
            cores=[e for e in EVENTS if e[0]=='core']; finals=[e for e in EVENTS if e[0]=='finalize']
            sizes=[min(384,total-start) for start in range(0,total,384)]
            assert [e[1] for e in cores]==sizes
            assert [e[1] for e in finals]==[max(32,n) if total>=32 and n<32 else n for n in sizes]
            for item in finals:
                kw=item[3];assert kw['norm_weight'] is layer._post_expert_norm_weight
                assert kw['eps']==norm.eps and kw['weight_first']==isinstance(norm,NormInput)
            if sizes and sizes[-1]<32 and total>=32:
                assert ('fill',(32-sizes[-1],8),-1) in EVENTS
            # Every core is finalized before another core can overwrite output.
            assert [e[0] for e in EVENTS if e[0] in ('core','finalize')]==['core','finalize']*len(sizes)
    assert not hasattr(layer,'gate_up_proj')
    assert layer._post_expert_norm_weight.tag=='ones'
    EVENTS.clear()
    layer._marlin_expert_outputs=lambda *a:(_ for _ in ()).throw(AssertionError('released Marlin path used'))
    layer._forward_marlin(Tensor((385,4096)),Tensor((385,8),'f32'),Tensor((385,8),'i32'),norm)
    assert [e[1] for e in EVENTS if e[0]=='core']==[384,1]
    stock=Layer(None)
    stock._marlin_expert_outputs=lambda x,s,i:(Tensor((x.shape[0],8,4096)),Tensor(i.shape,'i32'))
    EVENTS.clear();stock._forward_marlin(Tensor((16,4096)),Tensor((16,8),'f32'),Tensor((16,8),'i32'),norm)
    assert not any(e[0]=='core' for e in EVENTS)
    assert [e[1] for e in EVENTS if e[0]=='finalize']==[16]
    expect_error(lambda:ns['native_forward'](layer,Tensor((1,4096)),Tensor((1,8)),Tensor((1,8),'i32'),None),ValueError)
    ex.overlap.enable_two_batch_overlap=True
    expect_error(lambda:ns['native_forward'](layer,Tensor((1,4096)),Tensor((1,8)),Tensor((1,8),'i32'),norm))
    ex.overlap.enable_two_batch_overlap=False
    checks.append('24 chunk/norm cases; exact setter None-weight device; runtime policy changes fail closed')
    checks.append('actual _forward_marlin selects native only when installed; unchanged default selects stock')
    return {'checks':checks,'counts':{'rank_installs':8,'policy_declines':12,'metadata_declines':6,
                                    'failure_atomicity':5,'forward_cases':24},
            'limits':'Python control/metadata and source conformance only; no numeric CUDA, cross-stream or graph replay proof.'}


def run():
    return {'status':'PASS','source':source_checks(),'mocks':run_mocks(),
            'manifest_sha256':digest(HERE/'manifest.json'),'verifier_sha256':digest(__file__)}


if __name__=='__main__':
    result=run();(HERE/'cpu_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

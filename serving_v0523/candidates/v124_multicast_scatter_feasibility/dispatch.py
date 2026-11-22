"""Execute the pinned Hopper TP8 dispatch at the actual V64 workspace limit."""
import ast
import enum
import hashlib
import json
import re
import subprocess
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Optional

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
PIN = '297e54fd105e974cdf401e5b4d20253a22c16f90'
MIB = 1024 ** 2


def pinned(path):
    return subprocess.check_output(['git','-C',TRACK/'sglang','show',PIN+':python/sglang/'+path],text=True)


class Algo(enum.Enum):
    ONE_SHOT_PUSH=1
    TWO_SHOT_PUSH=2
    ONE_SHOT_PULL=3
    TWO_SHOT_PULL=4


def pick_factory(cap_mib):
    confpath='srt/distributed/device_communicators/configs/custom_all_reduce_v2.py'
    src=pinned(confpath)
    names={'Range','Heuristic','AllReduceConfig','two_shot_push_max_bytes','two_shot_push_slot_bytes','_pack_heuristic','_sm90_configs'}
    nodes=[n for n in ast.parse(src).body if getattr(n,'name','') in names]
    ns={'KB':1024,'MB':MIB,'cache':cache,'NamedTuple':NamedTuple,'Optional':Optional}
    exec(compile(ast.Module(nodes,type_ignores=[]),confpath,'exec'),ns)
    raw=ns['_sm90_configs'](132)[8]
    pull=min(cap_mib*MIB,raw.max_pull_bytes)
    push=min(cap_mib*MIB,raw.max_push_bytes(8))
    config=raw.clip(max_push_bytes=push,max_pull_bytes=pull,world_size=8)
    algos=pinned('srt/distributed/device_communicators/custom_all_reduce_v2.py')
    a=ast.parse(algos)
    cls=next(n for n in a.body if getattr(n,'name','')=='CustomAllReduceV2')
    pick=next(n for n in cls.body if getattr(n,'name','')=='_pick_config')
    picked=next(n for n in a.body if getattr(n,'name','')=='AllReduceConfig')
    env={'AllReduceAlgo':Algo,'NamedTuple':NamedTuple}
    exec(compile(ast.Module([picked,pick],type_ignores=[]),'pinned_pick_config','exec'),env)
    def native(rows,features,graph):
        c=env['_pick_config'](SimpleNamespace(config=config),rows*features*2,graph)
        return 'none' if c is None else 'multicast' if c.use_multicast else c.algo.name.lower()
    actual_slab=2*8*push+pull+128*config.num_pull_blocks
    return native,actual_slab


def report():
    env=pinned('srt/environ.py')
    decl=next(l for l in env.splitlines() if 'SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB =' in l)
    assert 'EnvInt(16 * 1024)' in decl
    group=pinned('srt/distributed/parallel_state.py')
    node=next(n for n in ast.walk(ast.parse(group)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='CAClass')
    assert not node.args and {a.arg for a in node.keywords}=={'group','device'}
    base=TRACK/'candidates/v120b_native_muse_prompt'
    expected=json.loads((base/'validation.json').read_text())['patch_sha256']
    patch=(base/'patch.diff').read_bytes();assert hashlib.sha256(patch).hexdigest()==expected
    for path in ('srt/environ.py','srt/distributed/parallel_state.py','srt/distributed/device_communicators/custom_all_reduce_v2.py','srt/distributed/device_communicators/configs/custom_all_reduce_v2.py'):
        assert ('diff --git a/python/sglang/'+path+' ').encode() not in patch
    p16,slab16=pick_factory(16);p128,slab128=pick_factory(128)
    permits={}
    for h in (4096,8192):
        case={}
        for graph in (False,True):
            choices=[r for r in range(8,16385,8) if p16(r,h,graph)=='multicast']
            assert choices==list(range(choices[0],choices[-1]+1,8))
            case['graph' if graph else 'eager']={'first':choices[0],'last':choices[-1]}
        permits[str(h)]=case
    assert permits=={'4096':{'eager':{'first':24,'last':2048},'graph':{'first':64,'last':2048}},
                     '8192':{'eager':{'first':16,'last':1024},'graph':{'first':32,'last':1024}}}
    samples={}
    for rows in (128,256,512,768,1024,1536,2048,4096,8192,16320,16384):
        samples[rows]={}
        for h in (4096,8192):
            samples[rows][h]={'mib':rows*h*2/MIB,'graph16':p16(rows,h,True),'eager16':p16(rows,h,False),'graph128':p128(rows,h,True)}
    historical={}
    for name,job in [('v41','67e2364305c041ae'),('v48b','8ebed2bced8942ff'),('v64','abd217baf0ed49fd')]:
        rp=TRACK/'runs';j=json.loads((rp/(job+'.json')).read_text())['result'];log=(rp/(job+'.log')).read_text()
        size=float(re.search(r'All Reduce config: symmetric_memory = ([\d.]+) MB',log).group(1))
        assert size==round((slab128 if name=='v48b' else slab16)/MIB,2)
        historical[name]={'job':job,'logged_symmetric_mib':size,'score':j['frontier']['score'],
                          'frontier':[x['ratio'] for x in j['frontier']['points']],'peak_delta_gb':j['gpu_memory']['over_stock_gb']}
        if name=='v64':
            historical[name]['logged_prefill_rows']=[int(x) for x in re.findall(r'Prefill batch, #new-seq: [0-9]+, #new-token: ([0-9]+)',log)]
    return {'pin':PIN,'parent':expected,'bf16_equal_row_aligned_gate_before_other_backends':permits,
            'actual_size_pairs_tested':2*2*(16384//8),'samples':samples,'receipts':historical,
            'workspace_additional_mib_for_128':(slab128-slab16)/MIB}


if __name__=='__main__':
    print(json.dumps(report(),indent=2))

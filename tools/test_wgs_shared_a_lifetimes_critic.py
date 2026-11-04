#!/usr/bin/env python3
"""Independent source-bound SS-A ring model; no CUDA or numerical emulation."""
import argparse
import ast
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BASE='247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0'
CANDIDATE=ROOT/'muse_moe_w4a16/candidates/native_wgs_shared_a_v15.py'
OUT=ROOT/'experiments/iterations/20261104-wgs-shared-a-lifetimes-critic.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def cuda(text):
    tree=ast.parse(text)
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets)))


def extract(source,name):
    # Mask strings/comments before balancing; inline PTX braces are not C++ scopes.
    masked=re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda m:' '*len(m.group()),source)
    match=re.search(r'\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{',masked)
    assert match,name
    start=masked.index('{',match.start());depth=1
    for end in range(start+1,len(masked)):
        depth+=(masked[end]=='{')-(masked[end]=='}')
        if depth==0:return source[match.start():end+1]
    raise AssertionError('unbalanced '+name)


def stages_for(a,b):
    out=[];segments=[]
    for pb,(g0,g1) in enumerate((a,b)):
        sp=((g1-1)//16)*16
        spans=[(sp,g1),(g0,sp)] if pb==0 and g1%16 and sp>g0 else [(g0,g1)]
        for start,end in spans:
            while start<end:
                stop=min(end,(start//16+1)*16)
                sid=len(segments);segments.append((pb,start//16,start%16,stop-start))
                for gs in range(start,stop):
                    out.append(dict(pb=pb,u=gs//16,st=gs%16,first=gs==start,last=gs==stop-1,segment=sid))
                start=stop
    return out,segments


@dataclass
class Ring:
    depth:int
    owner:dict=field(default_factory=dict)
    complete:set=field(default_factory=set)
    release:dict=field(default_factory=dict)
    issue:int=0

    def can_issue(self,n):
        return self.issue<n and (self.issue<self.depth or self.release.get(self.issue-self.depth,0)==65535)

    def put(self):
        q=self.issue;s=q%self.depth
        if s in self.owner:assert self.release.get(self.owner[s],0)==65535
        self.owner[s]=q;self.issue+=1

    def ready(self,q):return self.owner.get(q%self.depth)==q and q in self.complete

    def arrive(self,q,warp):
        bit=1<<warp;assert not self.release.get(q,0)&bit
        self.release[q]=self.release.get(q,0)|bit


@dataclass
class WG:
    q:int=0
    ci:int=0
    phase:str='enter'
    mask:int=0
    pending:list=field(default_factory=list)
    bank_values:dict=field(default_factory=dict)
    prev:int=-1
    releaseq:int=-1
    after_release:str=''
    pd:int=-1
    trace:list=field(default_factory=list)


def simulate(a,b,seed,ns=2,nx=2,banks=2,missing_weight_warp=False,premature_x=False,
             missing_publication=False):
    stages,segments=stages_for(a,b);n=len(stages)
    w=Ring(ns);x=Ring(nx);wgs=[WG() for _ in range(4)]
    rng=random.Random(seed);events=0;flushes=0;joined=0
    def check_A(g):
        for q,ci,bank in g.pending:
            assert g.bank_values[bank]==(q,ci,15),'A bank overwritten before retirement'
    while True:
        actions=[]
        for kind,ring in (('W',w),('X',x)):
            if ring.can_issue(n):actions.append(('issue',kind))
            for q in ring.owner.values():
                if q not in ring.complete:actions.append(('copy',kind,q))
        for i,g in enumerate(wgs):
            check_A(g)
            if g.phase=='done':continue
            if g.q==n:
                assert not g.pending and g.prev<0
                actions.append(('finish',i));continue
            if g.phase=='enter':
                if w.ready(g.q) and x.ready(g.q):actions.append(('enter',i))
            elif g.phase in ('reuse','store','publish','Wrelease','Xrelease'):
                for wp in range(4):
                    if not(g.mask>>wp)&1:
                        if not(missing_weight_warp and g.phase=='Wrelease' and i==3 and wp==3):
                            actions.append((g.phase,i,wp))
                if missing_publication and g.phase=='publish' and g.mask:
                    actions.append(('bad_issue',i))
            else:actions.append((g.phase,i))
        if all(g.phase=='done' for g in wgs):
            assert w.issue==x.issue==n
            assert all(w.release.get(q,0)==65535 and x.release.get(q,0)==65535 for q in range(n))
            expected=[(q,ci,jj,st['pb'],st['u'],st['st'])
                for q,st in enumerate(stages) for ci in range(8) for jj in range(2)]
            assert all(g.trace==expected for g in wgs)
            return dict(a=a,b=b,seed=seed,stages=n,segments=len(segments),events=events,
                joined_stage_acquires=joined,deferred_flushes=flushes,warp_releases=32*n)
        if not actions:raise AssertionError('deadlock: missing release or insufficient X lookahead')
        action=rng.choice(actions);name=action[0];events+=1
        assert events<50000,'event bound'
        if name=='issue':(w if action[1]=='W' else x).put();continue
        if name=='copy':
            ring=w if action[1]=='W' else x;ring.complete.add(action[2]);continue
        i=action[1];g=wgs[i]
        if name=='finish':
            if g.pd>=0:flushes+=1;g.pd=-1
            g.phase='done';continue
        st=stages[g.q]
        if name=='enter':
            assert w.ready(g.q) and x.ready(g.q)
            joined+=1;g.ci=0;g.phase='reuse';g.mask=0
        elif name=='reuse':
            g.mask|=1<<action[2]
            if g.mask==15:
                assert not any(bank==g.ci%banks for _,_,bank in g.pending),'A bank reused while live'
                g.phase='store'
                g.mask=0
        elif name=='Wrelease':
            w.arrive(g.q,i*4+action[2]);g.mask|=1<<action[2]
            if g.mask==15:g.phase='reuse';g.mask=0
        elif name=='store':
            # One event represents one warp's disjoint BF16 stores. Publication
            # is separately modeled; no MMA may see an incompletely written bank.
            bank=g.ci%banks
            assert not any(oldbank==bank for _,_,oldbank in g.pending)
            g.mask|=1<<action[2];g.bank_values[bank]=(g.q,g.ci,g.mask)
            if g.mask==15:g.phase='publish';g.mask=0
        elif name=='publish':
            # Per-warp event represents every writer's proxy fence and arrival.
            g.mask|=1<<action[2]
            if g.mask==15:g.phase='issue_mma'
        elif name in ('issue_mma','bad_issue'):
            assert g.mask==15,'missing collective publication'
            assert g.bank_values[g.ci%banks]==(g.q,g.ci,15)
            for jj in range(2):g.trace.append((g.q,g.ci,jj,st['pb'],st['u'],st['st']))
            g.pending.append((g.q,g.ci,g.ci%banks));g.phase='wait'
        elif name=='wait':
            if premature_x and g.ci==0 and g.prev>=0:
                assert not any(q==g.prev for q,_,_ in g.pending),'premature previous X release'
            # Worst legal wait1: retain exactly the most recent group.
            g.pending=g.pending[-1:]
            if g.ci==0 and g.prev>=0:
                g.releaseq=g.prev;g.prev=-1;g.after_release='after_wait';g.phase='Xrelease';g.mask=0
            else:g.phase='after_wait'
        elif name=='Xrelease':
            assert not any(q==g.releaseq for q,_,_ in g.pending),'live X read at release'
            x.arrive(g.releaseq,i*4+action[2]);g.mask|=1<<action[2]
            if g.mask==15:g.phase=g.after_release;g.mask=0
        elif name=='after_wait':
            if st['first'] and g.ci==3 and g.pd>=0:
                assert g.pd!=st['segment'];g.pd=-1;flushes+=1
            if g.ci<7:
                g.ci+=1;g.phase='Wrelease' if g.ci==6 else 'reuse';g.mask=0
            elif st['last']:g.phase='drain'
            else:g.prev=g.q;g.q+=1;g.phase='enter'
        elif name=='drain':
            g.pending=[];g.releaseq=g.q;g.prev=-1
            g.phase='Xrelease';g.mask=0;g.after_release='segment_end'
        elif name=='segment_end':
            assert not g.pending
            g.pd=st['segment'];g.q+=1;g.phase='enter'
            # Actual wgs_phase flushes its last pending segment before phase B.
            if g.q==n or stages[g.q]['pb']!=st['pb']:flushes+=1;g.pd=-1
        else:raise AssertionError(name)


def model_suite():
    patterns=[((0,0),(0,0)),((0,1),(0,1)),((0,32),(0,16)),((3,49),(7,35)),
        ((15,33),(15,17)),((25,65),(3,17)),((0,3),(0,0)),((0,0),(0,1))]
    rows=[simulate(a,b,seed) for a,b in patterns for seed in range(6)]
    negatives=[]
    for label,kw in [('single_A_bank',dict(banks=1)),('X_depth_one',dict(nx=1)),
        ('missing_16th_weight_release',dict(missing_weight_warp=True)),
        ('premature_previous_X_release',dict(premature_x=True)),
        ('missing_publication_barrier',dict(missing_publication=True))]:
        caught=None
        for seed in range(6):
            try:simulate((0,32),(0,16),seed,**kw)
            except AssertionError as error:caught=str(error);break
        assert caught,label+' negative not caught'
        negatives.append({'case':label,'caught':caught})
    return {'cases':rows,'event_count':sum(r['events'] for r in rows),'negative_controls':negatives}


def range_contract(source):
    costs=[]
    for phase in ('A','B'):
        m=re.search(r'^#define OKC_WGS'+phase+r' ([0-9, ]+)',source,re.M)
        costs.append(list(map(int,m.group(1).split(','))))
        assert len(costs[-1])==8
    cases=[]
    histograms=([1],[8],[9],[16],[64],[384],[1,8,9,16,24,32,40,48,56,64],
        [321]*9+[129]+[1]*22)
    for counts in histograms:
        jobs=[min(64,c-start) for c in counts for start in range(0,c,64)]
        for grid in (7,132):
            for pb in (0,1):
                upj=32 if pb==0 else 16;spj=upj*16
                per=[costs[pb][(c+7)//8-1] for c in jobs];prefix=[0]
                for cost in per:prefix.append(prefix[-1]+cost*spj)
                total=prefix[-1]
                def at(pos):
                    j=bisect.bisect_right(prefix,pos)-1
                    return j*spj+(pos-prefix[j])//per[j]
                boundaries=[at(total*c//grid) for c in range(grid)]+[len(jobs)*spj]
                shares=defaultdict(list)
                for c in range(grid):
                    a,b=boundaries[c:c+2]
                    ordered,segments=stages_for((a,b) if pb==0 else (0,0),(a,b) if pb else (0,0))
                    assert sorted(st['u']*16+st['st'] for st in ordered)==list(range(a,b))
                    for _,u,s0,nst in segments:shares[u].append((c,s0,s0+nst))
                assert len(shares)==len(jobs)*upj
                splits=0
                for u,parts in shares.items():
                    parts.sort(key=lambda p:p[1])
                    assert parts[0][1]==0 and parts[-1][2]==16
                    assert sum(e-s for _,s,e in parts)==16
                    assert all(a[2]==b[1] for a,b in zip(parts,parts[1:]))
                    if len(parts)>1:
                        splits+=1;owner=parts[0][0]
                        assert owner<160 and all(c>owner for c,_,_ in parts[1:])
                        # Actual inherited inverse owner, including exact correction.
                        j,off=divmod(u*16,spj);xe=prefix[j]+(off+1)*per[j]
                        inv=min(grid-1,(xe-1)*grid//total)
                        while inv+1<grid and total*(inv+1)//grid<xe:inv+=1
                        while inv>0 and total*inv//grid>=xe:inv-=1
                        assert inv==owner
                cases.append({'histogram':counts,'grid':grid,'phase':pb,'jobs':len(jobs),
                    'stages':boundaries[-1],'split_units':splits})
    return {'cases':cases,'case_count':len(cases),
        'claim':'Exact stage partition and inverse split owner; B finisher dependencies point to later CTAs.',
        'limits':'No global numerical fixup or GPU visibility emulation; unchanged flush/gather/range bodies are separately byte-bound.'}


def source_contract(parent,current):
    unchanged=('wgs_split','wgs_wproducer','wgs_xproducer','wgs_gather','wgs_epi_tile',
        'wgs_flush','wgs_phase','wgs_prefix','wgs_stage_at','wgs_range','wgs_cta_of','wgs_ranges','moe_fused')
    hashes={}
    for name in unchanged:
        p=extract(parent,name);c=extract(current,name)
        assert p==c,'inherited function changed: '+name
        hashes[name]=sha(c.encode())
    required=(
        '#define OKC_WGSNX 2', 'constexpr int WGS_NS = 2;',
        'static_assert(WGS_NS == 2 && WGS_NX == 2,',
        'constexpr int WGS_A_BANK_BYTES = 4 * 2 * 64 * 16 * 2;',
        'constexpr int WGS_A_BYTES = 2 * WGS_A_BANK_BYTES;',
        'constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;',
        '#define OKC_WGSFCI 3', '#define OKC_WGSGS 2',
        'return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS + okc::WGS_A_BYTES;')
    for text in required:assert text in current,text
    ps=extract(parent,'wgs_seg');cs=extract(current,'wgs_seg')
    def clean(s):return re.sub(r'\s+','',re.sub(r'//[^\n]*|/\*[\s\S]*?\*/','',s))
    def canonical_segment(seg,new):
        if new:
            bar='asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");'
            assert seg.count(bar)==2,'both explicit WG barriers required'
            before=seg.index(bar);store=seg.index('wgs_a_store(')
            fence=seg.index('asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");')
            after=seg.index(bar,before+len(bar))
            assert before<store<fence<after<seg.index('wg_fence();')
            for text in (
                'static_assert(GS == 2, "Shared-A banks preserve the incumbent GS2 schedule");',
                'const uint32_t aring_s = smem_u32(ring + WGS_NS * UNIT_BYTES + WGS_NX * WGS_XS);',
                'const uint32_t atile = aring_s + (ci & 1) * WGS_A_BANK_BYTES + wg * 4096;',
                'const uint64_t ad = wg_desc(atile, 128, 256);',bar,
                'asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");'):
                assert text in seg,text
                seg=seg.replace(text,'')
            store_text='wgs_a_store(atile + jj * 2048, wp, lane, bmul2l(r0, sgp), bmul2h(r1, sgp),\n                      bmul2l(r2, sgp), bmul2h(r3, sgp));'
        else:
            assert 'uint32_t af[2][GS][4];' in seg
            seg=seg.replace('uint32_t af[2][GS][4];','')
            store_text='af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);'
        assert store_text in seg
        seg=seg.replace(store_text,'EXACT_FRAGMENT_STORE;')
        calls=[]
        def normal(m):
            args=[x.strip() for x in m.group(2).split(',')]
            assert len(args)==(3 if new else 6)
            if new:
                assert args[1] in ('ad','ad + (uint64_t)(jj * 128)')
                a_index='first' if args[1]=='ad' else 'jj'
            else:
                assert args[1:5] in ([f'af[ci & 1][jj][{r}]' for r in range(4)],
                                    [f'af[0][0][{r}]' for r in range(4)])
                a_index='first' if args[1]=='af[0][0][0]' else 'jj'
            calls.append((m.group(1),args[0],args[-1],a_index))
            return f'{m.group(1)}({args[0]}, {args[-1]});'
        seg=re.sub(r'WGSMMA<NTT>::(mma0|mma)\(([\s\S]*?)\);',normal,seg)
        assert len(calls)==6
        return clean(seg),calls
    pn,pc=canonical_segment(ps,False);cn,cc=canonical_segment(cs,True)
    assert pn==cn and pc==cc,'segment differs beyond reviewed operand storage/publication'
    _,wrong_A=canonical_segment(cs.replace('ad + (uint64_t)(jj * 128)','ad'),True)
    assert wrong_A!=pc,'wrong second-k16 descriptor negative not detected'
    # Removing either required barrier must be rejected even though the event
    # model treats each warpgroup wait as collective, rather than lane-level GPU execution.
    bar='asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");'
    for which in (0,1):
        positions=[m.start() for m in re.finditer(re.escape(bar),cs)]
        bad=cs[:positions[which]]+cs[positions[which]+len(bar):]
        try:canonical_segment(bad,True)
        except AssertionError:pass
        else:raise AssertionError('missing barrier negative not detected')
    wrappers=current[current.index('template <int NTT> struct WGSMMA;'):current.index('// Exact original RS fragment pairs')]
    instructions=re.findall(r'wgmma\.mma_async\.sync\.aligned\.m64n(\d+)k16\.f32\.bf16\.bf16[^\n]+',wrappers)
    assert list(map(int,instructions))==[n for n in range(8,65,8) for _ in range(2)]
    for n in range(1,9):
        section=wrappers.split(f'template <> struct WGSMMA<{n}> {{')[1].split('\n};')[0]
        assert section.count('"l"(adesc), "l"(bdesc)')==2
        assert section.count('0, 1, 1, 0, 0;')==1 and section.count('1, 1, 1, 0, 0;')==1
        assert section.count('"=f"')==4*n and section.count('"+f"')==4*n
    return {'unchanged_function_sha256':hashes,'segment_sha256':sha(cs.encode()),
        'changed_source_contract':'PASS_EXACT_SCOPED_CANONICAL_RESTORATION',
        'MMA_shape_variants':list(range(8,65,8)),'missing_barrier_negative_controls':2,
        'wrong_second_k16_descriptor_negative_detected':True,
        'native_descriptor_scatter':native_layout(current)}


def native_layout(source):
    helper=extract(source,'wgs_a_store');desc=extract(source,'wg_desc')
    pattern=r'asm volatile\("st\.shared\.u32 \[%0\], %1;\\n" :: "r"\(([^)]*)\), "r"\((r[0-3])\) : "memory"\);'
    transformed,count=re.subn(pattern,lambda m:f'store({m.group(1)}, {m.group(2)});',helper)
    assert count==4,'actual helper scatter extraction'
    prefix=r'''
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#define CHECK(x) do { if(!(x)) std::exit(1); } while(0)
uint32_t memory[8192], seen[8192];
void store(uint32_t a,uint32_t v){CHECK(a%4==0 && a<32768 && !seen[a/4]);seen[a/4]=1;memory[a/4]=v;}
'''
    suffix=r'''
int main(){
 for(int bank=0;bank<2;bank++)for(int wg=0;wg<4;wg++)for(int jj=0;jj<2;jj++){
  int tile=bank*16384+wg*4096+jj*2048,tag=(bank*8+wg*2+jj)*1024;
  auto d=wg_desc(161536+tile,128,256);
  CHECK((d&16383)==(161536+tile)/16 && ((d>>16)&16383)==8 && ((d>>32)&16383)==16);
  for(int wt=0;wt<128;wt++){
   int wp=wt/32,l=wt%32;uint32_t r[4];
   for(int q=0;q<4;q++){
    int m=16*wp+l/4+8*(q&1),k=2*(l&3)+8*(q/2);
    r[q]=(tag+m*16+k)|(uint32_t(tag+m*16+k+1)<<16);
   }
   wgs_a_store(tile,wp,l,r[0],r[1],r[2],r[3]);
  }
  for(int m=0;m<64;m++)for(int k=0;k<16;k++){
   int off=tile+(m/8)*256+(k/8)*128+(m%8)*16+(k%8)*2;
   CHECK(((memory[off/4]>>(16*(k&1)))&65535)==uint32_t(tag+m*16+k));
  }
 }
 for(int i=0;i<8192;i++)CHECK(seen[i]);
 std::puts("16384 BF16 coordinates; 16 descriptors; exact actual helper");
}
'''
    program=prefix+'uint64_t '+desc+'\nvoid '+transformed+suffix
    with tempfile.TemporaryDirectory(prefix='okc-ss-lifetime-critic-') as td:
        exe=Path(td)/'layout'
        for bad in (False,True):
            code=program.replace('tile + off + 256','tile + off + 128',1) if bad else program
            r=subprocess.run(['clang++','-x','c++','-std=c++17','-O2','-o',str(exe),'-'],input=code,text=True,capture_output=True,timeout=30)
            assert r.returncode==0,r.stderr
            r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            assert (r.returncode!=0) if bad else (r.returncode==0),r.stderr
    return {'actual_helper_sha256':sha(helper.encode()),'actual_descriptor_sha256':sha(desc.encode()),
        'native_translation_sha256':sha(program.encode()),'BF16_coordinates':16384,'descriptors':16,
        'wrong_row_stride_negative_detected':True}


def run(candidate=CANDIDATE,expected_sha=None):
    candidate=Path(candidate);raw=candidate.read_bytes();actual=sha(raw)
    assert expected_sha and actual==expected_sha,'exact expected candidate SHA required'
    parent=(ROOT/'experiments/candidates/muse_moe_w4a16'/f'{BASE}.py').read_bytes()
    assert sha(parent)==BASE
    contract=source_contract(cuda(parent),cuda(raw))
    assert contract['changed_source_contract']!='PENDING_DRAFT_REVIEW','draft binding not finished'
    return {'status':'PASS_CPU_MODEL_NOT_GPU_PROOF','candidate_sha256':actual,'parent_sha256':BASE,
        'verifier_sha256':sha(Path(__file__).read_bytes()),'source_contract':contract,'model':model_suite(),
        'range_contract':range_contract(cuda(raw)),
        'limits':['Independent actors represent producer issue/completion and collective WG operations; no CUDA memory model is emulated.',
                 'Wait-group retirement is a WG-collective event: four warps returning from wait1 at unequal times are not separately simulated. Pre-overwrite-barrier justification rests on the explicit source/PTX contract; its negative is a source-conformance check, not a simulated GPU hazard witness.',
                 'Cross-CTA input readiness and split-fixup arithmetic are inherited source contracts, not emulated global numerical data.',
                 'Zero-stage/empty CTA ranges are modeled; global nj==0 routing is not proven. Inherited wgs_ranges has no explicit nj==0 guard; official reference defaults to busiest rank and rejects zero-local-output cases.',
                 'No compiler/resource/numerical/performance claim.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-only',action='store_true')
    parser.add_argument('--candidate',type=Path,default=CANDIDATE);parser.add_argument('--expected-sha')
    args=parser.parse_args()
    if args.model_only:
        report=model_suite();print(json.dumps({'scope':'MODEL_ONLY_NO_SOURCE_PASS',
            'cases':len(report['cases']),'events':report['event_count'],'negative_controls':report['negative_controls']}))
    else:
        report=run(args.candidate,args.expected_sha);OUT.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'saved':str(OUT),'candidate_sha256':report['candidate_sha256']}))

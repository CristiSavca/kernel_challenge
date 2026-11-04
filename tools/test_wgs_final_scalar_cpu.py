#!/usr/bin/env python3
"""Bounded actual-source checks for WGS final scalar ownership; no CUDA execution."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import build_wgs_final_scalar_candidate as builder
import test_wgs_shared_a_cpu as common
from test_wgs_wait2_cpu import literal, preprocess

OUT = builder.ROOT / 'experiments/iterations/20261104-wgs-final-scalar-cpu.json'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def finalizer(src):
    a = src.index('#if OKC_WGS && OKC_WGSFINSS\n')
    return src[a:src.index('#elif OKC_FINCOL\n', a)]


def conformance(parent, candidate):
    ps = literal(ast.parse(parent), 'CUDA_SRC')
    cs = literal(ast.parse(candidate), 'CUDA_SRC')
    restored = candidate
    for new, old in ((builder.OWNER_NEW, builder.OWNER_OLD),
                     (builder.CONSUMER_NEW, builder.CONSUMER_OLD),
                     (builder.HOST + '\n', ''), (builder.CALL, ''),
                     ('okc_moe_wgs_final_scalar_v19_', 'okc_moe_kperm3_v9_'),
                     ('verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence', 'verbose=False,'),
                     (builder.HEADER_NEW, builder.HEADER_OLD)):
        assert restored.count(new) == 1, new[:100]
        restored = restored.replace(new, old)
    assert restored == parent, 'complete parent byte restoration failed'
    pf, cf = finalizer(ps), finalizer(cs)
    assert cf.replace(builder.OWNER_NEW, builder.OWNER_OLD).replace(builder.CONSUMER_NEW, builder.CONSUMER_OLD) == pf
    pp, cp = pf.split('      CTA_SYNC();'), cf.split('      CTA_SYNC();')
    assert len(pp) == len(cp) == 4
    assert pp[0] == cp[0] and pp[3] == cp[3], 'row reads/squares/warp reduction or tail barrier changed'
    assert pp[1].replace(builder.OWNER_OLD, builder.OWNER_NEW) == cp[1]
    assert pp[2].replace(builder.CONSUMER_OLD, builder.CONSUMER_NEW) == cp[2]
    builds = 0
    pt = ast.parse(parent)
    for name in ('TINY_DEFINES', 'TINY2_DEFINES', 'MID_DEFINES', 'MAIN_FALLBACK_DEFINES'):
        for mul in (0, 1):
            for prmt in (0, 1):
                flags = [*literal(pt, name), f'-DOKC_MULDEC={mul}', f'-DOKC_PRMTSC={prmt}',
                         '-DOKC_DQ7=1', '-DOKC_NS=3', '-DOKC_NSF=3']
                assert preprocess(ps, flags) == preprocess(cs, flags), (name, mul, prmt)
                builds += 1
    return dict(complete_parent_byte_restoration=True, restored_parent_sha256=sha(restored.encode()),
                non_wgs_preprocessed_builds=builds, finalizer_barriers_unchanged=3,
                row_square_warp_sum_and_output_FMA_cast_source_identical=True), ps, cs


def phases(src, prefix):
    parts = finalizer(src).split('      CTA_SYNC();')
    assert len(parts) == 4, 'three unchanged finalizer barriers required'
    owner = parts[1]
    consumer = parts[2]
    coefficient_line = next(line for line in consumer.splitlines() if 'const float r =' in line)
    consumer = consumer.replace(coefficient_line, coefficient_line +
                                f'\n          coefficient({int(prefix == "candidate")}, tid, sl, r);')
    return ('void ' + prefix + '_owner(Params&p,Smem&S,Shared&red,int tid,int tk){\n' + owner + '\n}\n'
            + 'void ' + prefix + '_consumer(Params&p,Smem&S,Shared&red,int tid,int tk){\n'
            + 'int qv[8]; float4 cached_a[8],cached_b[8];\n'
            + 'for(int sl=0;sl<8;sl++){qv[sl]=S.pos[tk*8+sl]; cached_a[sl]=values(tid,sl,0); cached_b[sl]=values(tid,sl,1);}\n'
            + consumer + '\n}\n')


def native_finalizer(ps, cs):
    code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
constexpr int HD=4096,NT=512,NW=16;
struct float4{float x,y,z,w;}; struct uint4{uint32_t x,y,z,w;};
struct __nv_bfloat162{uint32_t bits;};
uint32_t bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
uint16_t bf(float x){uint32_t b=bits(x);if((b&0x7fffffff)>0x7f800000)return uint16_t((b>>16)|64);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
__nv_bfloat162 __floats2bfloat162_rn(float x,float y){return {uint32_t(bf(x))|(uint32_t(bf(y))<<16)};}
uint64_t rsq_calls=0,weight_reads=0;
float rsqrtf(float x){rsq_calls++;return 1.f/std::sqrt(x);}
uint32_t coefficient_bits[8];uint64_t coefficient_checks=0;
void coefficient(int candidate,int tid,int sl,float r){
 if(!candidate&&tid==0)coefficient_bits[sl]=bits(r);
 else assert(coefficient_bits[sl]==bits(r));
 if(candidate)coefficient_checks++;
}
struct Smem{std::vector<int16_t>pos;};
struct Weights{std::vector<float>v;Smem*S;
 float operator[](int i){assert(i>=0&&i<(int)v.size());assert(S->pos[i]>=0);weight_reads++;return v[i];}};
struct Params{Weights topk_w;uint16_t*out;};
struct Shared;
struct Slot{Shared*s;int i;operator float()const;Slot&operator=(float v);};
struct Shared{
 float v[136];int written[8]={};int phase=0,epoch=0,owner=-1;uint64_t scalar_reads=0;
 Slot operator[](int i){assert(i>=0&&i<136);return {this,i};}
 void publish(){for(int i=0;i<8;i++)assert(written[i]==epoch);phase=1;}
};
Slot::operator float()const{if(i>=128){assert(s->phase==1&&s->written[i-128]==s->epoch);s->scalar_reads++;}return s->v[i];}
Slot&Slot::operator=(float v){assert(s->phase==0&&i>=128&&i<136&&i-128==s->owner);s->v[i]=v;s->written[i-128]=s->epoch;return *this;}
float4 values(int tid,int sl,int half){
 float z=float(((tid*13+sl*17+half*3)%31)-15)*0.125f;
 return {z,z*0.5f+0.03125f,-z*0.25f,z*0.0625f-0.015625f};
}
''' + phases(ps, 'parent') + phases(cs, 'candidate') + r'''
uint64_t cases=0,threads=0,live_routes=0,old_calls=0,new_calls=0,stores=0;
void one(int mask,int tk,int epoch,Smem&S,Params&p0,Params&p1,Shared&r0,Shared&r1){
 int R=0;
 for(int sl=0;sl<8;sl++){
  bool live=(mask>>sl)&1;S.pos[tk*8+sl]=live?int16_t(sl):-1;R+=live;
  p0.topk_w.v[tk*8+sl]=p1.topk_w.v[tk*8+sl]=sl==0?-0.f:float(sl-4)*0.03125f;
 }
 for(Shared*r:{&r0,&r1}){
  r->phase=0;r->epoch=epoch;r->scalar_reads=0;
  for(int w=0;w<16;w++)for(int sl=0;sl<8;sl++){
   // Directed nonnegative mixed-magnitude partials make reduction order observable.
   r->v[w*8+sl]=(w==((sl+epoch)%16))?16777216.f:float(1+((w+epoch)%3));
  }
  for(int sl=0;sl<8;sl++)r->v[128+sl]=std::nanf("");
 }
 uint64_t c=rsq_calls,wr=weight_reads;
 for(int i=0;i<512;i++){int tid=(i*73)%512;r0.owner=tid;parent_owner(p0,S,r0,tid,tk);}
 r0.publish();
 for(int i=0;i<512;i++){int tid=(i*29)%512;parent_consumer(p0,S,r0,tid,tk);}
 assert(rsq_calls-c==uint64_t(512*R)&&weight_reads-wr==uint64_t(512*R));old_calls+=rsq_calls-c;
 c=rsq_calls;wr=weight_reads;
 for(int i=0;i<512;i++){int tid=(i*73)%512;r1.owner=tid;candidate_owner(p1,S,r1,tid,tk);}
 r1.publish();
 for(int i=0;i<512;i++){int tid=(i*29)%512;candidate_consumer(p1,S,r1,tid,tk);}
 assert(rsq_calls-c==uint64_t(R)&&weight_reads-wr==uint64_t(R));new_calls+=rsq_calls-c;
 assert(r0.scalar_reads==uint64_t(512*R)&&r1.scalar_reads==uint64_t(512*R));
 for(int col=0;col<HD;col++)assert(p0.out[tk*HD+col]==p1.out[tk*HD+col]);
 // Explicit end-of-token barrier before scratch is reused on the next iteration.
 r0.phase=r1.phase=2;cases++;threads+=512;live_routes+=R;stores+=16;
}
int main(){
 Smem S;S.pos.resize(384*8,-1);std::vector<uint16_t>a(384*HD,0xa5a5),b(384*HD,0xa5a5);
 Params p0{{std::vector<float>(384*8),&S},a.data()},p1{{std::vector<float>(384*8),&S},b.data()};
 Shared r0{},r1{};int epoch=0;
 for(int mask=0;mask<256;mask++){
  // CTA0 at grid132 for M384, followed by an M1 chunk remainder. Reuse same slots.
  for(int tk:{0,132,264})one(mask,tk,++epoch,S,p0,p1,r0,r1);
  int rotated=((mask<<1)|(mask>>7))&255;one(rotated,0,++epoch,S,p0,p1,r0,r1);
 }
 assert(old_calls==new_calls*512);
 printf("%llu %llu %llu %llu %llu %llu %llu\n",(unsigned long long)cases,(unsigned long long)threads,
  (unsigned long long)live_routes,(unsigned long long)old_calls,(unsigned long long)new_calls,(unsigned long long)stores,
  (unsigned long long)coefficient_checks);
}
'''
    out, _ = common.compile_run(code)
    names = ('token_cases', 'consumer_thread_pairs', 'live_route_cases', 'parent_scalar_evaluations',
             'candidate_scalar_evaluations', 'owner_slot_stores_both_versions', 'FP32_coefficient_bit_comparisons')
    return dict(zip(names, map(int, out.split())))


def token_assignment(ps, cs):
    loop = 'for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x)'
    assert finalizer(ps).count(loop) == finalizer(cs).count(loop) == 1
    sizes = (0, 1, 8, 16, 32, 128, 132, 133, 256, 257, 384, 385, 769)
    coverage = 0
    for m in sizes:
        rows = []
        for base in range(0, m, 384):
            chunk = min(384, m - base)
            for cta in range(132):
                rows.extend(base + t for t in range(cta, chunk, 132))
        assert sorted(rows) == list(range(m))
        coverage += len(rows)
    cols = [tid * 8 + i for tid in range(512) for i in range(8)]
    assert cols == list(range(4096))
    return dict(physical_M=list(sizes), rows_once=coverage, columns_once=4096, grid=132,
                max_chunk=384, inherited_zero_job_support_not_claimed=True)


def host_mock(cs):
    code = (inspect.getsource(common.host_mock).replace('def host_mock(cs):', 'def adapted(cs):')
            .replace('okc_shared_a_resources', 'okc_final_scalar_resources')
            .replace('194304', '231168').replace('SSAR ', 'FSR '))
    ns = dict(common.__dict__)
    exec(code, ns)
    return ns['adapted'](cs)


def run(candidate=builder.CANDIDATE, expected_sha=None):
    candidate = Path(candidate)
    raw = candidate.read_bytes()
    digest = sha(raw)
    if expected_sha:
        assert digest == expected_sha
    parent = builder.PARENT.read_bytes()
    assert sha(parent) == builder.PARENT_SHA
    assert builder.build().encode() == raw
    scope, ps, cs = conformance(parent.decode(), raw.decode())
    trace = native_finalizer(ps, cs)
    negatives = []
    for name, old, new in (
        ('missing_owner_live_guard', 'if (S.pos[tk * 8 + tid] >= 0)', 'if (true)'),
        ('wrong_owner_slot', 'red[NW * 8 + tid] = r;', 'red[NW * 8 + ((tid + 1) & 7)] = r;'),
        ('reversed_FP32_sum_order', 'for (int w = 0; w < NW; ++w)', 'for (int w = NW - 1; w >= 0; --w)'),
        ('old_consumer_recomputes_coefficient', builder.CONSUMER_NEW, builder.CONSUMER_OLD),
    ):
        cf = finalizer(cs)
        assert cf.count(old) == 1
        mutated = cs.replace(cf, cf.replace(old, new))
        try:
            native_finalizer(ps, mutated)
        except common.NativeCheckFailed:
            negatives.append(name)
        else:
            raise AssertionError('negative accepted: ' + name)
    cf = finalizer(cs)
    missing_barrier = cf.replace('      CTA_SYNC();', '', 1)
    try:
        phases(cs.replace(cf, missing_barrier), 'invalid')
    except AssertionError:
        negatives.append('missing_finalizer_barrier')
    else:
        raise AssertionError('missing barrier accepted')
    result = dict(status='PASS', source_sha256=digest, parent_sha256=builder.PARENT_SHA,
                  cuda_sha256=sha(cs.encode()), conformance=scope, native_finalizer=trace,
                  token_assignment=token_assignment(ps, cs), host_resources=host_mock(cs),
                  negative_controls=negatives,
                  limits=('Actual owner and consumer source executes natively with FP32 partial fixtures, '
                          'a 1/sqrt CPU shim and BF16 conversion shim; this is not CUDA rsqrt/FMA/BF16 proof. '
                          'Earlier y-square and warp-shuffle reduction source is unchanged, not emulated. '
                          'All256 route masks execute three CTA0 token iterations and a rotated-mask chunk remainder; '
                          'other token/column assignments are separately enumerated. Barriers are modeled phase boundaries, '
                          'not a GPU memory-order/progress simulation. No GPU correctness, resources or speed established.'))
    assert sha(candidate.read_bytes()) == digest
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, default=builder.CANDIDATE)
    ap.add_argument('--expected-sha')
    ap.add_argument('--output', type=Path, default=OUT)
    args = ap.parse_args()
    result = run(args.candidate, args.expected_sha)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], source_sha256=result['source_sha256'],
                          report_sha256=sha(args.output.read_bytes()), output=str(args.output)), indent=2))

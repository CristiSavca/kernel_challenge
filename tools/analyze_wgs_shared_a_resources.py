#!/usr/bin/env python3
"""Offline source-pinned SSAR admission/warning audit; no polling or promotion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import analyze_wgs_shared_a_result as performance
from analyze_decoder_selected_resources import ROW_RE

ROOT=Path(__file__).resolve().parents[1]
JOB='100bb9bbce8f4539'
SOURCE='93c2d0fbfe1d37792389402b335d34f8e00328e84ce25ad8ea732b9726d08a38'
EVALUATOR='c8a380059e3d1f0afaa9914d6d48b09cdc83efc7e110e7301bca3413720092c0'
EVALUATOR_INVENTORY_SHA='68c76b0c18f7154905a802d227b084e15c4b95e17d0d67fa40e9fef3d6ab7e11'
FIELDS='v big ntl dyn regs local static maxdyn occ set attr cap'.split()
INT=r'-?\d{1,20}'
MARKER=re.compile('SSAR '+r' '.join(k+'=('+INT+')' for k in FIELDS))
SYMBOL=re.compile(r'void okc::moe_fused<(true|false)>\(okc::Params\)')
require=performance.require


def parse_log(log):
    require(isinstance(log,str) and len(log.encode())<=512*1024,'log type/bound')
    records=[];malformed=[];warnings=[];build=[]
    for number,line in enumerate(log.splitlines(),1):
        if re.match(r'^\s*(ptxas|nvcc)(\s|:)',line) and re.search(
                r'C75\d\d|Potential Performance Loss|warning|wgmma.*serialized',line,re.I):
            require(len(warnings)<128,'warning count bound')
            ids=re.findall(r'moe_fusedILb([01])E',line)
            warnings.append({'line':number,'text':line[:1024],
                'BIG_if_symbol_present':int(ids[0]) if len(ids)==1 else None,
                'compiled_module_identity':'not encoded by this symbol'})
        if re.match(r'^\[\d+/\d+\]',line):build.append({'line':number,'text':line[:512]})
        if 'SSAR' not in line:continue
        # Runtime exception text is not a malformed resource record.
        if 'shared-A WGS resource admission failed; inspect SSAR records' in line:continue
        try:
            require(len(records)<128,'resource record count bound')
            m=MARKER.fullmatch(line);require(m is not None,'unknown/truncated marker')
            r=dict(zip(FIELDS,map(int,m.groups())))
            require(r['v']==1 and r['big'] in (0,1),'unknown schema/BIG')
            problems=[]
            if r['ntl']!=576 or r['dyn']!=194304:problems.append('wrong_fixed_launch_geometry')
            if any(r[k]!=0 for k in ('set','attr','cap')):problems.append('CUDA_API_failure')
            if r['attr']==0:
                if r['regs']<=0 or min(r[k] for k in ('local','static','maxdyn'))<0:problems.append('invalid_attributes')
                if r['maxdyn']<r['dyn']:problems.append('dynamic_exceeds_reported_limit')
            if r['cap']==0 and r['occ']<1:problems.append('insufficient_admission')
            records.append({'id':len(records),'line':number,**r,'issues':problems,
                'reported_admission':'FAIL' if problems else 'PASS',
                'attributes_usable':r['attr']==0 and 'invalid_attributes' not in problems,
                'occupancy_usable':r['cap']==0,
                'scope':'first launch of some compiled WGS module; not benchmark/state identity'})
        except ValueError as e:malformed.append({'line':number,'text':line[:1024],'reason':str(e)})
    groups=defaultdict(list)
    for r in records:groups[r['big']].append(r)
    conflicts=[]
    for big,rows in groups.items():
        if len({tuple(r[k] for k in FIELDS) for r in rows})>1:
            conflicts.append({'BIG':big,'record_ids':[r['id'] for r in rows],
                'reason':'different records cannot be assigned to distinct modules because marker lacks module ID'})
    pairs=[];orphans=[];previous=None
    for r in records:
        if r['big']==0:
            if previous is not None:orphans.append(previous['id'])
            previous=r
        elif previous is None:orphans.append(r['id'])
        else:pairs.append([previous['id'],r['id']]);previous=None
    if previous is not None:orphans.append(previous['id'])
    failures=[r['id'] for r in records if r['issues']]
    complete=bool(pairs) and not (malformed or conflicts or orphans)
    gate='FAIL' if failures else 'PASS_RETAINED_PAIRS' if complete else 'UNKNOWN_INCOMPLETE_OR_CONFLICTING'
    # Quarantine rows in contradictory identities rather than allowing a row's
    # raw PASS label to serve as a clean identity-level resource conclusion.
    conflict_ids={i for c in conflicts for i in c['record_ids']}
    for r in records:r['eligible_for_clean_pair']=not r['issues'] and r['id'] not in conflict_ids and r['id'] not in orphans
    return {'gate':gate,'records':records,'paired_record_ids':pairs,'orphan_record_ids':orphans,
        'missing_BIG':[b for b in (0,1) if b not in groups],'conflicts':conflicts,'malformed':malformed,
        'failed_record_ids':failures,'compiler_diagnostics':{
            'positive_warning_records':warnings,'visible_build_steps':build,
            'absence_of_other_warnings':'UNKNOWN: original compiler-output completeness is not certified',
            'causal_limit':'Warning presence is a compiler fact, not measured latency or parent serialization evidence.'},
        'policy':'Local bytes, register count and serialization warnings are reported; none alone is a rejection rule.',
        'coverage_limit':'Identical repeated BIG0/1 pairs are retained; module/cache/weight/benchmark identities are not encoded. Occupancy is admission capacity, not achieved residency.'}


def profile_activation(result):
    rows=[];fail=False;unknown=False
    for p in (r for r in result['records'] if r['event']=='profile'):
        M=p['case']['bs'];expected=M>=32;parsed=[];unparsed=[]
        report=p.get('report','');require(isinstance(report,str) and len(report)<=131072,'profile report bound')
        for line in report.splitlines():
            m=ROW_RE.fullmatch(line)
            if not m:
                if 'moe_fused' in line:unparsed.append(line[:1024])
                continue
            symbol,_,_,calls=m.groups();name=SYMBOL.fullmatch(symbol)
            parsed.append({'symbol':symbol,'calls':int(calls),'BIG':name.group(1)=='true' if name else None})
        if unparsed or len(parsed)!=1 or parsed[0]['calls']!=1 or parsed[0]['BIG'] is None:
            status='UNKNOWN_PROFILE_SHAPE';unknown=True
        elif parsed[0]['BIG']!=expected:status='FAIL_BIG_DISPATCH_MISMATCH';fail=True
        else:status='CONSISTENT_SINGLE_VISIBLE_KERNEL'
        rows.append({'index':p['index'],'M':M,'affected':M>16,'expected_BIG':expected,
            'status':status,'rows':parsed,'unparsed_relevant_lines':unparsed,
            'runtime_arguments':'UNKNOWN: profile omits module, flags, pointers, grid and pmax',
            'SS_A_activation':'UNKNOWN' if M>16 else 'UNCHANGED_SMALL_SOURCE_PATH'})
    return {'gate':'FAIL_DISPATCH' if fail else 'UNKNOWN_PROFILE_SHAPE' if unknown else 'VISIBLE_DISPATCH_CONSISTENT_WGS_IDENTITY_UNKNOWN',
        'captures':rows,'limit':'MAIN selection depends on runtime scale eligibility sfx. The fallback has the same moe_fused symbol; SSAR pairs do not bind each benchmark preparation. No new reference/nvfp4 input-chain certification is performed.'}


def analyze(job=JOB,source_sha=SOURCE,root=ROOT):
    require(job==JOB and source_sha==SOURCE,'this resource contract is pinned to one job/source')
    run=performance.read_run(job,source_sha,'check',root)
    base={'schema':1,'status':run['status'],'job':job,'source_sha256':source_sha,
        'provenance':run['provenance'],'automatic_promotion_allowed':False,'metrics':{}}
    if run['status']=='PENDING':return {**base,'resource_gate':'PENDING','activation_gate':'PENDING'}
    _,data=performance.bounded_file(f'experiments/runs/{job}.json',root)
    response=json.loads(data)['response'];result=response.get('result') or {}
    tail=result.get('log_tail') or '';require(isinstance(tail,str),'log tail type')
    log=tail;relation='receipt_tail_only';conflicts=[];files={}
    path=(response.get('output') or {}).get('log')
    if path:
        p,raw=performance.bounded_file(path,root);require(len(raw)<=512*1024,'saved log bound')
        external=raw.decode();files={'saved_log':str(p),'saved_log_sha256':performance.digest(raw)}
        if external==tail:relation='identical'
        elif external and (not tail or external.startswith(tail) or external.endswith(tail)):
            log=external;relation='saved_log_contains_tail'
        elif tail and (not external or tail.startswith(external) or tail.endswith(external)):
            relation='receipt_tail_contains_saved_log'
        else:relation='conflict';conflicts.append('saved_log_and_receipt_tail_disagree')
    resources=parse_log(log)
    base.update(resources=resources,resource_gate=resources['gate'],
        log_provenance={**files,'selected_log_sha256':performance.digest(log.encode()),'relationship':relation,
                        'artifact_conflicts':conflicts})
    if conflicts:base['resource_gate']='UNKNOWN_ARTIFACT_CONFLICT'
    if run['status']!='PASS':return {**base,'activation_gate':'UNAVAILABLE_OFFICIAL_GATE_FAILED',
        'official_failure':run.get('reason'),'decision':'REJECT_OFFICIAL_GATE',
        'limit':'Resource observations do not invent correctness, benchmark or profile outcomes.'}
    require(result['evaluator']['code_sha256']==EVALUATOR,'evaluator identity mismatch')
    evaluator_digest=performance.digest(json.dumps(result['evaluator'],sort_keys=True,separators=(',',':')).encode())
    require(evaluator_digest==EVALUATOR_INVENTORY_SHA,'full evaluator inventory/image mismatch')
    base['evaluator_inventory_sha256']=evaluator_digest
    require(result['gpu']=='H200','GPU class mismatch')
    active=profile_activation(result);base.update(activation=active,activation_gate=active['gate'])
    base['decision']='REJECT_RESOURCE_OR_VISIBLE_DISPATCH' if base['resource_gate']=='FAIL' or active['gate']=='FAIL_DISPATCH' else 'DIAGNOSTIC_ONLY_NO_PROMOTION'
    return base


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',default=JOB)
    p.add_argument('--source-sha256',default=SOURCE);p.add_argument('--output',type=Path);a=p.parse_args()
    try:r=analyze(a.job,a.source_sha256)
    except (OSError,ValueError,KeyError,TypeError) as e:r={'status':'EVIDENCE_REJECTED','error':str(e),'metrics':{},'automatic_promotion_allowed':False}
    text=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')

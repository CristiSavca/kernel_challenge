"""Follow each physical M8 accumulator quartet through four K16 calls and compare the PTX, scales and launch ABI."""
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import sys


HERE=Path(__file__).resolve().parent


def block(src,opening):
    pos=src.index(opening)
    start=src.index("{",pos); level=1
    for i in range(start+1,len(src)):
        level+=(src[i]=="{")-(src[i]=="}")
        if level==0:return src[pos:i+1]
    raise ValueError(opening)


def check_sources(before,after):
    op="static void fma(uint64_t &desc, uint32_t *b, float *d, bool pred = true)"
    asm=block(after,op)
    assert asm==block(before,op)
    for literal in ['setp.ne.b32 p, %9, 0','wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16',
                    '{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0',
                    '"+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])',
                    '"r"(b[0]),  "r"(b[1]),  "r"(b[2]),  "r"(b[3])','"l"(desc)']:
        assert literal in asm,literal
    old=block(before,"struct MuseOverlapMma :")
    assert old==block(after,"struct MuseOverlapMma :")
    assert 'if constexpr (bands == 2) Base::transform_b(buffer, iteration)' in old
    assert 'MmaOpClass::fma(desc, regs_b[buffer][j][0], regs_c[0][j][0], true)' in old
    new=block(after,"struct MuseFullStageMma :")
    assert 'transform_b(' not in new and 'held[4][2]' in new
    assert 'held[iteration][j][i] = regs_b[iteration % 2][j][0][i]' in new
    assert 'MmaOpClass::fma(desc, held[iteration][j], regs_c[0][j][0], true)' in new
    desc=lambda x:re.search(r'const uint32_t address = [^;]+;',x).group().replace('stage_id *','stage *')
    assert desc(new)==desc(old) and "iteration * 2 * sizeof(int4)" in desc(new)
    loop="for (uint32_t warp_iter_id = 0;"
    p=block(after,loop)
    old_loop=block(before,loop)
    prefetch="if (warp_iter_id + 1 < Ctx::kWarpIters || slice_iters > 1) s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);"
    norm=lambda x:" ".join(x.replace(prefetch,"").split())
    assert norm(p)==norm(old_loop) and p.count(prefetch)==old_loop.count(prefetch)==1
    assert p.index("mma.run(")<p.index("consumer.arrive(stage_id)") and p.index(prefetch)<p.index("mma.transform_b(")
    assert 'if (warp_iter_id == Ctx::kWarpIters - 2)' in p and 'consumer.wait_stage((stage_id + 1) % kNumStages)' in p
    assert '#define MUSE_FULL_K_STAGE' in after and '#define MUSE_PAIR_LAST_K_SLICES' in after
    assert "if (iteration == 3)" in new and new.index('wgmma_commit()')<new.index('wgmma_wait<0>()')<new.index('warpgroup_fence_operand(held[k][j][i])')
    assert re.search(r'for \(unsigned k = 0; k < 4; \+\+k\)',new)
    return sha256(asm.encode()).hexdigest()


def inspect(sass,per_group):
    pattern=r'/\*([0-9a-f]+)\*/\s+(?:(@!?U?P\d+)\s+)?(HGMMA\.[\w.]+|WARPGROUP\.DEPBAR\.LE)\s+([^;\n]+)\s*;'
    events=re.findall(pattern,sass)
    assert sass.count('HGMMA.')==32
    assert sass.count('WARPGROUP.DEPBAR.')==32//per_group
    logical=[]; held=set(); pair=None; issues=0; drains=[]; groups=[]
    for pc,pred,op,args in events:
        if op.startswith('WARPGROUP'):
            assert not pred and args.strip()=='gsb0, 0x0' and len(held)==4*per_group
            drains.append(issues);held.clear();continue
        assert not pred and op=='HGMMA.64x8x16.F32.BF16'
        form=re.fullmatch(r'R(\d+), R(\d+), gdesc\[UR(\d+)\], R(\d+)(, gsb0)?',args.strip())
        assert form,(hex(int(pc,16)),args)
        dst,src,desc,acc=map(int,form.groups()[:4])
        assert dst==acc and src%4==0 and dst%4==0 and not (set(range(src,src+4))&set(range(dst,dst+4)))
        assert held.isdisjoint(range(src,src+4));held.update(range(src,src+4))
        if issues%8==0:
            groups.append([]); pair=[]
        if len(pair)<2:
            assert dst not in pair;pair.append(dst)
        # A single mistaken same-shape accumulation into the other quartet changes
        # this chain while preserving the number of instructions and full drains.
        assert dst==pair[issues%2]
        logical.append((issues//8,issues%2,(issues%8)//2,dst,acc))
        groups[-1].append(src)
        issues+=1
    assert not held and drains==list(range(per_group,33,per_group)) and len(logical)==32
    return logical,drains,groups


def rejects(fn):
    try:fn()
    except AssertionError:return True
    raise AssertionError("corrupt control passed")


def main(scratch):
    result={}
    authority=json.loads((HERE/"machine.json").read_bytes())["m8"]
    for tag in ("w13_m8_n256","w2_m8_n128"):
        records=[]
        for variant,dir_name,size in (("parent","built",4),("selected","physical",8)):
            stem=scratch/dir_name/tag
            src=stem.with_suffix(".cu").read_bytes();sass=stem.with_suffix(".sass").read_bytes()
            for s,b in [('source',src),('sass',sass),('cubin',stem.with_suffix(".cubin").read_bytes())]:
                assert sha256(b).hexdigest()==authority[tag][variant][s]
            stream,drains,held=inspect(sass.decode(),size)
            records.append((src.decode(),sass.decode(),stream,drains,held))
        old,new=records;cuda_hash=check_sources(old[0],new[0])
        assert old[2]==new[2]
        symbol=lambda s:re.search(r'Function : (\S+)',s).group(1)
        assert symbol(old[1])==symbol(new[1])
        # Change both source and destination of a single accumulator on the
        # third K slice: still eight correct-shape instructions before drain.
        hit=list(re.finditer(r'HGMMA\.64x8x16.F32.BF16 R(\d+), R\d+, gdesc\[UR\d+\], R\1',new[1]))[4]
        oldR=hit[1];other=str(new[2][5][3]);assert oldR!=other
        swapped=new[1][:hit.start()]+hit[0].replace("R"+oldR+",", "R"+other+",",1).rsplit("R",1)[0]+"R"+other+new[1][hit.end():]
        assert rejects(lambda:inspect(swapped,8))
        altered=new[0].replace('%8, p, 1, 1, 0','%8, p, 1, -1, 0')
        assert altered!=new[0] and rejects(lambda:check_sources(old[0],altered))
        alternate=new[0].replace('= regs_b[iteration % 2][j][0][i]', '= regs_b[(iteration + 1) % 2][j][0][i]')
        assert alternate!=new[0] and rejects(lambda:check_sources(old[0],alternate))
        result[tag]=dict(ptx_fma_sha256=cuda_hash,launch_symbol_sha256=sha256(symbol(new[1]).encode()).hexdigest(),
                  unchanged_ordered_accumulator_updates=len(new[2]),stages=len(new[4]),destinations=sorted({q[3] for q in new[2]}),
                  parent_drains_after=old[3],candidate_drains_after=new[3],candidate_source_quartet_bases_by_stage=new[4],
                  flipped_input_scale_rejected=True,reused_wrong_pingpong_buffer_rejected=True,redirected_accumulator_rejected=True,
                  cubin_sha256=authority[tag]["selected"]["cubin"])
    native=(scratch/'humming/include/humming/mma/wgmma.cuh').read_bytes()
    transform=block(native.decode(),'void transform_b(')
    assert transform.index('dequant<ElementB')<transform.index('arith.may_apply_bs_and_zp_on_b(')
    s2r=(scratch/'humming/include/humming/memory/s2r_pipeline.cuh').read_bytes()
    mem=block(s2r.decode(),'void load_stage_iter(')
    for expr in ('(stage_id + iter_id / Ctx::kWarpIters) % kNumStages','iter_id = iter_id % Ctx::kWarpIters','buffer_id = iter_id % 2','loader_b.load(smem.stages[stage_id].b','loader_bs.load(smem.stages[stage_id].bs'):
        assert expr in mem,expr
    # Scalar cancellation only discriminates reordering; it does not reproduce
    # the ISA's unspecified accumulation implementation inside each K16 dot.
    f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
    def dot(values):
        acc=0.
        for v in values:acc=f32(acc+v)
        return acc
    fixed=dot([2**24,1,-2**24,1]);permuted=dot([2**24,-2**24,1,1]);assert (fixed,permuted)==(1.,2.)
    print(json.dumps(dict(images=result,ordinary_transform_sha256=sha256(native).hexdigest(),
        s2r_header_sha256=sha256(s2r).hexdigest(),scalar_order_control=dict(native_order=fixed,swapped_middle_updates=permuted)),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))

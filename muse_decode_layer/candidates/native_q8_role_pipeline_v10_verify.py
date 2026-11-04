"""CPU-only ownership/layout checks for the untested Q8 role-pipeline prototype."""
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path


PARENT_SHA = "80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c"
ROOT = Path(__file__).resolve().parents[2]


def sw128(row, head, column):
    return row * 128 + (((head * 4 + column // 16) ^ (row & 7)) * 16) + column % 16


def layout_proof():
    # PTX WGMMA e4m3 register-A coordinates; independently compare to incumbent
    # score-accumulator -> pa packing and existing PV key permutation.
    owner = {}
    for head in range(2):
        for tid in range(128):
            lane, warp = tid % 32, tid // 32
            for half in range(2):
                for reg in range(4):
                    row = 16 * warp + lane // 4 + 8 * (reg % 2)
                    for byte in range(4):
                        col = 32 * half + 4 * (lane % 4) + 16 * (reg // 2) + byte
                        address = sw128(row, head, col)
                        assert address not in owner
                        # Incumbent e4m3x4f order: [s0,s1,s4,s5] etc.
                        index = 16 * half + 8 * (reg // 2) + 2 * (reg % 2) + (0, 1, 4, 5)[byte]
                        source_key = 8 * (index // 4) + 2 * (lane % 4) + index % 2
                        logical_key = 16 * (col // 16) + 8 * ((col % 4) // 2) + 2 * ((col % 16) // 4) + col % 2
                        assert source_key == logical_key
                        owner[address] = head, row, logical_key
        old_k_addresses = {sw128(row, head, column) for row in range(64) for column in range(64)}
        p_addresses = {address for address, value in owner.items() if value[0] == head}
        assert p_addresses == old_k_addresses
    assert set(owner) == set(range(8192))
    # Shared descriptor decoding: a 128B-swizzled atom has eight128B rows;
    # the stride descriptor advances1024B per eight rows, and the K32 half
    # advances32B inside its swizzled atom. Basehead64 participates in XOR.
    for head in range(2):
        for row in range(64):
            for col in range(64):
                linear = (row // 8) * 1024 + (row % 8) * 128 + head * 64 + col
                decoded = linear ^ ((row & 7) << 4)
                assert decoded == sw128(row, head, col)
    return {"p_bytes": len(owner), "head_disjoint_bytes_each": 4096,
            "rs_to_ss_elements_checked": 8192}


def schedule_proof(seed, segment_lengths):
    """Randomized legal interleavings of loader/QK/PV state machines.

    QK has separate head/half publication; Kempty requires BOTH QK and PV
    arrivals. V's two-slot credit is returned only after PV completion.
    Metadata and P remain live until the same K credit returns.
    """
    rng = random.Random(seed)
    events = 0
    for count in segment_lengths:
        class Barrier:
            def __init__(self, expected):
                self.expected, self.pending, self.phase, self.generation = expected, 0, 0, 0
            def arrive(self, amount):
                self.pending += amount
                assert self.pending <= self.expected
                if self.pending == self.expected:
                    self.pending = 0
                    self.phase ^= 1
                    self.generation += 1
            def ready(self, old_phase, generation):
                return self.phase != old_phase and self.generation == generation

        full = [Barrier(128) for _ in range(4)]
        empty = [Barrier(256) for _ in range(4)]
        ready = [[[Barrier(128) for _ in range(2)] for _ in range(2)] for _ in range(4)]
        vempty = [[Barrier(128) for _ in range(2)] for _ in range(2)]
        k = [None] * 4
        v = [[None] * 2 for _ in range(2)]
        published = set()
        qdone, pdone = set(), set()
        loaded = qtile = ptile = 0
        qhead = qhalf = phead = phalf = 0
        qprepared = False
        issued = set()
        for _ in range(count * 80 + 20):
            if ptile == count and qtile == count and loaded == count:
                break
            actions = []
            if loaded < count and k[loaded % 4] is None:
                actions.append("load")
            if qtile < loaded and k[qtile % 4] == qtile:
                if not qprepared and v[qhead][qtile % 2] is None:
                    actions.append("prepare")
                elif qprepared:
                    actions.append("publish")
            if (ptile, phead, phalf) in published:
                actions.append("pv")
            if phalf == 2:
                # Actual implementation can accumulate both heads before wait0;
                # this model conservatively tracks their reads until tile done.
                actions = [action for action in actions if action != "pv"]
                actions.append("advance_pv_head")
            if phead == 2:
                actions = [action for action in actions if action not in ("pv", "advance_pv_head")]
                actions.append("pv_complete")
            assert actions, (seed, count, loaded, qtile, ptile, k, v)
            action = rng.choice(actions)
            events += 1
            if action == "load":
                assert loaded < 4 or loaded - 4 in qdone & pdone
                if loaded >= 4:
                    assert empty[loaded % 4].ready(((loaded // 4) - 1) & 1, loaded // 4)
                k[loaded % 4] = loaded
                full[loaded % 4].arrive(128)
                loaded += 1
            elif action == "prepare":
                assert qtile < 2 or qtile - 2 in pdone
                assert full[qtile % 4].ready((qtile // 4) & 1, qtile // 4 + 1)
                if qtile >= 2:
                    assert vempty[qhead][qtile % 2].ready(((qtile - 2) // 2) & 1, qtile // 2)
                v[qhead][qtile % 2] = qtile
                qprepared = True
            elif action == "publish":
                assert k[qtile % 4] == qtile
                published.add((qtile, qhead, qhalf))
                ready[qtile % 4][qhead][qhalf].arrive(128)
                qhalf += 1
                if qhalf == 2:
                    qhead += 1
                    qhalf = 0
                    qprepared = False
                if qhead == 2:
                    qdone.add(qtile)
                    empty[qtile % 4].arrive(128)
                    if qtile in pdone:
                        k[qtile % 4] = None
                    qtile += 1
                    qhead = 0
            elif action == "pv":
                assert k[ptile % 4] == ptile and v[phead][ptile % 2] == ptile
                assert ready[ptile % 4][phead][phalf].ready((ptile // 4) & 1, ptile // 4 + 1)
                assert phalf == 0 or (ptile, phead, 0) in issued
                issued.add((ptile, phead, phalf))
                published.remove((ptile, phead, phalf))
                phalf += 1
            elif action == "advance_pv_head":
                phead += 1
                phalf = 0
            else:
                assert all((ptile, head, half) in issued for head in range(2) for half in range(2))
                for head in range(2):
                    assert v[head][ptile % 2] == ptile
                    v[head][ptile % 2] = None
                    vempty[head][ptile % 2].arrive(128)
                pdone.add(ptile)
                empty[ptile % 4].arrive(128)
                if ptile in qdone:
                    k[ptile % 4] = None
                ptile += 1
                phead = phalf = 0
        else:
            raise AssertionError("state machine did not drain")
        assert not published and all(slot is None for slot in k)
        assert all(slot is None for head in v for slot in head)
        # These drained lifetimes authorize resetting the union and phases.
        assert qdone == pdone == set(range(count))
        assert all(bar.pending == 0 for bar in full + empty)
    return events


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


E4M3 = [((byte & 7) * 2.0 ** -9 if byte < 8 else
         (1 + (byte & 7) / 8) * 2.0 ** ((byte >> 3) - 7)) for byte in range(127)]


def quant(x):
    i = bisect_left(E4M3, x)
    if i >= 127:
        i = 126
    elif i and (x - E4M3[i - 1] < E4M3[i] - x or
                (x - E4M3[i - 1] == E4M3[i] - x and (i - 1) % 2 == 0)):
        i -= 1
    return i, E4M3[i]


def rescale_proof(seed):
    """Prove deferred per-half rescale is identical to incumbent operation order.

    CPU exp/FP8 conversion is an oracle shared by the two schedules, not a
    claim to emulate MUFU approximation or tensor accumulation bit-for-bit.
    The comparison tests handoff order, including second-half moves and rows
    masked entirely; the GPU numerical suite remains mandatory.
    """
    rng = random.Random(seed)
    rows, tiles = 16, 6  # one warp's two row groups; all128threads covered by byte proof
    scores = [[[[f32(rng.uniform(-2, 2) + (40 if t == 3 else 0) + (12 * t if k >= 32 else 0)) for k in range(64)]
                 for r in range(rows)] for h in range(2)] for t in range(tiles)]
    for t in range(tiles):
        for h in range(2):
            scores[t][h][0][:32] = [-math.inf] * 32
            scores[t][h][1][32:] = [-math.inf] * 32
            scores[t][h][3] = [-math.inf] * 64
            if t == 0:
                scores[t][h][2] = [-math.inf] * 64
    values = [[[f32(rng.choice((-1., -.25, .5, 2.))) for _ in range(3)]
               for _ in range(64)] for _ in range(2)]
    offsets = [[0.] * rows for _ in range(2)]
    direct = [[[0.] * 4 for _ in range(rows)] for _ in range(2)]
    deferred = [[[0.] * 4 for _ in range(rows)] for _ in range(2)]
    deferred_offset = [[0.] * rows for _ in range(2)]
    queue = []
    moved_count = 0
    no_move_count = 0
    moved_halves = set()
    for t in range(tiles):
        for h in range(2):
            if t == 0:
                offsets[h] = [max(row) if max(row) != -math.inf else 0. for row in scores[t][h]]
            for half in range(2):
                part = [row[32 * half:32 * (half + 1)] for row in scores[t][h]]
                def pack():
                    return [[quant(f32(2. ** f32(score - offsets[h][r]))) for score in row]
                            for r, row in enumerate(part)]
                packed = pack()
                moved = t != 0 and any(code > 0x78 for row in packed for code, _ in row)
                alpha = [1.] * rows
                if moved:
                    moved_count += 1
                    moved_halves.add(half)
                    for r in range(rows):
                        new = max(max(part[r]), offsets[h][r])
                        alpha[r] = f32(2. ** f32(offsets[h][r] - new))
                        offsets[h][r] = new
                        direct[h][r] = [f32(x * alpha[r]) for x in direct[h][r]]
                    packed = pack()
                else:
                    no_move_count += 1
                probs = [[v for _, v in row] for row in packed]
                for r in range(rows):
                    for k, prob in enumerate(probs[r]):
                        vv = values[h][32 * half + k] + [1.]
                        for d in range(4):
                            direct[h][r][d] = f32(direct[h][r][d] + prob * vv[d])
                queue.append((h, half, moved, alpha, probs, list(offsets[h])))
    for h, half, moved, alpha, probs, offset in queue:
        if moved:
            for r in range(rows):
                deferred[h][r] = [f32(x * alpha[r]) for x in deferred[h][r]]
        for r in range(rows):
            for k, prob in enumerate(probs[r]):
                vv = values[h][32 * half + k] + [1.]
                for d in range(4):
                    deferred[h][r][d] = f32(deferred[h][r][d] + prob * vv[d])
        if half == 1:
            deferred_offset[h] = offset
    assert direct == deferred
    assert offsets == deferred_offset
    assert moved_halves == {0, 1}
    assert moved_count and no_move_count
    return moved_count, no_move_count


def host_syntax_and_layout(parent, child, fragment):
    compiler = shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("clang++ is required for the CPU layout/syntax proof")
    def smem(source):
        begin = source.index("template <int L, bool F>\nstruct Smem {")
        end = source.index("\n};", begin) + 3
        return source[begin:end]
    flags = {name: 1 for name in ("OKC_SM90", "PV8ON", "OKC_PV8FLAT", "OKC_QSS", "OKC_Q8RAW",
                                 "OKC_PCHK", "OKC_LAZYMAX", "OKC_PVSPLIT", "OKC_XP_ARRALL",
                                 "OKC_SETMAXNREG", "OKC_MERGE3")}
    flags.update({name: 0 for name in ("OKC_CONT", "OKC_DYN", "OKC_PTR", "OKC_PROF", "OKC_PK2",
                                      "OKC_RAWTHR", "OKC_EXH8", "OKC_F2", "OKC_XP_NOEXP",
                                      "OKC_EXPPOLY", "OKC_PCHKTEST", "OKC_PVW", "OKC_QDIRECT")})
    stub = r'''
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
using std::min; using std::max;
#define __device__
#define __forceinline__ inline
#define __shfl_xor_sync(mask,x,v) (x)
#define __any_sync(mask,x) (x)
using __nv_bfloat16=unsigned short;
struct {int x;} threadIdx;
struct uint2 {uint32_t x,y;}; struct uint4 {uint32_t x,y,z,w;};
uint2 make_uint2(uint32_t,uint32_t);
constexpr int STAGE_BYTES=8192, MAXB=512, MAXSEG=64, ROWB=128, TILE=64;
using ix_t=long long;
struct Params {int coop,resident,window,rope,Hkv,nc; float *ws_acc,*ws_l;};
template<int L,bool F>struct Cfg {
  static constexpr bool WS=true,S2=false,F2=false;
  static constexpr int STAGES=F && L==8 ? 4 : 3;
  static constexpr int NP=128;
};
uint32_t smem_u32(const void*),prmt(uint32_t,uint32_t,uint32_t),pack_e4m3x4(uint32_t,uint32_t),e4m3x4f(float,float,float,float);
uint64_t desc_sw64(uint32_t),desc_sw128(uint32_t);
void fence_proxy_async(), wgsync(int), wgmma_fence(),wgmma_commit(),mbar_fence_init(),pdl_launch_dependents(),__syncthreads();
void mbar_wait_full(uint64_t*,int),mbar_arrive(uint64_t*),mbar_init_n(uint64_t*,int),reg_fence(float&);
void wgmma_m64n64k32_e4m3_ss(float*,uint64_t,uint64_t,int),ldsm_x4_t(uint32_t,uint32_t&,uint32_t&,uint32_t&,uint32_t&);
float quad_max(float),quad_sum(float),ex2(float);
void red_v4(float*,float,float,float,float),red_f32(float*,float);
template<int N> void wgmma_wait();
template<bool W,int N>void csync();
'''
    stub += "".join(f"#define {k} {v}\n" for k, v in flags.items())
    stub += smem(parent).replace("struct Smem {", "struct ParentSmem {") + "\n" + smem(child) + "\n"
    stub += fragment + r'''
using S8=Smem<8,true>;
static_assert(sizeof(S8)==113664);
static_assert(offsetof(S8,q8_role_scratch_v10)==offsetof(S8,kvst));
static_assert(offsetof(S8,gst)==offsetof(S8,kvst)+4096);
static_assert(offsetof(S8,q8_role_scratch_v10)%alignof(Q8RoleMetaV10)==0);
static_assert(sizeof(Smem<8,true>)==sizeof(ParentSmem<8,true>));
static_assert(sizeof(Smem<8,false>)==sizeof(ParentSmem<8,false>));
static_assert(sizeof(Smem<1,true>)==sizeof(ParentSmem<1,true>));
static_assert(sizeof(Smem<1,false>)==sizeof(ParentSmem<1,false>));
'''
    result = subprocess.run([compiler, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
                            input=stub, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def main():
    parent = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA
    result = layout_proof()
    result["schedule_seeds"] = 128
    result["schedule_events"] = sum(schedule_proof(seed, [1, 2, 3, 4, 5, 9, 37]) for seed in range(128))
    moves = [rescale_proof(seed) for seed in range(16)]
    result["rescale_moves"] = sum(x[0] for x in moves)
    result["rescale_fast_halves"] = sum(x[1] for x in moves)
    result["register_pool"] = (96 + 112 + 32) * 128
    result["metadata_bytes"] = 4096 + 2048 + 256 + 128 + 32
    assert result["metadata_bytes"] <= 8192
    sys.path.insert(0, str(ROOT / "tools"))
    from build_q8_role_pipeline_v10 import BASE, build
    rebuilt = build()
    assert BASE.with_suffix(".py").read_text() == rebuilt
    host_syntax_and_layout(parent.read_text(), rebuilt, BASE.with_suffix(".cuh").read_text())
    result["host_syntax_and_exact_smem_layout"] = "pass (CUDA/PTX not compiled)"
    result["candidate_sha256"] = hashlib.sha256(rebuilt.encode()).hexdigest()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

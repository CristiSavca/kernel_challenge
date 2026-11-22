"""Model page-one SWA certificate eligibility with pinned allocation and archived lengths."""
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "serving_v0523/sglang"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
TRACE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
TILE, WINDOW, WIDTH, PARTITIONS = 64, 2047, 8, 66
LAYERS, KV_ROW_BYTES = 46, 2 * 2 * 64
LAYOUTS = (0, 4096, 256, 64, 32)
WAVES = (0, 4, 16, 32, 48)
DIGESTS = {}


def read_pin(relative):
    raw = subprocess.check_output(["git", "-C", BASE, "show", PIN + ":" + relative])
    DIGESTS[relative] = sha256(raw).hexdigest()
    return raw.decode()


def method(source, name, namespace=None):
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    space = dict(namespace or {})
    exec("from __future__ import annotations\n" + ast.unparse(node), space)
    return space[name]


class Rows(np.ndarray):
    """NumPy adapter for the two PyTorch integer-tensor methods used by these bodies."""
    def numel(self):
        return self.size

    def to(self, dtype):
        return self.astype(dtype, copy=False)


class SlotList:
    def __init__(self, rows):
        self.free_pages = rows
        self.release_pages = rows[:0]
        self.need_sort = False


class TwoPools:
    page_size = 1


class IntegerOps:
    int1, uint32 = np.bool_, np.uint32
    where, minimum, min = staticmethod(np.where), staticmethod(np.minimum), staticmethod(np.min)
    full = staticmethod(np.full)

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int32).view(Rows)


def mask32(origins, start, end, limit):
    assert 0 <= start < end <= len(origins)
    assert end - start <= 32
    word = sum(1 << i for i, slot in enumerate(origins[start:end]) if 0 < slot <= limit)
    return np.uint32(word)


def new_file_source(patch, path):
    head = f"diff --git a/{path} b/{path}"
    section = patch.split(head, 1)[1].split("\ndiff --git ", 1)[0]
    assert "new file mode" in section
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++ "))


def bind_native():
    global RESERVE, WINDOW_RUN
    prefix = "python/sglang/srt/mem_cache/"
    token = read_pin(prefix + "allocator/token.py")
    hybrid = read_pin(prefix + "allocator/swa.py")
    SlotList.alloc = method(token, "alloc")
    SlotList.available_size = method(token, "available_size")
    TwoPools.alloc = method(hybrid, "alloc")
    TwoPools.set_full_to_swa_mapping = method(hybrid, "set_full_to_swa_mapping", {"torch": SimpleNamespace(int64=np.int64)})
    RESERVE = method(read_pin(prefix + "allocation_sizing.py"), "page_aligned_decode_alloc_lens")
    source = read_pin("python/sglang/srt/speculative/dflash_info_v2.py")
    assert "reserve = 2 * block_size" in source
    patch = (HERE / "patch.diff").read_text()
    source = new_file_source(patch, "python/sglang/kernels/ops/attention/paged_verify_attention.py")
    WINDOW_RUN = method(source, "_window_tile_run", {"tl": IntegerOps, "_tile_origin_mask": mask32})
    assert all(fragment in patch for fragment in (
        "swa_pages.to(tl.int64) - cols % 64", "page_table is metadata.swa_page_table",
        "getattr(metadata, \"swa_verify_tile_origins\", None)", "window_size[0] == self.sliding_window_size",
        "if self.use_sliding_window_kv_pool", "window_left + max_q + tile_n - 1",
    ))


def make_pool(size, fragment, rng):
    # Logical full slots are deliberately shuffled for every scenario, independently
    # of SWA. Each free list contains every positive physical slot exactly once.
    full = rng.permutation(np.arange(1, size + 1)).view(Rows)
    swa = np.arange(1, size + 1, dtype=np.int64)
    if fragment:
        blocks = swa.reshape(-1, fragment)
        # Random phase is independent of request logical tile boundaries.
        swa = np.roll(blocks[rng.permutation(len(blocks))].ravel(), int(rng.integers(fragment)))
    pool = TwoPools()
    pool.full_attn_allocator = SlotList(full)
    pool.swa_attn_allocator = SlotList(swa.view(Rows))
    pool.full_to_swa_index_mapping = np.zeros(size + 1, dtype=np.int64).view(Rows)
    return pool


def measure(rows, lengths, offsets, pool):
    first = [max(0, length - WIDTH - WINDOW) for length in lengths]
    counts = [(length + TILE - 1) // TILE - lo // TILE for length, lo in zip(lengths, first)]
    partition_size = (sum(counts) + PARTITIONS - 1) // PARTITIONS
    base = 0
    cnt = Counter(sampled_batches=1, sampled_request_blocks=len(rows))
    for full_slots, length, lo, count, offset in zip(rows, lengths, first, counts, offsets):
        head = lo // TILE
        first_whole = (lo + TILE - 1) // TILE
        end_whole = length // TILE
        cert = np.full(count, -1, dtype=np.int64)
        physical = pool.full_to_swa_index_mapping[full_slots[first_whole*TILE-offset:end_whole*TILE-offset]]
        complete = physical.reshape(-1, TILE)
        valid = (np.diff(complete, axis=1) == 1).all(axis=1) & (complete[:, 0] > 0)
        cert[first_whole-head:end_whole-head] = np.where(valid, complete[:, 0], -1)
        assert count <= (WINDOW + WIDTH + 126) // 64
        cnt["tiles"] += count
        cnt["live_rows"] += length - lo
        cnt["maybe_affine_tiles"] += end_whole - first_whole
        max_slot = len(pool.full_to_swa_index_mapping) - 1
        for part in range(base // partition_size, (base + count - 1) // partition_size + 1):
            cursor = head + max(0, part * partition_size - base)
            end = head + min(count, (part + 1) * partition_size - base)
            cnt["intersections"] += 1
            cnt["single_tile_intersections"] += int(end - cursor == 1)
            while cursor < end:
                stop, direct = WINDOW_RUN(cert, cursor, end, lo, length, head, max_slot + 1 - TILE, TILE)
                stop, direct = int(stop), bool(direct)
                assert cursor < stop <= end
                choices = cert[cursor-head:stop-head] > 0
                assert np.all(choices == direct)
                cnt["runs"] += 1
                if direct:
                    cnt["direct_tiles"] += stop - cursor
                    cnt["direct_runs"] += 1
                    cnt["single_direct_runs"] += int(stop - cursor == 1)
                    if stop - cursor >= 4:
                        cnt["tiles_in_direct_runs_of_at_least_four"] += stop - cursor
                cursor = stop
        base += count
    return cnt


def simulate(jobs, fragment, seed):
    by_phase = defaultdict(Counter)
    jobs = sorted(jobs, key=lambda r: r["t0"])
    limits = [r["prompt"] + r["completion"] - 1 for r in jobs]
    progress = [r["prompt"] for r in jobs]
    # Allocate independent recent prompt suffixes. Missing oldest history cannot
    # enter SWA. Keeping a 64-row margin preserves arbitrary alignment.
    offsets = [max(0, p - WINDOW - 2 * TILE) for p in progress]
    capacity = ((sum(n - o + 2*WIDTH for n, o in zip(limits, offsets)) + 4095) // 4096 + 1) * 4096
    pool = make_pool(capacity, fragment, np.random.default_rng(seed))
    table = [np.empty(limit - o + 2*WIDTH, dtype=np.int64) for limit, o in zip(limits, offsets)]
    for i, (p, o) in enumerate(zip(progress, offsets)):
        table[i][:p-o] = pool.alloc(p-o)
    reqs = [SimpleNamespace(kv=SimpleNamespace(kv_committed_len=p, kv_allocated_len=p)) for p in progress]
    observed = progress.copy()
    for wave in range(max(WAVES) + 1):
        active = [i for i in range(len(jobs)) if progress[i] < limits[i]]
        if not active:
            break
        live = [reqs[i] for i in active]
        for i in active:
            reqs[i].kv.kv_committed_len = observed[i]
        before, after, total = RESERVE(live, reserve=2*WIDTH, page_size=1)
        allocated = pool.alloc(total)
        assert allocated is not None
        cursor = 0
        for i, lo, hi in zip(active, before, after):
            offset = offsets[i]
            table[i][lo-offset:hi-offset] = allocated[cursor:cursor+hi-lo]
            cursor += hi - lo
            reqs[i].kv.kv_allocated_len = hi
            assert progress[i] + WIDTH <= hi
        if wave in WAVES:
            phase = "first" if wave == 0 else "later"
            by_phase[phase].update(measure([table[i] for i in active], [progress[i]+WIDTH for i in active],
                                           [offsets[i] for i in active], pool))
        for i in active:
            observed[i] = progress[i]
            commit = 1 + (3*wave + 5*jobs[i]["user"] + jobs[i]["turn"]) % 7
            progress[i] = min(limits[i], progress[i] + commit)
    return by_phase


def summarize(c):
    return dict(c, direct_tile_fraction=c["direct_tiles"]/c["tiles"],
                direct_payload_fraction=TILE*c["direct_tiles"]/c["live_rows"],
                affine_ceiling_fraction=c["maybe_affine_tiles"]/c["tiles"],
                direct_tiles_in_runs_of_four_share=c["tiles_in_direct_runs_of_at_least_four"]/max(1,c["direct_tiles"]),
                runs_per_intersection=c["runs"]/c["intersections"])


def boundary_cases():
    checks = Counter()
    for length in [*range(1, 80), *range(2015, 2140), *range(131008, 131088)]:
        lowest = max(0, length - WIDTH - WINDOW)
        base = lowest // TILE
        end = (length + TILE - 1) // TILE
        for salt in (0, 1):
            # Certify the widest geometry only. Exercise the descriptor bound and
            # an internal rejected tile as well as independently narrower windows.
            certs = np.array([1 + TILE*(t-base) if t*TILE >= lowest and
                              (t+1)*TILE <= length and (t+salt)%7 else -1
                              for t in range(base, end)])
            limit = 1920 if salt else 3000
            for q in range(min(length, WIDTH) + 1):
                start_key = max(0, length - q - WINDOW)
                start = start_key // TILE if q else end
                expected = [t for t in range(start, end) if t*TILE >= start_key and
                            (t+1)*TILE <= length and 0 < certs[t-base] <= limit]
                observed = []
                for block in range(start, end, 7):
                    cursor, bound = block, min(block+7, end)
                    while cursor < bound:
                        stop, direct = WINDOW_RUN(certs, cursor, bound, start_key, length, base, limit, TILE)
                        assert cursor < stop <= bound
                        if direct:
                            observed.extend(range(cursor, int(stop)))
                        cursor = int(stop)
                assert observed == expected
                checks["sampled_rows"] += 1
                checks["all_tiles"] += end-start
                checks["direct_tiles"] += len(observed)
                checks["empty_rows"] += int(q == 0)
    return checks


def main():
    bind_native()
    client = next(TRACE.glob("*v41__load.json"))
    stats = json.loads(client.read_bytes())["requests"]
    cohorts = defaultdict(list)
    for record in stats:
        cohorts[record["load"], record["rep"], record["turn"]].append(record)
    results = {}
    for fragment in LAYOUTS:
        cur = defaultdict(lambda: defaultdict(Counter))
        for i, ((load, _, _), jobs) in enumerate(sorted(cohorts.items())):
            for phase, cnt in simulate(jobs, fragment, 1027+i).items():
                cur[load][phase].update(cnt)
        results["fresh" if fragment == 0 else f"shuffled_{fragment}"] = {
            n: {p:summarize(c) for p,c in phases.items()} for n, phases in cur.items()}
    floors = {}
    profile = next((ROOT/"experiments/candidates/muse_attention_decode/profiles").glob("87a9bdd*.c21*.json"))
    for row in json.loads(profile.read_bytes())["benchmarks"]:
        case = row["case"]
        if case["layer"] == "sliding" and case["load"].startswith("verify"):
            b = case["batch"]
            payload = (WINDOW+WIDTH)*KV_ROW_BYTES*LAYERS*b
            floors[b] = dict(logical_kv_bytes_per_gpu=payload, payload_only_ms_at_4p8TBs=payload/4.8e9,
                            separate_program_original_net_us=row["baseline_ns"]/1000,
                            separate_program_candidate_net_us=row["median_ns"]/1000,
                            separate_program_all_46_layers_delta_ms=LAYERS*(row["baseline_ns"]-row["median_ns"])/1e6)
    print(json.dumps(dict(patch_sha256=sha256((HERE/"patch.diff").read_bytes()).hexdigest(), base=PIN,
        upstream_source_sha256=DIGESTS, trace_sha256=sha256(client.read_bytes()).hexdigest(),
        trace_request_count=len(stats), cohorts=len(cohorts), waves=WAVES, tile_rows=TILE,
        variable_row_oracle=boundary_cases(),
        kv_row_bytes_per_gpu=KV_ROW_BYTES, persistent_swa_origin_bytes_at_bs32=32*34*4,
        modeled_eligibility=results, unrelated_h200_context=floors,
        notes="First/later snapshots model independent overlapping prompt suffixes with same-turn cohorts. Native page-one full/SWA allocation + mapping via NumPy, native DFLASH reserve lengths, synthetic 1..7 commits, scalar native V218 run body. Prefix reuse, radix eviction/reuse, elapsed time and judge physical addresses are unobserved."), indent=2))


if __name__ == "__main__":
    main()


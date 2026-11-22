"""Run the production tree KV compaction and its guard with bounds-checked CPU pool storage."""
from __future__ import annotations

import ast
import itertools
import json
import random
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import torch


class Tensor:
    """Typed storage and the small CUDA tensor interface exercised by the production dispatch."""
    def __init__(self, data, dtype):
        self.array = data
        self.dtype = dtype
        self.shape = data.shape
        self.ndim = data.ndim
        self.device = 'cuda:0'
        self.is_cuda = True

    def view(self, *shape):
        return Tensor(self.array.reshape(*shape), self.dtype)

    def numel(self):
        return self.array.size

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def stride(self, axis):
        return self.array.strides[axis] // self.array.itemsize

    def data_ptr(self):
        return int(self.array.ctypes.data)

    def element_size(self):
        return self.array.itemsize


class MHATokenToKVPool:
    pass


class SWAKVPool:
    pass


def triton_grid(checkout, rng):
    sim = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'v70_dflash_selector_tree/cpu_check.py'))
    path = checkout / 'python/sglang/srt/speculative/dflash_tree_kv.py'
    pid, (kernel,) = sim['program'](path, ('_compact_tree_rows',))
    tl = kernel.__globals__['tl']
    tl.pointer_type = lambda dtype: 'word_pointer'
    tl.uint32 = np.uint32
    tl.debug_barrier = lambda: None
    raw_load = tl.load
    ptr = sim['Pointer']
    memories = {}

    class Address(int):
        def to(self, dtype):
            assert dtype == 'word_pointer'
            return ptr(memories[int(self)].view(np.uint32).reshape(-1))

    addresses = set()

    def load(p, mask=True, other=0):
        val = raw_load(p, mask, other)
        return Address(int(val)) if id(p.data) in addresses else val

    tl.load = load

    class Launch:
        def __getitem__(self, grid):
            def run(pointers, strides, mapping, slots, order, lengths,
                    width, rows, words, translate, num_warps):
                assert num_warps == 1
                addresses.add(id(pointers.array))
                coords = list(itertools.product(*(range(n) for n in grid)))
                rng.shuffle(coords)
                for xyz in coords:
                    pid[:] = xyz
                    kernel(ptr(pointers.array), ptr(strides.array),
                           None if mapping is None else ptr(mapping.array),
                           ptr(slots.array), ptr(order.array), ptr(lengths.array),
                           width, rows, words, translate)
                addresses.discard(id(pointers.array))
            return run

    includes = {'_supports_mha', 'tree_kv_compaction_supported', '_compact_mha', 'compact_tree_kv'}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in includes]
    for name, cls in [('memory_pool', MHATokenToKVPool), ('swa_memory_pool', SWAKVPool)]:
        mod = types.ModuleType('sglang.srt.mem_cache.' + name)
        setattr(mod, cls.__name__, cls)
        sys.modules[mod.__name__] = mod
    ns = {
        '_compact_tree_rows': Launch(), 'torch': torch,
        'triton': types.SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                                        next_power_of_2=lambda a: 1 << (a-1).bit_length()),
    }
    exec(compile(tree, str(path), 'exec'), ns)
    return ns['tree_kv_compaction_supported'], ns['compact_tree_kv'], memories


def run(checkout):
    rng = np.random.default_rng(101031)
    supported, compact, memories = triton_grid(checkout, random.Random(91))
    value_count = cases = slots_checked = restore_pages = 0

    def mha(n_slots, layer_shapes):
        pool = MHATokenToKVPool()
        pool.kv_cache_layout = 'nhd'
        pool.is_quantized_kv_cache = False
        pool.layer_num = len(layer_shapes)
        pool.store_dtype = torch.uint8 # legacy FP8 cache is stored through a uint8 view
        pool.head_num, pool.head_dim, pool.v_head_dim = 2, 64, 128
        pool.k_buffer = []
        pool.v_buffer = []
        for k, v in layer_shapes:
            assert (k, v) == (pool.head_dim, pool.v_head_dim)
            for bank, dim in ((pool.k_buffer, k), (pool.v_buffer, v)):
                # Includes bytes for NaNs, signed zero and all bitwise FP8 patterns.
                t = Tensor(rng.integers(256, size=(n_slots, pool.head_num, dim), dtype=np.uint8), torch.uint8)
                bank.append(t)
                memories[t.data_ptr()] = t.array
        banks = pool.k_buffer + pool.v_buffer
        pool.data_ptrs = Tensor(np.array([t.data_ptr() for t in banks], np.uint64), torch.uint64)
        pool.data_strides = Tensor(np.array([t.stride(0) for t in banks], np.int64), torch.int64)
        return pool

    def bounds_case(batch, page, width, kind, mixed):
        nonlocal value_count, cases, slots_checked, restore_pages
        pages_per_req = 4
        n_pages = batch * pages_per_req + 2
        all_pages = rng.permutation(np.arange(1, n_pages))
        req_pages = all_pages[:batch*pages_per_req].reshape(batch, pages_per_req)
        positions = np.arange(page*pages_per_req)
        table = req_pages[:, positions//page]*page + positions%page
        indices = np.arange(batch)
        prefix = rng.choice([0, page-1, page-2, page, page+1, 2*page-1], size=batch)
        slots = table[indices[:,None], prefix[:,None] + np.arange(width)]
        assert np.unique(slots).size == slots.size
        protected_table = table.copy()
        lengths = rng.integers(1, width+1, size=batch, dtype=np.int32)
        order = np.empty((batch,width), np.int64)
        for r in indices:
            if mixed:
                path = [0]+sorted(rng.choice(range(1,width),size=int(lengths[r])-1,replace=False))
            else:
                # Fully adversarial permutation, including cycles, beyond valid tree paths.
                lengths[r] = width
                path = list(rng.permutation(width))
            order[r] = path+[i for i in range(width) if i not in path]
        perm = rng.permutation(np.arange(1, n_pages))
        mapping = np.zeros(n_pages*page, np.int64)
        mapping[page:] = (perm[:,None]*page + np.arange(page)).flatten()
        full = mha(n_pages*page, [(64,128)] * (0 if kind=='swa_only' else 2))
        swa = mha(n_pages*page, [(64,128)] * (0 if kind=='full_only' else 1))
        pool = SWAKVPool()
        pool.full_kv_pool, pool.swa_kv_pool = full,swa
        pool.full_to_swa_index_mapping = Tensor(mapping,torch.int64)
        if kind=='ordinary':
            pool = full
            pools = ((full,slots),)
        else:
            pools = ((full,slots),(swa,mapping[slots]))
        assert supported(pool)
        # Prefix acceptance is a snapshot operation, even if source and target cells overlap.
        expected = {}
        for sub, physical in pools:
            for t in sub.k_buffer+sub.v_buffer:
                data = t.array.copy()
                for r,length in enumerate(lengths):
                    dst = physical[r,:length]
                    src = physical[r,order[r,:length]]
                    data[dst] = t.array[src].copy()
                expected[t.data_ptr()] = data
        compact(pool, Tensor(slots.copy(),torch.int64), Tensor(order,torch.int64), Tensor(lengths,torch.int32))
        assert np.array_equal(table, protected_table)
        for sub,physical in pools:
            for t in sub.k_buffer+sub.v_buffer:
                ref = expected[t.data_ptr()]
                assert np.array_equal(t.array,ref), (batch,page,width,kind,mixed)
                # Whole-page backup/reload must retain the canonical accepted prefix,
                # including steps crossing a page boundary and permuted SWA backing.
                for r,length in enumerate(lengths):
                    rows = physical[r,:length]
                    for page_id in np.unique(rows//page):
                        cpu_copy = t.array[page_id*page:(page_id+1)*page].copy()
                        assert np.array_equal(cpu_copy,ref[page_id*page:(page_id+1)*page])
                        restore_pages += 1
                value_count += ref.size
        slots_checked += slots.size
        cases += 1
        memories.clear()

    for args in itertools.product((1, 3), (16,160), (2,7,8),
                                  ('ordinary','hybrid','full_only','swa_only'), (True,False)):
        bounds_case(*args)
    # Run a full sequence through the real acceptance kernel and compactor.
    # Requests share a complete immutable cached page and own private tails.
    # Re-admit a new request into the middle request's physical tail partway.
    sim = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'v70_dflash_selector_tree/cpu_check.py'))
    pid, (accept,) = sim['program'](checkout / 'python/sglang/kernels/ops/speculative/dflash_tree.py',
                                     ('_accept_tree',))
    ptr = sim['Pointer']
    batch, width, page = 3, 8, 160
    page_ids = np.array([[1,7,5,10,12], [1,3,11,9,8], [1,6,13,4,2]])
    x = np.arange(page_ids.shape[1] * page)
    table = page_ids[:,x//page] * page + x%page
    immutable_table = table.copy()
    nslots = (page_ids.max() + 1) * page
    pool = SWAKVPool()
    pool.full_kv_pool = mha(nslots,[(64,128)])
    pool.swa_kv_pool = mha(nslots,[(64,128)])
    mapping = np.arange(nslots)
    mapping[page:] = (rng.permutation(np.arange(1,nslots//page))[:,None] * page
                      + np.arange(page)).reshape(-1)
    pool.full_to_swa_index_mapping = Tensor(mapping,torch.int64)
    target_banks = [(b,t) for b in pool.full_kv_pool.k_buffer + pool.full_kv_pool.v_buffer
                    for t in (table,)] + [(b,mapping[table]) for b in
                    pool.swa_kv_pool.k_buffer + pool.swa_kv_pool.v_buffer]
    prefixes = np.array([page,2*page-4,2*page-1],np.int64)
    visited = set()
    backups = 0
    for step in range(32):
        if step==16:
            prefixes[1] = page
        sequence_order = rng.permutation(batch)
        prefix = prefixes[sequence_order].copy()
        slots = table[sequence_order[:,None],prefix[:,None]+np.arange(width)]
        tokens = np.arange(1, 1+batch*width).reshape(batch,width) + step*100
        parents = np.full_like(tokens,-1)
        predict = np.full_like(tokens,99999)
        expected_paths = []
        for r in range(batch):
            path = [0] + sorted(rng.choice(range(1,width),rng.integers(0,width), replace=False))
            for j in range(1,width):
                parents[r,j] = rng.integers(0,j)
            for a,b in zip(path,path[1:]):
                parents[r,b] = a
                predict[r,a] = tokens[r,b]
            expected_paths.append(path)
        lens = np.zeros(batch,np.int32)
        bonus = np.zeros(batch,np.int64)
        out = np.zeros((batch,width),np.int64)
        new = np.zeros(batch,np.int64)
        order = np.full_like(out,-7)
        for r in range(batch):
            pid[0]=r
            accept(*[ptr(a) for a in (tokens,parents,predict,prefix,lens,bonus,out,new,order)],width,width)
            assert order[r,:lens[r]].tolist() == expected_paths[r]
            assert bonus[r] == predict[r,expected_paths[r][-1]]
            assert out[r,lens[r]-1] == bonus[r] and int(new[r]) == prefix[r]+len(expected_paths[r])
            visited.add(int(lens[r]))
        snaps = []
        for b,path_table in target_banks:
            physical = path_table[sequence_order[:,None],prefix[:,None]+np.arange(width)]
            verify_values = rng.integers(256,size=b.array[physical].shape,dtype=np.uint8)
            b.array[physical] = verify_values
            old = b.array.copy()
            for r,path in enumerate(expected_paths):
                old[physical[r,:len(path)]] = verify_values[r,path]
            snaps.append(old)
        compact(pool,Tensor(slots,torch.int64),Tensor(order,torch.int64),Tensor(lens,torch.int32))
        for (b,path_table),snap in zip(target_banks,snaps):
            assert np.array_equal(b.array,snap)
            # Simulate HiCache backing and reloading each completed physical
            # page at its original canonical offsets, including a shared prefix.
            for logical in range(batch):
                fullpages = (int(new[logical])//page)*page
                page_slots = path_table[sequence_order[logical],:fullpages].reshape(-1,page)
                for live_page in page_slots:
                    assert np.all(np.diff(live_page)==1)
                    copy = b.array[live_page].copy()
                    assert np.array_equal(copy,snap[live_page])
                    backups += 1
        assert np.array_equal(table,immutable_table)
        prefixes[sequence_order] = new
    assert visited == set(range(1,9))
    guards = 0
    invalid = mha(20, [(64,128)])
    for field, bad in [('is_quantized_kv_cache',True),('kv_cache_layout','hnd'),
                       ('layer_num',2),('head_dim',65),('head_num',3)]:
        old = getattr(invalid,field)
        setattr(invalid,field,bad)
        assert not supported(invalid)
        setattr(invalid,field,old)
        guards += 1
    assert not supported(types.SimpleNamespace())
    guards += 1
    assert not supported(mha(20,[]))
    guards += 1
    return dict(production_kernel_cases=cases, compared_bytes=value_count,
                reservation_slots=slots_checked, compared_whole_page_copies=restore_pages,
                rejected_layouts=guards, sequential_accept_steps=32,
                shared_prefix_whole_page_copies=backups, committed_lengths=sorted(visited),torch=torch.__version__, cpu=True)


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2))

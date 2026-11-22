"""Replay the frozen Triton metadata and attention programs with CPU-backed tensors."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import ml_dtypes
import numpy as np
import triton
import triton.language as tl
from triton.runtime.interpreter import InterpreterBuilder
from triton.tools.tensor_descriptor import TensorDescriptor

from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def replay(source):
    rng = np.random.default_rng(218)
    counts = collections.Counter()
    loaded = []
    witness = {}
    native_load = InterpreterBuilder.create_descriptor_load
    native_dot = InterpreterBuilder.create_dot

    def observed_load(self, descriptor, indices, *flags):
        loaded.append(tuple(int(np.asarray(i.data).item()) for i in indices))
        return native_load(self, descriptor, indices, *flags)

    def bf16_dot(self, a, b, acc, *flags):
        # The host interpreter represents BF16 as uint16; both versions use the same correction.
        if a.dtype == b.dtype == tl.bfloat16:
            left = a.data.view(ml_dtypes.bfloat16).astype("f4")
            right = b.data.view(ml_dtypes.bfloat16).astype("f4")
            return type(acc)(left @ right + acc.data, acc.dtype.scalar)
        return native_dot(self, a, b, acc, *flags)

    with tempfile.TemporaryDirectory() as tmp:
        new, prior, meta = [load_jit_module(source / name, Path(tmp)) for name in
                            ("paged_verify_attention.py", "prior.py", "metadata.py")]
        InterpreterBuilder.create_descriptor_load = observed_load
        InterpreterBuilder.create_dot = bf16_dot
        try:
            for window, b, parts, width, rows in [
                (2047, 1, 1, 8, 4317), (2047, 4, 7, 8, 2551),
                (127, 3, 4, 8, 397), (63, 4, 3, 8, 138), (1, 2, 3, 8, 129),
                (0, 2, 3, 8, 129), (2047, 1, 2, 8, 192), (255, 3, 1, 4, 800),
            ]:
                test_shape(rng, counts, loaded, new, prior, meta, window, b, parts, width, rows, witness)
        finally:
            InterpreterBuilder.create_descriptor_load = native_load
            InterpreterBuilder.create_dot = native_dot
    assert counts["TMA_loads"] and counts["fragmented_tiles"] and counts["expired_prefixes"] and counts["zero_query_rows"]
    return dict(triton=triton.__version__, physical_h200=False, counts=dict(counts), witness=witness,
                sha256={n: hashlib.sha256((source/n).read_bytes()).hexdigest()
                        for n in ("paged_verify_attention.py", "prior.py", "metadata.py")})


def test_shape(rng, counts, loaded, new, prior, meta, window, b, parts, width, rows, witness):
    dim, heads, groups = 64, 2, 4
    tile_m = width * groups
    span = min(rows, window + width + 63)
    columns = triton.cdiv(span, 64)
    row_pitch = columns + 3
    full_pitch = rows + 6
    orig = np.full((b, row_pitch), -777, "i4")
    full_orig = np.full((b, triton.cdiv(rows, 128)), -777, "i4")
    source = np.arange(1, b*rows+1, dtype="i4").reshape(b, rows)
    page = np.full((b, full_pitch), -777, "i4")
    swa = page.copy()
    capacity = b * (rows+256) + 65
    key_pitch, value_pitch = heads * dim + 16, heads * dim + 32
    weights = [rng.integers(-2, 3, (capacity, p)).astype(ml_dtypes.float8_e4m3fn)
               for p in (key_pitch, value_pitch)]
    for bank in weights:
        bank[0] = np.nan
    k, v = [buf(a.view("u1"), tl.float8e4nv) for a in weights]
    kd, vd = [TensorDescriptor(bank, [capacity, heads * dim], [pitch, 1], [64, dim])
              for bank, pitch in ((k, key_pitch), (v, value_pitch))]
    ks = np.linspace(.7, 1.1, b * heads, dtype="f4").reshape(b, heads)
    vs = np.linspace(.95, .66, b * heads, dtype="f4").reshape(b, heads)

    # The very same allocations are refreshed on each replay, with a new request permutation and live prefix.
    for step in range(3):
        indices = np.roll(np.arange(b, dtype="i4"), step)
        length = np.array([max(width, rows-((r*71+step*53) % (rows-width+1))) for r in range(b)], dtype="i4")
        qlen = np.full(b, width, "i4")
        if step == 1 and b > 1:
            qlen[-1] = 0
            qlen[0] = 1
        if step == 2:
            qlen[0] = width-1
        cu = np.r_[0, qlen.cumsum()].astype("i4")
        delta = width if step != 1 else 0
        slot_map = np.zeros(b*rows+1, "i4")
        actual = {}
        for r in range(b):
            slot = indices[r]
            addresses = 1 + (b-1-slot)*(rows+256) + (23+step*9) + np.arange(rows, dtype="i4")
            first = max(0, int(length[r])-width-window)
            addresses[:first] = 0
            for t in range(triton.cdiv(first, 64), length[r]//64):
                if step and (t+r) % (3 if step==1 else 2) == 0:
                    i = t*64
                    addresses[i+7], addresses[i+37] = addresses[i+37], addresses[i+7]
                    counts["fragmented_tiles"] += 1
            slot_map[source[slot]] = addresses
            actual[r] = addresses
            counts["expired_prefixes"] += int(first > 0)
        prev = orig.copy()
        lens = np.zeros(b, "i4")
        prefix = np.zeros(b+1, "i4")
        meta._fused_decode_metadata_kernel[(b, triton.cdiv(rows, 128))](
            buf(length-delta, tl.int32), 1, buf(source, tl.int32), rows, 1, buf(indices, tl.int32), 1,
            buf(lens, tl.int32), 1, buf(prefix, tl.int32), 1, buf(page, tl.int32), full_pitch, 1,
            buf(swa, tl.int32), full_pitch, 1, buf(slot_map, tl.int32), 1, None, 0, None, 0, b, rows,
            PAGE_SIZE=1, SEQ_LEN_DELTA=delta, BLOCK_COLS=128, SKIP_PAGE_TABLE=False, ZERO_TAIL=False,
            WINDOW_LEFT=-1, origins=buf(full_orig, tl.int32), ORIGIN_STRIDE=full_orig.shape[1],
            swa_origins=buf(orig, tl.int32), SWA_ORIGIN_STRIDE=row_pitch, SWA_WINDOW=window, VERIFY_WIDTH=width)
        np.testing.assert_array_equal(lens, length)
        np.testing.assert_array_equal(prefix, np.r_[0, length.cumsum()])
        expected = collections.Counter()
        for r, n in enumerate(length):
            n = int(n)
            first = max(0, n-width-window)
            base = first//64
            np.testing.assert_array_equal(page[r,:n], source[indices[r],:n])
            np.testing.assert_array_equal(swa[r,:n], actual[r][:n])
            for t in range(base, triton.cdiv(n, 64)):
                a = actual[r][t*64:min(t*64+64,n)]
                good = len(a)==64 and t*64>=first and a[0]>0 and np.array_equal(a, a[0]+np.arange(64))
                prev[r,t-base] = a[0] if good else -1
                if good and qlen[r] and t*64 >= n-qlen[r]-window:
                    for h in range(heads):
                        expected[(int(a[0]), h*dim)] += 2
                counts["refreshed_certificates"] += 1
            counts["translated_page_ids"] += n
        np.testing.assert_array_equal(orig, prev)
        counts["zero_query_rows"] += int((qlen==0).sum())
        counts["verifier_rounds"] += 1
        q = rng.integers(-2, 3, (b*width, heads*groups, dim)).astype(ml_dtypes.float8_e4m3fn)
        outs = []
        loaded.clear()
        for mod in (prior, new):
            acc = np.full((parts+b-1, heads, tile_m, dim), -71, "f4")
            lse = np.full(acc.shape[:-1], -91, "f4")
            out = np.full(q.shape, 91, "u2")
            direct = {} if mod is prior else dict(Origins=buf(orig, tl.int32), KDesc=kd, VDesc=vd,
                                                   ORIGIN_STRIDE=row_pitch, CAPACITY=capacity)
            mod._partition_attention[(parts, heads)](
                buf(q.view("u1"),tl.float8e4nv), k, v, buf(swa,tl.int32), buf(lens,tl.int32), buf(cu,tl.int32),
                buf(ks,tl.float32), buf(acc,tl.float32), buf(lse,tl.float32), full_pitch, 1,
                key_pitch, key_pitch, dim, value_pitch, value_pitch, dim, heads, 1, b, heads, groups, dim,
                math.log2(math.e)/8, window, width, tile_m, 64, parts, triton.next_power_of_2(b), **direct)
            mod._merge_partitions[(b, heads, tile_m)](
                buf(acc,tl.float32), buf(lse,tl.float32), buf(lens,tl.int32), buf(cu,tl.int32),
                buf(vs,tl.float32), buf(out,tl.bfloat16), heads, 1, b, heads, groups, dim, window,
                tile_m, 64, parts, triton.next_power_of_2(b), 16)
            outs.append((acc, lse, out))
        assert collections.Counter(loaded) == expected, (window,b,step,collections.Counter(loaded)-expected,expected-collections.Counter(loaded))
        for a, b_arr in zip(*outs):
            np.testing.assert_array_equal(a.view("u4") if a.dtype==np.float32 else a,
                                          b_arr.view("u4") if a.dtype==np.float32 else b_arr)
        if window == 2047 and b == 4 and step == 1:
            words = out[:int(cu[-1])].reshape(-1)[:2048]
            assert len(words) == 2048 and len(np.unique(words)) > 100 and loaded
            witness.update(dtype="BF16 little-endian words", source="First 2,048 live, merged elements, B4 W8 sliding window 2047, replay 1",
                           sha256=hashlib.sha256(words.astype("<u2").tobytes()).hexdigest(), words=words.tolist())
        counts["TMA_loads"] += len(loaded)
        counts["fp32_partial_and_lse_words"] += acc.size + lse.size
        counts["bf16_output_words"] += out.size
        for r, qn in enumerate(qlen):
            if qn == 0:
                continue
            i = int(cu[r] + qn-1)
            pos = int(length[r]-1)
            ids = actual[r][max(pos-window,0):pos+1]
            head = r%heads
            sl = slice(head*dim,(head+1)*dim)
            logits = (weights[0][ids,sl].astype("f4") @ q[i,head*groups].astype("f4")) * (ks[r,head]/8)
            probs = np.exp(logits-logits.max())
            ref = (probs @ weights[1][ids,sl].astype("f4")) / probs.sum() * vs[r,head]
            np.testing.assert_allclose(out[i,head*groups].view(ml_dtypes.bfloat16).astype("f4"),ref,rtol=.04,atol=.025)
            counts["dense_reference_words"] += dim


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]).resolve()),indent=2))

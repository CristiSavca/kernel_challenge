"""Bounded source/FP32 semantics check; does not compile or emulate Triton GPU lowering."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
CANDIDATE_SHA = "8ab1cf33ef5b4a4eea8721625ad446358fcbd603dfe1277eee17b7e76a53d232"
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_route_payload_v20.py"
OLD = "        idx = tl.argmax(sel, 0)\n        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)"
NEW = "        _, idx, val = tl.reduce((sel, e, sc), 0, _route_argmax_payload)"


def f32(x):
    try:
        return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def bits(x):
    return struct.unpack("I", struct.pack("f", x))[0]


def from_bits(x):
    return struct.unpack("f", struct.pack("I", x))[0]


class Scalar(float):
    def __add__(self, other):
        return Scalar(f32(float(self) + f32(other)))


class Vec:
    def __init__(self, values):
        self.v = list(values)

    def op(self, other, fn):
        rhs = other.v if isinstance(other, Vec) else [other] * len(self.v)
        return Vec(fn(a, b) for a, b in zip(self.v, rhs, strict=True))

    def __add__(self, other):
        return self.op(other, lambda a, b: f32(a + b))

    def __truediv__(self, other):
        return self.op(other, lambda a, b: f32(a / b))

    def __eq__(self, other):
        return self.op(other, lambda a, b: a == b)


def parent_combine(v1, i1, v2, i2):
    # Exact predicate in Triton v3.8.0 language/standard.py, lines 137--145.
    tie = v1 == v2 and i1 < i2
    gt = v1 > v2 or tie
    return (v1, i1) if gt else (v2, i2)


class TL:
    int32 = "int32"
    float32 = "float32"
    static_range = range

    def __init__(self, tree="adjacent", flush_add=False):
        self.tree, self.flush_add = tree, flush_add

    def arange(self, lo, hi):
        return Vec(range(lo, hi))

    def zeros(self, shape, dtype):
        return Vec([0 if dtype == self.int32 else 0.0] * shape[0])

    def where(self, pred, a, b):
        if not isinstance(pred, Vec):
            return a if pred else b
        aa = a.v if isinstance(a, Vec) else [a] * len(pred.v)
        bb = b.v if isinstance(b, Vec) else [b] * len(pred.v)
        return Vec(x if p else y for p, x, y in zip(pred.v, aa, bb, strict=True))

    def tree_reduce(self, values, combine):
        while len(values) > 1:
            if self.tree == "adjacent":
                pairs = zip(values[::2], values[1::2], strict=True)
            else:
                n = len(values) // 2
                pairs = zip(values[:n], values[n:], strict=True)
            values = [combine(*a, *b) for a, b in pairs]
        return values[0]

    def argmax(self, x, axis):
        assert axis == 0
        return self.tree_reduce(list(zip(x.v, range(len(x.v)))), parent_combine)[1]

    def sum(self, x, axis):
        assert axis == 0

        def add(a, b):
            if self.flush_add:
                a = 0.0 if 0 < abs(a) < 2**-126 else a
                b = 0.0 if 0 < abs(b) < 2**-126 else b
            c = f32(a + b)
            if self.flush_add and 0 < abs(c) < 2**-126:
                c = 0.0
            return (c,)

        return Scalar(self.tree_reduce([(v,) for v in x.v], add)[0])

    def reduce(self, args, axis, combine):
        assert axis == 0 and len(args) == 3
        return self.tree_reduce(list(zip(*(a.v for a in args), strict=True)), combine)


def loop_code(source, name):
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == name)
    start = next(i for i, x in enumerate(node.body)
                 if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == "sel")
    stop = next(i for i in range(start, len(node.body)) if isinstance(node.body[i], ast.Expr)
                and isinstance(node.body[i].value, ast.Call)
                and isinstance(node.body[i].value.func, ast.Attribute) and node.body[i].value.func.attr == "store")
    return compile(ast.Module(body=node.body[start:stop], type_ignores=[]), name, "exec")


def execute(code, tl, scores, bias, helper=None):
    env = {"tl": tl, "sc": Vec(scores), "biasv": Vec(bias), "e": Vec(range(256)),
           "_route_argmax_payload": helper}
    exec(code, env)
    return env["ids"].v, env["wv"].v


def equal_result(a, b):
    assert a[0] == b[0], (a[0], b[0])
    assert all((math.isnan(x) and math.isnan(y)) or bits(x) == bits(y)
               for x, y in zip(a[1], b[1], strict=True)), (a[1], b[1])


def run():
    source = CANDIDATE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == CANDIDATE_SHA
    parent = (ROOT / f"experiments/candidates/muse_decode_layer/{PARENT_SHA}.py").read_text()
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    tree = ast.parse(source)
    helper_node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_route_argmax_payload")
    lines = source.splitlines(keepends=True)
    begin = helper_node.decorator_list[0].lineno - 1
    end = helper_node.end_lineno
    helper_text = "".join(lines[begin:end]) + "\n\n"
    restored = source.replace(helper_text, "", 1).replace(NEW, OLD)
    assert source.count(NEW) == 2 and restored == parent
    assert ast.unparse(helper_node.body[0]) == "tie = value1 == value2 and index1 < index2"
    assert ast.unparse(helper_node.body[1]) == "gt = value1 > value2 or tie"
    helper_node.decorator_list = []
    helper_code = compile(ast.Module(body=[helper_node], type_ignores=[]), "actual_helper", "exec")
    old_codes = [loop_code(parent, n) for n in ("_route", "_route_split")]
    new_codes = [loop_code(source, n) for n in ("_route", "_route_split")]
    rng = random.Random(20261104)
    cases = [([0.5] * 256, [0.0] * 256), ([0.0] * 256, [0.0] * 256)]
    special = [0.0, from_bits(1), from_bits(0x007fffff), from_bits(0x00800000),
               f32(0.5), from_bits(0x3f7fffff), 1.0]
    for shift in range(16):
        cases.append(([special[(i + shift) % len(special)] for i in range(256)],
                      [f32((i % 13 - 6) / 8) for i in range(256)]))
    cases.append(([0.5] * 256, [from_bits(0x7f7fffff) if i % 2 else -from_bits(0x7f7fffff) for i in range(256)]))
    for _ in range(64):
        cases.append(([from_bits(rng.randrange(0x3f800001)) for _ in range(256)],
                      [f32(rng.uniform(-1, 1)) for _ in range(256)]))
    finite_count = 0
    for order in ("adjacent", "halves"):
        tl = TL(order)
        ns = {"tl": tl}
        exec(helper_code, ns)
        for scores, bias in cases:
            for old, new in zip(old_codes, new_codes, strict=True):
                a = execute(old, tl, scores, bias)
                b = execute(new, tl, scores, bias, ns["_route_argmax_payload"])
                equal_result(a, b)
                expected_ids = sorted(range(256), key=lambda i: (-f32(scores[i] + bias[i]), i))[:8]
                assert b[0] == expected_ids
                finite_count += 1
        ids, weights = execute(new_codes[0], tl, *cases[0], ns["_route_argmax_payload"])
        assert ids == list(range(8)) and weights == [0.125] * 8
    # NaNs are diagnostic inputs outside the finite sigmoid-output proof. Compare
    # same trees only, treating NaNs by classification rather than payload bits.
    nan_count = 0
    for order in ("adjacent", "halves"):
        tl = TL(order); ns = {"tl": tl}; exec(helper_code, ns)
        for payload_nan in (False, True):
            for nan_at in (0, 1, 127, 255):
                scores, bias = [0.5] * 256, [0.0] * 256
                (scores if payload_nan else bias)[nan_at] = math.nan
                for old, new in zip(old_codes, new_codes, strict=True):
                    equal_result(execute(old, tl, scores, bias),
                                 execute(new, tl, scores, bias, ns["_route_argmax_payload"]))
                    nan_count += 1
    # Real limitations, not successful-domain assertions.
    left = parent_combine(*parent_combine(2.0, 0, math.nan, 1), 1.0, 2)[1]
    right = parent_combine(2.0, 0, *parent_combine(math.nan, 1, 1.0, 2))[1]
    assert left != right
    tl = TL(); ns = {"tl": tl}; exec(helper_code, ns)
    z_old = execute(old_codes[0], tl, [-0.0] * 256, [0.0] * 256)
    z_new = execute(new_codes[0], tl, [-0.0] * 256, [0.0] * 256, ns["_route_argmax_payload"])
    assert bits(z_old[1][0]) != bits(z_new[1][0])
    tl = TL(flush_add=True); ns = {"tl": tl}; exec(helper_code, ns)
    d_old = execute(old_codes[0], tl, [from_bits(1)] * 256, [0.0] * 256)
    d_new = execute(new_codes[0], tl, [from_bits(1)] * 256, [0.0] * 256, ns["_route_argmax_payload"])
    assert bits(d_old[1][0]) != bits(d_new[1][0])
    # Directed negative controls must detect a wrong tie direction or payload.
    bad_tie = lambda v1, i1, p1, v2, i2, p2: (v1, i1, p1) if (v1 > v2 or (v1 == v2 and i1 > i2)) else (v2, i2, p2)
    tl = TL()
    assert execute(new_codes[0], tl, *cases[0], bad_tie)[0] != list(range(8))
    ns = {"tl": tl}; exec(helper_code, ns)
    actual = ns["_route_argmax_payload"]
    wrong_payload = lambda *a: (*actual(*a)[:2], actual(*a)[0])
    scores, bias = cases[2]
    good = execute(new_codes[0], tl, scores, bias, actual)
    bad = execute(new_codes[0], tl, scores, bias, wrong_payload)
    assert any(bits(a) != bits(b) for a, b in zip(good[1], bad[1], strict=True))
    return {"source_sha256": CANDIDATE_SHA, "parent_sha256": PARENT_SHA,
            "restoration": "all parent bytes restored", "finite_fp32_comparisons": finite_count,
            "same_tree_nan_classification_comparisons": nan_count, "negative_controls": 2,
            "counterexamples": ["NaN comparator depends on reduction tree", "negative-zero payload differs",
                                "flushed-add model differs for subnormal sigmoid payload"],
            "limits": "CPU IEEE-RNE shim; no sigmoid/transcendental emulation, GPU compilation, reduction-layout, FTZ, NaN-payload or timing proof"}


if __name__ == "__main__":
    report = run()
    path = ROOT / "experiments/iterations/20261104-route-payload-cpu.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

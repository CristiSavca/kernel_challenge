"""Execute V73's projection and checkpoint loader with CPU array doubles."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.data = self

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def size(self, axis=None):
        return self.shape if axis is None else self.shape[axis]

    def dim(self):
        return self.ndim

    def numel(self):
        return self.value.size

    def narrow(self, axis, start, size):
        indices = [slice(None)] * self.ndim
        indices[axis] = slice(start, start + size)
        return Tensor(self.value[tuple(indices)])

    def copy_(self, other):
        assert self.shape == other.shape
        np.copyto(self.value, other.value)

    def new_empty(self, shape):
        return Tensor(np.empty(shape, dtype=self.value.dtype))

    def reshape(self, *shape):
        return Tensor(self.value.reshape(*shape))


def bf16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + ((bits >> 16) & 1) + 0x7FFF
    return (rounded & 0xFFFF0000).view(np.float32)


def linear(inputs, weight):
    # All output columns use the identical complete K reduction.
    product = np.einsum("mk,nk->mn", inputs.value, weight.value, optimize=False)
    return Tensor(bf16(product))


class Linear:
    def __init__(self, in_features, out_features, bias):
        assert not bias
        self.in_features, self.out_features = in_features, out_features
        self.weight = Tensor(np.zeros((out_features, in_features), np.float32))

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        return linear(inputs, self.weight)


class Group:
    def __init__(self, rank, size):
        self.rank, self.world_size = rank, size
        self.expected = None
        self.calls = 0

    def all_gather(self, partial, dim):
        assert dim == -1
        width = self.expected.shape[-1] // self.world_size
        np.testing.assert_array_equal(
            partial.value, self.expected[:, self.rank * width:(self.rank + 1) * width]
        )
        self.calls += 1
        return Tensor(self.expected.copy())


def extract(path, names, env, class_name=None):
    program = ast.parse(path.read_text(), str(path))
    scope = program.body
    if class_name is not None:
        scope = next(node.body for node in scope if getattr(node, "name", None) == class_name)
    program.body = [node for node in scope if getattr(node, "name", None) in names]
    assert len(program.body) == len(names)
    program.body.insert(0, ast.ImportFrom("__future__", [ast.alias("annotations")], 0))
    ast.fix_missing_locations(program)
    exec(compile(program, str(path), "exec"), env)
    return [env[name] for name in names]


def load_environment(root):
    current = SimpleNamespace(group=None)
    linear_module = ModuleType("sglang.srt.layers.linear")
    linear_module.resolve_linear_parallel_group = lambda _: (
        current.group.rank, current.group.world_size
    )
    sys.modules[linear_module.__name__] = linear_module
    environment = {"is_cpu": lambda: False}
    default, loader = extract(
        root / "model_loader/weight_utils.py",
        ("default_weight_loader", "sharded_weight_loader"),
        environment,
    )

    def attributes(tensor, values):
        for name, value in values.items():
            setattr(tensor, name, value)

    env = {
        "nn": SimpleNamespace(Linear=Linear), "F": SimpleNamespace(linear=linear),
        "sharded_weight_loader": loader, "set_weight_attrs": attributes,
    }
    projection, = extract(
        root / "layers/dflash_context_projection.py", ("DFlashContextProjection",), env
    )
    load, project = extract(
        root / "models/dflash.py", ("load_weights", "project_target_hidden"),
        {"DFlashContextProjection": projection, "default_weight_loader": default},
        "DFlashDraftModel",
    )
    return current, projection, default, load, project


def constructor_program(path):
    tree = ast.parse(path.read_text(), str(path))
    model = next(node for node in tree.body if getattr(node, "name", None) == "DFlashDraftModel")
    init = next(node for node in model.body if getattr(node, "name", None) == "__init__")
    branch = next(
        node for node in init.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.quantizable"
        and any(isinstance(child, ast.Attribute) and child.attr == "fc" for child in ast.walk(node))
    )
    return compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")


def check_constructor(root):
    branch = constructor_program(root / "models/dflash.py")
    marked = []

    def sharded(*args):
        marked.append(args)
        return SimpleNamespace(kind="sharded")

    ordinary = lambda *args, **kwargs: SimpleNamespace(kind="ordinary")
    replicated = lambda *args, **kwargs: SimpleNamespace(kind="quantized")
    spec = SimpleNamespace(speculative_dflash_tp_context_projection=True)
    group = SimpleNamespace(world_size=8)
    platform = SimpleNamespace(cuda=True)
    selected = 0
    for quantized in (False, True):
        for enabled in (False, True):
            spec.speculative_dflash_tp_context_projection = enabled
            for cuda in (False, True):
                platform.cuda = cuda
                for tp in (1, 2, 3, 4, 8):
                    group.world_size = tp
                    for hidden in (64, 65, 96):
                        model = SimpleNamespace(quantizable=quantized, num_context_features=5)
                        exec(
                            branch,
                            {"self": model, "prefix": "", "hidden_size": hidden,
                             "quant_config": None, "get_spec": lambda: spec,
                             "is_cuda": lambda: platform.cuda,
                             "get_parallel": lambda: SimpleNamespace(tp_group=group),
                             "DFlashContextProjection": sharded,
                             "nn": SimpleNamespace(Linear=ordinary), "ReplicatedLinear": replicated},
                        )
                        expected = (
                            "quantized" if quantized else
                            "sharded" if enabled and cuda and tp > 1 and hidden % tp == 0 else
                            "ordinary"
                        )
                        assert model.fc.kind == expected
                        selected += 1
    return selected


def check_projection(root):
    current, projection, default, load, project = load_environment(root)
    rng = np.random.default_rng(873)
    shapes = 0
    load_checks = 0
    for hidden, taps in ((16, 1), (32, 3), (64, 5)):
        input_width = hidden * taps
        weights = Tensor(bf16(rng.normal(size=(hidden, input_width))))
        for tp in (2, 4, 8):
            modules = []
            for rank in range(tp):
                group = Group(rank, tp)
                current.group = group
                fc = projection(input_width, hidden, group)
                model = SimpleNamespace(
                    projector_type="linear", is_nemotron_35_draft=False,
                    num_context_features=taps, config=SimpleNamespace(hidden_size=hidden),
                    quantizable=False, fc=fc, hidden_norm=lambda x: x,
                    named_parameters=lambda fc=fc: [("fc.weight", fc.weight)],
                )
                # The loader must freeze the draft group at construction,
                # even when loading or running from a target scope later.
                current.group = Group(91, 97)
                for name in ("fc.weight", "model.fc.weight", "encoder.fc.weight"):
                    load(model, [(name, weights)])
                    start, stop = rank * hidden // tp, (rank + 1) * hidden // tp
                    np.testing.assert_array_equal(fc.weight.value, weights.value[start:stop])
                    load_checks += 1
                for wrong in ((hidden + 1, input_width), (hidden, input_width+1),
                              (hidden // tp, input_width)):
                    try:
                        load(model, [("fc.weight", Tensor(np.zeros(wrong)))])
                    except ValueError:
                        load_checks += 1
                    else:
                        raise AssertionError("Invalid checkpoint projection shape accepted")
                modules.append((model, group))

            for tokens in (0, 1, 7, 8, 16, 32, 256, 513):
                # Include an ordinary noncontiguous input view.
                data = bf16(rng.normal(size=(tokens, input_width * 2)))[:, ::2]
                inputs = Tensor(data)
                expected = linear(inputs, weights).value
                for model, group in modules:
                    calls = group.calls
                    group.expected = expected
                    result = project(model, inputs)
                    np.testing.assert_array_equal(result.value, expected)
                    assert group.calls == calls + (tokens > 0)
                    shapes += 1
                    for malformed in (np.zeros((tokens, input_width+1)), np.zeros((tokens, 1, input_width))):
                        try:
                            project(model, Tensor(malformed))
                        except ValueError:
                            load_checks += 1
                        else:
                            raise AssertionError("Invalid target hidden shape accepted")
    return shapes, load_checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    root = parser.parse_args().source_root / "python/sglang/srt"
    constructor = check_constructor(root)
    outputs, loaders = check_projection(root)
    print(f"Passed {constructor} constructor cases, {outputs} projection/gather cases,")
    print(f"and {loaders} checkpoint/shape checks using the extracted V73 source.")

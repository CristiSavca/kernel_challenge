"""Exercise sharded DFLASH convolution projection with the pinned source and CPU arrays."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run(source_root):
    here = Path(__file__).resolve().parent
    fixture = here.parent / "v73_dflash_tp_context_projection/cpu_check.py"
    env = {"__name__": "v73_fixture"}
    # V78 generalizes the original projection class to two DFLASH call sites.
    exec(compile(fixture.read_text().replace("DFlashContextProjection", "DFlashGatheredLinear"),
                 str(fixture), "exec"), env)
    srt = source_root / "python/sglang/srt"
    inherited = [env["check_constructor"](srt), *env["check_projection"](srt)]
    current, projection, _, load, _ = env["load_environment"](srt)
    Tensor, Linear, Group = (env[name] for name in ("Tensor", "Linear", "Group"))
    bf16, linear = env["bf16"], env["linear"]
    Tensor.__getitem__ = lambda self, key: Tensor(self.value[key])
    modules = SimpleNamespace(Module=object, Linear=Linear, Parameter=lambda data: Tensor(data))
    torch = SimpleNamespace(zeros=lambda *shape: np.zeros(shape, dtype=np.float32))
    conv, = env["extract"](
        srt / "models/dflash.py", ("DFlashGroupedConv",),
        {"nn": modules, "torch": torch, "DFlashGatheredLinear": projection,
         "get_parallel": lambda: SimpleNamespace(tp_group=current.group)},
    )
    # Confirm the actual factory admits only opt-in, unquantized CUDA models.
    tree = ast.parse((srt / "models/dflash.py").read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "grouped_conv")
    kw = next(n.value for n in ast.walk(factory) if isinstance(n, ast.keyword)
              and n.arg == "shard_filter")
    for flag in (False, True):
        for cuda in (False, True):
            for quant in (False, True):
                enabled = eval(compile(ast.Expression(kw), "<factory>", "eval"),
                               {"is_cuda": lambda: cuda, "self": SimpleNamespace(quantizable=quant),
                                "get_spec": lambda: SimpleNamespace(speculative_dflash_tp_conv_projection=flag)})
                assert enabled == (flag and cuda and not quant)

    rng = np.random.default_rng(7803)
    gates = loaders = forwards = 0
    for groups, taps in ((3, 1), (4, 2), (8, 3), (16, 2)):
        channels, width = groups * 4, 2 * taps * groups
        weight = Tensor(bf16(rng.normal(size=(width, channels))))
        for size in (1, 2, 3, 4, 8):
            for rank in range(size):
                group = Group(rank, size)
                current.group = group
                for flag in (False, True):
                    model = conv(channels, 8, taps, 4, shard_filter=flag)
                    proj = model.kernel_projection
                    expected_shard = flag and size > 1 and width % size == 0
                    assert isinstance(proj, projection) == expected_shard
                    gates += 1
                    # Exercise real attention and MLP checkpoint names and rejection.
                    aliases = ("layers.0.attention_conv.kernel_projection.weight",
                               "layers.4.mlp_conv.kernel_projection.weight")
                    holder = SimpleNamespace(projector_type="linear", named_parameters=lambda: [
                        (name, proj.weight) for name in aliases])
                    for name in aliases:
                        for prefix in ("", "model."):
                            load(holder, [(prefix + name, weight)])
                            loaders += 1
                    if expected_shard:
                        for shape in ((width+1, channels), (width, channels+1), (width//size, channels)):
                            try:
                                load(holder, [(aliases[0], Tensor(np.zeros(shape)))])
                            except ValueError:
                                loaders += 1
                            else:
                                raise AssertionError("Malformed filter checkpoint accepted")
                    # The unchanged convolution receives both sides and all taps in checkpoint order.
                    seen = {}
                    def observe(x, delta, side):
                        seen[side] = delta.value.copy()
                        return x
                    model._convolve = observe
                    for rows in (0, 1, 7, 8, 16, 32, 256):
                        inputs = Tensor(bf16(rng.normal(size=(rows, 2 * channels)))[:, ::2])
                        group.expected = linear(inputs, weight).value
                        before = group.calls
                        output, coefficients = model.prepare(inputs)
                        assert model.finish(output, coefficients) is inputs
                        by_side = group.expected.reshape(rows, 2, taps, groups)
                        for side in (0, 1):
                            np.testing.assert_array_equal(seen[side], by_side[:, side])
                        assert group.calls - before == int(expected_shard and rows > 0)
                        forwards += 1
    print(f"V73 inherited {inherited}; V78 factory 8, projections {gates}, "
          f"checkpoint cases {loaders}, prepare/finish pairs {forwards}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    run(parser.parse_args().source_root)

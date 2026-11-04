"""CPU-only metadata boundary tests; no Torch, Triton, CUDA or Modal imports."""

import json
import types
import unittest

from decoder_policy_metadata import collect_decoder_policy


class TensorLike:
    def _forbidden(self, *args, **kwargs):
        raise AssertionError("unsupported object was inspected")
    __str__ = __repr__ = __iter__ = __len__ = __bool__ = _forbidden
    item = cpu = numpy = tolist = detach = _forbidden


class EvilDict(dict):
    def _forbidden(self, *args, **kwargs):
        raise AssertionError("dict subclass was inspected")
    get = __iter__ = __len__ = __getitem__ = _forbidden


def entries(mapping):
    return mapping["entries"]


class MetadataTests(unittest.TestCase):
    def test_realistic_policy_and_tuple_keys(self):
        module = types.ModuleType("fake_decoder")
        module.PDL, module.FORCE_GLUEV2 = True, False
        module.MOE_DEFINES = ("-DOKC_WGS=1",)
        module._MA_EXTS = {("-DOKC_WGS=1", "-DOKC_PRMTSC=1"): TensorLike()}
        module.__getattr__ = lambda name: (_ for _ in ()).throw(AssertionError("module hook called"))
        state = {
            "mm_pick": {(16, "o"): "bfp", (128, "pr"): True},
            "moe_pick": {16: "agent"}, "gluev2": {16: False}, "marlin_cfg": {16: (16, False)},
            "bfp": {"choice": {("pooler", 128): ("rq_tma",)}, "disabled": {("o", "wide", 128)},
                    "old_paths": False, "comp": {"pooler": True}, "tune_log": [("pooler", 128, [("torch", 24.2)])]},
            "sk2b": {16: {"grid": 256, "cap": 264, "sums": TensorLike(), "bar": TensorLike()}},
            "ub2": {"weights": TensorLike()}, "fa3": TensorLike(),
            "dense_chain_log": [{"T": 16, "selected": ("bfp", "o", ("sk2",)), "score": 1.03}],
            "private_weights": TensorLike(),
        }
        output = collect_decoder_policy(module, state)
        self.assertEqual(entries(output["state"]["mm_pick"])[0],
                         {"key": {"kind": "sequence", "items": [16, "o"], "omitted": 0}, "value": "bfp"})
        self.assertEqual(output["features"]["ub2"], True)
        self.assertEqual(output["module_policy"]["FORCE_GLUEV2"], False)
        self.assertEqual(output["built_expert_defines"]["keys"]["items"][0]["items"],
                         ["-DOKC_WGS=1", "-DOKC_PRMTSC=1"])
        serialized = json.dumps(output, allow_nan=False)
        self.assertNotIn("private_weights", serialized)
        self.assertNotIn('"sums"', serialized)
        self.assertEqual(state["mm_pick"][(16, "o")], "bfp")

    def test_unsupported_leaf_is_explicit_not_stringified(self):
        result = collect_decoder_policy({}, {"mm_pick": {(16, "o"): TensorLike()}})
        self.assertEqual(entries(result["state"]["mm_pick"])[0]["value"],
                         {"status": "unavailable", "reason": "unsupported_type"})
        json.dumps(result, allow_nan=False)

    def test_mapping_subclass_rejected_without_methods(self):
        result = collect_decoder_policy(EvilDict(), EvilDict())
        self.assertEqual(result["state"]["status"], "unavailable")
        self.assertEqual(result["module_policy"]["status"], "unavailable")

    def test_cycles_depth_collection_and_string_limits(self):
        cycle = []
        cycle.append(cycle)
        result = collect_decoder_policy({}, {"mm_pick": {"cycle": cycle, "deep": [[[[1]]]],
                                                         "long": "x" * 40}, "moe_pick": list(range(10))},
                                        max_depth=3, max_items=3, max_string=8)
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn('"reason": "cycle"', encoded)
        self.assertIn('"reason": "depth_limit"', encoded)
        self.assertIn('"omitted_chars": 32', encoded)
        self.assertEqual(result["state"]["moe_pick"]["omitted"], 7)

    def test_node_budget_does_not_expand_remaining_tree(self):
        tree = [list(range(100)) for _ in range(100)]
        result = collect_decoder_policy({}, {"mm_pick": tree}, max_nodes=10, max_items=64)
        self.assertEqual(result["nodes_used"], 10)
        self.assertEqual(len(result["state"]["mm_pick"]["items"]), 1)
        self.assertLess(len(json.dumps(result)), 7000)

    def test_dense_selection_survives_large_original_tune_log(self):
        result = collect_decoder_policy({}, {
            "bfp": {"tune_log": list(range(1000))},
            "dense_chain_log": [{"T": 128, "selected": "rq_tma", "score": 1.03}],
        }, max_nodes=100, max_items=128)
        selection = result["dense_chain_log"]["dense_chain_log"]
        self.assertEqual(selection["omitted"], 0)
        self.assertEqual({entry["key"]: entry["value"] for entry in selection["items"][0]["entries"]},
                         {"T": 128, "selected": "rq_tma", "score": 1.03})
        self.assertGreater(result["bfp_tune_log"]["tune_log"]["omitted"], 0)
        self.assertEqual(result["nodes_used"], 100)

    def test_json_special_values_and_scalar_subclass(self):
        class CustomInt(int):
            __repr__ = TensorLike._forbidden
        result = collect_decoder_policy({}, {"mm_pick": [float("nan"), float("inf"), 1 << 200, CustomInt(3)]})
        reasons = [entry["reason"] for entry in result["state"]["mm_pick"]["items"]]
        self.assertEqual(reasons, ["non_finite_float", "non_finite_float", "integer_too_large", "unsupported_type"])
        json.dumps(result, allow_nan=False)

    def test_set_order_and_limit_validation(self):
        result = collect_decoder_policy({}, {"bfp": {"disabled": {("z", 3), ("a", 1)}}})
        self.assertEqual(result["bfp"]["disabled"]["items"][0]["items"], ["a", 1])
        with self.assertRaises(ValueError):
            collect_decoder_policy({}, {}, max_nodes=True)
        with self.assertRaises(ValueError):
            collect_decoder_policy({}, {}, max_depth=0)
        result = collect_decoder_policy({}, {}, max_nodes=1000000, max_items=1000000)
        self.assertEqual(result["limits"]["max_nodes"], 16384)
        self.assertEqual(result["limits"]["max_items"], 256)


if __name__ == "__main__":
    unittest.main()

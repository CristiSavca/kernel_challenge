"""Bounded CPU-only decoder-policy metadata; never inspect tensor/object values.

Use collect_decoder_policy(module, prepared_state) immediately after prepare.
Only exact built-in scalar/container types are serialized. Tuple-keyed maps
become entry lists. Unsupported objects are explicitly unavailable; there is
no str/repr fallback and no module function is invoked.
"""

import itertools
import json
import math
import types


STATE_FIELDS = ("mm_pick", "moe_pick", "gluev2", "marlin_cfg", "pbmax", "att_ok", "ma_ok", "nsm")
BFP_FIELDS = ("choice", "disabled", "old_paths", "sm90", "comp")
FEATURE_FIELDS = ("bfp", "bank", "fa3", "ub2", "uv")
MODULE_FIELDS = (
    "PDL", "PDL_MAX_T", "BFP_PDL_N", "FORCE_GLUEV2", "FORCE_MM", "FORCE_PBMAX",
    "PIN_PICK", "BFP_MAX_T", "UV", "UV_MAX_T", "UB2", "SK2B", "SK2B_UN_TOKR", "SK2B1",
    "MOE_DEFINES", "MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "MOE_MID_DEFINES",
    "MOE_TINY_MAX_M", "MOE_TINY2_MAX_M", "MOE_MID_MAX_M", "MAIN_FALLBACK_DEFINES",
    "_MA_SFX", "_MA_MAIN_NS", "DQ7", "MOE_MULDEC", "MOE_HAS_PDL",
)
_MISSING = object()


def _unavailable(reason):
    return {"status": "unavailable", "reason": reason}


class _Encoder:
    def __init__(self, max_depth, max_items, max_nodes, max_string):
        self.max_depth, self.max_items = max_depth, max_items
        self.max_nodes, self.max_string = max_nodes, max_string
        self.remaining = max_nodes
        self.active = set()

    def encode(self, value, depth=0):
        if self.remaining <= 0:
            return _unavailable("node_budget")
        self.remaining -= 1
        kind = type(value)
        if value is _MISSING:
            return _unavailable("missing")
        if value is None or kind is bool:
            return value
        if kind is int:
            return value if value.bit_length() <= 128 else _unavailable("integer_too_large")
        if kind is float:
            return value if math.isfinite(value) else _unavailable("non_finite_float")
        if kind is str:
            if len(value) <= self.max_string:
                return value
            return {"status": "truncated", "value": value[:self.max_string],
                    "omitted_chars": len(value) - self.max_string}
        if kind not in (dict, list, tuple, set, frozenset):
            return _unavailable("unsupported_type")
        if depth >= self.max_depth:
            return _unavailable("depth_limit")
        identity = id(value)
        if identity in self.active:
            return _unavailable("cycle")
        self.active.add(identity)
        try:
            count = len(value)
            items = []
            iterator = value.items() if kind is dict else iter(value)
            for item in itertools.islice(iterator, self.max_items):
                if self.remaining <= 0:
                    break
                if kind is dict:
                    key, child = item
                    items.append({"key": self.encode(key, depth + 1),
                                  "value": self.encode(child, depth + 1)})
                else:
                    items.append(self.encode(item, depth + 1))
            if kind in (set, frozenset):
                # Sort only already-encoded JSON values; never compare original
                # objects or invoke their repr/str. A truncated subset is marked.
                items.sort(key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
            return {"kind": "dict" if kind is dict else "set" if kind in (set, frozenset) else "sequence",
                    "entries" if kind is dict else "items": items,
                    "omitted": count - len(items)}
        finally:
            self.active.remove(identity)


def collect_decoder_policy(module, state, *, max_depth=8, max_items=64, max_nodes=4096, max_string=256):
    """Return bounded JSON-safe policy data, without CUDA calls or tensor reads.

    module must be an exact ModuleType or a plain namespace dict; state must
    be a plain dict. Missing/unsupported data is explicit. Limits are hard
    capped even when a caller supplies a larger number. Logs are collected
    last so they cannot exhaust the budget before core dispatch metadata;
    dense-chain selection evidence has priority over the original tune log.
    """
    requested = (max_depth, max_items, max_nodes, max_string)
    if any(type(value) is not int or value < 1 for value in requested):
        raise ValueError("metadata limits must be positive built-in integers")
    max_depth, max_items = min(max_depth, 12), min(max_items, 256)
    max_nodes, max_string = min(max_nodes, 16384), min(max_string, 1024)
    encoder = _Encoder(max_depth, max_items, max_nodes, max_string)
    namespace = vars(module) if type(module) is types.ModuleType else module if type(module) is dict else None
    source = state if type(state) is dict else None
    bfp = source.get("bfp") if source is not None else None
    bfp = bfp if type(bfp) is dict else None
    result = {"schema": 1, "limits": {"max_depth": max_depth, "max_items": max_items,
                                        "max_nodes": max_nodes, "max_string": max_string}}

    def fields(mapping, names):
        if mapping is None:
            return _unavailable("unsupported_or_missing_mapping")
        return {name: encoder.encode(mapping.get(name, _MISSING)) for name in names}

    result["state"] = fields(source, STATE_FIELDS)
    result["bfp"] = fields(bfp, BFP_FIELDS)
    result["module_policy"] = fields(namespace, MODULE_FIELDS)
    result["features"] = ({name: name in source and source[name] is not None for name in FEATURE_FIELDS}
                          if source is not None else _unavailable("unsupported_or_missing_mapping"))
    result["fused_boundaries"] = {}
    for name in ("sk2b", "sk2b1"):
        available = source.get(name, _MISSING) if source is not None else _MISSING
        if type(available) is dict:
            # Project the whitelist first: sums/bar tensors never reach encoder.
            projected = [{"bucket": bucket,
                          "policy": {key: entry.get(key, _MISSING) for key in ("grid", "cap")}
                          if type(entry) is dict else _unavailable("unsupported_type")}
                         for bucket, entry in itertools.islice(available.items(), max_items)]
            result["fused_boundaries"][name] = {
                "buckets": encoder.encode(projected), "omitted_buckets": len(available) - len(projected)}
        else:
            result["fused_boundaries"][name] = encoder.encode(available)
    extensions = namespace.get("_MA_EXTS", _MISSING) if namespace is not None else _MISSING
    if type(extensions) is dict:
        keys = list(itertools.islice(extensions.keys(), max_items))
        result["built_expert_defines"] = {"keys": encoder.encode(keys), "omitted": len(extensions) - len(keys)}
    else:
        result["built_expert_defines"] = encoder.encode(extensions)
    result["dense_chain_log"] = fields(source, ("dense_chain_log",))
    result["bfp_tune_log"] = fields(bfp, ("tune_log",))
    result["nodes_used"] = max_nodes - encoder.remaining
    return result

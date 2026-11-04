# V27 serving ownership adapter. The CUDA arithmetic above is frozen V26.
# Initial execution domain: one ordinary SGLang forward lane per model/device.
# PDMUX/TBO/SP/offload/compile are declined before weight replacement.
class NativeUnsupported(ValueError):
    """A qualification decline before any source-weight ownership transfer."""


def _serial_policy():
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    execution = get_exec()
    config = execution.graph.cuda_graph_config
    if config is None:
        return False  # require resolved phase backends before installation
    if any(getattr(config, phase).backend not in ("full", "breakable", "disabled")
           for phase in ("decode", "prefill")):
        return False
    return not (
        get_disagg().enable_pdmux
        or execution.overlap.enable_two_batch_overlap
        or execution.overlap.enable_single_batch_overlap
        or execution.offload.cpu_offload_gb > 0
        or execution.offload.offload_group_size > 0
        or execution.features.enable_memory_saver
        or execution.graph.enable_torch_compile
        or execution.moe.enable_eplb
        or get_parallel().enable_layernorm_sp
    )


class NativeModelOwner:
    """Model lifetime owns scratch, never a process-global or stream-ID cache.

    Ordinary scheduler forwards and graph replay must remain serialized on the
    model's forward lane. Capture uses another stream but the same lane. External
    concurrent use of one lane is unsupported; separate owners/lanes need their
    own storage. Cooperative grid admission does not isolate workspace writes.
    """
    def __init__(self):
        self._workspaces = {}

    def workspace(self, device, *, lane="serial"):
        device = torch.device(device)
        key = (device.type, device.index, lane)
        if key not in self._workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V27 workspace must be created before capture")
            workspace = Workspace(device)
            torch.cuda.synchronize(device)
            self._workspaces[key] = workspace
        return self._workspaces[key]


_V27_RAW = (
    ("w13", "gate_up_proj"),
    ("w2", "down_proj"),
    ("w13_sf", "gate_up_proj_weight_scale"),
    ("w2_sf", "down_proj_weight_scale"),
    ("w13_gs", "gate_up_proj_weight_scale_2"),
    ("w2_gs", "down_proj_weight_scale_2"),
)
_V27_BUFFERS = tuple("_native_v27_weight_" + str(i) for i in range(5))


def _layer_eligible(layer):
    owner = getattr(layer, "_native_v27_owner", None)
    if owner is None or not _serial_policy():
        return False
    if (not layer.use_marlin or layer.expert_tp or layer.use_megamoe
            or layer.params_dtype != torch.bfloat16
            or layer.hidden_size != 4096
            or layer.intermediate_size_per_partition != 4096
            or layer.top_k != 8 or layer.global_num_experts != 256
            or layer.quant_config.group_size != 16
            or not 0 < layer.num_local_experts <= 32
            or not 0 <= layer.local_expert_offset <= 256 - layer.num_local_experts):
        return False
    # Muse constructs this norm before postload. Its cached effective weight is
    # refreshed by the original setter at forward time, not frozen into the bank.
    norm = getattr(layer, "post_expert_norm", None)
    if norm is None or not callable(getattr(norm, "_get_effective_weight", None)):
        return False
    if any(name not in layer._parameters for _, name in _V27_RAW):
        return False
    device = layer.gate_up_proj.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        return False
    # Native's contiguous rank mapping must agree with the actual stock map.
    # EPLB is excluded above; this is preparation metadata, never input routing.
    mapping = layer._expert_map
    if mapping is None or mapping.dtype != torch.int32 or mapping.shape != (256,):
        return False
    expected = torch.full((256,), -1, dtype=torch.int32, device=mapping.device)
    lo = layer.local_expert_offset
    expected[lo:lo + layer.num_local_experts] = torch.arange(
        layer.num_local_experts, dtype=torch.int32, device=mapping.device
    )
    if not torch.equal(mapping, expected):
        return False
    if any(hasattr(layer, name) for name in _V27_BUFFERS):
        raise RuntimeError("V27 derived buffer name collision")
    return True


def try_install_native(layer):
    """Return False on unsupported metadata/format; never catch device failures.

    Build and initialize the bank and workspace before touching source slots.
    The narrow commit rolls registry changes back if any installation step
    raises. Unexpected CUDA/compile/OOM errors propagate with originals intact.
    """
    if getattr(layer, "_native_v27_bank", None) is not None:
        raise RuntimeError("V27 cannot prepare an already converted layer twice")
    if not _layer_eligible(layer):
        return False
    raw = {key: getattr(layer, name) for key, name in _V27_RAW}
    try:
        bank = prepare_native(
            raw, local_start=layer.local_expert_offset,
            local_count=layer.num_local_experts,
            source_is_local=True, globals_prepared=False,
        )
    except NativeUnsupported:
        return False
    owner = layer._native_v27_owner
    owner.workspace(bank["device"])
    parameters = dict(layer._parameters)
    buffers = dict(layer._buffers)
    nonpersistent = set(layer._non_persistent_buffers_set)
    try:
        for name, tensor in zip(_V27_BUFFERS, bank["w"]):
            layer.register_buffer(name, tensor, persistent=False)
        # Metadata points at the registered tensors, not additional copies.
        object.__setattr__(layer, "_native_v27_bank", bank)
        for _, name in _V27_RAW:
            delattr(layer, name)
    except Exception:
        layer._parameters.clear()
        layer._parameters.update(parameters)
        layer._buffers.clear()
        layer._buffers.update(buffers)
        layer._non_persistent_buffers_set.clear()
        layer._non_persistent_buffers_set.update(nonpersistent)
        if "_native_v27_bank" in layer.__dict__:
            object.__delattr__(layer, "_native_v27_bank")
        raise
    return True


def native_forward(layer, hidden_states, topk_scores, topk_ids,
                   post_expert_norm, pre_expert_norm=None):
    """Core then original stock finalizer per chunk, before scratch reuse.

    No per-call host inspection of routes/activations. No fallback to released
    Marlin weights; changed execution configuration fails explicitly.
    """
    if not _serial_policy():
        raise RuntimeError("V27 execution policy changed after weight replacement")
    if post_expert_norm is None:
        raise ValueError("V27 requires the existing post-expert norm")
    if pre_expert_norm is not None:
        hidden_states = pre_expert_norm(hidden_states)
    bank = layer._native_v27_bank
    if hidden_states.device != bank["device"]:
        raise RuntimeError("V27 model device changed after preparation")
    if layer._post_expert_norm is not post_expert_norm:
        layer.set_post_expert_norm(post_expert_norm)
    if layer._post_expert_norm_weight is None:
        raise RuntimeError("V27 requires the original effective-weight finalizer path")
    total = hidden_states.shape[0]
    if total == 0:
        return hidden_states.new_empty((0, layer.hidden_size))
    workspace = layer._native_v27_owner.workspace(bank["device"])
    scores = topk_scores.to(torch.float32).contiguous()
    result = None if total <= MMAX else torch.empty_like(hidden_states)
    for start in range(0, total, MMAX):
        end = min(start + MMAX, total)
        y, local = expert_outputs(
            hidden_states[start:end], topk_ids[start:end], bank, workspace
        )
        chunk_scores = scores[start:end]
        if total >= 32 and end - start < 32:
            y = workspace.expert_out[:32]
            local = workspace.local_ids[:32]
            local[end - start:].fill_(-1)
            padded_scores = torch.zeros((32, 8), dtype=torch.float32,
                                        device=hidden_states.device)
            padded_scores[:end - start].copy_(chunk_scores)
            chunk_scores = padded_scores
        value = fused_post_expert_norm_weighted_sum(
            y, chunk_scores, norm_weight=layer._post_expert_norm_weight,
            eps=layer._post_expert_norm_eps,
            weight_first=layer._post_expert_norm_weight_first,
            slot_expert_ids=local,
        )[:end - start]
        if result is None:
            return value
        result[start:end].copy_(value)
    return result

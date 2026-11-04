# V25 keeps the stock serving finalizer unchanged, including its BF16/FP32 order.
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum

MMAX = 384
EMAX = MMAX * 8
JMAX = 32 + EMAX // 64 + 8
WGS_XROWS = EMAX + 8 * JMAX
WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
_EXTS = {}

def _ext(device, prmt):
    # Device-specific cache; no mutable PDL mode and no global fast-math/FTZ.
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, bool(prmt))
    if key not in _EXTS:
        with torch.cuda.device(index):
            if torch.cuda.get_device_capability(index) != (9, 0):
                raise ValueError("V25 requires SM90 with cooperative launch support")
            defs = ["-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_MULDEC=1",
                    "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_FNST=4", f"-DOKC_PRMTSC={int(prmt)}"]
            mod = load_inline(name=f"okc_serving_wgs_v25_d{index}_p{int(prmt)}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                              functions=["okc_init", "okc_run"],
                              extra_cuda_cflags=["-O3", "-Xptxas", "-v,-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", *defs],
                              verbose=False)
            _EXTS[key] = (mod, int(mod.okc_init()))
    return _EXTS[key]


def _marlin_scale_lut(device):
    # Same scalar path as stock process_scales, with the layout-only permutation
    # omitted. Retain only generic BF16 bit constants, not decoded model weights.
    sf = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8).view(torch.float8_e4m3fn)
    half = sf.to(torch.bfloat16).to(torch.float16)
    shifted = ((half * 128).view(torch.int16).to(torch.int32) << 1) & 0xffff
    byte = (shifted >> 8) & 255
    bits = ((byte & 0x80) << 7) | ((byte & 0x7f) << 4)
    return (bits | (bits << 16)).to(torch.int32).contiguous()


def prepare_native(weights, *, local_start=0, local_count=32, source_is_local=False, globals_prepared=False):
    """Pack one local WGS bank; no source tensors or execution scratch retained.

    This explicit preparation must precede capture. A future serving installer
    must transfer ownership before freeing stock buffers; this function never
    mutates a Module or frees caller-owned weights.
    """
    if not (0 <= local_start and 0 < local_count <= 32 and local_start + local_count <= 256):
        raise ValueError("V25 supports contiguous ownership of <=32 of256 experts")
    start = 0 if source_is_local else local_start
    stop = start + local_count
    device = weights["w13"].device
    for name, shape, dtype in (("w13", (8192,2048), torch.uint8), ("w2", (4096,2048), torch.uint8),
                               ("w13_sf", (8192,256), torch.float8_e4m3fn), ("w2_sf", (4096,256), torch.float8_e4m3fn)):
        t = weights[name]
        if t.dtype != dtype or t.device != device or tuple(t.shape[1:]) != shape or t.shape[0] < stop:
            raise ValueError(f"Unsupported V25 weight metadata: {name}")
    scales = [weights[k][start:stop] for k in ("w13_sf", "w2_sf")]
    # Stock NVFP4 expects nonnegative finite scales. Decline unusual layers
    # before packing/ownership transfer; zero and subnormal bytes use exact LUT.
    if any(bool((s.contiguous().view(torch.uint8) > 0x7e).any()) for s in scales):
        raise ValueError("V25 declines negative/nonfinite scales before bank replacement")
    prmt = all(not bool((s.contiguous().view(torch.uint8) < 8).any()) for s in scales)
    globals_out = []
    for name in ("w13_gs", "w2_gs"):
        gs = weights[name][start:stop]
        if name == "w13_gs" and gs.dim() == 2 and gs.shape[1] == 2 and not globals_prepared:
            gs = gs[:,0]  # same gate-column collapse as stock Marlin
        if gs.shape != (local_count,) or gs.device != device:
            raise ValueError("V25 expects one global scalar per local expert")
        if globals_prepared:
            if gs.dtype != torch.bfloat16:
                raise ValueError("Prepared Marlin global scalars must remain BF16")
            globals_out.append(gs.clone().contiguous())
        else:
            if gs.dtype != torch.float32:
                raise ValueError("Checkpoint global scalars must be FP32")
            globals_out.append((gs.to(torch.bfloat16) * (2.0 ** 119)).contiguous())
    if any(not bool(torch.isfinite(gs).all()) for gs in globals_out):
        raise ValueError("V25 requires finite prepared BF16 global scalars")
    rb = torch.arange(512, device=device)
    g = torch.arange(8, device=device)
    rowsA = torch.stack((8 * rb[:,None] + g[None,:], 4096 + 8 * rb[:,None] + g[None,:]),1).reshape(-1)
    rowsA = _wgs_kperm_rows(rowsA)
    rowsB = torch.arange(4096, device=device)
    banks = []
    for code_name, scale_name, rows in (("w13","w13_sf",rowsA),("w2","w2_sf",rowsB)):
        row_count = rows.numel()
        expert_bytes = row_count * 4096 * 9 // 16
        target = torch.empty((local_count, expert_bytes), dtype=torch.uint8, device=device)
        # Bound packing intermediates to2048 output rows, avoiding whole-bank
        # lists/cat and the parent's eight retained layouts.
        for e in range(local_count):
            for r0 in range(0,row_count,2048):
                selected = rows[r0:r0+2048]
                codes = weights[code_name][start+e].index_select(0,selected)
                sf = weights[scale_name][start+e].index_select(0,selected)
                local_rows = torch.arange(selected.numel(), device=device)
                packed = _wgs_copy(codes,sf,local_rows,_POS_MUL)
                offset = r0 * 4096 * 9 // 16
                target[e,offset:offset+packed.numel()].copy_(packed)
        banks.append(target.view(-1))
    lut = _marlin_scale_lut(device)
    mod, grid = _ext(device, prmt)
    torch.cuda.synchronize(device)  # finish only weight preparation before capture/other streams
    return dict(w=[banks[0],banks[1],*globals_out,lut],lo=local_start,cnt=local_count,device=device,mod=mod,grid=grid,prmt=prmt)


class Workspace:
    """Explicit call/graph-owned storage; never share across overlapping calls.

    Distinct workspaces may execute on separate streams through cooperative
    launch. Reuse of this object and its captured graphs requires serial order.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.scratch = [torch.zeros(8+1024,dtype=torch.int32,device=device),
                        torch.zeros(WGS_FIXF,dtype=torch.float32,device=device),
                        torch.empty(2*WGS_XROWS*4096,dtype=torch.bfloat16,device=device)]
        self.expert_out = torch.empty((MMAX,8,4096),dtype=torch.bfloat16,device=device)
        self.local_ids = torch.empty((MMAX,8),dtype=torch.int32,device=device)


def expert_outputs(x, ids, bank, workspace):
    """Return BF16 route outputs plus the mask; remote output bytes are undefined.

    Call the stock masked finalizer before reusing this workspace. No input-
    dependent host branch, mapping copy, or input-derived persistent cache.
    """
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != 4096 or x.shape[0] > MMAX:
        raise ValueError("V25 core expects BF16 [M<=384,4096]")
    if x.device != bank['device'] or x.device != workspace.device or ids.device != x.device or ids.shape != (x.shape[0],8):
        raise ValueError("V25 core tensor device/shape mismatch")
    if ids.dtype not in (torch.int32,torch.int64):
        raise ValueError("V25 requires integer router IDs in the caller's -1..255 domain")
    x = x.contiguous(); ids = ids.to(torch.int32).contiguous()
    if ids.data_ptr() & 15: ids = ids.clone()
    m = x.shape[0]
    y, local = workspace.expert_out[:m], workspace.local_ids[:m]
    if m:
        bank['mod'].okc_run(bank['w'],workspace.scratch,x,ids,y,local,bank['lo'],bank['cnt'],bank['grid'])
    return y,local


def prepare(weights):
    bank = prepare_native(weights)
    # Component ownership is explicit. A full-model installer should provide
    # one safely serialized model resource, not create this workspace perlayer.
    ws = Workspace(bank['device'])
    return {'bank':bank,'workspace':ws,'norm_w':(weights['post_norm_w'].float()+1.0).to(torch.bfloat16)}


def kernel_fn(data,state):
    cfg=data['config'];bank=state['bank'];x=data['hidden_states'];ids=data['topk_ids']
    if cfg['local_start'] != bank['lo'] or cfg['local_count'] != bank['cnt']:
        raise ValueError("V25 rank ownership mismatch")
    scores=data['topk_weights'].to(torch.float32).contiguous()
    if x.shape[0] == 0:return torch.empty_like(x)
    result = None if x.shape[0] <= MMAX else torch.empty_like(x)
    for start in range(0,x.shape[0],MMAX):
        end=min(start+MMAX,x.shape[0])
        y,local=expert_outputs(x[start:end],ids[start:end],bank,state['workspace'])
        chunk_scores=scores[start:end]
        if x.shape[0] >= 32 and end-start < 32:
            # Preserve the original whole-call large-T finalizer branch on a
            # short final chunk. Only padding rows are masked; no real route changes.
            y=state['workspace'].expert_out[:32]
            local=state['workspace'].local_ids[:32]
            local[end-start:].fill_(-1)
            padded_scores=torch.zeros((32,8),dtype=torch.float32,device=x.device)
            padded_scores[:end-start].copy_(chunk_scores)
            chunk_scores=padded_scores
        value=fused_post_expert_norm_weighted_sum(y,chunk_scores,norm_weight=state['norm_w'],eps=1e-8,
                                                 weight_first=True,slot_expert_ids=local)[:end-start]
        if result is None:return value
        result[start:end].copy_(value)
    return result

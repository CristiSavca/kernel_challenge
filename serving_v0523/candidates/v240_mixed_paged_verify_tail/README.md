# V240 mixed paged verification tails

V240 runs the existing Hopper paged-verification kernel on the short, causal DFlash requests in a mixed full-attention layer. V239 had already separated these requests from the long prompts and sent them to native paged FA3. V240 selects the same attention implementation that these requests can use during standalone target verification. Its H200 elapsed time and token agreement are unmeasured.

The exported patch is 392,725 bytes against the public SGLang pin. Its SHA-256 is `bae5ddc9545ed49734a56137609bf96985cdeae2125a539d6f2fdf32e45637ab`. The official CLI exported with two context lines. Forward and reverse application of the full patch and the V239 delta reconstructed the exact trees. [Validation, rules and flags](validation.json)

## Dispatch and memory

The mixed scheduler must certify independent top-1 causal request chains. The attention backend checks the effective spec algorithm, uniform configured verify width, static Hopper FP8 policy, scale and mask support, local-attention exclusions and zero logit cap. Muse uses the native RadixAttention logit-cap default of zero. Static prefill replay carries the live algorithm, role split and causal certificate from the original forward. The inherited launcher separately checks cache geometry, output layout and batch size up to 64. Any decline returns the tail to V239's native FA3 call.

The tail consumes its actual physical page-table rows and rebased query offsets. Every request retains its own scales. It writes directly to its BF16 slice, then the prompt phase packs the long requests. The launcher declines before allocating if its FP32 partial vectors and log-sums plus the existing FP8 writer vectors exceed 32 MiB. This bounds these live tensors. The CUDA allocator's reserved memory and the entire model peak require hardware observation. At H200 geometry with 132 SMs, two KV heads, eight query heads per KV head and width eight, the tail uses 4.19 MiB for one request and 5.14 MiB for 31. Tail execution and the dense prompt pack run successively on the same stream. [Workspace and ABI tests](limits.json)

## Local results

The normal Muse → Breakable callback → attention backend → physical cache writer was exercised on 17 parent-candidate pairs. With 1, 3, 15 and 31 short requests, the candidate entered the real host launcher and its two existing Triton bodies, executed using checked CPU pointers. All 409,600 resulting BF16 elements were compared with an independent float64 dense causal reference. The largest per-case relative L2 error was 0.002244. The comparison covered scaled and unscaled FP8 and 51,180,288 visible K/V bytes across the 17 pairs. The masked, uncertified, softcapped, nonuniform, ordinary, sliding, algorithm and oversize controls preserved the parent behavior. [Model and kernel replay](dispatch.json)

An additional 168 launcher tests covered exact workspace boundaries and native geometry limits. Seventy-two calls through the native fallback ABI validated 36 legacy-provider retries, 72 distinct-storage copybacks and 76,689,408 BF16 destination components. [Checks](limits.py)

The archived V41 scheduler replay identified 58 possible prompt packs fitting V239's cap at 16 users and 115 at 32 users. They correspond to 513 and 2,022 short request-steps per full layer, respectively, conditional on admission. These are potential dispatch opportunities on another candidate's real chronology. No throughput gain is inferred from their count. [Physical trace and assumptions](../v239_split_mixed_full_attention/work.json)

Run `dispatch.py --stage /path/to/sglang` and `limits.py /path/to/sglang` using CPU PyTorch and NumPy. The dispatch script also accepts `--candidate` for integrated exports. No challenge check, scored submission or GPU allocation was made for V240.

# V111 paged prefill, sequence-parallel draft projection and FP8 verify writes

**November 18 dispatch correction.** The prompt-projection startup check in this frozen patch rejects the registered Muse logits processor. The earlier CPU fixture used the base class and missed that distinction. V120b enables the prompt optimization for the native model and tests both classes. [Correction and regression](../v120b_native_muse_prompt/README.md)

V111 composes three changes on the scored V64 implementation. V106 supplies page-160 full attention and compact sliding-window FP8 prefill. V108 projects all five target taps on their owning sequence-parallel ranks and gathers the smaller projected rows for the draft. V110 encodes the target verification query while scattering the new FP8 key and value rows to their physical slots. The target never uses that scatter during prefill. The original native eight-token DFLASH2 verification width stays in use.

The source has nine flag overrides. They consist of V106's eight flags plus `--speculative-dflash-sp-prefill-context`. V110's encoding table and dispatch are selected by device, cache type, model stage and forward mode. See [the frozen flags and patch hashes](validation.json).

The component patches produce the same Git tree in either order. All single-component files keep their original bytes. The two shared attention backend methods, `__init__` and `forward_extend`, match an independent three-way merge of each method's parsed source. Applying the full export and reversing it restores the pinned SGLang tree. [Composition check](composition.json)

- The draft projection passed 69 CPU/Gloo cases at TP2, TP4 and TP8, including output trimming, hooks on one rank, cache positions and slots. A separate 40,960-wide reference matched the BF16 result at every partition size. [Projection receipt](prefill_projection.json)
- The prefill dispatch suite passed 90 cases at six physical page sizes. Its independent SWA oracle checked 19,229,952 FP8 bytes, 77,312 attention output scalars and 80 eager/graph page table constructions. [Dispatch](prefill_dispatch.json) · [Oracle](paged_attention_oracle.json)
- The verification dispatch suite ran 12 eligible and 84 fallback cases with the sliding-window prefill feature enabled. It matched 10,420,224 FA query bytes, 57,487,360 complete pool storage bytes and 20,840,448 recorded output bytes. [Verification receipt](verify_dispatch.json)
- The included prefill and verification kernel source files are byte-identical to V106 and V110. Their component receipts cover ahead-of-time Hopper compilation. [V106 SM90](../v106_paged_swa_suffix_prefill/sm90.json) · [V110 SM90](../v110_verify_fp8_qkv_store/sm90.json)

No integrated H200 performance or model output measurement exists. V64 measured 1.4119× overall and 1.5919× at 16 users. A matched high-load private comparison would test whether the three changes reduce the first-turn and decode costs together within the 2 GB peak-memory limit.

The November 19 runner opens at 00:03 UTC. It selects V111 at 32 users while V87 is queued or running and V64’s scored result is confirmed. A V87 pass selects V89 at four users. Scored V64 reached 20,547.4 tokens/s/GPU, 43.3 median decode tokens/s/user, and 110.0 s first-turn p90 at 32 users. The prior V100c and V106 attempts share the same private slot. [Runner receipt](../../runs/nov19_decode_runner.launch.json) · [Official reference](../../runs/abd217baf0ed49fd.json)

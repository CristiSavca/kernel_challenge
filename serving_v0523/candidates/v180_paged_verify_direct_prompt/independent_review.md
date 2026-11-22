# V180 forward-path check

The V178b direct prompt helper and stock query conversion are identical in the combined source. Its V68 verification selector requires TARGET_VERIFY and a DFLASH speculative batch, while V178b requires ordinary EXTEND. In an eligible prompt, the selector leaves its result empty, direct prefix completion runs, and native FA3 receives the dense FP8 view and adjusted lengths along with its original query, scaling, soft cap and window. In a target verify pass the prompt producer declines before reading or mutating K/V.

The independent physical-slot/double-precision reference ran separately against the V180 tree. All 11,593,728 K/V bytes, 3,072 grouped-query attention outputs and 262,144 FP8 conversions matched. It exercised cold/warm, full/sliding, padded/live/warmup and strided/aliased BF16 inputs. [Result](independent_attention.json) · [Common source fixture](../v178b_live_direct_prompt/independent_attention.py)

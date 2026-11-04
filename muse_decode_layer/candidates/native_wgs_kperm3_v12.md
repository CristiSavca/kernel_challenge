# Isolated KPERM3 decoder experiment

This is frozen decoder `80a4eaa3` with the lossless MoE layout from standalone
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Its source SHA256 is `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
All other operators, flags, routing, scratch ownership and dispatch remain intact.
The only host change outside weight packing is a unique extension name.

The standalone private job `bcb761dfca264db0` passed five explicit tests and six
checked timed cases with no flags. Its aggregate 2.389836x versus the parent's
private 2.360502x / public 2.356104x overlaps the reported confidence intervals.
Changed M32/128/256 case speedup ratios improved 2.84/3.39/3.22% against the
private parent; unchanged controls do not establish a clean causal effect.
See `experiments/iterations/20261104-moe-kperm3-official-critic.json`.

One isolated decoder check tests whether that modest signal transfers to actual
decoder routing and PDL overlap. It is not a first-place-scale hypothesis or a
public promotion. Reassociation can change floating-point output and requires
all 13 official tests plus 12 benchmark correctness checks. Inspect each load
and control; reject an aggregate gain consistent with noise or regressions.

Root and independent critic ran `tools/test_decoder_kperm3_integration.py` with
the bounded NumPy CPU environment. It checks exact unchanged AST nodes, checked
CUDA equivalence, decoder MULDEC helper mapping, and independently recovers
50,331,648 FP4 code nibbles and 3,145,728 scale bytes from actual W13/W2 packed
outputs. Static GO is not GPU validation. Only weights are permuted in prepare;
activation gathering remains timed. All stored scale bytes retain eight bits.

Upstream notices are embedded in the standalone file and this decoder candidate;
the donor and licenses remain under `third_party/public_scout/keell0renz-890809c`.

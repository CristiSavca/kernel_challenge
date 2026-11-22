#!/bin/sh
# Pass one scratch directory with images/ & humming/include/ and the native NVCC toolchain bin dir.
set -eu
root=$(cd "$1" && pwd)
bin=$2
compat=$3
for phase in w13 w2; do
  for variant in base v2; do
    case "$phase-$variant" in
      w13-base) name=GroupedDecodeGateUpKernel;;
      w13-v2) name=GroupedResidualPromptGateUpKernel;;
      w2-base) name=PromptWeightPipeline;;
      w2-v2) name=ResidualPromptWeightPipeline;;
    esac
    mkdir -p "$root/$variant"
    stem="$root/$variant/$phase"
    cp "$root/images/${name}_176.cu" "$stem.cu"
    "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math \
      --diag-suppress=39,161,174,177,940,1444 -I"$compat" -I"$root/humming/include" \
      -gencode=arch=compute_90a,code=sm_90a -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda \
      --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp \
      "$stem.cu" -o "$stem.cubin" > "$stem.log" 2>&1
    "$bin/cuobjdump" -sass "$stem.cubin" > "$stem.sass"
  done
done

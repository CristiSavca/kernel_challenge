#!/bin/sh
# Scratch contains images and the published Humming include directory.
set -eu
scratch=$1
bin=$2
compat=$3
for src in "$scratch"/images/*.cu; do
    stem=${src%.cu}
    if "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math \
        --diag-suppress=39,161,174,177,940,1444 -I"$compat" -I"$scratch/include" \
        -gencode=arch=compute_90a,code=sm_90a -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda \
        --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp \
        "$src" -o "$stem.cubin" > "$stem.log" 2>&1; then
        "$bin/cuobjdump" -sass "$stem.cubin" > "$stem.sass"
    else
        cat "$stem.log" >&2
        exit 1
    fi
done


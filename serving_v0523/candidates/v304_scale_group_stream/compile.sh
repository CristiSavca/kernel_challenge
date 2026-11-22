#!/bin/sh
set -eu
bin=$1
include=$2
compat=$3
shift 3
for src in "$@"; do
  stem=${src%.cu}
  if ! "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math \
    --diag-suppress=39,161,174,177,940,1444 -I"$compat" -I"$include" -gencode=arch=compute_90a,code=sm_90a \
    -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda \
    --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp "$src" -o "$stem.cubin" >"$stem.log" 2>&1; then
    head -c 1000 "$stem.log"
    exit 1
  fi
  "$bin/cuobjdump" -sass "$stem.cubin" >"$stem.sass"
done

#!/bin/sh
set -eu
# Run after vendor_codegen.py. Arguments: rendered .cu directory, extracted
# Humming package, CUDA-toolkit root. HOST_COMPAT is optional on newer glibc.
build=$1
vendor=$2
toolkit=$3
for source in "$build"/*.cu; do
    name=${source%.cu}
    "$toolkit/bin/nvcc" -cubin "$source" -o "$name.cubin" -std=c++17 -DHUMMING_DEBUG_KERNEL=0 \
      --ptxas-options=--register-usage-level=10 --diag-suppress=39,161,174,177,940,1444 \
      --expt-relaxed-constexpr --expt-extended-lambda -Xptxas=-v -arch=sm_90a --use_fast_math \
      -I "$vendor/include" ${HOST_COMPAT:+-I"$HOST_COMPAT"} -O3 \
      -Xcompiler=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp >"$name.log" 2>&1
    "$toolkit/bin/cuobjdump" -sass "$name.cubin" > "$name.sass"
done

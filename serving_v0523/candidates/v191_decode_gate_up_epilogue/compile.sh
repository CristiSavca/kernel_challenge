set -eu
cd /tmp/v191
bin=$HOME/nvcc126/toolchain/usr/local/cuda-12.9/bin
for file in original/*.cu fusion/*.cu both/*.cu; do
    name=${file%.cu}
    "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math --diag-suppress=39,161,174,177,940,1444 -I"$HOME/nvcc126/compat" -Iinclude -gencode=arch=compute_90a,code=sm_90a -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp "$file" -o "$name.cubin" >"$name.log" 2>&1 || { cat "$name.log"; exit 1; }
    "$bin/cuobjdump" -sass "$name.cubin" >"$name.sass"
    grep -E 'bytes stack frame|Used .* registers' "$name.log" || true
done

set -eu
cd /tmp/v199
bin=$HOME/nvcc126/toolchain/usr/local/cuda-12.9/bin
for f in rendered/*.cu; do
    stem=${f%.cu}
    if "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math --diag-suppress=39,161,174,177,940,1444 -I"$HOME/nvcc126/compat" -Ihumming/include -gencode=arch=compute_90a,code=sm_90a -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp "$f" -o "$stem.cubin" >"$stem.log" 2>&1; then
        "$bin/cuobjdump" -sass "$stem.cubin" >"$stem.sass"
    else
        sed -n '1,3p' "$stem.log" >&2
    fi
done

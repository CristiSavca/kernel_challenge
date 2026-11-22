"""Exercise the production occupancy function with real cuda-bindings structures and substituted driver calls."""
import ast
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from cuda.bindings import driver

source = Path(sys.argv[1]).read_bytes()
node = next(n for n in ast.parse(source).body if getattr(n,"name",None)=="_cluster_grid")
ns = {}
exec(compile(ast.Module([node],[]),"production","exec"), ns)
hum = ModuleType("humming")
hum.ops = SimpleNamespace(get_kernel_smem_size=lambda k:156672)
sys.modules["humming"] = hum
kernel = SimpleNamespace(kernel_id=17, kernel_name="humming", kernel_filename="image.cubin", num_threads=384)
names = ("cuModuleLoad","cuModuleGetFunction","cuFuncSetAttribute","cuOccupancyMaxActiveClusters","cuModuleUnload")
cases = []
for fault in (None, *names):
    for active in ((0,1,48,62,64,66,70) if fault is None else (64,)):
        visited = []
        def fake(name):
            def call(*args):
                visited.append(name)
                if name == "cuModuleLoad":
                    assert args==(b"image.cubin",)
                    val=111
                elif name == "cuModuleGetFunction":
                    assert args==(111,b"humming")
                    val=123
                elif name == "cuFuncSetAttribute":
                    assert args==(123,driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,156672)
                    val=None
                elif name == "cuOccupancyMaxActiveClusters":
                    func,config=args
                    assert func==123 and (config.gridDimX,config.gridDimY,config.gridDimZ)==(132,1,1)
                    assert (config.blockDimX,config.blockDimY,config.blockDimZ)==(384,1,1)
                    assert config.sharedMemBytes==156672 and config.numAttrs==1
                    a=config.attrs[0]
                    assert a.id==driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                    cluster=a.value.clusterDim
                    assert (cluster.x,cluster.y,cluster.z)==(2,1,1)
                    val=active
                else:
                    assert args==(111,)
                    val=None
                return (int(name==fault),) if val is None else (int(name==fault),val)
            return call
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(driver,name,side_effect=fake(name)))
            try:
                ctas=ns["_cluster_grid"](kernel,132)
            except (ValueError,RuntimeError):
                assert fault or not active
                ctas=None
            else:
                assert ctas==2*min(66,active) and not fault and active>0
        assert ("cuModuleUnload" in visited)==(fault!="cuModuleLoad")
        cases.append(dict(failure=fault,active=active,ctas=ctas,visits=visited))
print(json.dumps(dict(version=version("cuda-bindings"), model_src_sha256=hashlib.sha256(source).hexdigest(), cases=cases,
                      scope="Real CUDA Python launch-config and attribute memory layout; substituted low-level driver returns, no GPU."),indent=2))


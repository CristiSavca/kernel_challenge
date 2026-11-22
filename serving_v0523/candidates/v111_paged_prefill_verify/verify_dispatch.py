"""Run V110's physical-pool dispatch fixture with V106's prefill feature enabled."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

source = Path(__file__).resolve().parent.parent / "v110_verify_fp8_qkv_store/dispatch_check.py"
spec = importlib.util.spec_from_file_location("verify_fixture", source)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class Extend:
    def __eq__(self, mode):
        return not mode.is_target_verify()


class Run(fixture.Run):
    def __init__(self, *args):
        super().__init__(*args)
        self.backend._pack_swa_fp8_prefill = True
        self.ns["ForwardMode"] = SimpleNamespace(EXTEND=Extend())


if __name__ == "__main__":
    fixture.Run = Run
    fixture.main()

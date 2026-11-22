"""Exercise submission gates with fake CLI responses; never contacts the judge."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "submit_nov17_integrated_checks.py"
spec = importlib.util.spec_from_file_location("integrated_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
scratch = Path(tempfile.mkdtemp(prefix="okc-v68-runner-test-"))
now = runner.START + timedelta(minutes=1)
cases = []
commands = []


def folder(name):
    path = scratch / name
    path.mkdir()
    return path


def fake(*, left=3, reset=runner.RESET, quota_exit=0, behavior="success"):
    def execute(command, **kwargs):
        assert kwargs["capture_output"] and kwargs["text"]
        if command[1:] == ["serve", "status", "--json"]:
            return SimpleNamespace(returncode=quota_exit, stderr="", stdout=json.dumps(
                {"limits": {"checks": {"left": left}}, "resets_at": reset}))
        assert command[1:3] == ["serve", "check"]
        assert command[-1] == "--no-wait" and "submit" not in command
        commands.append(command)
        if behavior == "timeout":
            raise subprocess.TimeoutExpired(command, 120, output=b"job 1234567890abcdef\n")
        return SimpleNamespace(returncode=1 if behavior == "nonzero" else 0,
                               stdout="job 1234567890abcdef\n", stderr="")
    return execute


assert [c.name for c in runner.CANDIDATES] == ["v64", "v62", "v68"]
for candidate in runner.CANDIDATES:
    patch = runner.ROOT / "serving_v0523/candidates" / candidate.directory / "patch.diff"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == candidate.sha256
assert runner.V64.flags == runner.V68.flags == {
    "--moe-runner-backend": "humming", "--speculative-moe-runner-backend": "marlin",
    "--enable-layernorm-sp": "", "--swa-full-tokens-ratio": "0.25"}
assert runner.V62.flags == {"--swa-full-tokens-ratio": "0.25"}
cases += ["ordered exact frozen sources", "exact flags including boolean SP"]
path = folder("success")
offers = runner.decisions(path)
assert len(offers) == 3 and all(offer[2] == 16 for offer in offers)
for offer in offers:
    assert runner.submit_once(offer, okc="okc-sim", now=now, runs=path,
                              execute=fake()) == "submitted 1234567890abcdef"
assert not runner.decisions(path)
calls = len(commands)
assert runner.submit_once(offers[0], okc="okc-sim", now=now, runs=path,
                          execute=fake()) == "prior attempt exists"
assert len(commands) == calls
cases += ["three private checks only", "exclusive receipts suppress duplicates"]
for label, when in (("early", runner.START - timedelta(seconds=1)), ("late", runner.END)):
    assert runner.submit_once(offers[0], okc="okc-sim", now=when, runs=folder(label),
                              execute=lambda *a, **k: 1 / 0) == "outside submission window"
    cases.append(label + " window refused without CLI")
for index, left in enumerate((0, -1, True, "3", None)):
    assert runner.submit_once(offers[0], okc="okc-sim", now=now, runs=folder("quota" + str(index)),
                              execute=fake(left=left)) == "quota unavailable"
    cases.append("quota rejected " + repr(left))
for label, kwargs, expected in (
    ("reset", {"reset": "2026-11-17T00:00:00Z"}, "quota unavailable"),
    ("quota_failure", {"quota_exit": 1}, "quota request failed"),
):
    assert runner.submit_once(offers[0], okc="okc-sim", now=now, runs=folder(label),
                              execute=fake(**kwargs)) == expected
    cases.append(label + " rejected")
wrong_root = folder("wrong_source")
wrong_patch = wrong_root / "serving_v0523/candidates" / runner.V64.directory / "patch.diff"
wrong_patch.parent.mkdir(parents=True)
wrong_patch.write_text("not the frozen patch")
assert runner.submit_once(offers[0], okc="okc-sim", now=now, root=wrong_root,
                          runs=folder("hash_receipts"), execute=lambda *a, **k: 1 / 0
                          ) == "patch hash mismatch"
cases.append("hash mismatch refused before CLI")
for behavior in ("timeout", "nonzero"):
    path = folder(behavior)
    result = runner.submit_once(offers[0], okc="okc-sim", now=now, runs=path,
                                execute=fake(behavior=behavior))
    assert "no retry" in result
    receipt = path / (offers[0][0] + ".submit.txt")
    assert "job 1234567890abcdef" in receipt.read_text()
    calls = len(commands)
    assert runner.submit_once(offers[0], okc="okc-sim", now=now, runs=path,
                              execute=fake()) == "prior attempt exists"
    assert len(commands) == calls
    cases.append(behavior + " preserves ambiguity and blocks retry")
cases.append("no scored endpoint")
report = {
    "state": "CPU_PASS", "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    "cases": cases, "commands": commands[:3], "scratch": str(scratch), "gpu_actions": False,
}
(HERE / "nov17_integrated_runner.cpu.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"passed": len(cases), "scratch": str(scratch), "sha256": report["script_sha256"]}))

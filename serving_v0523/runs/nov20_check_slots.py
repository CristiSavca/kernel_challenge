"""Shared evidence and short-lived arbitration for the November 20 private checks."""
import fcntl
import json
import re
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from math import isfinite

import submit_nov19_decode_checks as previous

from submit_nov17_integrated_checks import Candidate, V64
import submit_nov20_v129_check as first

GRACE = first.START + timedelta(minutes=10)
LOCK = "nov20_v132_v141_arbitration.lock"
V141 = Candidate("v141", "v141_adaptive_expert_projection",
                 "dacec92848e669b47532fb53738c869f89efdfdf538a5a12415a20b98c3532a4", V64.flags)
V141_NAME = "v141_nov20_check32"
SECOND = {"per_day": 3, "used": 1, "left": 2}
THIRD = {"per_day": 3, "used": 2, "left": 1}


@contextmanager
def arbitrate(runs):
    with (runs / LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            # Closing the file releases the cross-process lock after one poll.
            yield True


def prior_attempt(runs, candidate, receipt_name, exclude_own=False):
    tag = re.compile(r"(?<![a-z0-9])" + re.escape(candidate.name) + r"(?![a-z0-9])", re.I)
    for path in sorted(runs.rglob("*")):
        if not path.is_file() or (exclude_own and path == runs / (receipt_name + ".submit.txt")):
            continue
        name = path.name
        submitted = name.endswith((".submit.txt", ".submit.json", ".submitted_at.txt"))
        if tag.search(name) and (submitted or ("check" in name or ".skip." in name) and name.endswith((".txt", ".json"))):
            return name
        if not submitted and not first.JUDGE_FILE.fullmatch(name):
            continue
        raw = path.read_text()
        if submitted:
            line, _, tail = raw.partition("\n")
            try:
                header = json.loads(line)
            except ValueError:
                header = None
            if isinstance(header, dict) and header.get("candidate") and header.get("sha256"):
                if (tag.search(str(header["candidate"])) or header["sha256"] == candidate.sha256
                        or tag.search(str(header.get("directory", "")))):
                    return name
                raw = tail
        if tag.search(raw) or candidate.sha256 in raw:
            return name
    return None


def accepted_receipt(runs, candidate, users, name, window=None):
    path = runs / (name + ".submit.txt")
    if (runs / (name + ".skip.json")).exists():
        return None
    try:
        lines = path.read_text().splitlines()
        if not lines or lines[-1] != "CLI exit code: 0" or sum(line.startswith("CLI exit code:") for line in lines) != 1:
            return None
        meta = json.loads(lines[0])
        if (not isinstance(meta, dict) or type(meta.get("users")) is not int
                or (meta.get("candidate"), meta.get("sha256"), meta["users"], meta.get("flags"))
                != (candidate.name, candidate.sha256, users, candidate.flags)):
            return None
        if window is None:
            if candidate == first.V129:
                window = first.START, first.END
            elif candidate == previous.V120b:
                window = previous.START, previous.END
        if window is not None:
            when = datetime.fromisoformat(meta["attempted_at"])
            if not when.tzinfo or not window[0] <= when < window[1]:
                return None
        body = "\n".join(lines[1:])
        matches = re.findall(r"(\S+)/patch\.diff as a check run at (\d+) users?\b", body)
        if len(matches) != 1 or (Path(matches[0][0]).name, int(matches[0][1])) != (candidate.directory, users):
            return None
        jobs = set(re.findall(r"\bjob ([0-9a-f]{16})\b", body))
        confirmed = re.findall(
            r"^(?:✓[ \t]+)?submitted[ \t]+(\S+)/patch\.diff as a check run at (\d+) users?\b"
            r"[^\n]*?\bjob ([0-9a-f]{16})\b", body, re.MULTILINE,
        )
        if len(confirmed) != 1 or jobs != {confirmed[0][2]}:
            return None
        return confirmed[0][2]
    except (ValueError, TypeError, KeyError, FileNotFoundError):
        return None


def dated_receipt(runs, req):
    candidate, users, name, begin, end = req
    return accepted_receipt(runs, candidate, users, name, (begin, end))


def bound_receipt(runs, req):
    """Bind a specific official job to one CLI acceptance, including its declared users."""
    job = dated_receipt(runs, req)
    if job is None:
        return None
    canonical = runs / (req[2] + ".submit.txt")
    for path in runs.rglob("*"):
        if (path == canonical or not path.is_file()
                or not path.name.endswith((".submit.txt", ".submit.json", ".submitted_at.txt"))):
            continue
        text = path.read_text()
        first_line, _, body = text.partition("\n")
        try:
            header = json.loads(first_line)
        except ValueError:
            header = None
        if isinstance(header, dict) and header.get("candidate") and header.get("sha256"):
            text = body
        # A separate job can normally reuse these exact patch bytes at another load or date.
        if re.search(r"\bjob " + job + r"\b", text):
            return None
    return job


def official_state(req, job, okc, execute, *, runs=None):
    candidate, users, _, begin, end = req
    response = execute([okc, "serve", "status", job, "--json"], capture_output=True, text=True, timeout=60)
    if response.returncode:
        return None
    data = json.loads(response.stdout)
    if not isinstance(data, dict):
        return None
    stamp = data.get("created_at")
    if (data.get("serving_flags") != candidate.flags or data.get("duplicate_of") is not None
            or type(stamp) not in (int, float) or not isfinite(stamp) or not begin.timestamp() <= stamp < end.timestamp()):
        return None
    pending = data.get("status") in ("queued", "running")
    reported = data.get("users")
    if reported is not None and (type(reported) is not int or reported != users):
        return None
    if pending:
        # The judge omits users on queued checks. Its job ID binds the dated
        # accepted CLI receipt, which records the requested patch, flags and load.
        if data.get("result") is not None:
            return None
        if reported is None and (runs is None or bound_receipt(runs, req) != job):
            return None
    elif data.get("status") != "done" and reported is None:
        return None
    result = data.get("result")
    if isinstance(result, dict):
        loads = result.get("concurrency")
        if loads is not None and (not isinstance(loads, list) or len(loads) != 1
                                  or type(loads[0]) is not int or loads[0] != users):
            return None
        if data.get("status") != "done" and result.get("passed") is True:
            return None
    state, _ = previous.parent_choice(candidate, job, users, "passed", "failed", okc, lambda *a, **kw: response, pending="pending")
    return state

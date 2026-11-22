"""Bind one separately accepted scored run before using the final private-check allocation."""
import hashlib
import json
import math
import re
from datetime import datetime

SUFFIXES=(".submit.txt",".submit.json",".submitted_at.txt")
SUBMITTED=re.compile(r"^(?:✓\s+)?submitted\s+(\S+/patch\.diff) to serving\s+\d+ files? changed\s+· job ([0-9a-f]{16})\s*$")
SHA256=re.compile(r"[0-9a-f]{64}")
TEAM="CristiSavca"


def header(raw):
    try:
        value=json.loads(raw.partition(b"\n")[0])
    except (ValueError, UnicodeError):
        return {}
    return value if isinstance(value,dict) else {}


def window_time(meta):
    stamps=[datetime.fromisoformat(meta[k]) for k in ("attempted_at","utc") if k in meta]
    if not stamps or any(t.tzinfo is None or t!=stamps[0] for t in stamps):
        raise ValueError("unknown receipt time")
    return stamps[0]


def inventory(runs,own):
    # The worker creates its own durable claim only after the decisive reads.
    # All other receipt paths and contents remain bound through that creation.
    return {p:p.read_bytes() for p in runs.rglob("*") if p!=own and p.name.lower().endswith(SUFFIXES) and p.is_file()}


def local_score(runs,root,records,allowed,begin,end):
    scores=[]
    for path,raw in records.items():
        if path in allowed:
            continue
        meta=header(raw)
        try:
            stamp=window_time(meta)
        except (KeyError,TypeError,ValueError,OverflowError):
            stamp=None
        if not ("nov20" in path.name.lower() or (stamp is not None and begin<=stamp<end)
                or (stamp is None and begin.timestamp()<=path.stat().st_mtime<end.timestamp())):
            continue
        if stamp is None or not begin<=stamp<end or not path.name.endswith(".submit.txt"):
            return None
        try:
            lines=raw.decode().splitlines()
        except UnicodeDecodeError:
            return None
        accepted=[line for line in lines[1:] if re.match(r"^(?:✓\s+)?submitted\b",line)]
        if (not lines or lines[-1]!="CLI exit code: 0" or
                sum(line.startswith("CLI exit code:") for line in lines)!=1 or len(accepted)!=1):
            return None
        match=SUBMITTED.fullmatch(accepted[0])
        if not match or any("as a check run" in line for line in lines[1:]):
            return None
        path_text,job=match.groups()
        next_ids=re.findall(r"\bokc serve status ([0-9a-f]{16})\b","\n".join(lines[1:]))
        if any(other!=job for other in next_ids):
            return None
        patch=(root/path_text).resolve()
        candidates=(root/"serving_v0523/candidates").resolve()
        if patch.parent.parent!=candidates or patch.name!="patch.diff" or not patch.is_file():
            return None
        digest=meta.get("sha256")
        flags=meta.get("flags")
        if (not isinstance(digest,str) or not SHA256.fullmatch(digest) or
                hashlib.sha256(patch.read_bytes()).hexdigest()!=digest or
                not isinstance(flags,dict) or any(not isinstance(k,str) or not k.startswith("--") or not isinstance(v,str)
                                                  for k,v in flags.items()) or
                meta.get("mode","run") not in ("run","scored") or meta.get("users") is not None or
                ("candidate" in meta and meta["candidate"]!=patch.parent.name.split("_",1)[0]) or
                any(k in meta and meta[k]!=job for k in ("job","job_id","id"))):
            return None
        # A renamed acceptance or two claims for the same job cannot supply a binding.
        pattern=re.compile(rb"\bjob\s+"+job.encode()+rb"\b")
        if any(p!=path and pattern.search(content) for p,content in records.items()):
            return None
        if {s.decode() for s in re.findall(rb"\bjob\s+([0-9a-f]{16})\b",raw)}!={job}:
            return None
        scores.append((job,stamp,flags,digest))
    return scores if len(scores)<=1 else None


def confirmed(scores,okc,execute,now):
    if scores is None:
        return False
    for job,stamp,flags,digest in scores:
        resp=execute([okc,"serve","status",job,"--json"],capture_output=True,text=True,timeout=60)
        if resp.returncode:
            return False
        try:
            d=json.loads(resp.stdout)
        except (TypeError, ValueError, UnicodeError):
            return False
        if not isinstance(d,dict):
            return False
        created=d.get("created_at")
        if ((d.get("id"),d.get("board"),d.get("mode"),d.get("team"))!=(job,"serving","run",TEAM)
                or type(created) not in (int,float) or not math.isfinite(created)
                or not stamp.timestamp()<=created<=now().timestamp() or d.get("serving_flags")!=flags
                or d.get("duplicate_of") is not None or d.get("refunded",False) is not False
                or d.get("users") is not None):
            return False
        result=d.get("result")
        if d.get("status") in ("queued","running"):
            if result is not None:
                return False
        elif d.get("status")=="done" and isinstance(result,dict):
            if not isinstance(result.get("patch"),dict) or result["patch"].get("sha256")!=digest or result.get("serving_flags")!=flags:
                return False
        else:
            return False
    return True

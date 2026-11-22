#!/usr/bin/env python3
"""Closed-loop multi-turn load for an SGLang server, shaped like the serving judge's workload.

Every repetition starts `users` conversations at the same moment. A conversation is `turns` chat requests; each
appends the reply and a new user message, so later turns reuse the server's prefix cache. Requests are greedy,
streamed, and ask for a fixed number of output tokens (ignore_eos), like the judge's.

Known from the judge's result JSON: 13 turns; first turn 74,160 tokens, 753 new tokens a turn and 220 output tokens,
all means with "varied" lengths; POST /v1/chat/completions; tokens/s per GPU counts prompt plus output tokens.
Assumed here: the length distributions (uniform around those means), the text (source files from the image) and a
cache flush between repetitions. Calibrate against a judged build before trusting absolute numbers.

  okc_node_loadgen.py run --url http://127.0.0.1:30000 --tokenizer /models/X --corpus /sgl-workspace/sglang \
      --loads 1,4,16,32 --reps 3,3,3,1 --out result.json
  okc_node_loadgen.py compare base.json new.json
  okc_node_loadgen.py mock --port 30000        # a fake server, to test this script without a GPU
"""
import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".rs", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".toml", ".yaml", ".txt"}


# ---- workload ---------------------------------------------------------------------------------------------------

class Tokens:
    """Token pool cut from a text corpus; `mock` counts whitespace-separated words instead."""

    def __init__(self, tokenizer: str, corpus: str, cache: str | None, pool_tokens: int):
        self.mock = tokenizer == "mock"
        self.tok = None
        if not self.mock:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(tokenizer)
        if cache and Path(cache).exists():
            self.pool = json.loads(Path(cache).read_text())
            return
        self.pool = []
        files = sorted(p for p in Path(corpus).rglob("*") if p.suffix in TEXT_SUFFIXES and p.is_file())
        for path in files:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if len(text) < 400:
                continue
            self.pool.extend(text.split() if self.mock else self.tok(text, add_special_tokens=False).input_ids)
            if len(self.pool) >= pool_tokens:
                break
        if len(self.pool) < 200_000:
            raise SystemExit(f"corpus {corpus} gave only {len(self.pool)} tokens")
        if cache:
            Path(cache).write_text(json.dumps(self.pool))

    def text(self, rng: random.Random, n: int) -> str:
        start = rng.randrange(0, len(self.pool) - n)
        window = self.pool[start:start + n]
        return " ".join(window) if self.mock else self.tok.decode(window)


def conversation_lengths(rng: random.Random, a) -> tuple[int, list[int], list[int]]:
    first = rng.randint(a.first_min, a.first_max)
    new = [rng.randint(round(0.4 * a.new_mean), round(1.6 * a.new_mean)) for _ in range(a.turns - 1)]
    out = [rng.randint(a.out_min, a.out_max) for _ in range(a.turns)]
    room = a.context - a.context_margin - sum(new) - sum(out)
    return min(first, room), new, out


def build_conversation(tokens: Tokens, a, load: int, rep: int, user: int) -> dict:
    rng = random.Random(f"{a.seed}/{load}/{rep}/{user}")
    first, new, out = conversation_lengths(rng, a)
    tag = f"[session {a.seed}-{load}-{rep}-{user}] "  # a distinct first token run, so conversations share no prefix
    return {"first": tag + tokens.text(rng, first), "new": [tokens.text(rng, n) for n in new], "out": out}


# ---- client -----------------------------------------------------------------------------------------------------

def find_key(node, key):
    if isinstance(node, dict):
        if node.get(key):
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, key)
            if found:
                return found
    return None


async def chat(session, a, model: str, messages: list, n_out: int) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": n_out, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = None
    content, reasoning, usage, spec = [], [], {}, None
    async with session.post(a.url + "/v1/chat/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece, thought = delta.get("content") or "", delta.get("reasoning_content") or ""
                if first is None and (piece or thought):
                    first = time.perf_counter()
                content.append(piece)
                reasoning.append(thought)
            usage = chunk.get("usage") or usage
            spec = find_key(chunk, "spec_tokens_details") or spec
    t1 = time.perf_counter()
    if isinstance(spec, list):
        spec = spec[0]
    details = usage.get("prompt_tokens_details") or {}
    return {"text": "".join(content) or "".join(reasoning), "t0": t0, "ttft": (first or t1) - t0, "dur": t1 - t0,
            "prompt": usage.get("prompt_tokens", 0), "cached": details.get("cached_tokens") or 0,
            "completion": usage.get("completion_tokens", 0), "asked": n_out,
            "verify": (spec or {}).get("spec_verify_ct", 0)}


async def run_conversation(session, a, model, conv, meta, records, origin, errors):
    messages = [{"role": "user", "content": conv["first"]}]
    for turn in range(a.turns):
        try:
            got = await chat(session, a, model, messages, conv["out"][turn])
        except Exception as exc:  # one failed request ends its conversation, not the rung
            errors.append({**meta, "turn": turn, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            return
        text = got.pop("text")
        got.update(meta, turn=turn, t0=got["t0"] - origin, reply=text[:400])
        records.append(got)
        if turn + 1 < a.turns:
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": conv["new"][turn]}]


async def run_load(a, tokens: Tokens) -> dict:
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.read_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    result = {"shape": {k: getattr(a, k) for k in ("turns", "first_min", "first_max", "new_mean", "out_min", "out_max",
                                                      "context", "seed", "flush")}, "gpus": a.gpus, "rungs": []}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, read_bufsize=1 << 22) as session:
        async with session.get(a.url + "/v1/models") as resp:
            model = (await resp.json())["data"][0]["id"]
        for load, reps in zip(a.loads, a.reps):
            records, walls, errors = [], [], []
            for rep in range(reps):
                convs = [build_conversation(tokens, a, load, rep, user) for user in range(load)]
                if a.flush:
                    async with session.post(a.url + "/flush_cache") as resp:
                        await resp.read()
                    await asyncio.sleep(2)
                origin = time.perf_counter()
                await asyncio.gather(*(run_conversation(session, a, model, conv, {"load": load, "rep": rep, "user": u},
                                                         records, origin, errors) for u, conv in enumerate(convs)))
                walls.append(time.perf_counter() - origin)
                print(f"load {load} rep {rep}: {walls[-1]:.1f} s", flush=True)
            rung = summarize(records, walls, a.gpus) if records else {}
            rung |= {"concurrency": load, "errors": errors[:8], "failed_conversations": len(errors)}
            result["rungs"].append(rung)
            result.setdefault("requests", []).extend(records)
            if a.out:
                Path(a.out).write_text(json.dumps(result))
            print(json.dumps({k: v for k, v in rung.items() if k != "rep_tok_s_per_gpu"}), flush=True)
    return result


# ---- metrics ----------------------------------------------------------------------------------------------------

def quantile(values: list, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list, walls: list, gpus: int) -> dict:
    per_rep = []
    for rep, wall in enumerate(walls):
        total = sum(r["prompt"] + r["completion"] for r in records if r["rep"] == rep)
        per_rep.append(total / wall / gpus)
    mean_rate = sum(per_rep) / len(per_rep)
    out = sum(r["completion"] for r in records)
    prompt = sum(r["prompt"] for r in records)
    verify = sum(r["verify"] for r in records)
    decode = [r["completion"] / (r["dur"] - r["ttft"]) for r in records if r["dur"] > r["ttft"]]
    return {
        "reps": len(walls), "requests": len(records), "wall_s": [round(w, 2) for w in walls],
        "tok_s_per_gpu": round(mean_rate, 1), "rep_tok_s_per_gpu": [round(x, 1) for x in per_rep],
        "spread": round((max(per_rep) - min(per_rep)) / mean_rate, 4),
        "tok_s_per_user_e2e": round(out / sum(r["dur"] for r in records), 2),
        "tok_s_per_user_e2e_p50": round(quantile([r["completion"] / r["dur"] for r in records], 0.5), 2),
        "tok_s_per_user_p50": round(quantile(decode, 0.5), 2),
        "ttft_first_p90_s": round(quantile([r["ttft"] for r in records if r["turn"] == 0], 0.9), 3),
        "ttft_later_p90_s": round(quantile([r["ttft"] for r in records if r["turn"] > 0], 0.9), 3),
        "first_turn_tokens_mean": round(sum(r["prompt"] for r in records if r["turn"] == 0)
                                        / max(1, sum(1 for r in records if r["turn"] == 0))),
        "cache_hit": round(sum(r["cached"] for r in records) / max(1, prompt), 4),
        "acceptance": round(out / verify, 2) if verify else None,
        "short": sum(1 for r in records if r["completion"] != r["asked"]),
    }


def frontier(rungs: list, speed: float) -> float:
    """Tokens/s per GPU this build delivers at a per-user speed: linear between its rungs in log speed, clamped."""
    points = sorted((r["tok_s_per_user_e2e"], r["tok_s_per_gpu"]) for r in rungs)
    if speed <= points[0][0]:
        return points[0][1]
    if speed >= points[-1][0]:
        return points[-1][1]
    for (s0, t0), (s1, t1) in zip(points, points[1:]):
        if s0 <= speed <= s1:
            return t0 + (t1 - t0) * math.log(speed / s0) / math.log(s1 / s0)
    return points[-1][1]


def compare(base: dict, new: dict) -> dict:
    rows = []
    mine = {r["concurrency"]: r for r in new["rungs"]}
    for b in base["rungs"]:
        n = mine.get(b["concurrency"])
        if n is None:
            continue
        rows.append({"concurrency": b["concurrency"], "base_tok_s_per_gpu": b["tok_s_per_gpu"],
                     "new_tok_s_per_gpu": n["tok_s_per_gpu"], "same_load_ratio": n["tok_s_per_gpu"] / b["tok_s_per_gpu"],
                     "at_base_speed_ratio": frontier(new["rungs"], b["tok_s_per_user_e2e"]) / b["tok_s_per_gpu"]})
    geo = lambda key: math.exp(sum(math.log(r[key]) for r in rows) / len(rows))  # noqa: E731
    out = {"rows": rows, "same_load_geomean": geo("same_load_ratio"), "at_base_speed_geomean": geo("at_base_speed_ratio")}
    # First turns have identical prompts in both runs, so their greedy replies should agree.
    key = lambda r: (r["load"], r["rep"], r["user"])  # noqa: E731
    mine = {key(r): r.get("reply") for r in new.get("requests", []) if r["turn"] == 0}
    pairs = [(r.get("reply"), mine[key(r)]) for r in base.get("requests", []) if r["turn"] == 0 and mine.get(key(r))]
    pairs = [(a, b) for a, b in pairs if a]
    if pairs:
        prefix = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))  # noqa: E731
        out["first_turn_replies"] = {"compared": len(pairs), "identical": sum(a == b for a, b in pairs) / len(pairs),
                                     "mean_common_prefix": sum(prefix(a, b) / max(1, len(a)) for a, b in pairs) / len(pairs)}
    return out


# ---- mock server ------------------------------------------------------------------------------------------------

def mock_app(prefill_tok_s: float, step_s: float, accept: int):
    """One GPU queue: prefill of uncached tokens is serial, decode steps are shared by all running requests."""
    from aiohttp import web
    gpu = asyncio.Lock()
    seen: dict = {}

    async def chat_handler(request):
        body = await request.json()
        words = [len(m["content"].split()) for m in body["messages"]]
        key = body["messages"][0]["content"][:64]
        prompt, cached = sum(words), seen.get(key, 0)
        cached = min(cached, prompt)
        async with gpu:
            await asyncio.sleep((prompt - cached) / prefill_tok_s)
        n = body["max_tokens"]
        seen[key] = prompt + n
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        sent = 0
        while sent < n:
            take = min(accept, n - sent)
            async with gpu:
                await asyncio.sleep(step_s)
            sent += take
            chunk = {"choices": [{"delta": {"content": "w " * take}}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        usage = {"prompt_tokens": prompt, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}}
        final = {"choices": [], "usage": usage, "sglext": {"spec_tokens_details": {"spec_verify_ct": math.ceil(n / accept)}}}
        await resp.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")
        return resp

    async def models(_):
        return web.json_response({"data": [{"id": "mock"}]})

    async def flush(_):
        seen.clear()
        return web.Response(text="ok")

    app = web.Application(client_max_size=1 << 28)
    app.add_routes([web.post("/v1/chat/completions", chat_handler), web.get("/v1/models", models),
                    web.post("/flush_cache", flush), web.get("/health", flush)])
    return app


# ---- cli --------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--url", default="http://127.0.0.1:30000")
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--pool-cache")
    run.add_argument("--pool-tokens", type=int, default=6_000_000)
    run.add_argument("--loads", default="1,4,16,32")
    run.add_argument("--reps", default="3,3,3,1")
    run.add_argument("--gpus", type=int, default=8)
    run.add_argument("--turns", type=int, default=13)
    run.add_argument("--first-min", type=int, default=38_320)
    run.add_argument("--first-max", type=int, default=110_000)
    run.add_argument("--new-mean", type=int, default=753)
    run.add_argument("--out-min", type=int, default=40)
    run.add_argument("--out-max", type=int, default=400)
    run.add_argument("--context", type=int, default=131_072)
    run.add_argument("--context-margin", type=int, default=6_000)  # chat template and re-tokenisation drift
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--no-flush", dest="flush", action="store_false")
    run.add_argument("--read-timeout", type=float, default=1800)
    run.add_argument("--out")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base")
    cmp_.add_argument("new")
    mock = sub.add_parser("mock")
    mock.add_argument("--port", type=int, default=30000)
    mock.add_argument("--prefill-tok-s", type=float, default=14_060)
    mock.add_argument("--step-ms", type=float, default=17.3)
    mock.add_argument("--accept", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "compare":
        print(json.dumps(compare(json.loads(Path(a.base).read_text()), json.loads(Path(a.new).read_text())), indent=1))
    elif a.cmd == "mock":
        from aiohttp import web
        web.run_app(mock_app(a.prefill_tok_s, a.step_ms / 1e3, a.accept), host="127.0.0.1", port=a.port, print=None)
    else:
        a.loads = [int(x) for x in a.loads.split(",")]
        a.reps = [int(x) for x in a.reps.split(",")]
        tokens = Tokens(a.tokenizer, a.corpus, a.pool_cache, a.pool_tokens)
        result = asyncio.run(run_load(a, tokens))
        if a.out:
            Path(a.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()

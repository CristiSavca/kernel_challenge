"""Replay the V205 four-slot producer/consumer protocol across distinct expert jobs.

    python3 independent_multijob.py [path/to/humming-0.1.12-x86_64.whl]

Without a wheel argument, the script reads the published wheel into memory.
"""
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from random import Random
import sys
from urllib.request import urlopen
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057"
    "b8afbaa21241c63342b5969bb74/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA = "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"


def bind_sources():
    manifest = json.loads((HERE / "validation.json").read_text())
    patch = (HERE / "patch.diff").read_bytes()
    assert sha256(patch).hexdigest() == manifest["patch_sha256"]
    assert sha256((HERE / "delta.diff").read_bytes()).hexdigest() == manifest["delta_sha256"]
    wheel = Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1 else urlopen(WHEEL_URL, timeout=15).read()
    assert sha256(wheel).hexdigest() == WHEEL_SHA
    with ZipFile(BytesIO(wheel)) as archive:
        kernel = archive.read("humming/include/humming/kernel/humming_ws.cuh").decode()
        pipeline = archive.read("humming/include/humming/memory/g2s_pipeline.cuh").decode()
        s2r = archive.read("humming/include/humming/memory/s2r_pipeline.cuh").decode()
    loop_sha = sha256(kernel.encode()).hexdigest()
    assert loop_sha.encode() in patch
    assert kernel.index("auto producer = ProducerPipeline(ctx)") < kernel.index("while (scheduler.get_next_block())")
    assert kernel.index("auto consumer = ConsumerPipeline(ctx)") < kernel.rindex("while (scheduler.get_next_block())")
    assert kernel.index("consumer.wait_stage<true>(kNumStages)") < kernel.index("s2r_pipe.load_stage_iter<true>(0, 0)")
    assert "load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages)" in kernel
    assert "stage_id = (stage_id + iter_id / Ctx::kWarpIters) % kNumStages" in s2r
    assert "phases[stage_id] ^= 1;" in pipeline and "mbarrier_arrive(&smem.math_mbar[stage_id])" in pipeline
    assert b"if (warp_iter_id + 1 < Ctx::kWarpIters || slice_iters > 1)" in patch
    assert b"if (warp_iter_id + 1 == Ctx::kWarpIters) consumer.arrive(stage_id)" in patch
    assert b"wgmma_wait<0>();" in patch
    return manifest["patch_sha256"], loop_sha


def instructions(jobs):
    producer = []
    consumers = [[] for _ in range(4)]
    for job, count in enumerate(jobs):
        producer.append(("previous_epilogue", job, 0))
        producer.extend(("publish", job, tile) for tile in range(min(3, count)))
        for tile in range(count):
            producer.append(("acknowledge", job, tile))
            if tile + 3 < count:
                producer.append(("publish", job, tile + 3))
        for operations in consumers:
            operations.extend([("wait", job, 0), ("s2r", job, 0)])
            for tile in range(count):
                operations.extend([("first_pair", job, tile), ("third", job, tile)])
                if tile + 1 < count:
                    operations.extend([("wait", job, tile + 1), ("s2r", job, tile + 1)])
                operations.extend([("drain", job, tile), ("release", job, tile)])
            operations.append(("epilogue", job, count))
    return [producer, *consumers]


def replay(jobs, random):
    streams = instructions(jobs)
    positions = [0] * 5
    slots = [None] * 4
    epochs = [0] * 4
    observed = [[0] * 4 for _ in range(4)]
    held = [None] * 4
    releases = {}
    fences = {-1: set(range(4))}
    events = 0
    while any(position < len(stream) for stream, position in zip(streams, positions)):
        eligible = []
        for index, (stream, pos) in enumerate(zip(streams, positions)):
            if pos == len(stream):
                continue
            op, job, tile = stream[pos]
            if op == "previous_epilogue" and len(fences.get(job - 1, ())) != 4:
                continue
            if op == "acknowledge" and len(releases.get((job, tile), ())) != 4:
                continue
            if op == "wait" and slots[tile % 4] != (job, tile):
                continue
            eligible.append(index)
        assert eligible, (jobs, "deadlock", positions)
        index = random.choice(eligible)
        op, job, tile = streams[index][positions[index]]
        positions[index] += 1
        warpgroup = index - 1
        token = (job, tile)
        slot = tile % 4
        if op == "publish":
            previous = slots[slot]
            assert previous is None or (len(releases.get(previous, ())) == 4 and previous not in held)
            slots[slot] = token
            epochs[slot] += 1
        if op == "wait":
            observed[warpgroup][slot] += 1
            assert observed[warpgroup][slot] == epochs[slot]
        if op in ("s2r", "first_pair", "third", "drain"):
            assert slots[slot] == token
        if op == "third":
            held[warpgroup] = token
        if op == "drain":
            assert held[warpgroup] == token
            held[warpgroup] = None
        if op == "release":
            assert held[warpgroup] is None
            releases.setdefault(token, set()).add(warpgroup)
        if op == "epilogue":
            fences.setdefault(job, set()).add(warpgroup)
        events += 1
    return events


def check_images():
    machine = json.loads((HERE / "sm90a.json").read_text())["kernels"]
    archive = json.loads((HERE / "cleanup.json").read_text())["guest"]["cubin_sha256"]
    count = 0
    for name, images in machine.items():
        for directory, kind in (("paired", "original"), ("paired_twice", "candidate")):
            image = images[kind]
            assert archive[f"{directory}/{name}.cubin"] == image["cubin_sha256"]
            assert image["registers"] == 96 and image["stack_spill_store_load_bytes"] == [0, 0, 0]
            count += 1
        assert images["candidate"]["group_sizes"] == [4] * 8
    return count


if __name__ == "__main__":
    patch_sha, loop_sha = bind_sources()
    jobs = list(product([1, 2, 3, 4, 5, 6, 7, 8, 16], repeat=2))
    jobs.extend([(16,) * 16, (4,) * 17, (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33)])
    random = Random(205)
    events = sum(replay(shape, random) for shape in jobs for _ in range(9))
    print(json.dumps(dict(
        command="python3 independent_multijob.py",
        physical_hardware=False,
        patch_sha256=patch_sha,
        published_wheel_sha256=WHEEL_SHA,
        published_loop_sha256=loop_sha,
        seed=205,
        math_warpgroups=4,
        job_sequences=len(jobs),
        schedules=len(jobs) * 9,
        interleaved_events=events,
        bound_sm90a_images=check_images(),
    ), indent=2))

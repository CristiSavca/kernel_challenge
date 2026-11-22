"""Count the memory operations and graph geometry for V261; no timing assumptions."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parent / "v248_shared_verify_raster/independent_reach.json"
PROFILE = HERE.parent / "v213_live_decode_expert_reduction/physical.json"


def geometry(slots, local):
    # Count in-range original global IDs and a resident int32 expert map. Invalid
    # global IDs suppress map reads. The same full metadata plan runs either way.
    assert 0 <= local <= slots <= 256
    return dict(slots=slots, live_rows=local, metadata_programs=1, old_gather_programs=min(slots,128),
                new_route_programs=slots, scalar_id_lanes=slots, vector_id_lanes=local*slots,
                extra_logical_id_and_map_bytes_int32=8*slots*(1+local),
                extra_logical_id_and_map_bytes_int64_ids_int32_map=12*slots*(1+local),
                unique_id_and_map_bytes_upper_bound_int32=4*(slots+min(256,slots)),
                old_row_key_and_first_offset_read_bytes=4*(local+min(slots,128)),
                common_hidden_read_write_bytes=2*2*4096*local)


def main():
    receipts = {p.name:sha256(p.read_bytes()).hexdigest() for p in (PROFILE,SAMPLES,HERE/"patch.diff")}
    observed = json.loads(PROFILE.read_bytes())["cases"]
    profile = next(r for r in observed if r["rows"]==32)
    assert profile["local_routes"]==39
    archive = json.loads(SAMPLES.read_bytes())
    per_running = []
    for row in archive["by_observed_running_count"].values():
        b = row["captured_capacity"]
        per_running.append(dict(ending_running=row["running"],samples=row["sample_count"],captured_bs=b,
                                pure_decode_slots=8*b,pure_decode_eligible=8*b<=256,
                                dflash_verify_slots=8*8*b,dflash_verify_eligible=8*8*b<=256))
    result = dict(inputs=receipts, eligible_width=[1024,4096],slots_max=256,layers=62,
                  launches_per_rank_staging_across_eligible_target_forward=[124,62],
                  snapshot_total=sum(row["samples"] for row in per_running),
                  width8_verify_eligible_snapshots=sum(row["samples"] for row in per_running if row["dflash_verify_eligible"]),
                  pure_decode_eligible_snapshots=sum(row["samples"] for row in per_running if row["pure_decode_eligible"]),
                  snapshot_counts_by_running=per_running,
                  uniform_balanced=[geometry(s,s//8) for s in (8,32,64,128,192,256)],
                  observed_H200_standalone_histogram=geometry(8*profile["rows"],profile["local_routes"]),
                  worst_case=geometry(256,256),
                  limits="Logical metadata access sizes exclude the common single-CTA plan, instructions, cache effects, transaction granularity and timing. Scheduler counts indicate ending geometry, not a batch census.")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

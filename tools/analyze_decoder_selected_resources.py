#!/usr/bin/env python3
"""Offline parser for the frozen selected-resource check. Never polls or promotes.

Only source-pinned, retained diagnostic records are decoded. Profile associations
are compatibility sets, never proof that a first-observation marker was a timed
benchmark invocation. No candidate import, subprocess, network or CUDA operation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
JOB = "c0b4201c5ad345d1"
SOURCE = "e444f46f945e03bb42dab8d80ed4869f8a7da6464f845ac09ac3cf8ea19473fd"
SK2_SCHEMA = "SK2R_SCHEMA t=MT,TOKR,NT,NWR,WSK,UPW,NS,COMP,HL,NACC,CL,OutBytes,PK,FUSE,BND x=M,N,K,pmax,UV,PDL,threads,grid,dyn a=regs,local,static,maxdyn,maxthreads occ=blocks_per_SM,SMs dev=tensor,current err=device,SMs,attrs,occ CLgt1_block_only"
MOE_SCHEMA = "MOER_SCHEMA f=NW,NS,WS,TY,SP,MUL,DQ,PR,FP,BK,SC,FC,L2,CP x=M,B,G,P,th,dyn a=R,L,S,MD,MT,O,SM d=tensor,current e=D,S,A,O g=last_arm"
T_FIELDS = "MT TOKR NT NWR WSK UPW NS COMP HL NACC CL OutBytes PK FUSE BND".split()
X_FIELDS = "M N K pmax UV PDL threads grid dynamic_shared_bytes".split()
A_FIELDS = "registers_per_thread local_bytes_per_thread static_shared_bytes max_dynamic_shared_bytes max_threads_per_block".split()
F_FIELDS = "NW NS WS TINY SPLIT MULDEC DQ7 PRMTSC FINPF WSBLK SKSC2 FTSC2 L2H CPF".split()
MOE_X = "first_M BIG first_grid first_PDL threads dynamic_shared_bytes".split()
ERROR_FIELDS = "device SMs attributes occupancy".split()
INT = r"-?\d{1,20}"
CSV = rf"{INT}(?:,{INT})*"
SK2_RE = re.compile(rf"SK2R g=({INT}) t=({CSV}) x=({CSV}) d=({INT}) a=({CSV}) occ=({CSV}) dev=({CSV}) err=({CSV})")
MOE_RE = re.compile(rf"MOER g=({INT}) f=({CSV}) x=({CSV}) a=({CSV}) d=({CSV}) e=({CSV})")
ROW_RE = re.compile(r"^(.*?)\s+(\d+(?:\.\d+)?)\s+[µu]s\s+([\d.]+)%\s+(\d+)\s*$")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def named(csv, fields):
    values = [int(x) for x in csv.split(",")]
    if len(values) != len(fields):
        raise ValueError(f"expected {len(fields)} fields, got {len(values)}")
    return dict(zip(fields, values))


def resource_status(record, a, errors, devices, blocks, sms):
    issues = record["issues"]
    if errors["device"] < 0 or errors["attributes"] < 0:
        issues.append("queried_API_has_invalid_status")
    for field, dependency in (("SMs", "device"), ("occupancy", "attributes")):
        if (errors[dependency] != 0 and errors[field] != -1) or (errors[dependency] == 0 and errors[field] < 0):
            issues.append("API_dependency_status_inconsistent:" + field)
    if errors["attributes"] == 0:
        if min(a.values()) < 0 or a["max_threads_per_block"] <= 0:
            issues.append("successful_attribute_query_has_invalid_values")
        if record["launch"]["dynamic_shared_bytes"] > a["max_dynamic_shared_bytes"]:
            issues.append("launch_dynamic_exceeds_reported_limit")
        if record["launch"]["threads"] > a["max_threads_per_block"]:
            issues.append("launch_threads_exceed_reported_limit")
    elif any(v != -1 for v in a.values()):
        issues.append("failed_attribute_query_has_non_sentinel_values")
    if (errors["SMs"] == 0 and sms <= 0) or (errors["SMs"] != 0 and sms != -1):
        issues.append("SM_count_status_inconsistent")
    if (errors["occupancy"] == 0 and blocks < 0) or (errors["occupancy"] != 0 and blocks != -1):
        issues.append("block_capacity_status_inconsistent")
    if errors["device"] == 0 and (devices[0] < 0 or devices[0] != devices[1]):
        issues.append("tensor_and_current_device_mismatch")
    if errors["device"] != 0 and devices[1] != -1:
        issues.append("failed_device_query_has_non_sentinel_value")
    record.update(attributes=a, query_status=errors, devices={"tensor": devices[0], "current": devices[1]},
                  blocks_per_SM=blocks, SMs=sms)
    record["all_queries_succeeded"] = all(v == 0 for v in errors.values())
    record["resource_record_usable"] = record["all_queries_succeeded"] and not issues
    record["calculated_device_block_capacity"] = blocks * sms if record["resource_record_usable"] else None


def decode_sk2(match, line_number):
    g, t, x, key_device, a, occ, dev, err = match.groups()
    t, x, a = named(t, T_FIELDS), named(x, X_FIELDS), named(a, A_FIELDS)
    occ, dev, err = named(occ, ("blocks", "sms")), named(dev, ("tensor", "current")), named(err, ERROR_FIELDS)
    record = {"id": f"SK2:{line_number}", "line": line_number, "kind": "SK2", "arming_generation": int(g),
              "template": t, "launch": x, "issues": []}
    issues = record["issues"]
    if int(g) <= 0 or any(t[k] <= 0 for k in T_FIELDS[:7] + ["NACC", "CL"]):
        raise ValueError("nonpositive generation/template geometry")
    if any(t[k] not in (0, 1) for k in ("COMP", "HL", "PK", "FUSE", "BND")) or t["OutBytes"] not in (2, 4):
        raise ValueError("invalid boolean/output type")
    if min(x[k] for k in ("M", "N", "K", "threads", "grid")) <= 0 or x["pmax"] < 0 or x["UV"] not in (0, 1, 2) or x["PDL"] < 0:
        raise ValueError("invalid runtime geometry")
    nw, r, ups = t["NWR"] * t["WSK"], t["NWR"] * t["NT"] * 8, t["WSK"] * t["UPW"]
    nesc = min(t["NS"], 4) if t["COMP"] else 1
    wby = nw * t["UPW"] * t["NT"] * (768 if t["COMP"] else 2048 if t["HL"] else 1024)
    sst = (ups * t["TOKR"] * 128 + wby + 1023) & ~1023
    toks, rp, cp = 16 * t["MT"], r + ((40 - r % 32) % 32), r + 1
    smem = (1024 + t["NS"] * sst + 16 * t["NS"] + 32 + nesc * ((toks * cp + 3) & ~3) * 4
            + ((t["CL"] - 1) * t["WSK"] * toks * rp * 4 if t["CL"] > 1 else 0) + x["pmax"] * 8 + 128)
    expected = {"threads": (nw + 1 + nesc) * 32, "grid": x["N"] // r * t["CL"], "dynamic_shared_bytes": smem}
    if x["N"] % r or any(x[k] != v for k, v in expected.items()):
        issues.append("launch_geometry_differs_from_pinned_source")
    if int(key_device) != dev["tensor"]:
        issues.append("dedup_key_device_mismatch")
    if x["UV"] == 2 and x["M"] % 2:
        issues.append("UV2_requires_even_physical_M")
    record.update(physical_M=x["M"], logical_T_if_UV=x["M"] // 2 if x["UV"] == 2 else x["M"],
                  capacity_scope="block_only_cluster_capacity_unmeasured" if t["CL"] > 1 else "CL1_block_capacity_not_achieved_residency",
                  derived_geometry={"rows_per_CTA": r, "escape_warps": nesc, **expected})
    resource_status(record, a, err, (dev["tensor"], dev["current"]), occ["blocks"], occ["sms"])
    return record


def decode_moe(match, line_number):
    g, f, x, a, dev, err = match.groups()
    f, x = named(f, F_FIELDS), named(x, MOE_X)
    av = named(a, A_FIELDS + ["blocks", "sms"])
    dev, err = named(dev, ("tensor", "current")), named(err, ERROR_FIELDS)
    record = {"id": f"MOE:{line_number}", "line": line_number, "kind": "small_MoE", "arming_generation": int(g),
              "compiled_flags": f, "launch": x, "issues": [], "sample_scope": "first_identity_sample_not_exhaustive_M_grid_PDL",
              "capacity_scope": "block_capacity_not_achieved_residency"}
    if int(g) <= 0 or f["TINY"] != 1 or f["WS"] != 1 or f["NW"] <= 0 or f["NS"] <= 0 or x["BIG"] not in (0, 1):
        raise ValueError("invalid small-MoE identity")
    if min(x[k] for k in ("first_M", "first_grid", "threads", "dynamic_shared_bytes")) <= 0:
        raise ValueError("invalid small-MoE geometry")
    if x["threads"] != (f["NW"] + 1) * 32:
        record["issues"].append("tiny_WS_threads_differ_from_pinned_source")
    resource_status(record, {k: av[k] for k in A_FIELDS}, err, (dev["tensor"], dev["current"]), av["blocks"], av["sms"])
    return record


def parse_log(log):
    """Bounded strict parse. Missing headers remain visible, never silently assumed."""
    if len(log.encode()) > 262144:
        raise ValueError("diagnostic log exceeds 256 KiB parser bound")
    records, malformed, stops, schemas = [], [], [], {"SK2": [], "small_MoE": []}
    malformed_count = 0
    for number, line in enumerate(log.splitlines(), 1):
        if not any(marker in line for marker in ("SK2R", "MOER")):
            continue
        if number > 4096:
            stops.append({"line": number, "kind": "parser", "reason": "line_bound"}); break
        try:
            if line == SK2_SCHEMA:
                schemas["SK2"].append(number); continue
            if line == MOE_SCHEMA:
                schemas["small_MoE"].append(number); continue
            if re.fullmatch(r"SK2R_STOP reason=[a-z_]+ bytes=\d{1,10} records=\d{1,10}", line):
                stops.append({"line": number, "kind": "SK2", "raw": line}); continue
            if line in ("MOER_STOP coverage=partial", "MOER_STOP module_budget_or_flags"):
                stops.append({"line": number, "kind": "small_MoE", "raw": line}); continue
            if len(records) >= 128:
                stops.append({"line": number, "kind": "parser", "reason": "record_bound"}); break
            match = SK2_RE.fullmatch(line)
            if match:
                records.append(decode_sk2(match, number)); continue
            match = MOE_RE.fullmatch(line)
            if match:
                records.append(decode_moe(match, number)); continue
            raise ValueError("unknown/truncated schema or marker syntax")
        except (ValueError, OverflowError) as error:
            malformed_count += 1
            if len(malformed) < 64:
                malformed.append({"line": number, "reason": str(error), "raw_prefix": line[:512]})
    for record in records:
        record["schema_seen"] = bool(schemas[record["kind"]])
        record["schema_before_record"] = any(n < record["line"] for n in schemas[record["kind"]])
        record["schema_basis"] = "retained_header_and_pinned_source" if record["schema_seen"] else "pinned_source_only_header_missing"
    conflicts, static_groups, occupancy_groups = [], {}, {}
    for record in records:
        # Static compiled attributes do not depend on M, UV, pmax, grid or PDL.
        # MoE first_M/grid/PDL are sample fields, absent from its actual dedup key.
        identity = (record["kind"], json.dumps(record.get("template", record.get("compiled_flags")), sort_keys=True),
                    record["launch"].get("BIG"), record["devices"]["tensor"])
        static_groups.setdefault(identity, []).append(record)
        occupancy_identity = identity + (record["launch"]["threads"], record["launch"]["dynamic_shared_bytes"])
        occupancy_groups.setdefault(occupancy_identity, []).append(record)
    for kind, groups in (("compiled_attributes_conflict", static_groups), ("same_launch_resource_capacity_conflict", occupancy_groups)):
        for group in groups.values():
            if kind == "compiled_attributes_conflict":
                values = {json.dumps(r["attributes"], sort_keys=True) for r in group if r["query_status"]["attributes"] == 0}
            else:
                values = {(r["blocks_per_SM"], r["SMs"]) for r in group if r["all_queries_succeeded"]}
            if len(values) <= 1:
                continue
            conflict_id = len(conflicts)
            conflicts.append({"id": conflict_id, "record_ids": [r["id"] for r in group], "reason": kind})
            for r in group:
                r["resource_record_usable"] = False
                r["calculated_device_block_capacity"] = None
                r.setdefault("conflict_ids", []).append(conflict_id)
                r["issues"].append(kind)
    return {"records": records, "schemas": schemas, "stops": stops, "malformed_count": malformed_count,
            "malformed": malformed, "identity_conflicts": conflicts,
            "missing_schema_kinds": [k for k in schemas if any(r["kind"] == k for r in records) and not schemas[k]],
            "complete_run_coverage": False,
            "coverage_limits": ["first_observations_only", "arming_epoch_is_not_weight_or_benchmark_identity",
                                "retained_tail_may_omit_earlier_records", "small_MoE_one_identity_per_module"],
            "records_with_API_errors": [r["id"] for r in records if not r["all_queries_succeeded"]],
            "MoE_STOP_scope": "global_incomplete_MoE_coverage_module_identity_unknown" if any(s["kind"] == "small_MoE" for s in stops) else None}


def visible_template(symbol):
    start = symbol.find("sk2_kernel<")
    if start < 0:
        return None
    tail = symbol[start + len("sk2_kernel<"):]
    end, ellipsis = tail.find(">"), tail.find("...")
    complete = end >= 0 and (ellipsis < 0 or end < ellipsis)
    if complete:
        tokens = tail[:end].split(",")
    elif ellipsis >= 0:
        prefix = tail[:ellipsis]
        # A final partial integer such as `1...` must not be mistaken for 1 vs16.
        tokens = prefix.split(",")[:-1]
    else:
        return {"values": [], "complete": False, "error": "unclosed_symbol_without_explicit_truncation"}
    values = []
    for index, token in enumerate(tokens):
        token = token.strip()
        if index >= 15:
            return {"values": values, "complete": False, "error": "too_many_template_arguments"}
        if token in ("true", "false"):
            values.append(int(token == "true"))
        elif index == 11 and token in ("__nv_bfloat16", "float"):
            values.append(2 if token == "__nv_bfloat16" else 4)
        elif re.fullmatch(r"\d{1,10}", token):
            values.append(int(token))
        else:
            return {"values": values, "complete": False, "error": "unknown_complete_template_token:" + token[:40]}
    return {"values": values, "complete": complete and len(values) == 15,
            "error": None if not complete or len(values) == 15 else "incomplete_full_symbol"}


def associate_profiles(captures, records):
    associations, warnings = [], []
    if len(captures) > 128:
        raise ValueError("profile capture bound exceeded")
    for capture in captures:
        case = capture.get("case") or {}
        batch, qlen = case.get("batch"), case.get("q_len", 1)
        logical_t = batch * qlen if type(batch) is int and type(qlen) is int and batch > 0 and qlen > 0 else None
        report = capture.get("report", "")
        if not isinstance(report, str) or len(report) > 131072:
            warnings.append({"index": capture.get("index"), "reason": "invalid_or_oversized_report"}); continue
        rows = []
        for line in report.splitlines()[:256]:
            if "sk2_kernel<" not in line and "okc::moe_fused<" not in line:
                continue
            match = ROW_RE.fullmatch(line)
            if not match:
                warnings.append({"index": capture.get("index"), "reason": "unparsed_relevant_profile_row", "raw_prefix": line[:256]}); continue
            symbol, _, _, calls = match.groups()
            row = {"symbol": symbol.strip(), "calls": int(calls), "benchmark_resources_established": False}
            visible = visible_template(symbol)
            if visible is None:
                row.update(kind="MoE", status="symbol_does_not_identify_compiled_flag_build",
                           compatible_record_ids=[], note="Do not equate first_M with exhaustive M coverage or map identical moe_fused names to a module.")
            else:
                compatible = []
                if not visible.get("error") and visible["values"]:
                    for r in records:
                        if r["kind"] != "SK2": continue
                        if [r["template"][k] for k in T_FIELDS[:len(visible["values"])]] != visible["values"]: continue
                        if logical_t is not None and r["physical_M"] != logical_t * (2 if r["launch"]["UV"] == 2 else 1): continue
                        compatible.append(r)
                templates = {tuple(r["template"][k] for k in T_FIELDS) for r in compatible}
                row.update(kind="SK2", visible_template=visible, compatible_record_ids=[r["id"] for r in compatible],
                           compatible_usable_record_ids=[r["id"] for r in compatible if r["resource_record_usable"]],
                           compatible_full_template_count=len(templates),
                           status="no_compatible_retained_record" if not compatible else "one_compatible_retained_record" if len(compatible) == 1 else "multiple_compatible_retained_records",
                           hidden_template_fields=T_FIELDS[len(visible["values"]):],
                           note="Compatibility only: hidden arguments, pmax, UV mode and arming epoch cannot be recovered from truncated profiler text.")
            rows.append(row)
        associations.append({"index": capture.get("index"), "case": {k: case[k] for k in ("batch", "q_len", "layer", "load") if k in case},
                             "logical_T": logical_t, "rows": rows})
    return {"captures": associations, "warnings": warnings, "timing_metrics_used": False,
            "mapping_scope": "compatible retained source-pinned records; never benchmark invocation identity"}


def bounded_read(path, limit, root):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError("artifact path outside repository")
    if path.stat().st_size > limit: raise ValueError("artifact size bound exceeded: " + str(path))
    return path, path.read_bytes()


def terminal_gate(response, result):
    records = result.get("records") or []
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("result.records must be a list of objects")
    counts = Counter(r.get("event") for r in records)
    expected = {"setup": 1, "test": 13, "benchmark": 12, "profile": 12}
    reasons = []
    if response.get("status") != "done": reasons.append("terminal_status_not_done")
    if result.get("passed") is not True: reasons.append("official_pass_not_true")
    if result.get("returncode") != 0: reasons.append("returncode_not_zero")
    if result.get("error") not in (None, ""): reasons.append("official_error_present")
    if result.get("status") != "ok": reasons.append("result_status_not_ok")
    if dict(counts) != expected: reasons.append("record_counts_differ_from_13_tests_12_benchmarks_12_profiles")
    if any(r.get("status") != "pass" for r in records if r.get("event") in ("setup", "test", "benchmark")) or any(
            r.get("status", "pass") != "pass" for r in records if r.get("event") == "profile"):
        reasons.append("nonpassing_or_missing_required_event_status")
    if result.get("checks") != {"ok": True, "flagged": [], "rejected": []}: reasons.append("official_checks_not_clean")
    if result.get("flags") != []: reasons.append("official_flags_present_or_missing")
    return {"passed": not reasons, "reasons": reasons, "actual_record_counts": dict(counts), "expected_record_counts": expected}


def analyze_receipt(path, root=ROOT):
    path, raw = bounded_read(path, 16 * 1024 * 1024, root)
    receipt = json.loads(raw)
    response = receipt.get("response") or {}
    report = {"schema": 1, "job": receipt.get("job"), "source_sha256": receipt.get("source_sha256"),
              "decision": "DIAGNOSTIC_ONLY_NO_PROMOTION", "receipt_sha256": sha(raw), "provenance_errors": []}
    errors = report["provenance_errors"]
    if receipt.get("job") != JOB or response.get("id") != JOB: errors.append("job_mismatch")
    if receipt.get("source_sha256") != SOURCE: errors.append("source_hash_mismatch")
    if receipt.get("problem") != "muse_decode_layer" or response.get("board") != "muse_decode_layer": errors.append("problem_mismatch")
    if response.get("mode") != "check": errors.append("not_private_check")
    try:
        source_path, source = bounded_read(receipt.get("source", ""), 4 * 1024 * 1024, root)
        if sha(source) != SOURCE: errors.append("source_file_hash_mismatch")
        report["source_path"] = str(source_path)
    except (OSError, ValueError) as error:
        errors.append("source_unavailable:" + str(error))
    if errors:
        report.update(status="PROVENANCE_REJECTED", resources=[], metrics={}); return report
    status = response.get("status")
    report["saved_status"] = status
    if status not in ("done", "failed", "rejected", "error", "cancelled"):
        report.update(status="PENDING", result_pending=True, resources=[], metrics={},
                      note="Saved receipt is not terminal; no polling, result parsing or inferred metrics."); return report
    result = response.get("result") or {}
    report.update(status="TERMINAL", result_pending=False, official_passed=result.get("passed"),
                  official_result_status=result.get("status"), returncode=result.get("returncode"),
                  official_error=result.get("error"), checks=result.get("checks"), flags=result.get("flags"),
                  metrics={}, score_deliberately_not_analyzed=True)
    report["run_audit_gate"] = terminal_gate(response, result)
    report["resource_evidence_eligible"] = report["run_audit_gate"]["passed"]
    warnings = report["artifact_warnings"] = []
    tail = result.get("log_tail") or ""
    if not isinstance(tail, str): raise ValueError("log_tail must be text")
    log, selected = tail, "receipt.result.log_tail"
    report["log_sources"] = {"tail_bytes": len(tail.encode()), "tail_sha256": sha(tail.encode())}
    log_path = (response.get("output") or {}).get("log")
    if log_path:
        try:
            lp, data = bounded_read(log_path, 262144, root)
            external = data.decode()
            report["log_sources"].update(external_path=str(lp), external_bytes=len(data), external_sha256=sha(data))
            if external == tail:
                report["log_sources"]["relationship"] = "identical"
            elif not external:
                report["log_sources"]["relationship"] = "saved_log_empty"
            elif not tail:
                log, selected = external, "saved_log"
                report["log_sources"]["relationship"] = "receipt_tail_empty"
            elif tail.startswith(external):
                report["log_sources"]["relationship"] = "saved_log_is_prefix_of_receipt_tail"
            elif external.endswith(tail):
                log, selected = external, "saved_log"
                report["log_sources"]["relationship"] = "receipt_tail_is_suffix_of_saved_log"
            elif tail.endswith(external):
                report["log_sources"]["relationship"] = "saved_log_is_suffix_of_receipt_tail"
            elif external.startswith(tail):
                log, selected = external, "saved_log"
                report["log_sources"]["relationship"] = "receipt_tail_is_prefix_of_saved_log"
            else:
                report["log_sources"]["relationship"] = "disagree"
                warnings.append("saved_log_and_receipt_tail_disagree_using_receipt_tail")
        except (OSError, ValueError, UnicodeError) as error:
            warnings.append("saved_log_unavailable:" + str(error))
    report["log_sources"]["selected"] = selected
    report["log_sources"]["tail_limit_possible"] = len(tail.encode()) >= 16000
    parsed = parse_log(log)
    for record in parsed["records"]:
        record["eligible_as_optimization_input"] = report["resource_evidence_eligible"] and record["resource_record_usable"]
    if not parsed["records"]: warnings.append("no_retained_resource_records")
    report["resources"] = parsed
    captures = [r for r in result.get("records", []) if r.get("event") == "profile"]
    profile_path = (response.get("profile") or {}).get("json")
    if profile_path:
        try:
            pp, data = bounded_read(profile_path, 16 * 1024 * 1024, root)
            profile = json.loads(data)
            if profile.get("job_id") != JOB or profile.get("problem") != "muse_decode_layer": raise ValueError("profile identity mismatch")
            _, ps = bounded_read(profile.get("submission", ""), 4 * 1024 * 1024, root)
            if sha(ps) != SOURCE: raise ValueError("profile submission hash mismatch")
            external_profiles = profile.get("profiles", [])
            if captures and external_profiles != captures: raise ValueError("external and embedded profiles disagree")
            captures = external_profiles
            report["profile_artifact"] = {"path": str(pp), "sha256": sha(data)}
        except (OSError, ValueError, UnicodeError) as error:
            warnings.append("external_profile_rejected:" + str(error))
    report["profile_associations"] = associate_profiles(captures, parsed["records"])
    report["counts"] = dict(Counter(r["kind"] for r in parsed["records"]))
    report["limitations"] = ["No performance comparison or promotion decision", "Capacity is not achieved occupancy or stall evidence",
                             "Local allocation does not establish spill traffic", "Attributes do not prove absence of WGMMA serialization",
                             "Missing/error/STOP/truncated records leave coverage unknown"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / f"experiments/runs/{JOB}.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = analyze_receipt(args.receipt)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"status": "INPUT_ERROR", "decision": "DIAGNOSTIC_ONLY_NO_PROMOTION", "error": str(error), "metrics": {}}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output: args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

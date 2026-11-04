#!/usr/bin/env python3
"""Focused offline fixtures for the exact-source resource parser; no CUDA."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_decoder_selected_resources as parser


def sk2(*, m=1, pmax=48, uv=0, regs=96, template=None, errors=(0, 0, 0, 0), attributes=None, occ=(1, 132)):
    t = list(template or (1, 8, 3, 1, 4, 2, 6, 1, 0, 1, 1, 2, 1, 0, 0))
    mt, tokr, nt, nwr, wsk, upw, ns, comp, hl, nacc, cl, ot, pk, fuse, bnd = t
    n, k = (4096, 8192) if fuse else (2304, 8192)
    nw, r, ups = nwr * wsk, nwr * nt * 8, wsk * upw
    nesc = min(ns, 4) if comp else 1
    sst = (ups * tokr * 128 + nw * upw * nt * (768 if comp else 2048 if hl else 1024) + 1023) & ~1023
    rp, cp, toks = r + ((40 - r % 32) % 32), r + 1, 16 * mt
    smem = 1024 + ns * sst + 16 * ns + 32 + nesc * ((toks * cp + 3) & ~3) * 4 + (cl - 1) * wsk * toks * rp * 4 + pmax * 8 + 128
    x = (m, n, k, pmax, uv, 2, (nw + 1 + nesc) * 32, n // r * cl, smem)
    a = attributes or (regs, 0, 1024, 231424, 1024)
    csv = lambda values: ",".join(map(str, values))
    return f"SK2R g=1 t={csv(t)} x={csv(x)} d=0 a={csv(a)} occ={csv(occ)} dev=0,0 err={csv(errors)}"


MOE = "MOER g=1 f=8,3,1,1,0,1,1,1,0,2,0,0,0,1 x=1,0,132,0,288,186528 a=144,0,1024,186528,1024,1,132 d=0,0 e=0,0,0,0"


def profile(symbol, *, batch=1, qlen=1):
    return {"index": 0, "case": {"batch": batch, "q_len": qlen, "load": "fixture", "layer": "sliding"},
            "report": symbol + "     10.1 µs  12.7%  1\n"}


class ParseFixtures(unittest.TestCase):
    def test_exact_source_schemas(self):
        source = (parser.ROOT / f"experiments/candidates/muse_decode_layer/{parser.SOURCE}.py").read_text()
        for schema in (parser.SK2_SCHEMA, parser.MOE_SCHEMA):
            self.assertIn('"' + schema + r'\n"', source)

    def test_valid_geometry_and_resource_capacity(self):
        result = parser.parse_log(parser.SK2_SCHEMA + "\n" + sk2())
        row = result["records"][0]
        self.assertTrue(row["resource_record_usable"])
        self.assertEqual(row["launch"]["dynamic_shared_bytes"], 167808)
        self.assertEqual(row["calculated_device_block_capacity"], 132)
        self.assertEqual(row["derived_geometry"]["grid"], 96)
        self.assertTrue(row["schema_before_record"])
        self.assertFalse(result["complete_run_coverage"])

    def test_missing_malformed_schema_and_truncated_record(self):
        result = parser.parse_log("SK2R_SCHEMA t=bad\n" + sk2() + "\n" + sk2()[:60])
        self.assertEqual(result["malformed_count"], 2)
        self.assertEqual(result["missing_schema_kinds"], ["SK2"])
        self.assertTrue(result["records"][0]["resource_record_usable"])
        self.assertEqual(result["records"][0]["schema_basis"], "pinned_source_only_header_missing")

    def test_api_errors_and_sentinel_validation(self):
        error = sk2(errors=(0, 0, 7, -1), attributes=(-1,) * 5, occ=(-1, 132))
        result = parser.parse_log(parser.SK2_SCHEMA + "\n" + error)
        row = result["records"][0]
        self.assertFalse(row["all_queries_succeeded"])
        self.assertFalse(row["resource_record_usable"])
        self.assertIsNone(row["calculated_device_block_capacity"])
        self.assertFalse(row["issues"])
        inconsistent = parser.parse_log(error.replace("a=-1,-1,-1,-1,-1", "a=96,-1,-1,-1,-1"))
        self.assertIn("failed_attribute_query_has_non_sentinel_values", inconsistent["records"][0]["issues"])

    def test_geometry_and_device_corruption(self):
        result = parser.parse_log(sk2().replace(",288,96,167808", ",320,96,167808").replace("dev=0,0", "dev=0,1"))
        self.assertIn("launch_geometry_differs_from_pinned_source", result["records"][0]["issues"])
        self.assertIn("tensor_and_current_device_mismatch", result["records"][0]["issues"])

    def test_moe_repeated_schema_stop_is_not_assigned_to_module(self):
        log = "\n".join((parser.MOE_SCHEMA, MOE, parser.MOE_SCHEMA, MOE.replace("f=8,3", "f=8,4"),
                          "MOER_STOP coverage=partial", "MOER_STOP module_budget_or_flags"))
        result = parser.parse_log(log)
        self.assertEqual(len(result["schemas"]["small_MoE"]), 2)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(len(result["stops"]), 2)
        self.assertTrue(all("record_ids" not in stop for stop in result["stops"]))
        self.assertTrue(all(r["sample_scope"].startswith("first_identity") for r in result["records"]))

    def test_conflicting_and_distinct_runtime_identities(self):
        result = parser.parse_log("\n".join((sk2(), sk2(regs=100), sk2(pmax=50))))
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(len(result["identity_conflicts"]), 1)
        self.assertTrue(all(not r["resource_record_usable"] and r["calculated_device_block_capacity"] is None for r in result["records"]))
        symbol = "void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _..."
        mapped = parser.associate_profiles([profile(symbol)], result["records"])["captures"][0]["rows"][0]
        self.assertEqual(mapped["status"], "multiple_compatible_retained_records")
        self.assertEqual(len(mapped["compatible_record_ids"]), 3)
        self.assertEqual(mapped["compatible_usable_record_ids"], [])
        self.assertFalse(mapped["benchmark_resources_established"])

    def test_moe_sample_fields_do_not_hide_compiled_conflict(self):
        other = MOE.replace("x=1,0,132,0", "x=8,0,264,1").replace("a=144,", "a=140,")
        rows = parser.parse_log(MOE + "\n" + other)
        self.assertEqual(len(rows["identity_conflicts"]), 1)
        self.assertTrue(all(not r["resource_record_usable"] for r in rows["records"]))

    def test_occupancy_conflict_uses_dynamic_shared_key(self):
        result = parser.parse_log("\n".join((sk2(), sk2(occ=(2, 132)), sk2(pmax=50, occ=(2, 132)))))
        self.assertEqual(len(result["identity_conflicts"]), 1)
        self.assertEqual([r["resource_record_usable"] for r in result["records"]], [False, False, True])

    def test_uv_physical_M_and_truncated_tokens(self):
        t = (1, 16, 4, 1, 8, 1, 5, 1, 0, 2, 1, 2, 0, 1, 0)
        rows = parser.parse_log(sk2(m=16, uv=2, template=t) + "\n" + sk2(m=16, uv=0, template=t))["records"]
        symbol = "void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, _..."
        cases = [profile(symbol, batch=1, qlen=8), profile(symbol, batch=16)]
        mapped = parser.associate_profiles(cases, rows)["captures"]
        self.assertEqual(mapped[0]["rows"][0]["compatible_record_ids"], [rows[0]["id"]])
        self.assertEqual(mapped[1]["rows"][0]["compatible_record_ids"], [rows[1]["id"]])
        self.assertFalse(mapped[0]["rows"][0]["benchmark_resources_established"])
        self.assertEqual(parser.visible_template("void sk2_kernel<1, 1...")["values"], [1])
        complete = parser.visible_template("void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, __nv_bfloat16, true, false, false>(...)")
        self.assertTrue(complete["complete"])

    def test_cluster_scope_and_moe_symbol_ambiguity(self):
        t = (1, 8, 1, 1, 8, 1, 4, 0, 1, 4, 3, 4, 1, 0, 0)
        rows = parser.parse_log(sk2(template=t) + "\n" + MOE)["records"]
        self.assertIn("cluster_capacity_unmeasured", rows[0]["capacity_scope"])
        mapped = parser.associate_profiles([profile("void okc::moe_fused<false>(okc::Params)")], rows)
        self.assertEqual(mapped["captures"][0]["rows"][0]["compatible_record_ids"], [])

    def test_bounds(self):
        with self.assertRaises(ValueError): parser.parse_log("x" * 262145)
        result = parser.parse_log("\n".join([sk2()] * 129))
        self.assertEqual(len(result["records"]), 128)
        self.assertEqual(result["stops"][-1]["reason"], "record_bound")


class ReceiptFixtures(unittest.TestCase):
    def test_terminal_gate_requires_explicit_test_status(self):
        records = [{"event": "setup", "status": "pass"}] + [{"event": "test", "status": "pass"} for _ in range(13)]
        records += [{"event": "benchmark", "status": "pass"} for _ in range(12)]
        records += [{"event": "profile"} for _ in range(12)]
        result = {"records": records, "passed": True, "returncode": 0, "error": None, "status": "ok",
                  "checks": {"ok": True, "flagged": [], "rejected": []}, "flags": []}
        self.assertTrue(parser.terminal_gate({"status": "done"}, result)["passed"])
        del records[1]["status"]
        gate = parser.terminal_gate({"status": "done"}, result)
        self.assertFalse(gate["passed"])
        self.assertIn("nonpassing_or_missing_required_event_status", gate["reasons"])

    def test_pending_provenance_and_terminal_log_disagreement(self):
        source = (parser.ROOT / f"experiments/candidates/muse_decode_layer/{parser.SOURCE}.py").read_bytes()
        with tempfile.TemporaryDirectory(prefix="okc-resource-parser-") as directory:
            root = Path(directory)
            (root / "source.py").write_bytes(source)
            receipt = {"job": parser.JOB, "problem": "muse_decode_layer", "source_sha256": parser.SOURCE,
                       "source": "source.py", "response": {"id": parser.JOB, "board": "muse_decode_layer", "mode": "check", "status": "running"}}
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            pending = parser.analyze_receipt(path, root)
            self.assertEqual(pending["status"], "PENDING")
            self.assertEqual(pending["resources"], [])
            self.assertEqual(pending["metrics"], {})
            bad = copy.deepcopy(receipt)
            bad["source_sha256"] = "0" * 64
            path.write_text(json.dumps(bad))
            self.assertEqual(parser.analyze_receipt(path, root)["status"], "PROVENANCE_REJECTED")
            receipt["response"].update(status="done", output={"log": "log.txt"},
                result={"passed": True, "log_tail": parser.SK2_SCHEMA + "\n" + sk2(), "records": [], "speedup": 999})
            (root / "log.txt").write_text("unrelated external log")
            path.write_text(json.dumps(receipt))
            terminal = parser.analyze_receipt(path, root)
            self.assertEqual(terminal["status"], "TERMINAL")
            self.assertEqual(len(terminal["resources"]["records"]), 1)
            self.assertIn("saved_log_and_receipt_tail_disagree_using_receipt_tail", terminal["artifact_warnings"])
            self.assertEqual(terminal["metrics"], {})
            self.assertNotIn("speedup", terminal)
            self.assertFalse(terminal["run_audit_gate"]["passed"])
            self.assertFalse(terminal["resources"]["records"][0]["eligible_as_optimization_input"])
            records = [{"event": "setup", "status": "pass"}] + [{"event": "test", "status": "pass"} for _ in range(13)]
            records += [{"event": "benchmark", "status": "pass"} for _ in range(12)]
            records += [{"event": "profile", "report": "", "index": index} for index in range(12)]
            receipt["response"]["result"].update(status="ok", returncode=0, error=None,
                checks={"ok": True, "flagged": [], "rejected": []}, flags=[], records=records)
            (root / "log.txt").write_text(receipt["response"]["result"]["log_tail"])
            receipt["response"]["result"]["log_tail"] += "\nUSDT profiler suffix\n"
            path.write_text(json.dumps(receipt))
            terminal = parser.analyze_receipt(path, root)
            self.assertTrue(terminal["run_audit_gate"]["passed"])
            self.assertTrue(terminal["resources"]["records"][0]["eligible_as_optimization_input"])
            self.assertEqual(terminal["log_sources"]["relationship"], "saved_log_is_prefix_of_receipt_tail")
            self.assertNotIn("saved_log_and_receipt_tail_disagree_using_receipt_tail", terminal["artifact_warnings"])
            receipt["response"]["result"]["passed"] = False
            path.write_text(json.dumps(receipt))
            failed = parser.analyze_receipt(path, root)
            self.assertFalse(failed["resource_evidence_eligible"])
            self.assertFalse(failed["resources"]["records"][0]["eligible_as_optimization_input"])
            (root / "source.py").write_text("changed")
            self.assertEqual(parser.analyze_receipt(path, root)["status"], "PROVENANCE_REJECTED")


if __name__ == "__main__":
    unittest.main()

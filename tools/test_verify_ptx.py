"""Source-binding/axiom rejection tests; --integration invokes real Lean proofs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import verify_ptx as v


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = v.ANCHOR.read_bytes()

    def test_actual_constants_and_independent_spec(self):
        bound = v.extract(self.source)
        self.assertEqual(bound["source_sha256"], v.ANCHOR_SHA)
        self.assertEqual(bound["constants"], dict(mask=0x81C081C0, rotate1=3,
                         rotate2=6, multiply_mask=0x001B001B, multiplier=4228))
        lean = v.generated_lean(bound)
        self.assertIn("actual3 q = spec3 q", lean)
        self.assertIn("* 4228", lean)

    def test_arithmetic_mutations_are_extracted_not_hidden_by_anchor(self):
        for before, after, key, expected in [
            (b"__funnelshift_r(q,q,3)", b"__funnelshift_r(q,q,4)", "rotate1", 4),
            (b"M=0x81C081C0u", b"M=0x81C081C1u", "mask", 0x81C081C1),
            (b"*4228u", b"*4224u", "multiplier", 4224),
        ]:
            with self.subTest(key=key):
                changed = v.extract(self.source.replace(before, after, 1))
                self.assertEqual(changed["constants"][key], expected)
                self.assertNotEqual(changed["source_sha256"], v.ANCHOR_SHA)

    def test_packing_mutations_fail_binding(self):
        for before, after in [(b"[4,0,1,3][bit]", b"[4,0,1,2][bit]"),
                              (b"code = n[..., h, q, up]", b"code = n[..., h, q, 1-up]"),
                              (b"_REG_HJ = [(0, 0)", b"_REG_HJ = [(0, 1)")]:
            with self.subTest(before=before):
                with self.assertRaisesRegex(v.VerificationError, "Packing specification binding changed"):
                    v.extract(self.source.replace(before, after, 1))

    def test_unmodeled_expression_type_shift_and_assignment_rejected(self):
        mutants = [self.source.replace(b"r0=q&M;", b"r0=(q+1)&M;", 1),
                   self.source.replace(b"void dq4(uint32_t q", b"void dq4(int32_t q", 1),
                   self.source.replace(b"__funnelshift_r(q,q,3)", b"__funnelshift_r(q,q,32)", 1),
                   self.source.replace(b"r0=q&M;", b"r0=q&M; r0=0;", 1),
                   self.source.replace(b"*4228u", b"*04228u", 1),
                   self.source + b"\nCUDA_SRC = ''\n",
                   self.source + b"\n_POS_MUL = {}\n",
                   self.source + b"\n_pack_codes = None\n"]
        for mutant in mutants:
            with self.subTest(sha=v.sha(mutant)):
                with self.assertRaises(v.VerificationError):
                    v.extract(mutant)

    def test_never_imports_candidate(self):
        # A top-level side effect is outside the narrow claim, but must not run.
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            malicious = f"\nopen({str(marker)!r}, 'w').write('bad')\n".encode()
            v.extract(self.source + malicious)
            self.assertFalse(marker.exists())

    def test_cpp_line_splicing_and_obvious_environment_mutations_rejected(self):
        continued = b"// hide assignment " + bytes([92, 10]) + b"r0=q&M;"
        mutants = [self.source.replace(b"r0=q&M;", continued, 1),
                   self.source.replace(b"r0=q&M;", continued.replace(b"\\\n", b"\\\r\n"), 1),
                   self.source.replace(b"constexpr uint32_t M=", b"#define __funnelshift_r(...) 0u\nconstexpr uint32_t M=", 1),
                   self.source + b"\n_REG_HJ[0] = (1,2)\n",
                   self.source + b"\n_POS_MUL.clear()\n"]
        for mutant in mutants:
            with self.subTest(sha=v.sha(mutant)):
                with self.assertRaises(v.VerificationError):
                    v.extract(mutant)

    def test_axiom_audit_rejects_native_custom_missing_and_duplicate(self):
        accepted = "'T' depends on axioms: [propext, Classical.choice, Quot.sound]"
        self.assertEqual(set(v.audit_axioms(accepted, ["T"])["T"]), v.ALLOWED_AXIOMS)
        self.assertEqual(v.audit_axioms("'T' does not depend on any axioms", ["T"]), {"T": []})
        for output in [accepted.replace("Quot.sound", "sorryAx"),
                       accepted.replace("Quot.sound", "T._native.bv_decide.ax_1"),
                       accepted.replace("Quot.sound", "custom_axiom"), "", accepted + accepted]:
            with self.subTest(output=output):
                with self.assertRaises(v.VerificationError):
                    v.audit_axioms(output, ["T"])


def integration():
    """Real universal proof and three wrong constants; no GPU or official calls."""
    source = v.ANCHOR.read_bytes()
    report = {"incumbent": v.verify(v.ANCHOR), "mutants": []}
    with tempfile.TemporaryDirectory(prefix="okc-lean-mutants-") as directory:
        for label, before, after in [
            ("rotate3_to4", b"__funnelshift_r(q,q,3)", b"__funnelshift_r(q,q,4)"),
            ("extra_mask_bit", b"M=0x81C081C0u", b"M=0x81C081C1u"),
            ("multiplier4228_to4224", b"*4228u", b"*4224u")]:
            path = Path(directory) / (label + ".py")
            path.write_bytes(source.replace(before, after, 1))
            try:
                v.verify(path)
            except v.VerificationError as exc:
                if "Lean rejected fragment" not in str(exc):
                    raise
                report["mutants"].append({"label": label, "source_sha256": v.sha(path.read_bytes()),
                                          "status": "rejected_by_lean", "evidence": str(exc)})
            else:
                raise AssertionError(f"Broken decoder was accepted: {label}")
    destination = v.ROOT / "experiments/iterations/20261104-lean-dq4-proof.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "incumbent": report["incumbent"]["status"],
                      "mutants": len(report["mutants"])}))


if __name__ == "__main__":
    if "--integration" in sys.argv:
        integration()
    else:
        unittest.main()

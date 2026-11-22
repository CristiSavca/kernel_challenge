"""Frozen source and ancestor evidence for the optional second private check."""
import hashlib
import json

from submit_nov17_integrated_checks import Candidate, V68


V204 = Candidate(
    "v204", "v204_mixed_prompt_weight_pipeline",
    "3a091e426aa4a867e2a66eefdcf190cd52f11a68fd874684514681b80bbae88e",
    V68.flags | {"--speculative-dflash-sp-prefill-context": "", "--speculative-dflash-mixed-prefill": ""},
)
# The old V168 runner recognizes this slot claim by name if it is restored on rollback.
CLAIM = "v168_or_v204_nov19_check16"
V68_REPORT = "67c0839fef1e9b068efd81c72b0b660f3d5abfd044fce5aa1093a6b2e5b0f505"
SOURCE = {
    "candidate": "V204", "base": "297e54fd105e974cdf401e5b4d20253a22c16f90",
    "parent": "V203", "parent_tree": "1180446a62b6d3536e90f35ab5a1b62a16889288",
    "parent_sha256": "06542edafffd499247ed33e1a932280b935f35e5269d57d363f008adbde1f334",
    "common_parent": "V198", "common_tree": "927369031684bbe39bd46354b552a1a32cbfbb2f",
    "expert_donor": "V199", "donor_tree": "dc2388522de1957f7f34e94ab5756da892f35f50",
    "donor_sha256": "6a2c175fb20c4b6a0c8304467997f72f26a34e87f5f122434e4c975e274a5ec0",
    "tree": "ad8f0b3a2d73e2153f5a5ae6eec773682528d09d",
    "patch_sha256": V204.sha256,
    "delta_sha256": "e78cce2efc0acb577842929968aaf189d25b50e29b8704d63cad2e420194204d",
    "launch_flags": V204.flags, "bytes": 291087, "official_export": "okc serve diff",
}


def verified(root, runs, v68_job):
    folder = root / "serving_v0523/candidates" / V204.directory
    try:
        manifest = json.loads((folder / "validation.json").read_text())
        if (not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in SOURCE.items())
                or any(manifest.get(k) is not True for k in ("strict_both_donor_orders", "strict_apply_reverse_full_delta"))):
            return False
        for path, expected in ((folder / "patch.diff", V204.sha256), (folder / "delta.diff", SOURCE["delta_sha256"]),
                               (runs / (v68_job + ".json"), V68_REPORT)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        return True
    except (OSError, ValueError):
        return False

"""Run V235's ownership fixture against the V291 manifest and production checkout."""
from pathlib import Path

here=Path(__file__).resolve().parent
fixture=here.parent/"v235_peer_mixed_sliding_prefill/independent_certificate.py"
exec(compile(fixture.read_text(),str(fixture),"exec"),{"__file__":str(Path(__file__).resolve())})

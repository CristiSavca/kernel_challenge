"""Run the real frozen gate/up writer and paired weight/scale descriptors at W13 native decode widths."""
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
base=here.parent/"v191_decode_gate_up_epilogue/independent_epilogue.py"
s=base.read_text().replace("header = CPP + function", 'header = CPP + "template<bool Prompt>\\n" + function').replace("path.write_text(header + MAIN)", 'path.write_text(header + MAIN.replace("assert(bank_map(", "assert(bank_map<false>("))')
exec(compile(s,str(base),"exec"))

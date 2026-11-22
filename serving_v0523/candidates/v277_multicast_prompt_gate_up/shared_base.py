"""Run inside the offline compiler workspace for the ordinary and multicast images."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

directory=Path(sys.argv[1])
bin=Path.home()/"nvcc126/toolchain/usr/local/cuda-12.9/bin"
records={}
for path in sorted(directory.glob("*.cubin")):
    elf=subprocess.check_output(["readelf","-SW",str(path)],text=True)
    line=next(l for l in elf.splitlines() if ".nv.shared._Z" in l)
    metadata=line.split("NOBITS",1)[1].split()
    image=subprocess.check_output([str(bin/"cuobjdump"),"-sass",str(path)],text=True)
    op=re.search(r"UTMALDG\.2D(?:\.MULTICAST)? \[(UR\d+)\]",image)
    assert op is not None
    reg=op[1]
    movs=re.findall(rf"(?:^|\s)UMOV {reg}, (0x[0-9a-fA-F]+) ;",image[:op.start()])
    assert movs and int(movs[-1],16)==int(metadata[2],16)==1024, (reg,movs,metadata)
    records[path.stem]=dict(cubin_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        static_section_size=int(metadata[2],16),section_alignment=int(metadata[-1]),first_A_destination_register=reg,
        first_A_destination=int(movs[-1],16),phase_in_1024_byte_swizzle_pattern=0)
assert len(records)==2
print(json.dumps(records,indent=2))

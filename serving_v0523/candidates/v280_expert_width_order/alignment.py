"""Read true SH size/alignment and the Humming base construction from both SM90a modules."""
from pathlib import Path
from hashlib import sha256
import json
import re
import struct
import sys

def inspect(path):
    b=path.read_bytes()
    assert b[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',b,40)[0]
    size,count,strings=struct.unpack_from('<HHH',b,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size) for i in range(count)]
    stringoff=sections[strings][4]
    shared=[]
    for h in sections:
        n=b[stringoff+h[0]:].split(b'\0',1)[0]
        if n.startswith(b'.nv.shared._Z'):
            shared.append(dict(size=h[5],align=h[8]))
    assert len(shared)==1
    asm=path.with_suffix('.sass').read_text()
    mov,add=tuple(int(x,16) for x in re.search(r'UMOV UR5, (0x[0-9a-f]+).*?UIADD3 UR4, UR5, (0x[0-9a-f]+)',asm,re.S).groups()) if 'UIADD3 UR4, UR5' in asm else (0,int(re.search(r'UMOV UR4, (0x[0-9a-f]+)',asm)[1],16))
    assert (add+mov)%1024==0 and shared[0]['size']==add+mov
    return dict(sha256=sha256(b).hexdigest(),ELF_shared=shared[0],sass_dynamic_base=mov+add,swizzle_phase=((mov+add)//128)%8)
images=[inspect(Path(arg)) for arg in sys.argv[1:]]
assert images[1]['ELF_shared']['align']==1024 and images[1]['sass_dynamic_base']==2048
print(json.dumps(dict(baseline=images[0],ordered=images[1],GPU=False),indent=2))

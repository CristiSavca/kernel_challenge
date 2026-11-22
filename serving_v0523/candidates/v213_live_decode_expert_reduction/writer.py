"""Prove the live FP32 prefix survives Humming's native XOR4 register-to-shared store transpose."""
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1])
p=s/'humming/include/humming/epilogue/smem_writer.cuh';src=p.read_text()
for text in ('__shfl_xor_sync(0xffffffff, val, 4)','uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;',
             'uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row;', 'part_regs[n * inner_m + m]'):
    assert text in src
counts={};total=0
for phase,groups in [('W13',2),('W2',4)]:
    for width in (8,16,24):
        checks=0
        for live in range(1,width+1):
            narrow=min(w for w in range(8,width+1,8) if w>=live)
            for t in range(512//groups):
                lane=t%32
                for band in range(2):
                    for column8 in range(width//8):
                        output_row=column8*8+(lane%4)*2+(lane%8)//4
                        for channel8 in range(2):
                            for component in (0,1):
                                partner=(lane//4)%2
                                reads_partner=(component==1-partner)
                                source_lane=lane^(4 if reads_partner else 0)
                                source_component=1-component if reads_partner else component
                                source_reg=4*column8+2*channel8+source_component
                                token=8*(source_reg//4)+2*(source_lane%4)+source_reg%2
                                assert token==output_row
                                if output_row<live:
                                    assert source_reg<narrow//2
                                    vector=(band*width//2+source_reg)//4
                                    assert 8*(vector%(width//8))+2*(source_lane%4)<live
                                    checks+=1
        counts[f'{phase}_{width}']=checks;total+=checks
print(json.dumps(dict(writer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),live_output_components=total,by_shape=counts),indent=2))

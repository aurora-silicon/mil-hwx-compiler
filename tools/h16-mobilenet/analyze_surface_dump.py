#!/usr/bin/env python3
"""Summarize spatial/channel fp16 differences in two padded CHW surfaces."""
import argparse
import json
from pathlib import Path
import struct


def ulp_key(value):
    return 0x8000 - (value & 0x7fff) if value & 0x8000 else 0x8000 + value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('got', type=Path)
    ap.add_argument('expected', type=Path)
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--height', type=int, required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--row-elements', type=int, required=True)
    ap.add_argument('--tile', type=int, default=16)
    args = ap.parse_args()
    count=args.channels*args.height*args.row_elements
    got_bytes=args.got.read_bytes(); expected_bytes=args.expected.read_bytes()
    if len(got_bytes)!=2*count or len(expected_bytes)!=2*count:
        raise SystemExit(f'expected two {2*count}-byte fp16 surfaces')
    got=struct.unpack(f'<{count}H',got_bytes)
    expected=struct.unpack(f'<{count}H',expected_bytes)
    at=lambda c,y,x:(c*args.height+y)*args.row_elements+x
    per_channel=[]; per_row=[]; per_column=[]
    for c in range(args.channels):
        per_channel.append(sum(got[at(c,y,x)]!=expected[at(c,y,x)]
                               for y in range(args.height) for x in range(args.width)))
    for y in range(args.height):
        per_row.append(sum(got[at(c,y,x)]!=expected[at(c,y,x)]
                           for c in range(args.channels) for x in range(args.width)))
    for x in range(args.width):
        per_column.append(sum(got[at(c,y,x)]!=expected[at(c,y,x)]
                              for c in range(args.channels) for y in range(args.height)))
    categories={'both_nonzero':0,'got_zero_only':0,'expected_zero_only':0}
    distances=[]
    for c in range(args.channels):
        for y in range(args.height):
            for x in range(args.width):
                a,b=got[at(c,y,x)],expected[at(c,y,x)]
                if a==b: continue
                if a in (0,0x8000) and b not in (0,0x8000):
                    categories['got_zero_only']+=1
                elif b in (0,0x8000) and a not in (0,0x8000):
                    categories['expected_zero_only']+=1
                else:
                    categories['both_nonzero']+=1
                distances.append(abs(ulp_key(a)-ulp_key(b)))
    tiles=[]
    for top in range(0,args.height,args.tile):
        row=[]
        for left in range(0,args.width,args.tile):
            row.append(sum(got[at(c,y,x)]!=expected[at(c,y,x)]
                           for c in range(args.channels)
                           for y in range(top,min(top+args.tile,args.height))
                           for x in range(left,min(left+args.tile,args.width))))
        tiles.append(row)
    logical=args.channels*args.height*args.width
    padding=args.channels*args.height*(args.row_elements-args.width)
    padding_wrong=sum(got[at(c,y,x)]!=expected[at(c,y,x)]
                      for c in range(args.channels) for y in range(args.height)
                      for x in range(args.width,args.row_elements))
    thresholds=(1,2,3,4,8,16,64,256,1024)
    print(json.dumps({
        'logical': {'exact':logical-len(distances),'total':logical,
                    'mismatches':len(distances)},
        'padding': {'exact':padding-padding_wrong,'total':padding,
                    'mismatches':padding_wrong},
        'mismatchCategories':categories,
        'ulp': {'maximum':max(distances,default=0),
                'atOrBelow':{str(n):sum(d<=n for d in distances) for n in thresholds}},
        'perChannelMismatches':per_channel,
        'exactChannels':[i for i,n in enumerate(per_channel) if not n],
        'perRowMismatches':per_row,
        'perColumnMismatches':per_column,
        'tileSize':args.tile,
        'tileMismatches':tiles,
    },indent=2))


if __name__=='__main__':
    main()

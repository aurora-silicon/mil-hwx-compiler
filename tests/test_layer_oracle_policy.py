#!/usr/bin/env python3
"""Privacy/failure policy and indexed coefficient proofs for native layers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
import mint_conv_probes as conv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",choices=("h13","h14"))
    args=parser.parse_args();count=indexed=0
    for target in (args.target,) if args.target else ("h13","h14"):
        for path in (ROOT/"research/oracles"/target).glob("layer_*.json"):
            r=json.loads(path.read_text())
            if not (r["case"].startswith("layer_conv") or r["family"].startswith("layer_")): continue
            c=r["compiler"]
            assert "host" not in c and c["tool"]=="ane-compile-hwx"
            assert c["command"][:3]==["ane-compile-hwx","CAPTURE_DIR","OUTPUT_DIR"]
            assert len(c["tool_sha256"])==64 and len(c["decoder_sha256"])==64
            if r["error"]:
                assert not r["task_descriptors"] and r["constant_section"] is None
            else:
                section=r["constant_section"]
                if section["size"]>256: assert section["nonzero_fp16_words"] is None
                assert len(section["sha256"])==64 and len(r["hwx_sha256"])==64
            if r["case"].startswith("layer_conv_indexed"):
                assert conv.covered(r,allow_multi_task=True),(target,r["case"])
                indexed+=1
            count+=1
    assert indexed>=20
    print(f"layer oracle policy: PASS ({count} records, {indexed} indexed packing proofs)")


if __name__=="__main__": main()

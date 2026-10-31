#!/usr/bin/env python3
"""Full native task/weight parity for the reported 128->512 3x3 @35 gather.

The upper output half runs first; the lower half reuses the cached input.
This is descriptor evidence, not a device numerical qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_l2_edge_parity import tasks
from mint_conv_probes import conv_case
from h13_td import decode_task, split_h14_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler",nargs="?",default="build/mil-hwxc")
    parser.add_argument("--target",choices=("H13","H14"),default="H13")
    args=parser.parse_args()
    compiler = str(Path(args.compiler).resolve())
    for cin,cout in ((128,512),(512,128)):
        for bias in (False, True):
            fact = json.loads((ROOT / "research/oracles" / args.target.lower() /
                f"layer_conv_k3_c{cin}_n{cout}_s35_st1_g1_bias{int(bias)}_same.json").read_text())
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                item = conv_case(3,cin,cout,35,1,1,bias)
                (root/"model.mil").write_text(item["mil"])
                (root/"weights.bin").write_bytes(item["weights"])
                subprocess.run([compiler,"--target",args.target,"--mil",str(root/"model.mil"),
                                "--model-root",tmp,"--output",str(root/"pkg")],
                               env=dict(os.environ,MILHWXC_H13_CONV_GENERIC="1"),
                               check=True,capture_output=True,timeout=120)
                manifest=json.loads((root/"pkg/manifest.json").read_text())
                program=manifest["programs"][0]
                assert program["taskDescriptors"]==2
                anec=(root/"pkg"/program["file"]).read_bytes()
                import struct
                first,count=struct.unpack_from("<II",anec,8)
                size=struct.unpack_from("<Q",anec,16)[0]
                from h13_td import split_h13_tasks
                split=(split_h13_tasks(anec[0x1000:0x1000+size],first//4-1,count)
                       if args.target=="H13" else split_h14_tasks(anec[0x1000:0x1000+size]))
                decoded=[decode_task(t,args.target.lower()) for t in split]
                from test_native_layer_cli import bound
                expected=[bound(t) for t in fact["task_descriptors"]] if args.target=="H13" else fact["task_descriptors"]
                assert decoded==expected,(args.target,bias,"full native task parity")
                constants=anec[0x1000+program["constantOffset"]:]
                assert hashlib.sha256(constants).hexdigest()==fact["constant_section"]["sha256"]
                registers=[{int(a,16):int(v,16) for b in task["blocks"].values() for a,v in b["words"].items()} for task in decoded]
                dst=0x17804 if args.target=="H13" else 0x1508
                split = 256 if cout==512 else 96
                assert registers[0][dst]==split*35*128 and registers[1][dst]==0
                assert registers[0][0x10]==cout-split and registers[1][0x10]==split
    if args.target=="H13":
        for ci,co,spatial,bias in ((128,512,40,False),(128,256,40,True),(128,1024,35,False),(512,128,40,False)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                item=conv_case(3,ci,co,spatial,1,1,bias)
                (root/"model.mil").write_text(item["mil"])
                (root/"weights.bin").write_bytes(item["weights"])
                run=subprocess.run([compiler,"--target","H13","--mil",str(root/"model.mil"),
                    "--model-root",tmp,"--output",str(root/"pkg")],capture_output=True,text=True,timeout=120)
                assert run.returncode and "h13.conv-outside-envelope" in run.stderr
    print(f"{args.target} camera conv @35: PASS (gather 256+256, mix 32+96, bias/no bias)")


if __name__ == "__main__":
    main()

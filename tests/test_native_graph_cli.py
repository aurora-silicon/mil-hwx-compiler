#!/usr/bin/env python3
"""Producer placement for zero-copy and materialized channel concat."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from layer_graph_cases import graph_case
from h13_l2_edge_parity import tasks
from h13_td import split_h14_tasks,decode_task


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("compiler",nargs="?",default="build/mil-hwxc")
    parser.add_argument("--target",choices=("H13","H14"),default="H13")
    args=parser.parse_args();compiler=str(Path(args.compiler).resolve())
    for shared,consumer,two,count in ((False,False,False,2),(True,False,False,4),
                                   (False,True,False,3),(True,True,False,5),(False,False,True,2)):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mil,blob=graph_case(shared,consumer,two_inputs=two)
            (root/"model.mil").write_text(mil);(root/"weights.bin").write_bytes(blob)
            run=subprocess.run([compiler,"--target",args.target,"--schedule","linked",
                "--mil",str(root/"model.mil"),"--model-root",tmp,"--output",str(root/"out")],
                capture_output=True,text=True,timeout=120)
            assert run.returncode==0,(args.target,shared,consumer,run.stderr)
            m=json.loads((root/"out/manifest.json").read_text());p=m["programs"][0]
            assert len(m["programs"])==1 and m["dispatchPlan"]==[0] and p["taskDescriptors"]==count
            assert m["intermediates"]==[]
            assert len(p["inputs"])==(2 if two else 1)
            assert p["layers"][2]["zeroCopy"]==(not shared)
            assert p["layers"][2]["taskDescriptors"]==(2 if shared else 0)
            if not shared and not consumer:
                assert m["scratch"]["bytes"]==0
            a=(root/"out"/p["file"]).read_bytes()
            if args.target=="H13":
                decoded=tasks(a)
                regs=[r for _,r in decoded]
                channels=[(h[8]>>12)&31 for h,_ in decoded]
                dst=0x17804
                if two: assert [h[8]&31 for h,_ in decoded]==[5,6]
            else:
                size=struct.unpack_from("<Q",a,16)[0]
                decoded=[decode_task(t,"h14") for t in split_h14_tasks(a[4096:4096+size])]
                regs=[{int(k,16):int(v,16) for b in t["blocks"].values() for k,v in b["words"].items()} for t in decoded]
                channels=[]
                for t in decoded:
                    bars={int(r["address"],16):(int(r["header"],16)>>23)&63 for r in t["records"]
                          if int(r["header"],16)&0xa0000000==0x20000000}
                    channels.append(bars.get(0x1508))
                dst=0x1508
                if two:
                    inputs=[]
                    for t in decoded:
                        inputs += [(int(r["header"],16)>>23)&63 for r in t["records"]
                            if int(r["address"],16)==0x1110 and int(r["header"],16)&0xa0000000==0x20000000]
                    assert inputs==[5,6]
            if not shared:
                assert regs[0][dst]==0 and regs[1][dst]==16*1024
                assert channels[:2]==([3,3] if consumer and args.target=="H13" else
                                     [7,7] if consumer else [4,4])
                if consumer:
                    assert m["scratch"]["regions"]["a"]["aliasOf"]=="z"
                    assert m["scratch"]["regions"]["b"]["offset"]==16*1024
                    assert channels[-1]==4
            else:
                concat_start=2
                assert regs[concat_start][dst]!=regs[concat_start+1][dst]
    print(f"{args.target} native graphs: PASS (zero-copy output/intermediate, shared producer copies)")


if __name__=="__main__": main()

#!/usr/bin/env python3
"""Native-layer DRAM edges, internal L2 tasks, constants and output spans."""
import argparse
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_l2_edge_parity import tasks
from h13_td import decode_task, split_h14_tasks
from mint_layer_probes import hardswish_arithmetic_case, resize_case, concat_case
from mint_conv_probes import conv_case


def activation_chain():
    mil = hardswish_arithmetic_case(16,16,16)["mil"]
    mil = mil.replace("    fp16 three", '    tensor<fp16, [1, 16, 16, 16]> z = sigmoid(x = x)[name = string("z")];\n    fp16 three')
    mil = mil.replace("add(x = x,", "add(x = z,").replace("mul(x = x,", "mul(x = z,")
    return mil


def resize_conv_chain():
    resize = resize_case("resize_nearest_neighbor",16,16,16,32,32)["mil"]
    conv = conv_case(5,16,32,32,1,1,False)
    body = conv["mil"].split("    ",1)[1].split("  } ->",1)[0]
    body = body.replace("x = x)", "x = r)")
    resize = resize.replace('> y = resize_', '> r = resize_').replace('  } -> (y);', '    ' + body + '  } -> (y);')
    return resize, conv["weights"]


def concat_conv_chain():
    mil = concat_case(16,16,16,16)["mil"].replace('> y = concat(', '> r = concat(')
    conv = conv_case(5,32,32,16,1,1,False)
    body = conv["mil"].split("    ",1)[1].split("  } ->",1)[0].replace("x = x)","x = r)")
    return mil.replace('  } -> (y);','    '+body+'  } -> (y);'),conv["weights"]


def camera_chain():
    gather=conv_case(3,128,512,35,1,1,True)
    mix=conv_case(3,512,128,35,1,1,True)
    body=mix["mil"].split("    ",1)[1].split("  } ->",1)[0]
    body="\n".join(line for line in body.splitlines() if
        " w = const(" in line or " b = const(" in line or " = conv(" in line)
    body=re.sub(r"\bw\b","w1",body)
    body=re.sub(r"\bb\b","b1",body)
    body=body.replace("weights.bin","weights1.bin").replace("x = x)","x = g)")
    mil=gather["mil"].replace('> y = conv(', '> g = conv(')
    return mil.replace('  } -> (y);','    '+body+'\n  } -> (y);'),{
        "weights.bin":gather["weights"],"weights1.bin":mix["weights"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler",nargs="?",default="build/mil-hwxc")
    parser.add_argument("--target",choices=("h13","h14"),default="h13")
    args = parser.parse_args()
    compiler = str(Path(args.compiler).resolve())
    for name, mil, weights in (("activation",activation_chain(),None),
                               ("resize-conv",*resize_conv_chain()),
                               ("concat-conv",*concat_conv_chain()),
                               ("camera",*camera_chain())):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"model.mil").write_text(mil)
            if isinstance(weights,dict):
                for path,payload in weights.items(): (root/path).write_bytes(payload)
            elif weights: (root/"weights.bin").write_bytes(weights)
            run = subprocess.run([compiler,"--target",args.target.upper(),"--schedule","linked",
                                  "--mil",str(root/"model.mil"),"--model-root",tmp,"--output",str(root/"out")],
                                 capture_output=True,text=True,timeout=120)
            assert run.returncode==0,(name,run.stderr)
            m=json.loads((root/"out/manifest.json").read_text())
            assert len(m["programs"])==1 and m["dispatchPlan"]==[0]
            p=m["programs"][0]
            anec=(root/"out"/p["file"]).read_bytes()
            if args.target=="h14":
                size=struct.unpack_from("<Q",anec,16)[0]
                decoded=[decode_task(t,"h14") for t in split_h14_tasks(anec[0x1000:0x1000+size])]
                assert len(decoded)==p["taskDescriptors"]
                for index,task in enumerate(decoded):
                    assert int(task["header_words"][0],16)&0xffff==index
                    assert bool(int(task["header_words"][2],16)&0x22)==(index==len(decoded)-1)
                def regs(t):
                    return {int(a,16):int(v,16) for b in t["blocks"].values() for a,v in b["words"].items()}
                def bars(t):
                    return {int(r["address"],16):(int(r["header"],16)>>23)&63 for r in t["records"]
                            if int(r["header"],16)&0xa0000000==0x20000000}
                reader=1 if name=="activation" else 2 if name=="camera" else len(decoded)-1
                r=regs(decoded[reader])
                assert bars(decoded[reader])[0x1110]==7
                assert r[0x1100]>>28==1 and (r[0x1100]>>16)&255==reader
                assert r[0x1108]==0xee
                assert bars(decoded[-1])[0x1508]==4
                assert struct.unpack_from("<I",anec,0x28+3*4)[0]*0x4000==m["scratch"]["bytes"]
                if name=="activation":
                    assert bars(decoded[2])[0x1110]==1
                    assert regs(decoded[2])[0x1110]==128
                    for t in decoded[3:]:
                        assert 0x1110 not in bars(t) and 0x1100 not in regs(t)
                if name=="concat-conv":
                    assert len(p["inputs"])==2
                    assert bars(decoded[0])[0x1110]==6
                    assert bars(decoded[1])[0x1110]==5
                if name=="camera":
                    assert len(decoded)==4 and len(p["inputs"])==1
                    assert [regs(t)[0x1508] for t in decoded]==[256*35*128,0,96*35*128,0]
                    assert [bars(t)[0x1508] for t in decoded]==[7,7,4,4]
                    assert 0x1110 not in bars(decoded[1]) and 0x1110 not in bars(decoded[3])
                continue
            decoded=tasks(anec)
            assert len(decoded)==p["taskDescriptors"]
            if name=="activation":
                assert len(decoded)==5 # sigmoid, four internal hardswish tasks
                # Scale staging reads the relocated constant BAR, not the
                # preceding layer's activation. Actual x read is task 2.
                h0,r0=decoded[1];h1,r1=decoded[2]
                assert h0[8]&31==1
                assert r0[0x13808]==2048
                assert h1[8]&31==3
                assert r1[0x13800]==0x3bb81 # producer is two tasks back
                assert r1[0x1f848]==2048+128
                assert not r0[0x17800]&1
                assert not decoded[3][1][0x13800]&1
            elif name=="resize-conv":
                assert len(decoded)==2
                assert decoded[0][0][8]>>12&31==3
                assert decoded[1][0][8]&31==3
                assert decoded[1][1][0x13800]==0x7bb81
            elif name=="concat-conv":
                assert len(decoded)==3 and len(p["inputs"])==2
                assert decoded[0][0][8]&31==6
                assert decoded[1][0][8]&31==5  # ANEC binding swaps input role 4->5
                assert decoded[2][1][0x13800]==0x7bb81
            else:
                assert len(decoded)==4 and len(p["inputs"])==1
                assert [r[0x17804] for _,r in decoded]==[256*35*128,0,96*35*128,0]
                assert [(h[8]>>12)&31 for h,_ in decoded]==[3,3,4,4]
                assert [r[0x17800] for _,r in decoded]==[0x3c1,0x3c1,0xc1,0xc1]
                assert decoded[2][1][0x13800]==0x7bb81
                assert not decoded[1][1][0x13800]&1 and not decoded[3][1][0x13800]&1
            for i,(header,_) in enumerate(decoded):
                assert bool(header[2]&0x22)==(i==len(decoded)-1)
                assert bool(header[8]&(1<<23))==(i!=len(decoded)-1)
    print(f"native {args.target.upper()} linked: PASS (resize->conv5, sigmoid->hardswish, concat->conv5, camera gather->mix)")


if __name__=="__main__": main()

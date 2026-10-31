#!/usr/bin/env python3
"""Whole-layer field/constant parity, alias spellings, and named refusals."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from generate_layer_templates import records
from h13_td import decode_task, split_h13_tasks, split_h14_tasks


def bound(task):
    header = [int(w, 16) for w in task["header_words"]]
    for shift in (0, 6, 12):
        channel = (header[8] >> shift) & 31
        if channel >= 4:
            channel = {4: 5, 5: 4, 6: 6}[channel]
            header[8] = (header[8] & ~(31 << shift)) | channel << shift
    header[0] = (header[0] & ~0x00ff0000) | 0x00400000
    return dict(task, header_words=[f"0x{w:08x}" for w in header])


def bound_h14(task):
    records = []
    for record in task["records"]:
        h = int(record["header"], 16)
        if h & 0xa0000000 == 0x20000000:
            address, slot = int(record["address"],16), (h >> 23) & 63
            if address in (0x1110,0x1128) and slot == 4: slot = 5
            elif address == 0x1508 and slot == 5: slot = 4
            h = (h & ~(63 << 23)) | (slot << 23)
        records.append(dict(record,header=f"0x{h:08x}"))
    return dict(task,records=records)


def compile_case(compiler, target, root, mil, success=True, artifact_format="anec"):
    (root / "model.mil").write_text(mil)
    run = subprocess.run([compiler, "--target", target.upper(), "--mil", str(root / "model.mil"),
                          "--model-root", str(root), "--format", artifact_format,
                          "--output", str(root / ("out" if artifact_format=="anec" else "out-hwx"))], capture_output=True, text=True, timeout=120)
    if not success:
        assert run.returncode != 0, run.stdout
        return
    assert run.returncode == 0, run.stderr
    return json.loads((root / ("out" if artifact_format=="anec" else "out-hwx") / "manifest.json").read_text())


def hwx_sections(data):
    sections={};cursor=32
    for _ in range(struct.unpack_from("<I",data,16)[0]):
        command,size=struct.unpack_from("<II",data,cursor)
        if command==0x19:
            segment=struct.unpack_from("<2I16s4Q4I",data,cursor)
            for index in range(segment[-2]):
                s=struct.unpack_from("<16s16s2Q8I",data,cursor+72+80*index)
                sections[(segment[2].rstrip(b"\0"),s[0].rstrip(b"\0"))]=data[s[4]:s[4]+s[3]]
        cursor+=size
    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler", nargs="?", default="build/mil-hwxc")
    parser.add_argument("--target", choices=("h13", "h14"))
    args = parser.parse_args()
    compiler = str(Path(args.compiler).resolve())
    count = 0
    cases = list(records())
    for target in ("h13", "h14"):
        for path in sorted((ROOT / "research/oracles" / target).glob("layer_upsample_*.json")):
            oracle = json.loads(path.read_text())
            if not oracle.get("error"):
                cases.append((target, None, None, oracle))
    for target, operation, shape, oracle in cases:
        if args.target and target != args.target:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = compile_case(compiler, target, root, oracle["mil"])
            assert len(manifest["programs"]) == 1, oracle["case"]
            program = manifest["programs"][0]
            assert program["taskDescriptors"] == len(oracle["task_descriptors"]), oracle["case"]
            assert not any("slice" in v for v in program["inputs"] + program["outputs"]), oracle["case"]
            anec = (root / "out" / program["file"]).read_bytes()
            first, tasks = struct.unpack_from("<II", anec, 8)
            size = struct.unpack_from("<Q", anec, 16)[0]
            stream = anec[0x1000:0x1000 + size]
            decoded = [decode_task(t, target) for t in (
                split_h13_tasks(stream, first // 4 - 1, tasks) if target == "h13" else split_h14_tasks(stream))]
            expected = [(bound if target == "h13" else bound_h14)(t) for t in oracle["task_descriptors"]]
            assert decoded == expected, (oracle["case"], "task field mismatch")
            constants = anec[0x1000 + program["constantOffset"]:]
            assert hashlib.sha256(constants).hexdigest() == oracle["constant_section"]["sha256"], oracle["case"]
            for binding, tensor in zip(program["inputs"] + program["outputs"], oracle["tensor_descriptors"]):
                n, c, h, w, plane, row = binding["nchw"]
                assert [n,c,h,w] == tensor["shape"]
                assert [plane*c,plane,row,2] == tensor["strides"]
            hwx=compile_case(compiler,target,root,oracle["mil"],artifact_format="hwx")
            hp=hwx["programs"][0]
            sections=hwx_sections((root/"out-hwx"/hp["file"]).read_bytes())
            hs=sections[(b"__TEXT",b"__text")]
            raw=(split_h13_tasks(hs,oracle["task_descriptors"][0]["size_bytes"]//4-1,tasks)
                 if target=="h13" else split_h14_tasks(hs))
            # H13 HWX uses the same explicit binding roles as its ANEC;
            # H14 HWX restores the oracle BAR roles from normalized ANEC.
            hwx_expected=expected if target=="h13" else oracle["task_descriptors"]
            assert [decode_task(t,target) for t in raw]==hwx_expected,(oracle["case"],"HWX task fields")
            assert hashlib.sha256(sections[(b"__TEXT",b"__const")]).hexdigest()==oracle["constant_section"]["sha256"]
            count += 1
    for target in (args.target,) if args.target else ("h13", "h14"):
        base = json.loads((ROOT / "research/oracles" / target /
            "layer_resize_nearest_neighbor_c16_16x16_32x32.json").read_text())["mil"]
        for source in (base.replace("int32(32)", "int32(48)").replace("16, 32, 32", "16, 48, 48"),
                       base.replace("[1, 16, 16, 16]", "[2, 16, 16, 16]").replace("[1, 16, 32, 32]", "[2, 16, 32, 32]"),
                       base.replace("int32(32)", "fp32(32)")):
            with tempfile.TemporaryDirectory() as tmp:
                compile_case(compiler, target, Path(tmp), source, success=False)
    print(f"native layer CLI: PASS ({count} exact field/constant points, ANEC and HWX)")


if __name__ == "__main__":
    main()

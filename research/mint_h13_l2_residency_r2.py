#!/usr/bin/env python3
"""Mint decoded-only H13 L2-residency round-2 oracle facts.

Apple-generated HWX is created only inside a TemporaryDirectory.  The
retained JSON contains hashes and decoded field values, never HWX bytes.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any

from mint_oracles import parse_hwx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research" / "oracles" / "h13-l2-residency-r2"
DEFAULT_TOOL = Path("/private/tmp/t3892.o8GtyF/ane-compile-hwx")
SOURCE_COMMIT = "e03ca24a025b1020e2e6abf6ee2542f7864f9329"


def blob(elements: int) -> bytes:
    payload = b"\x00\x38" * elements
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IQQ", result, 64, 0xDEADBEEF, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def layer(kind: str, channels_in: int, channels_out: int,
          stride: int = 1) -> dict[str, int | str]:
    if kind not in {"pw", "dw"}:
        raise ValueError(f"unknown layer kind {kind}")
    if kind == "dw" and channels_in != channels_out:
        raise ValueError("depthwise layers must preserve channels")
    return {
        "kind": kind,
        "channels_in": channels_in,
        "channels_out": channels_out,
        "kernel": 3 if kind == "dw" else 1,
        "groups": channels_in if kind == "dw" else 1,
        "stride": stride,
    }


def alternating(channels: int, count: int = 6) -> list[dict[str, int | str]]:
    return [layer("dw" if index % 2 == 0 else "pw", channels, channels)
            for index in range(count)]


def mobile_triplet(channels_in: int, channels_out: int,
                   spatial: int, stride: int) -> dict[str, Any]:
    return {
        "case": (f"mobile_dw_pw_dw_c{channels_in}_n{channels_out}_"
                 f"s{spatial}_st{stride}_h13"),
        "spatial": spatial,
        "layers": [
            layer("dw", channels_in, channels_in),
            layer("pw", channels_in, channels_out, stride),
            layer("dw", channels_out, channels_out),
        ],
        "purpose": "multi-task consumer and multi-task producer",
    }


def cases() -> list[dict[str, Any]]:
    result = [
        mobile_triplet(128, 256, 56, 2),
        mobile_triplet(256, 512, 28, 2),
        mobile_triplet(512, 1024, 14, 2),
        mobile_triplet(1024, 1024, 7, 1),
        mobile_triplet(1024, 1024, 16, 1),
        mobile_triplet(1024, 2048, 16, 1),
        mobile_triplet(1024, 4096, 16, 1),
    ]
    for spatial, channels in ((28, 256), (14, 512), (7, 1024)):
        result.append({
            "case": f"mobile_tail6_c{channels}_s{spatial}_h13",
            "spatial": spatial,
            "layers": alternating(channels),
            "purpose": "six-layer dependent/alias chain",
        })
    result.append({
        "case": "pw_dw_pw_c64_s7_h13",
        "spatial": 7,
        "layers": [layer("pw", 64, 64), layer("dw", 64, 64),
                   layer("pw", 64, 64)],
        "purpose": "single-task 7x7 edge forms",
    })
    for channels in (128, 160, 176, 192, 256):
        result.append({
            "case": f"pw_dw_pw_c{channels}_s56_h13",
            "spatial": 56,
            "layers": [layer("pw", channels, channels),
                       layer("dw", channels, channels),
                       layer("pw", channels, channels)],
            "purpose": "56x56 placement boundary",
        })
    return result


def make_mil(item: dict[str, Any], capture: Path) -> str:
    spatial = item["spatial"]
    channels = item["layers"][0]["channels_in"]
    current = "x"
    current_spatial = spatial
    lines: list[str] = []
    for index, spec in enumerate(item["layers"]):
        if spec["channels_in"] != channels:
            raise ValueError(f"layer {index} input does not match predecessor")
        kernel = int(spec["kernel"])
        groups = int(spec["groups"])
        stride = int(spec["stride"])
        outputs = int(spec["channels_out"])
        padding = (kernel - 1) // 2
        output_spatial = (current_spatial + stride - 1) // stride
        elements = outputs * (channels // groups) * kernel * kernel
        (capture / f"w{index}.bin").write_bytes(blob(elements))
        lines.extend([
            f'string pt{index} = const()[name = string("pt{index}"), '
            f'val = string("{"custom" if padding else "same"}")];',
            f'tensor<int32, [2]> st{index} = const()[name = string("st{index}"), '
            f'val = tensor<int32, [2]>([{stride}, {stride}])];',
            f'tensor<int32, [4]> pd{index} = const()[name = string("pd{index}"), '
            f'val = tensor<int32, [4]>([{padding}, {padding}, {padding}, {padding}])];',
            f'tensor<int32, [2]> dl{index} = const()[name = string("dl{index}"), '
            'val = tensor<int32, [2]>([1, 1])];',
            f'int32 gp{index} = const()[name = string("gp{index}"), val = int32({groups})];',
            f'tensor<fp16, [{outputs}, {channels // groups}, {kernel}, {kernel}]> '
            f'w{index} = const()[name = string("w{index}"), val = tensor<fp16, '
            f'[{outputs}, {channels // groups}, {kernel}, {kernel}]>'
            f'(BLOBFILE(path = string("@model_path/w{index}.bin"), offset = uint64(64)))];',
            f'tensor<fp16, [1, {outputs}, {output_spatial}, {output_spatial}]> c{index} = '
            f'conv(dilations = dl{index}, groups = gp{index}, pad = pd{index}, '
            f'pad_type = pt{index}, strides = st{index}, weight = w{index}, '
            f'x = {current})[name = string("c{index}")];',
        ])
        current = f"c{index}"
        channels = outputs
        current_spatial = output_spatial
    body = "\n    ".join(lines)
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {item['layers'][0]['channels_in']}, {spatial}, {spatial}]> x) {{
    {body}
  }} -> ({current});
}}
"""


def mint(item: dict[str, Any], tool: Path, output: Path) -> None:
    destination = output / f"{item['case']}.json"
    with tempfile.TemporaryDirectory(prefix=f"t4020-{item['case']}-") as root:
        temporary = Path(root)
        capture = temporary / "capture"
        compiled = temporary / "compiled"
        capture.mkdir()
        compiled.mkdir()
        mil = make_mil(item, capture)
        (capture / "model.mil").write_text(mil)
        result = subprocess.run(
            [str(tool), str(capture), str(compiled), "h13"],
            capture_output=True, text=True, timeout=900, check=False)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        raw = (compiled / "model.hwx").read_bytes()
        record = {
            "schema_version": 1,
            "case": item["case"],
            "target": "h13",
            "purpose": item["purpose"],
            "parameters": {
                "input_spatial": item["spatial"],
                "layers": item["layers"],
            },
            "source_commit": SOURCE_COMMIT,
            "mil_sha256": hashlib.sha256(mil.encode()).hexdigest(),
            "compiler_tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256(
                (ROOT / "research" / "h13_td.py").read_bytes()).hexdigest(),
            **parse_hwx(raw, "h13"),
        }
    output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{item['case']}: tasks={len(record['task_descriptors'])} "
          f"hwx_sha256={record['hwx_sha256']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [item for item in cases()
                if not args.case or fnmatch.fnmatch(item["case"], args.case)]
    if args.list:
        for item in selected:
            print(item["case"])
        return
    if not args.tool.is_file():
        raise SystemExit(f"missing compiler tool: {args.tool}")
    for item in selected:
        destination = args.output / f"{item['case']}.json"
        if destination.exists() and not args.force:
            print(f"{item['case']}: exists", flush=True)
            continue
        mint(item, args.tool, args.output)


if __name__ == "__main__":
    main()

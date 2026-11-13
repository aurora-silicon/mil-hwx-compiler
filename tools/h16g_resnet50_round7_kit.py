#!/usr/bin/env python3
"""Assemble the H16G ResNet-50 round-7 reviewer and morph kit.

The compiler-built ladder supplies the coherent fixes.  Round-6 objects are
kept as immutable controls, and diagnostic morphs patch exactly one existing
32-bit operand at a time; no record header or descriptor boundary is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

from h16g_load_grammar import check, object_fields, walk


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint(path: Path) -> tuple[bytes, dict, list[dict], dict]:
    data = path.read_bytes()
    fields = object_fields(data)
    facts = walk(fields, include_tasks=True, enforce_residual=False)
    tasks = facts.pop("_tasks")
    return data, fields, tasks, facts


def hex32(value: int | None) -> str | None:
    return None if value is None else f"0x{value:08x}"


def fields(task: dict, addresses: tuple[int, ...]) -> dict:
    return {
        "header": [hex32(value) for value in task["header"]],
        "registers": {f"0x{address:04x}": hex32(task["registers"].get(address))
                      for address in addresses},
    }


def task_diff(left: dict, right: dict) -> dict:
    keys = sorted(set(left["registers"]) | set(right["registers"]))
    return {
        "header": [{"word": index, "left": hex32(a), "right": hex32(b)}
                   for index, (a, b) in enumerate(zip(left["header"],
                                                       right["header"]))
                   if a != b],
        "registers": [{"address": f"0x{key:04x}",
                       "left": hex32(left["registers"].get(key)),
                       "right": hex32(right["registers"].get(key))}
                      for key in keys if left["registers"].get(key) !=
                      right["registers"].get(key)],
    }


def patch_header(ep: tuple, task_index: int, word: int, value: int) -> bytes:
    data, image_fields, tasks, _ = ep
    image = bytearray(data)
    task = tasks[task_index]
    struct.pack_into("<I", image,
                     image_fields["text_off"] + task["offset"] + 4 * word,
                     value)
    return bytes(image)


def patch_register(ep: tuple, task_index: int, address: int,
                   value: int) -> bytes:
    data, image_fields, tasks, _ = ep
    task = tasks[task_index]
    if address not in task["register_offsets"]:
        raise ValueError(f"task {task_index} lacks register 0x{address:x}")
    image = bytearray(data)
    struct.pack_into("<I", image,
                     image_fields["text_off"] +
                     task["register_offsets"][address], value)
    return bytes(image)


def write_morph(path: Path, payload: bytes) -> dict:
    path.write_bytes(payload)
    facts = check(path)
    facts.pop("path", None)
    return {"name": path.stem, "file": path.name,
            "sha256": sha256(path), "facts": facts}


def copy_case(source: Path, target: Path) -> None:
    shutil.copytree(source, target, symlinks=True)


def write_scripts(root: Path) -> None:
    scripts = {
        "run-one.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
R=${1:?usage: run-one.sh 01-stem|02-stem-maxpool|03-projection-block|04-first-stage|05-through-stage3|06-through-stage4|07-full [OUTPUT_DIR]}
case "$R" in 01-stem|02-stem-maxpool|03-projection-block|04-first-stage|05-through-stage3|06-through-stage4|07-full) ;; *) exit 64 ;; esac
O=${2:-/var/tmp/resnet50-h16g-rb7-$R}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$K/ladder/$R/program/program-0.hwx" --case "$K/ladder/$R/refs/input_00.bin:$K/ladder/$R/refs/logits_00.fp16" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$K/ladder/$R/refs/logits_00.fp16" | tee "$O/compare.log"
grep -E 'loaded|p50|PASS' "$O/run.log" || true
''',
        "run-one-bisect.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
C=${1:?usage: run-one-bisect.sh T1-stem-alone|T2-accepted-full|T3-fixed-stem-head|T5-head-alone [OUTPUT_DIR]}
case "$C" in T1-stem-alone) E=stem_00.surface.fp16 ;; T2-accepted-full|T3-fixed-stem-head|T5-head-alone) E=logits_00.fp16 ;; *) exit 64 ;; esac
O=${2:-/var/tmp/resnet50-h16g-rb7-$C}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$K/task-bisect/$C/program/program-0.hwx" --case "$K/task-bisect/$C/refs/input_00.bin:$K/task-bisect/$C/refs/$E" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$K/task-bisect/$C/refs/$E" | tee "$O/compare.log"
grep -E 'loaded|p50|PASS' "$O/run.log" || true
''',
        "run-one-boundary.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
B=${1:?usage: run-one-boundary.sh 01-stem|02-stem-maxpool|03-projection-block|04-first-stage [OUTPUT_DIR]}
case "$B" in 01-stem|02-stem-maxpool|03-projection-block|04-first-stage) ;; *) exit 64 ;; esac
O=${2:-/var/tmp/resnet50-h16g-rb7-boundary-$B}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$K/boundary-dumps/$B/program/program-0.hwx" --case "$K/boundary-dumps/$B/refs/input_00.bin:$K/boundary-dumps/$B/refs/boundary_00.surface.fp16" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$K/boundary-dumps/$B/refs/boundary_00.surface.fp16" | tee "$O/compare.log"
grep -E 'loaded|p50|PASS' "$O/run.log" || true
''',
        "run-one-morph.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
G=${1:?usage: run-one-morph.sh terminal-stem|maxpool NAME [OUTPUT_DIR]}; N=${2:?missing morph name}
case "$G" in terminal-stem) E="$K/task-bisect/T1-stem-alone/refs/stem_00.surface.fp16" ;; maxpool) E="$K/ladder/02-stem-maxpool/refs/logits_00.fp16" ;; *) exit 64 ;; esac
P="$K/morphs/$G/$N.hwx"; test -f "$P" || exit 64
O=${3:-/var/tmp/resnet50-h16g-rb7-$G-$N}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$P" --case "$K/ladder/01-stem/refs/input_00.bin:$E" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$E" | tee "$O/compare.log"
grep -E 'loaded|p50|PASS' "$O/run.log" || true
''',
    }
    for name, payload in scripts.items():
        path = root / name
        path.write_text(payload)
        path.chmod(0o755)


def write_verify(root: Path) -> None:
    payload = r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd); cd "$K"
shasum -a 256 -c SHA256SUMS
for object in ladder/*/program/program-0.hwx boundary-dumps/*/program/program-0.hwx task-bisect/T?-*/program/program-0.hwx morphs/*/*.hwx r6-baselines/*.hwx; do
  python3 h16g_validate.py "$object" >/dev/null
  python3 h16g_load_grammar.py "$object" >/dev/null
done
for json in ladder.json task-bisect/manifest.json terminal-result-fields.json maxpool-boundary-fields.json morphs/manifest.json STATUS.json; do python3 -m json.tool "$json" >/dev/null; done
python3 - <<'PY'
import json
from pathlib import Path
from h16g_load_grammar import object_fields, walk
root=Path('.')
def tasks(path):
    return walk(object_fields(path.read_bytes()),include_tasks=True,enforce_residual=False)['_tasks']
r6=json.loads((root/'r6-baselines/manifest.json').read_text())
for entry in r6['files']:
    import hashlib
    assert hashlib.sha256((root/'r6-baselines'/entry['file']).read_bytes()).hexdigest()==entry['sha256']
t1=tasks(root/'task-bisect/T1-stem-alone/program/program-0.hwx')[0]
assert t1['registers'][0x4148]==0x00500170
pool=tasks(root/'ladder/02-stem-maxpool/program/program-0.hwx')[1]
assert pool['registers'][0x5110]==128 and pool['registers'][0x5114]==7168
accepted=tasks(root/'task-bisect/T2-accepted-full/program/program-0.hwx')[1]
assert accepted['registers'][0x5110]==896 and accepted['registers'][0x5114]==50176
manifest=json.loads((root/'morphs/manifest.json').read_text())
for group in manifest['groups'].values():
    baseline=tasks(root/'morphs'/group['directory']/group['baseline'])[group['task']]
    for entry in group['one_word']:
        candidate=tasks(root/'morphs'/group['directory']/entry['file'])[group['task']]
        diffs=sum(a!=b for a,b in zip(baseline['header'],candidate['header']))
        keys=set(baseline['registers'])|set(candidate['registers'])
        diffs+=sum(baseline['registers'].get(k)!=candidate['registers'].get(k) for k in keys)
        assert diffs==1,(entry['name'],diffs)
PY
for script in run-one.sh run-one-bisect.sh run-one-boundary.sh run-one-morph.sh verify.sh; do sh -n "$script"; done
echo PASS
'''
    path = root / "verify.sh"
    path.write_text(payload)
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rb6-kit", type=Path, required=True)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--load-grammar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must not exist")
    root = args.out_dir
    root.mkdir(parents=True)

    for name in ("LICENSE", "ane-run.py", "compare.py", "cpu-reference.py",
                 "h16-sections.py"):
        shutil.copy2(args.rb6_kit / name, root / name)
    shutil.copy2(args.validator, root / "h16g_validate.py")
    shutil.copy2(args.load_grammar, root / "h16g_load_grammar.py")
    for name in ("source", "ladder", "boundary-dumps"):
        shutil.copytree(args.ladder_dir / name, root / name, symlinks=True)
    shutil.copy2(args.ladder_dir / "ladder.json", root / "ladder.json")

    task_bisect = root / "task-bisect"
    t1 = task_bisect / "T1-stem-alone"
    copy_case(root / "boundary-dumps/01-stem", t1)
    (t1 / "refs/boundary_00.surface.fp16").rename(
        t1 / "refs/stem_00.surface.fp16")
    copy_case(args.rb6_kit / "task-bisect/T2-accepted-full",
              task_bisect / "T2-accepted-full")
    copy_case(root / "ladder/01-stem", task_bisect / "T3-fixed-stem-head")
    copy_case(args.rb6_kit / "task-bisect/T5-head-alone",
              task_bisect / "T5-head-alone")

    r6_t1 = endpoint(args.rb6_kit /
                     "task-bisect/T1-stem-alone/program/program-0.hwx")
    r7_t1 = endpoint(t1 / "program/program-0.hwx")
    wrapped = endpoint(root / "ladder/01-stem/program/program-0.hwx")
    terminal_addresses = (0x4148, 0x414c, 0x4150, 0x4154, 0x4158, 0x415c,
                          0x5100, 0x5108, 0x510c, 0x5110, 0x5114, 0x5138,
                          0x5900, 0x5904, 0x5908, 0x5918, 0x5928, 0x592c)
    t1_package = json.loads((t1 / "program/manifest.json").read_text())
    t1_artifact = t1_package["artifacts"][0]
    t1_output = next(buffer for buffer in
                     t1_artifact["h16Execute"]["buffers"]
                     if buffer["class"] == 1)
    t1_binding = next(binding for binding in t1_artifact["bindings"]
                      if binding["role"] == "output")
    t1_registers = r7_t1[2][0]["registers"]
    t1_dma_extent = (63 * t1_registers[0x5114] +
                     111 * t1_registers[0x5110] + 2 * 112)
    terminal_table = {
        "schema": "mil-hwxc.h16g-resnet-terminal-result.v1",
        "r6_terminal": fields(r6_t1[2][0], terminal_addresses),
        "r7_terminal": fields(r7_t1[2][0], terminal_addresses),
        "wrapped_stem": fields(wrapped[2][0], terminal_addresses),
        "r6_to_r7": task_diff(r6_t1[2][0], r7_t1[2][0]),
        "dart_mapped_result": {
            "bar_slot": 7,
            "offset_bytes": t1_output["offsetBytes"],
            "mapping_bytes": t1_output["mappingBytes"],
            "allocation_bytes": t1_binding["allocationBytes"],
            "dma_extent_bytes": t1_dma_extent,
            "extent_within_mapping": t1_dma_extent <=
                                     t1_output["mappingBytes"],
        },
        "conclusion": "terminal TileDMA keeps its external surface fields; the sparse producer ResultCfg is invariant",
    }
    (root / "terminal-result-fields.json").write_text(
        json.dumps(terminal_table, indent=2) + "\n")

    r6_pool = endpoint(args.rb6_kit /
                       "ladder/02-stem-maxpool/program/program-0.hwx")
    r7_pool = endpoint(root / "ladder/02-stem-maxpool/program/program-0.hwx")
    accepted = endpoint(task_bisect /
                        "T2-accepted-full/program/program-0.hwx")
    pool_addresses = (0x14, 0x18, 0x1c, 0x28, 0x34, 0x3c, 0x4100,
                      0x4104, 0x4114, 0x4118, 0x4148, 0x414c, 0x5100,
                      0x5110, 0x5114, 0x5138)
    pool_table = {
        "schema": "mil-hwxc.h16g-resnet-maxpool-boundary.v2",
        "r6_pool_to_mean": fields(r6_pool[2][1], pool_addresses),
        "r7_pool_to_mean": fields(r7_pool[2][1], pool_addresses),
        "accepted_pool_to_conv": fields(accepted[2][1], pool_addresses),
        "r6_to_r7": task_diff(r6_pool[2][1], r7_pool[2][1]),
        "conclusion": "the result DMA row and plane must match the planar probe-mean source; the accepted convolution consumer remains blocked",
    }
    (root / "maxpool-boundary-fields.json").write_text(
        json.dumps(pool_table, indent=2) + "\n")

    morph_root = root / "morphs"
    terminal = morph_root / "terminal-stem"
    pool = morph_root / "maxpool"
    terminal.mkdir(parents=True)
    pool.mkdir(parents=True)
    (terminal / "00-rb6-baseline.hwx").write_bytes(r6_t1[0])
    (pool / "00-rb6-baseline.hwx").write_bytes(r6_pool[0])
    terminal_one = [
        write_morph(terminal / "c01-dead-execution-cost.hwx",
                    patch_header(r6_t1, 0, 1, r6_t1[2][0]["header"][1] + 1)),
        write_morph(terminal / "t01-l2-result-config.hwx",
                    patch_register(r6_t1, 0, 0x4148,
                                   r7_t1[2][0]["registers"][0x4148])),
        write_morph(terminal / "t02-result-dma-row.hwx",
                    patch_register(r6_t1, 0, 0x5110, 224)),
        write_morph(terminal / "t03-result-dma-plane.hwx",
                    patch_register(r6_t1, 0, 0x5114, 224 * 112)),
        write_morph(terminal / "t04-surface-direction.hwx",
                    patch_register(r6_t1, 0, 0x5138, 0x08302031)),
        write_morph(terminal / "t05-result-dma-enable.hwx",
                    patch_register(r6_t1, 0, 0x5100, 0x010000f1)),
    ]
    terminal_one[0]["expected"] = "byte-identical output; scheduling hint only"
    for entry in terminal_one[1:]:
        entry["expected"] = "discriminating candidate"
    (terminal / "99-rb7-fixed.hwx").write_bytes(r7_t1[0])
    pool_one = [
        write_morph(pool / "c01-dead-execution-cost.hwx",
                    patch_header(r6_pool, 1, 1,
                                 r6_pool[2][1]["header"][1] + 1)),
        write_morph(pool / "p01-result-dma-row.hwx",
                    patch_register(r6_pool, 1, 0x5110,
                                   r7_pool[2][1]["registers"][0x5110])),
        write_morph(pool / "p02-result-dma-plane.hwx",
                    patch_register(r6_pool, 1, 0x5114,
                                   r7_pool[2][1]["registers"][0x5114])),
        write_morph(pool / "p03-piece-width-16.hwx",
                    patch_register(r6_pool, 1, 0x3c,
                                   (r6_pool[2][1]["registers"][0x3c] & ~15) | 4)),
        write_morph(pool / "p04-piece-width-32.hwx",
                    patch_register(r6_pool, 1, 0x3c,
                                   (r6_pool[2][1]["registers"][0x3c] & ~15) | 8)),
        write_morph(pool / "p05-surface-direction.hwx",
                    patch_register(r6_pool, 1, 0x5138, 0x01302031)),
    ]
    pool_one[0]["expected"] = "byte-identical output; scheduling hint only"
    for entry in pool_one[1:]:
        entry["expected"] = "discriminating candidate"
    (pool / "99-rb7-fixed.hwx").write_bytes(r7_pool[0])
    morph_manifest = {"order": "least-likely-to-wedge first", "runs": 3,
                      "groups": {
        "terminal-stem": {"directory": "terminal-stem", "task": 0,
            "baseline": "00-rb6-baseline.hwx", "one_word": terminal_one,
            "fixed": "99-rb7-fixed.hwx"},
        "maxpool": {"directory": "maxpool", "task": 1,
            "baseline": "00-rb6-baseline.hwx", "one_word": pool_one,
            "fixed": "99-rb7-fixed.hwx"},
    }}
    (morph_root / "manifest.json").write_text(
        json.dumps(morph_manifest, indent=2) + "\n")

    baselines = root / "r6-baselines"
    baselines.mkdir()
    baseline_sources = {
        "T2-accepted-full.hwx": args.rb6_kit / "task-bisect/T2-accepted-full/program/program-0.hwx",
        "T3-fixed-stem-head.hwx": args.rb6_kit / "task-bisect/T3-fixed-stem-head/program/program-0.hwx",
        "T5-head-alone.hwx": args.rb6_kit / "task-bisect/T5-head-alone/program/program-0.hwx",
        "ladder-01-stem.hwx": args.rb6_kit / "ladder/01-stem/program/program-0.hwx",
        "S4-result-dma-layout.hwx": args.rb6_kit / "maxpool-switches/S4-result-dma-layout.hwx",
    }
    baseline_manifest = []
    for name, source in baseline_sources.items():
        shutil.copy2(source, baselines / name)
        baseline_manifest.append({"file": name, "sha256": sha256(source)})
    (baselines / "manifest.json").write_text(json.dumps(
        {"source": "round 6", "files": baseline_manifest}, indent=2) + "\n")

    cases = []
    for name in ("T1-stem-alone", "T2-accepted-full",
                 "T3-fixed-stem-head", "T5-head-alone"):
        path = task_bisect / name / "program/program-0.hwx"
        facts = check(path); facts.pop("path", None)
        cases.append({"name": name, "sha256": sha256(path), "facts": facts})
    (task_bisect / "manifest.json").write_text(json.dumps({
        "purpose": "round-7 terminal result write-out and accepted controls",
        "order": [entry["name"] for entry in cases], "cases": cases,
    }, indent=2) + "\n")

    write_scripts(root)
    write_verify(root)
    (root / "README.md").write_text(f"""# H16G ResNet-50 rule-built round 7

Reviewer kit for `t-6709`, commit `{args.commit}`. Run T1, T2, T3, T5,
ladder 01-04, then the terminal-stem and maxpool morphs in manifest order.
Rungs 05-07 remain in the kit to re-measure the prior non-finite/hang ceiling.
Each diagnostic candidate changes one 32-bit word; controls request three cold
runs. Stop and recover after any LOAD/EXECUTE failure, EIO, or timeout.
Acceptance is relL2 <= 0.02, byte-identical cold runs, and zero DART faults.
`verify.sh` is offline-only.
""")
    (root / "STATUS.json").write_text(json.dumps({
        "task": "t-6709-mil-hwx-compiler-h16g-rule-built-resnet-50-round",
        "branch": "feature/t-6076-h16g-resnet50-rule-built-r7",
        "commit": args.commit, "hardware_run": "reviewer-owned",
        "target_hardware_requested": False,
    }, indent=2) + "\n")

    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(root)}\n" for path in paths))
    subprocess.run([str(root / "verify.sh")], check=True)
    print(json.dumps({"out_dir": str(root), "files": len(paths),
                      "sha256sums_sha256": sha256(root / "SHA256SUMS")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

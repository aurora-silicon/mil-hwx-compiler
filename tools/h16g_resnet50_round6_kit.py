#!/usr/bin/env python3
"""Assemble the H16G ResNet-50 round-6 reviewer kit.

The input ladder must have been built by h16g_resnet50_ladder.py from the
round-6 compiler.  This assembler adds the accepted control, the fixed stem
write-out, max-pool field switches, direct boundary dumps, runner scripts,
and an offline verifier without embedding compiler-output fields in source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from h16g_load_grammar import check, object_fields, walk


SWITCHES = (
    ("S1-execution-cost", (1,), ()),
    ("S2-live-bank-mask", (7,), ()),
    ("S3-result-base", (), (0x414C,)),
    ("S4-result-dma-layout", (), (0x5110, 0x5114)),
)


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


def patch_switch(fixed: tuple, old: tuple, header_words: tuple[int, ...],
                 registers: tuple[int, ...]) -> bytes:
    data, fields, tasks, _ = fixed
    target, source = tasks[1], old[2][1]
    image = bytearray(data)
    for word in header_words:
        struct.pack_into("<I", image,
                         fields["text_off"] + target["offset"] + 4 * word,
                         source["header"][word])
    for address in registers:
        if address not in target["register_offsets"] or \
                address not in source["registers"]:
            raise ValueError(f"pool switch lacks register 0x{address:x}")
        struct.pack_into("<I", image,
                         fields["text_off"] + target["register_offsets"][address],
                         source["registers"][address])
    return bytes(image)


def copy_case(source: Path, target: Path) -> None:
    shutil.copytree(source, target)


def write_scripts(root: Path) -> None:
    scripts = {
        "run-one.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
R=${1:?usage: run-one.sh RUNG [OUTPUT_DIR]}
case "$R" in 01-stem|02-stem-maxpool|03-projection-block|04-first-stage|05-through-stage3|06-through-stage4|07-full) ;; *) exit 64 ;; esac
O=${2:-/var/tmp/resnet50-h16g-rb6-$R}; mkdir -p "$O"
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
O=${2:-/var/tmp/resnet50-h16g-rb6-$C}; mkdir -p "$O"
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
O=${2:-/var/tmp/resnet50-h16g-rb6-boundary-$B}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$K/boundary-dumps/$B/program/program-0.hwx" --case "$K/boundary-dumps/$B/refs/input_00.bin:$K/boundary-dumps/$B/refs/boundary_00.surface.fp16" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$K/boundary-dumps/$B/refs/boundary_00.surface.fp16" | tee "$O/compare.log"
grep -E 'loaded|p50|PASS' "$O/run.log" || true
''',
        "run-one-switch.sh": r'''#!/bin/sh
set -eu
K=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
S=${1:?usage: run-one-switch.sh S1-execution-cost|S2-live-bank-mask|S3-result-base|S4-result-dma-layout [OUTPUT_DIR]}
case "$S" in S1-execution-cost|S2-live-bank-mask|S3-result-base|S4-result-dma-layout) ;; *) exit 64 ;; esac
O=${2:-/var/tmp/resnet50-h16g-rb6-$S}; mkdir -p "$O"
set +e
timeout 300 python3 -I "$K/ane-run.py" --hwx "$K/maxpool-switches/$S.hwx" --case "$K/ladder/02-stem-maxpool/refs/input_00.bin:$K/ladder/02-stem-maxpool/refs/logits_00.fp16" --runs 3 --max-ulp 65535 --out-dir "$O/output" >"$O/run.log" 2>&1
rc=$?; set -e
if [ "$rc" -ne 0 ] || grep -Eq 'H16_LOAD|H16_EXEC|timed out|EIO' "$O/run.log"; then echo 'FAIL: recover before another object' >&2; exit 1; fi
python3 -I "$K/compare.py" "$O/output/output_00.bin" "$K/ladder/02-stem-maxpool/refs/logits_00.fp16" | tee "$O/compare.log"
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
for object in ladder/*/program/program-0.hwx boundary-dumps/*/program/program-0.hwx task-bisect/T?-*/program/program-0.hwx maxpool-switches/*.hwx; do
  python3 h16g_validate.py "$object" >/dev/null
  python3 h16g_load_grammar.py "$object"
done
for json in ladder.json task-bisect/manifest.json maxpool-field-diff.json STATUS.json; do python3 -m json.tool "$json" >/dev/null; done
python3 - <<'PY'
from pathlib import Path
from h16g_load_grammar import object_fields, walk
root=Path('.')
def tasks(path):
    return walk(object_fields(path.read_bytes()),include_tasks=True,enforce_residual=False)['_tasks']
accepted=tasks(root/'task-bisect/T2-accepted-full/program/program-0.hwx')[1]
fixed=tasks(root/'ladder/02-stem-maxpool/program/program-0.hwx')[1]
assert accepted['header']==fixed['header'] and accepted['registers']==fixed['registers']
t1=tasks(root/'task-bisect/T1-stem-alone/program/program-0.hwx')[0]['registers']
assert t1[0x5110]==0x100 and t1[0x5114]==0x7000
assert [len(tasks(root/f'boundary-dumps/{name}/program/program-0.hwx')) for name in
        ('01-stem','02-stem-maxpool','03-projection-block','04-first-stage')]==[1,2,7,15]
assert (root/'task-bisect/T1-stem-alone/refs/stem_00.surface.fp16').stat().st_size==1835008
PY
for script in run-one.sh run-one-bisect.sh run-one-boundary.sh run-one-switch.sh verify.sh; do sh -n "$script"; done
echo PASS
'''
    path = root / "verify.sh"
    path.write_text(payload)
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rb5-kit", type=Path, required=True)
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
        shutil.copy2(args.rb5_kit / name, root / name)
    shutil.copy2(args.validator, root / "h16g_validate.py")
    shutil.copy2(args.load_grammar, root / "h16g_load_grammar.py")
    shutil.copytree(args.ladder_dir / "source", root / "source", symlinks=True)
    shutil.copytree(args.ladder_dir / "ladder", root / "ladder", symlinks=True)
    shutil.copytree(args.ladder_dir / "boundary-dumps", root / "boundary-dumps",
                    symlinks=True)
    shutil.copy2(args.ladder_dir / "ladder.json", root / "ladder.json")

    task_bisect = root / "task-bisect"
    t1 = task_bisect / "T1-stem-alone"
    copy_case(root / "boundary-dumps/01-stem", t1)
    (t1 / "refs/boundary_00.surface.fp16").rename(
        t1 / "refs/stem_00.surface.fp16")
    copy_case(args.rb5_kit / "task-bisect/T2-accepted-full",
              task_bisect / "T2-accepted-full")
    copy_case(root / "ladder/01-stem", task_bisect / "T3-fixed-stem-head")
    copy_case(args.rb5_kit / "task-bisect/T5-head-alone",
              task_bisect / "T5-head-alone")

    accepted = endpoint(task_bisect /
                        "T2-accepted-full/program/program-0.hwx")
    old = endpoint(args.rb5_kit /
                   "ladder/02-stem-maxpool/program/program-0.hwx")
    fixed_path = root / "ladder/02-stem-maxpool/program/program-0.hwx"
    fixed = endpoint(fixed_path)
    switches = root / "maxpool-switches"
    switches.mkdir()
    switch_entries = []
    for name, header_words, registers in SWITCHES:
        path = switches / f"{name}.hwx"
        path.write_bytes(patch_switch(fixed, old, header_words, registers))
        facts = check(path);facts.pop("path", None)
        switch_entries.append({"name": name, "file": path.name,
                               "sha256": sha256(path), "facts": facts})

    diff = {
        "schema": "mil-hwxc.h16g-resnet-maxpool-diff.v1",
        "accepted_vs_fixed": task_diff(accepted[2][1], fixed[2][1]),
        "old_rule_built_vs_fixed": task_diff(old[2][1], fixed[2][1]),
        "equal_fields": ["3x3 window", "stride 2", "SAME padding edges",
                         "source base", "source channel stride",
                         "source row stride", "source plane spans",
                         "operation class"],
        "corrected_fields": ["execution cost", "live-bank mask",
                             "result base", "result DMA row stride",
                             "result DMA plane stride"],
        "switches": switch_entries,
    }
    (root / "maxpool-field-diff.json").write_text(
        json.dumps(diff, indent=2) + "\n")

    cases = []
    for name in ("T1-stem-alone", "T2-accepted-full",
                 "T3-fixed-stem-head", "T5-head-alone"):
        path = task_bisect / name / "program/program-0.hwx"
        facts = check(path);facts.pop("path", None)
        cases.append({"name": name, "sha256": sha256(path), "facts": facts})
    (task_bisect / "manifest.json").write_text(json.dumps({
        "purpose": "round-6 fixed stem write-out and accepted controls",
        "order": [entry["name"] for entry in cases], "cases": cases,
    }, indent=2) + "\n")

    write_scripts(root)
    write_verify(root)
    (root / "README.md").write_text(f"""# H16G ResNet-50 rule-built round 6

Reviewer kit for `t-6678`, commit `{args.commit}`. Run the fixed T1 and
accepted controls first, then boundary dumps 01-04, the four max-pool
single-field negative switches, and ladder rungs 01-07. The max-pool diff is
`maxpool-field-diff.json`; the corrected task matches the accepted task.

Stop and recover after any LOAD/EXECUTE failure, EIO, or timeout. Acceptance
is relL2 <= 0.05, no hang, and zero DART faults. `verify.sh` is offline-only.
""")
    (root / "STATUS.json").write_text(json.dumps({
        "task": "t-6678-mil-hwx-compiler-h16g-rule-built-resnet-50-round",
        "branch": "feature/t-6076-h16g-resnet50-rule-built-r6",
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

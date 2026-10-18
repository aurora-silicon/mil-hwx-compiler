#!/usr/bin/env python3
"""Materialize, compile and dry-run the H13 first-execution ladder.

Every rung is self-contained: deterministic inputs and weights from
rungs.json, expected outputs from tools/h13_reference.py, one compiled ANEC
package, and the libane call sequence tools/h13_run_linux.py will issue.

    python3 tests/h13_first_run/first_run.py            # every rung, dry run
    python3 tests/h13_first_run/first_run.py --rung 1    # one rung
    H13_DEADLINE_SECONDS=120 python3 tests/h13_first_run/first_run.py --rung 1 --execute

`--execute` submits the exact dry-run package through
tests/run_h13_linux_hardware.sh, which refuses to submit unless preflight.sh
passes against the reviewed identity file (ANE_REVIEWED_IDENTITIES). A dry
run needs no device and no ANE host.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import libane_overrun  # noqa: E402
COMPILER = Path(os.environ.get("ANE_COMPILER_BIN") or ROOT / "build/mil-hwxc")
REFERENCE = ROOT / "tools/h13_reference.py"
RUNNER = ROOT / "tools/h13_run_linux.py"
HARDWARE = ROOT / "tests/run_h13_linux_hardware.sh"


def values(spec):
    """The rung's deterministic pattern, rounded once to fp16."""
    scale, modulus, offset = spec["scale"], spec["modulus"], spec["offset"]
    return b"".join(
        struct.pack("<e", scale * ((index % modulus if modulus else index) - offset))
        for index in range(spec["count"]))


def blob(payload):
    """A BLOBFILE with the subheader the compiler reads at offset 64."""
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


def run(command, **kwargs):
    result = subprocess.run([str(item) for item in command], text=True,
                            capture_output=True, **kwargs)
    if result.returncode:
        raise SystemExit(f"FAIL {' '.join(str(item) for item in command)}\n"
                         f"{result.stdout}{result.stderr}")
    return result.stdout


def outputs(mil):
    """The MIL's returned tensor names, from its last `-> (...)` clause."""
    text = mil.read_text()
    clause = text[text.rindex("-> (") + 4:]
    return clause[:clause.index(")")].replace(",", " ").split()


def materialize(rung, work):
    directory = work / rung["name"]
    shutil.rmtree(directory, ignore_errors=True)
    for name in ("models", "inputs", "expected"):
        (directory / name).mkdir(parents=True)
    mil = directory / "model.mil"
    shutil.copyfile(KIT / "rungs" / rung["name"] / "model.mil", mil)
    for spec in rung["blobs"]:
        (directory / "models" / spec["file"]).write_bytes(blob(values(spec)))
    bindings = []
    for spec in rung["inputs"]:
        path = directory / "inputs" / f"{spec['name']}.fp16"
        path.write_bytes(values(spec))
        bindings += ["--input", f"{spec['name']}={path}"]
    expected = []
    for name in outputs(mil):
        expected += ["--output",
                     f"{name}={directory / 'expected' / (name + '.fp16')}"]
    run([sys.executable, REFERENCE, mil, "--model-root", directory / "models",
         *bindings, *expected])
    return directory, mil, bindings


def check(rung, work):
    directory, mil, bindings = materialize(rung, work)
    package = directory / "pkg"
    print(run([COMPILER, "--target", "H13", "--mil", mil,
               "--model-root", directory / "models", "--output", package]).strip())
    # Same provenance record the wrapper writes, so --package can verify the
    # package was produced by exactly the binary the wrapper attests.
    (package / "compiler.sha256").write_text(
        f"{hashlib.sha256(COMPILER.read_bytes()).hexdigest()}  mil-hwxc\n")
    manifest = json.loads((package / "manifest.json").read_text())
    for program in manifest["programs"]:
        found = libane_overrun.overruns((package / program["file"]).read_bytes())
        if found:
            channel, need, have = found[0]
            raise SystemExit(f"FAIL rung {rung['id']} {program['file']}: libane would refuse it, "
                             f"surface channel {channel} needs {need} bytes but is allocated {have}")
    encoders = Counter(program["encoder"] for program in manifest["programs"])
    if encoders != Counter(rung["encoders"]):
        raise SystemExit(f"FAIL rung {rung['id']} encoders {dict(encoders)} "
                         f"differ from {rung['encoders']}")
    descriptors = sum(program["taskDescriptors"] for program in manifest["programs"])
    if descriptors != rung["taskDescriptors"]:
        raise SystemExit(f"FAIL rung {rung['id']} has {descriptors} task "
                         f"descriptors, expected {rung['taskDescriptors']}")
    written = [f"{name}={directory / 'pkg' / (name + '.out.fp16')}"
               for name in outputs(mil)]
    plan = json.loads(run([sys.executable, RUNNER, package, "--mil", mil,
        "--model-root", directory / "models", "--dry-run", *bindings,
        *[item for name in written for item in ("--output", name)]]))
    if not all(program["libaneCalls"] for program in plan["programs"]):
        raise SystemExit(f"FAIL rung {rung['id']} plan carries no libane calls")
    (directory / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    calls = sum(len(program["libaneCalls"]) for program in plan["programs"])
    print(f"PASS rung {rung['id']} {rung['name']}: {dict(encoders)}, "
          f"{descriptors} task descriptors, {len(plan['programs'])} dispatched "
          f"programs, {calls} libane calls, plan {directory / 'plan.json'}")
    print("  hardware: H13_DEADLINE_SECONDS=120 "
          "ANE_REVIEWED_IDENTITIES=<reviewed identities file> "
          f"ANE_CHECKOUT=~/src/omarchy-ane bash {HARDWARE.relative_to(ROOT)} "
          f"--package {directory / 'pkg'} {mil} {directory / 'models'} "
          + " ".join(f"{spec['name']}={directory / 'inputs' / (spec['name'] + '.fp16')}"
                     for spec in rung["inputs"]))
    return directory, mil


def execute(rung, work):
    directory, mil = check(rung, work)
    command = ["bash", HARDWARE, "--package", directory / "pkg", mil,
               directory / "models",
               *[f"{spec['name']}={directory / 'inputs' / (spec['name'] + '.fp16')}"
                 for spec in rung["inputs"]]]
    print(f"submitting rung {rung['id']}: {' '.join(str(item) for item in command)}")
    result = subprocess.run([str(item) for item in command])
    if result.returncode:
        raise SystemExit(f"FAIL rung {rung['id']} hardware run "
                         f"exited {result.returncode}; stop the ladder here")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", type=int, action="append", default=[],
                        help="rung id; repeatable, defaults to every rung")
    parser.add_argument("--work", type=Path, default=Path("/tmp/h13-first-run"))
    parser.add_argument("--execute", action="store_true",
                        help="submit through tests/run_h13_linux_hardware.sh")
    arguments = parser.parse_args()
    catalog = json.loads((KIT / "rungs.json").read_text())["rungs"]
    selected = [rung for rung in catalog
                if not arguments.rung or rung["id"] in arguments.rung]
    missing = set(arguments.rung) - {rung["id"] for rung in catalog}
    if missing:
        raise SystemExit(f"no such rung: {sorted(missing)}")
    arguments.work.mkdir(parents=True, exist_ok=True)
    for rung in selected:
        (execute if arguments.execute else check)(rung, arguments.work)
    print(f"H13 first-run ladder: {len(selected)} rung(s) "
          f"{'submitted' if arguments.execute else 'dry-run'} OK")


if __name__ == "__main__":
    main()

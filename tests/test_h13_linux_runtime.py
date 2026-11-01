#!/usr/bin/env python3
"""Linux runner regression checks: real libane binding, pre-dispatch
rejection, and benchmark orchestration.

No test here opens the ANE device or submits hardware work. The binding test
resolves symbols inside a real libane_python.so (library load only). Scripted
adapter/program objects exercise orchestration and validation ordering; their
outputs are test data and never evidence about hardware.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "research")]

import h13_run_linux
from h13_reference import encode_fp16, evaluate, fp16
from research import inspect_anec
from timeouts import scaled_timeout

COMPILER = ROOT / "build/mil-hwxc"
CANDIDATE_LIBRARIES = [
    os.environ.get("H13_LIBANE_LIBRARY"),
    str(Path.home() / "src/omarchy-ane/bindings/python/dylib/libane_python.so"),
    "/usr/lib/libane_python.so",
]


def find_real_library():
    for candidate in CANDIDATE_LIBRARIES:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def test_adapter_loads_real_libane():
    """The binding must resolve against a real libane_python.so: the dunder
    symbols break under class-body name mangling (`self.lib.__ane_src_size`
    looks up `_LibANEAdapter__ane_src_size`, which libane never exports), and
    send/read are void entry points."""
    library = find_real_library()
    if library is None:
        raise unittest.SkipTest(
            "no libane_python.so available; build one to cover the binding")
    adapter = h13_run_linux.LibANEAdapter(library)
    assert callable(adapter._init)
    assert callable(adapter._free)
    assert callable(adapter._src_size)
    assert callable(adapter._dst_size)
    assert callable(adapter._exec)
    assert adapter._send.restype is None  # libane send/read return void
    assert adapter._read.restype is None


def test_adapter_rejects_library_without_libane_abi():
    with tempfile.TemporaryDirectory(prefix="h13-noabi-") as directory:
        source = Path(directory) / "empty.c"
        library = Path(directory) / "empty.so"
        source.write_text("int unrelated(void) { return 7; }\n")
        try:
            subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)],
                           check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise unittest.SkipTest(
                "no working gcc; cannot build the negative-case library")
        try:
            h13_run_linux.LibANEAdapter(library)
        except RuntimeError as error:
            assert "does not export the libane ABI" in str(error)
        else:
            raise AssertionError("a library without libane symbols was accepted")


def test_call_sequence_matches_real_libane_abi():
    """The published plan is a review artifact for the driver owner: it must
    name the calls the real library actually exports, in the order the runner
    issues them."""
    program = {
        "file": "0.anec",
        "constantBytes": 4096,
        "constantOffset": 0,
        "inputs": [{"name": "x", "allocationBytes": 128},
                   {"name": "z", "allocationBytes": 128}],
        "outputs": [{"name": "y", "allocationBytes": 256}],
    }
    calls = h13_run_linux.LibANEAdapter.call_sequence(program)
    assert calls[0].startswith("pyane_init(")
    assert not any("bind_kernel" in call or "exec_loop" in call for call in calls)
    assert calls[1].startswith("__ane_src_size(handle, 0)")
    assert calls[2].startswith("__ane_src_size(handle, 1)")
    assert calls[3].startswith("__ane_dst_size(handle, 0)")
    assert calls[4].startswith("__ane_send(handle, buffer, 0)")
    assert calls[5].startswith("__ane_send(handle, buffer, 1)")
    assert calls[6] == "ane_exec(handle)"
    assert calls[7].startswith("__ane_read(handle, buffer, 0)")
    assert calls[-1] == "pyane_free(handle)"


class ScriptedAdapter:
    """Stands in for LibANEAdapter inside PreparedProgram: counts transfers
    and reports configured surface sizes. Orchestration only."""

    def __init__(self, src_sizes, dst_sizes, free_result=0, exec_result=0):
        self.device = 0
        self.freed = 0
        self.sent = []
        self.read = []
        self.exec_calls = 0
        self.src_sizes = src_sizes
        self.dst_sizes = dst_sizes
        self.free_result = free_result
        self.exec_result = exec_result

    @staticmethod
    def _failure(name, result):
        return RuntimeError(f"{name} failed: {result}")

    def _init(self, path, device):
        return 0x1000

    def _src_size(self, handle, index):
        return self.src_sizes[index]

    def _dst_size(self, handle, index):
        return self.dst_sizes[index]

    def _send(self, handle, buffer, index):
        self.sent.append(index)

    def _read(self, handle, buffer, index):
        self.read.append(index)
        buffer[0] = 0x41  # prove the readback surfaces the device-visible data

    def _exec(self, handle):
        self.exec_calls += 1
        return self.exec_result

    def _free(self, handle):
        self.freed += 1
        return self.free_result


def test_prepared_program_rejects_bad_sizes_before_any_transfer():
    adapter = ScriptedAdapter(src_sizes=[4], dst_sizes=[8])
    try:
        h13_run_linux.PreparedProgram(adapter, b"anec", [9 * b"x"], [8])
    except RuntimeError as error:
        assert "input 0 size 4 differs from manifest 9" in str(error)
    else:
        raise AssertionError("a short input was accepted")
    assert adapter.sent == [] and adapter.exec_calls == 0 and adapter.freed == 1

    adapter = ScriptedAdapter(src_sizes=[4], dst_sizes=[8])
    try:
        h13_run_linux.PreparedProgram(adapter, b"anec", [4 * b"x"], [3])
    except RuntimeError as error:
        assert "output 0 size 8 differs from manifest 3" in str(error)
    else:
        raise AssertionError("a short output was accepted")
    assert adapter.sent == [] and adapter.freed == 1


def test_prepared_program_cleanup_failure_keeps_primary_error():
    """A failing pyane_free during rejection must not hide why the program
    was rejected: the primary failure propagates, the cleanup is its cause."""
    adapter = ScriptedAdapter(src_sizes=[4], dst_sizes=[8], free_result=5)
    try:
        h13_run_linux.PreparedProgram(adapter, b"anec", [9 * b"x"], [8])
    except RuntimeError as error:
        assert "input 0 size 4 differs" in str(error)
        assert error.__cause__ is not None and "pyane_free" in str(error.__cause__)
    else:
        raise AssertionError("a short input was accepted")
    assert adapter.freed == 1


def test_execute_preserves_primary_failure_over_cleanup_failure():
    """A failed submission must surface even when freeing the handle also
    fails: the ane_exec error propagates with the pyane_free error as cause."""
    adapter = ScriptedAdapter(src_sizes=[4], dst_sizes=[8],
                              free_result=9, exec_result=7)
    try:
        h13_run_linux.LibANEAdapter.execute(adapter, b"anec", [4 * b"x"], [8])
    except RuntimeError as error:
        assert "ane_exec failed: 7" in str(error)
        assert error.__cause__ is not None and "pyane_free" in str(error.__cause__)
    else:
        raise AssertionError("a failed submission was accepted")
    assert adapter.freed == 1


def test_prepared_program_run_transfer_read_close():
    adapter = ScriptedAdapter(src_sizes=[4, 4], dst_sizes=[8])
    program = h13_run_linux.PreparedProgram(adapter, b"anec", [4 * b"a", 4 * b"b"], [8])

    try:
        program.refill([5 * b"x"])
    except RuntimeError as error:
        assert "takes 2 input surfaces, got 1" in str(error)
    else:
        raise AssertionError("a wrong refill count was accepted")
    assert adapter.sent == []  # refusals never reach a transfer

    try:
        program.refill([4 * b"X", b"yy"])
    except RuntimeError as error:
        assert "input 1 surface is 2 bytes, prepared buffer is 4" in str(error)
    else:
        raise AssertionError("a short refill surface was accepted")
    assert adapter.sent == []

    program.refill([4 * b"X", 4 * b"Y"])
    outputs = program.run_once()
    assert adapter.sent == [0, 1]
    assert adapter.exec_calls == 1
    assert adapter.read == [0]
    assert outputs == [b"A" + bytes(7)]  # readback overwrote the prepared buffer
    program.close()
    program.close()  # idempotent
    assert adapter.freed == 1


def test_prepared_program_in_place_reuses_owned_surfaces():
    adapter = ScriptedAdapter(src_sizes=[4], dst_sizes=[8])
    program = h13_run_linux.PreparedProgram(adapter, b"anec", [4 * b"a"], [8])
    try:
        program.input_buffers[0][:] = b"bcde"
        outputs = program.run_in_place()
        assert outputs is program.output_buffers
        assert outputs[0].raw == b"A" + bytes(7)
        outputs[0][0] = b"Z"
        again = program.run_in_place()
        assert again is outputs and again[0].raw[0:1] == b"A"
        assert adapter.sent == [0, 0] and adapter.read == [0, 0]
        assert adapter.exec_calls == 2
    finally:
        program.close()
    assert adapter.freed == 1


class Closable:
    def __init__(self, fail=False):
        self.closed = 0
        self._fail = fail

    def close(self):
        self.closed += 1
        if self._fail:
            raise RuntimeError("close failed")


def test_close_all_closes_every_handle_despite_first_failure():
    """_close_all must not stop at the first failing handle: the second is
    still closed, and the primary failure wins over the cleanup error."""
    failing = Closable(fail=True)
    healthy = Closable()
    primary = ValueError("compare failed")
    try:
        h13_run_linux._close_all([failing, healthy], primary)
    except ValueError as error:
        assert error is primary
        assert error.__cause__ is not None and "close failed" in str(error.__cause__)
    else:
        raise AssertionError("cleanup error was swallowed")
    assert failing.closed == 1 and healthy.closed == 1

    failing = Closable(fail=True)
    healthy = Closable()
    try:
        h13_run_linux._close_all([failing, healthy], None)
    except RuntimeError as error:
        assert "close failed" in str(error)
    else:
        raise AssertionError("cleanup error was swallowed")
    assert healthy.closed == 1


class DeviceProbe:
    """Counts how often the runner reached a device-carrying execute call."""

    def __init__(self):
        self.execute_calls = 0

    def execute(self, anec, inputs, output_sizes):
        self.execute_calls += 1
        return [bytes(size) for size in output_sizes]


def add_source():
    def kind(dimensions):
        return f"tensor<fp16, [{', '.join(map(str, dimensions))}]>"
    shape = (1, 64, 8, 8)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({kind(shape)} x, {kind(shape)} z) {{
    {kind(shape)} y = add(x = x, y = z)[name = string("y")];
  }} -> (y);
}}
'''


def compile_add_package(root):
    mil = root / "add.mil"
    package = root / "package"
    mil.write_text(add_source())
    compiled = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(package)],
        capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return mil, package


def compile_runtime_matmul_package(root):
    mil = ROOT / "tests/h13_first_run/rungs/07-runtime-matmul-64/model.mil"
    package = root / "runtime-matmul-package"
    compiled = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(package)],
        capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return package


def compile_constant_broadcast_package(root):
    shape = (1, 64, 8, 8)
    constant_shape = (1, 64, 1, 1)

    def kind(dimensions):
        return f"tensor<fp16, [{', '.join(map(str, dimensions))}]>"

    mil = root / "constant-broadcast.mil"
    package = root / "constant-broadcast-package"
    mil.write_text(f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({kind(shape)} x) {{
    {kind(constant_shape)} z = const()[name = string("z"), val =
        {kind(constant_shape)}(BLOBFILE(path = string("@model_path/bias.bin"),
                                       offset = uint64(64)))];
    {kind(shape)} y = add(x = x, y = z)[name = string("y")];
  }} -> (y);
}}
''')
    payload = struct.pack("<e", 0.5) * 64
    blob = bytearray(128 + len(payload))
    struct.pack_into("<IIQQ", blob, 64, 0xDEADBEEF, 1, len(payload), 128)
    blob[128:] = payload
    (root / "bias.bin").write_bytes(blob)
    compiled = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(package)],
        capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return package


def add_inputs(root):
    values = [fp16(0.5 + (index % 5) * 0.25) for index in range(4096)]
    others = [fp16(0.25 * (index % 9) - 1.0) for index in range(4096)]
    x_path = root / "x.fp16"
    z_path = root / "z.fp16"
    x_path.write_bytes(encode_fp16(values))
    z_path.write_bytes(encode_fp16(others))
    return {"x": x_path, "z": z_path}


def test_run_package_rejects_bad_request_before_device():
    with tempfile.TemporaryDirectory(prefix="h13-reject-") as directory:
        root = Path(directory)
        mil, package = compile_add_package(root)
        inputs = add_inputs(root)
        outputs = {"y": root / "y.fp16"}
        probe = DeviceProbe()

        try:
            h13_run_linux.run_package(package, mil, root, {"x": inputs["x"]},
                                      outputs, probe, deadline_seconds=60)
        except ValueError as error:
            assert "inputs must be exactly" in str(error)
        else:
            raise AssertionError("a missing input was accepted")

        short = root / "short.fp16"
        short.write_bytes(bytes(64))
        try:
            h13_run_linux.run_package(package, mil, root,
                                      {"x": short, "z": inputs["z"]},
                                      outputs, probe, deadline_seconds=60)
        except ValueError as error:
            assert "byte count differs" in str(error)
        else:
            raise AssertionError("a short input was accepted")

        try:
            h13_run_linux.run_package(package, mil, root, inputs,
                                      {"wrong": outputs["y"]}, probe,
                                      deadline_seconds=60)
        except ValueError as error:
            assert "outputs must be exactly y" in str(error)
        else:
            raise AssertionError("a wrong output request was accepted")

        assert probe.execute_calls == 0


def test_run_package_rejects_unsafe_task_metadata_before_device():
    def reject(mutator, message):
        with tempfile.TemporaryDirectory(prefix="h13-metadata-reject-") as directory:
            root = Path(directory)
            mil, package = compile_add_package(root)
            mutator(package)
            probe = DeviceProbe()
            try:
                h13_run_linux.run_package(
                    package, mil, root, add_inputs(root),
                    {"y": root / "y.fp16"}, probe, deadline_seconds=60)
            except ValueError as error:
                assert message in str(error)
            else:
                raise AssertionError("unsafe task metadata was accepted")
            assert probe.execute_calls == 0

    def remove_scratch(package):
        path = package / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("scratchBytes", None)
        for program in manifest["programs"]:
            program.pop("scratchBytes")
        path.write_text(json.dumps(manifest))

    def rewrite_task_word(package, index, transform):
        manifest = json.loads((package / "manifest.json").read_text())
        path = package / manifest["programs"][0]["file"]
        payload = bytearray(path.read_bytes())
        offset = inspect_anec.HEADER_BYTES + index * 4
        struct.pack_into("<I", payload, offset,
                         transform(struct.unpack_from("<I", payload, offset)[0]))
        path.write_bytes(payload)

    reject(remove_scratch, "scratchBytes")
    reject(lambda package: rewrite_task_word(
        package, 0, lambda word: word & ~0x00ff0000), "NID")
    reject(lambda package: rewrite_task_word(
        package, 8, lambda word: (word & ~0x1f) | 3), "channel 3")
    reject(lambda package: rewrite_task_word(
        package, 8, lambda word: (word & ~0x1f) | 1), "channel 1")
    reject(lambda package: rewrite_task_word(
        package, 8, lambda word: (word & ~0x1f) | 7), "channel 7")


def test_inspector_accepts_only_kernel_backed_constant_source():
    with tempfile.TemporaryDirectory(prefix="h13-kernel-source-") as directory:
        package = compile_constant_broadcast_package(Path(directory))
        manifest, _ = inspect_anec.load_package(package)
        program = manifest["programs"][0]
        assert program["encoder"] == inspect_anec.PARITY_BROADCAST
        assert len(program["inputs"]) == 1 and program["constantBytes"] > 0

        path = package / program["file"]
        payload = bytearray(path.read_bytes())
        offset = inspect_anec.HEADER_BYTES + 32
        selectors = struct.unpack_from("<I", payload, offset)[0]
        assert selectors & 0x1f == 1
        struct.pack_into("<I", payload, offset,
                         (selectors & ~(0x1f << 12)) | (1 << 12))
        path.write_bytes(payload)
        try:
            inspect_anec.load_package(package)
        except ValueError as error:
            assert "noncanonical channel 1" in str(error)
        else:
            raise AssertionError("kernel BAR1 was accepted as a destination")


def test_inspector_accounts_for_tile_rounded_scratch():
    with tempfile.TemporaryDirectory(prefix="h13-scratch-accounting-") as directory:
        package = compile_runtime_matmul_package(Path(directory))
        manifest, allocations = inspect_anec.load_package(package)
        scratch = manifest["programs"][0]["scratchBytes"]
        allocation = allocations[0]
        assert scratch == inspect_anec.TILE_BYTES
        assert allocation["scratchBytes"] == inspect_anec.TILE_BYTES
        assert allocation["totalBytes"] == sum(allocation[key] for key in (
            "commandAndConstantsBytes", "inputBytes", "outputBytes", "scratchBytes"))

        inspected = subprocess.run(
            [sys.executable, str(ROOT / "research/inspect_anec.py"), str(package)],
            capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
        assert inspected.returncode == 0, inspected.stdout + inspected.stderr
        report = json.loads(inspected.stdout)["bufferAllocation"]
        assert report["scratchBytes"] == inspect_anec.TILE_BYTES
        assert report["totalBytes"] == sum(report[key] for key in (
            "commandAndConstantsBytes", "inputBytes", "outputBytes", "scratchBytes"))


class ScriptedProgram:
    """PreparedProgram stand-in with fixed output surfaces."""

    def __init__(self, events, outputs, close_error=None):
        self._events = events
        self._outputs = outputs
        self._close_error = close_error
        self.refills = 0
        self.closed = 0

    def refill(self, inputs):
        self.refills += 1

    def transfer_in(self):
        self._events.append(("transfer_in", id(self)))

    def submit(self):
        pass

    def transfer_out(self):
        return self._outputs

    def close(self):
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


def _with_scripted_programs(outputs, events, close_error=None):
    remaining = iter(outputs)
    programs = []

    def factory(adapter, anec, inputs, output_sizes):
        program_outputs = next(remaining)
        assert [len(item) for item in program_outputs] == output_sizes
        program = ScriptedProgram(events, program_outputs, close_error)
        events.append(("prepare", id(program)))
        programs.append(program)
        return program

    saved = h13_run_linux.PreparedProgram
    h13_run_linux.PreparedProgram = factory
    return programs, saved


def _correct_surfaces(mil, package, root, inputs):
    expected = evaluate(mil.read_text(), root,
                        {name: path.read_bytes() for name, path in inputs.items()})
    manifest, _ = inspect_anec.load_package(package)
    tensors = manifest["tensors"]
    outputs = []
    for index in manifest["dispatchPlan"]:
        program_outputs = []
        for binding in manifest["programs"][index]["outputs"]:
            dense = inspect_anec.dense_slice(expected[binding["name"]], binding, tensors)
            program_outputs.append(bytes(inspect_anec.convert_tensor(
                binding, dense, True)))
        outputs.append(program_outputs)
    return expected, outputs


def _zero_surfaces(surfaces):
    return [[bytes(len(surface)) for surface in outputs] for outputs in surfaces]


def test_benchmark_publishes_only_correct_gated_report():
    with tempfile.TemporaryDirectory(prefix="h13-bench-") as directory:
        root = Path(directory)
        mil, package = compile_add_package(root)
        inputs = add_inputs(root)
        outputs = {"y": root / "y.fp16"}
        expected, correct_surfaces = _correct_surfaces(mil, package, root, inputs)

        for surfaces, should_publish in ((_zero_surfaces(correct_surfaces), False),
                                         (correct_surfaces, True)):
            events = []
            programs, saved = _with_scripted_programs(surfaces, events)
            try:
                if not should_publish:
                    try:
                        h13_run_linux.benchmark_package(
                            package, mil, root, inputs, outputs, object(),
                            warmup=1, iterations=2, deadline_seconds=60)
                    except ValueError as error:
                        assert "differs" in str(error)
                    else:
                        raise AssertionError("an incorrect benchmark published timings")
                    assert programs and all(program.closed == 1 for program in programs)
                    continue
                result, report = h13_run_linux.benchmark_package(
                    package, mil, root, inputs, outputs, object(),
                    warmup=2, iterations=3, deadline_seconds=60)
                assert report["schema"] == "mil-hwxc.h13-linux-benchmark.v1"
                assert report["warmup"] == 2 and report["iterations"] == 3
                assert set(report["referenceCriteria"]) == {"y"}
                assert result == expected
                assert len(report["programs"]) == len(correct_surfaces)
                for program in report["programs"]:
                    assert program["iterations"] == 3 and program["warmup"] == 2
                    assert program["setupSeconds"] >= 0.0
                    for scope in ("transferSeconds", "submitSeconds",
                                  "readbackSeconds", "totalSeconds"):
                        assert len(program[scope]["samples"]) == 3
                        assert program[scope]["minSeconds"] >= 0.0
                assert len(report["endToEnd"]["samples"]) == 3
                assert all(program.refills == 5 for program in programs)
                assert all(program.closed == 1 for program in programs)
                assert ("Final output unpacking and correctness checks are outside every timed window."
                        in report["timingScope"])
                assert ("Reference evaluation is also outside every timed window."
                        in report["timingScope"])
            finally:
                h13_run_linux.PreparedProgram = saved


def test_benchmark_prepares_before_measuring_and_closes_on_failure():
    """With warmup 0, preparation precedes measurement and cleanup is total."""
    with tempfile.TemporaryDirectory(prefix="h13-bench0-") as directory:
        root = Path(directory)
        mil, package = compile_add_package(root)
        inputs = add_inputs(root)
        outputs = {"y": root / "y.fp16"}
        _, correct_surfaces = _correct_surfaces(mil, package, root, inputs)

        events = []
        programs, saved = _with_scripted_programs(correct_surfaces, events)
        try:
            _, report = h13_run_linux.benchmark_package(
                package, mil, root, inputs, outputs, object(),
                warmup=0, iterations=2, deadline_seconds=60)
            first_transfer = next(index for index, event in enumerate(events)
                                  if event[0] == "transfer_in")
            prepared = [index for index, event in enumerate(events)
                        if event[0] == "prepare"]
            assert prepared and max(prepared) < first_transfer
            assert report["warmup"] == 0 and len(report["endToEnd"]["samples"]) == 2
            assert all(program.refills == 2 and program.closed == 1
                       for program in programs)
        finally:
            h13_run_linux.PreparedProgram = saved

        events = []
        programs, saved = _with_scripted_programs(
            _zero_surfaces(correct_surfaces), events,
            close_error=RuntimeError("pyane_free failed: 5"))
        try:
            try:
                h13_run_linux.benchmark_package(
                    package, mil, root, inputs, outputs, object(),
                    warmup=0, iterations=1, deadline_seconds=60)
            except ValueError as error:
                assert "differs" in str(error)
            else:
                raise AssertionError("an incorrect benchmark published timings")
            assert all(program.closed == 1 for program in programs)
        finally:
            h13_run_linux.PreparedProgram = saved


def test_benchmark_reads_program_artifacts_only_before_iterations():
    with tempfile.TemporaryDirectory(prefix="h13-bench-reads-") as directory:
        root = Path(directory)
        mil, package = compile_add_package(root)
        inputs = add_inputs(root)
        outputs = {"y": root / "y.fp16"}
        _, correct_surfaces = _correct_surfaces(mil, package, root, inputs)
        events = []
        programs, saved_program = _with_scripted_programs(correct_surfaces, events)
        saved_local_file = inspect_anec.local_file
        reads = []

        def counted_local_file(*args):
            reads.append(args[1])
            return saved_local_file(*args)

        inspect_anec.local_file = counted_local_file
        try:
            h13_run_linux.benchmark_package(
                package, mil, root, inputs, outputs, object(),
                warmup=0, iterations=2, deadline_seconds=60)
        finally:
            inspect_anec.local_file = saved_local_file
            h13_run_linux.PreparedProgram = saved_program
        program_reads = [name for name in reads if name.endswith(".anec")]
        assert len(program_reads) == 2 * len(correct_surfaces), program_reads
        assert all(program.closed == 1 for program in programs)

def test_cli_refuses_native_timing_flags_on_dry_run():
    runner = ROOT / "tools/h13_run_linux.py"
    base = [sys.executable, str(runner), "/nonexistent/package",
            "--mil", "/nonexistent/mil", "--model-root", "/nonexistent",
            "--input", "x=/nonexistent", "--output", "y=/nonexistent"]
    run = subprocess.run(base + ["--dry-run", "--benchmark-json", "/tmp/h13-bench.json"],
                         capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
    assert run.returncode != 0
    assert "--dry-run never reports native timings" in run.stderr

    run = subprocess.run(base + ["--warmup", "2"],
                         capture_output=True, text=True, timeout=scaled_timeout(30), check=False)
    assert run.returncode != 0
    assert "require --benchmark-json" in run.stderr


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    passed = skipped = 0
    for test in tests:
        try:
            test()
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"SKIP {test.__name__}: {reason}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"PASS {passed} H13 Linux runtime tests; SKIP {skipped}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Host orchestration only: source-emitted H13/H14 packages, no device."""
import copy
import ctypes
import json
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_h13_linux_runtime as old
from research import inspect_anec
import h13_run_linux as runner


class Adapter(old.ScriptedAdapter):
    def __init__(self, fail_exec=0, fail_load=0, bad_output=False):
        super().__init__([16384, 16384], [16384])
        self.loaded = 0
        self.current = {}
        self.sizes = {}
        self.fail_exec, self.fail_load = fail_exec, fail_load
        self.bad_output = bad_output

    def _init(self, path, device):
        self.loaded += 1
        if self.loaded == self.fail_load:
            return None
        self.current[self.loaded] = {}
        fields = inspect_anec.HEADER.unpack_from(Path(path).read_bytes())
        tiles = fields[7:39]
        self.sizes[self.loaded] = ([tiles[i] * 16384 for i in (5, 6)],
                                   [tiles[4] * 16384])
        return self.loaded

    def _src_size(self, handle, index):
        sizes = self.sizes[handle][0]
        return sizes[index] if index < len(sizes) else 0

    def _dst_size(self, handle, index):
        sizes = self.sizes[handle][1]
        return sizes[index] if index < len(sizes) else 0

    def _send(self, handle, buffer, index):
        self.sent.append((handle, index))
        self.current[handle][index] = buffer.raw

    def _exec(self, handle):
        self.exec_calls += 1
        if self.exec_calls == self.fail_exec:
            return -5
        count = len(self.current[handle][0]) // 2
        a, b = [struct.unpack(f'<{count}e', self.current[handle][i]) for i in (0, 1)]
        self.result = struct.pack(f'<{count}e', *[(x + y if handle == 1 else x * y)
                                             for x, y in zip(a, b)])
        if self.bad_output:
            self.result = bytes(self.sizes[handle][1][0])
        return 0

    def _read(self, handle, buffer, index):
        ctypes.memmove(buffer, self.result, len(buffer))


def package(root, target):
    mil = root / 'model.mil'
    source = old.add_source().replace('y = add', 'tmp = add').replace(
        '  } -> (y);',
        '    tensor<fp16, [1, 64, 8, 8]> y = mul(x = tmp, y = z)[name = string("y")];\n  } -> (y);')
    mil.write_text(source)
    dest = root / 'package'
    result = subprocess.run([str(old.COMPILER), '--mil', str(mil), '--model-root', str(root),
                             '--target', target, '--output', str(dest)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return mil, dest, old.add_inputs(root), {'y': root / 'y.fp16'}


class ResidentTests(unittest.TestCase):
    def test_both_targets_reuse_handles_and_recompute_changed_inputs(self):
        for target in ('H13', 'H14'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mil, dest, inputs, outputs = package(root, target)
                adapter = Adapter()
                dense = {k: p.read_bytes() for k, p in inputs.items()}
                graph = runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
                for n in range(3):
                    dense['x'] = old.encode_fp16([float(n)] * 4096)
                    expected = old.evaluate(mil.read_text(), root, dense)
                    self.assertEqual(graph(dense), expected)
                self.assertEqual(adapter.loaded, 2)
                self.assertEqual(adapter.exec_calls, 6)
                graph.close(); graph.close()
                self.assertEqual(adapter.freed, 2)
                with self.assertRaisesRegex(RuntimeError, 'closed or retired'):
                    graph(dense)

    def test_failed_submission_retires_entire_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil, dest, inputs, outputs = package(root, 'H14')
            adapter = Adapter(fail_exec=2)
            graph = runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
            dense = {k: p.read_bytes() for k, p in inputs.items()}
            original = copy.deepcopy(dense)
            with self.assertRaisesRegex(RuntimeError, 'ane_exec failed'):
                graph(dense)
            self.assertEqual(adapter.freed, 2)
            self.assertEqual(dense, original)
            with self.assertRaisesRegex(RuntimeError, 'closed or retired'):
                graph(dense)
            self.assertEqual(adapter.exec_calls, 2)

    def test_partial_preparation_failure_closes_prior_handle_without_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil, dest, inputs, outputs = package(root, 'H14')
            adapter = Adapter(fail_load=2)
            with self.assertRaises(RuntimeError):
                runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
            self.assertEqual(adapter.freed, 1)
            self.assertEqual(adapter.sent, [])

    def test_parity_failure_preserves_output_and_retires(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil, dest, inputs, outputs = package(root, 'H14')
            adapter = Adapter(bad_output=True)
            graph = runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
            with self.assertRaises(ValueError):
                graph({k: p.read_bytes() for k, p in inputs.items()})
            self.assertTrue((dest / 'y.failed.fp16').is_file())
            self.assertEqual(adapter.freed, 2)

    def test_h14_metadata_corruption_refused_before_load(self):
        mutations = [lambda m: m.update(target='H16G'),
                     lambda m: m['programs'][0].update(firstTaskBytes=4),
                     lambda m: m['programs'][0].update(encoder='apple-parity-conv'),
                     lambda m: m.update(dispatchPlan=[1, 0]),
                     lambda m: m['programs'][0]['inputs'][0].update(allocationBytes=2)]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mil, dest, inputs, outputs = package(root, 'H14')
                path = dest / 'manifest.json'
                metadata = json.loads(path.read_text()); mutate(metadata)
                path.write_text(json.dumps(metadata))
                adapter = Adapter()
                with self.assertRaises(ValueError):
                    runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
                self.assertEqual(adapter.loaded, 0)

    def test_existing_fallback_recomputes_whole_graph_and_never_retries(self):
        library = old.find_real_library()
        if library is None:
            self.skipTest('build libane Python binding first')
        module_path = library.parent.parent / 'python/ane/fallback.py'
        spec = importlib.util.spec_from_file_location('resident_fallback', module_path)
        fallback_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fallback_module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil, dest, inputs, outputs = package(root, 'H14')
            adapter = Adapter(fail_exec=2)
            graph = runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
            dense = {k: p.read_bytes() for k, p in inputs.items()}
            cpu = lambda request: old.evaluate(mil.read_text(), root, request)
            fallback = fallback_module.Fallback(cpu, native=graph, library=library)
            expected = cpu(dense)
            for _ in range(2):
                actual, call = fallback.run(dense)
                self.assertEqual(actual, expected)
                self.assertEqual(call.backend, 0)
            self.assertEqual(adapter.exec_calls, 2)
            self.assertEqual(adapter.freed, 2)
            self.assertEqual(fallback.stats()['cpu_calls'], 2)

    def test_resident_uses_validated_snapshot_after_package_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil, dest, inputs, outputs = package(root, 'H14')
            adapter = Adapter()
            expected_blobs = [p.read_bytes() for p in sorted(dest.glob('*.anec'))]
            original = runner.PreparedProgram
            received = []
            def prepare(adapter, anec, inputs, sizes):
                # Change files after validation but before the first load.
                for path in dest.glob('*.anec'):
                    path.write_bytes(b'changed after validation')
                received.append(anec)
                return original(adapter, anec, inputs, sizes)
            with patch.object(runner, 'PreparedProgram', side_effect=prepare):
                graph = runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
            self.assertEqual(received, expected_blobs)
            dense = {k: p.read_bytes() for k, p in inputs.items()}
            self.assertEqual(graph(dense), old.evaluate(mil.read_text(), root, dense))
            graph.close()

    def test_extra_native_ports_refused_before_transfer(self):
        adapter = old.ScriptedAdapter([16, 16], [16])
        with self.assertRaisesRegex(RuntimeError, 'input count differs'):
            runner.PreparedProgram(adapter, b'fixture', [bytes(16)], [16])
        self.assertEqual(adapter.sent, [])
        self.assertEqual(adapter.freed, 1)

    def test_h14_alias_metadata_keeps_task_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, dest, _, _ = package(root, 'H14')
            path = dest / 'manifest.json'
            metadata = json.loads(path.read_text())
            metadata['tensors']['alias'] = dict(metadata['tensors']['y'], aliasOf='y')
            path.write_text(json.dumps(metadata))
            manifest, _ = inspect_anec.load_package(dest, allow_h14=True)
            self.assertEqual(manifest['target'], 'H14')

    def test_h14_task_corruption_refused_before_load(self):
        for corruption in ('size', 'register'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mil, dest, inputs, outputs = package(root, 'H14')
                path = dest / 'program-0.anec'
                data = bytearray(path.read_bytes())
                if corruption == 'size':
                    struct.pack_into('<I', data, 0x1000, 0x7ff0000)
                else:
                    last_header = struct.unpack_from('<I', data, 0x1000 + 28)[0]
                    offset = 0x1000 + (36 if last_header & 3 == 3 else 32)
                    record = struct.unpack_from('<I', data, offset)[0]
                    struct.pack_into('<I', data, offset, (record & ~0x7fff) | 0x7fff)
                path.write_bytes(data)
                adapter = Adapter()
                with self.assertRaises(ValueError):
                    runner.ResidentGraph(dest, mil, root, inputs, outputs, adapter, 60)
                self.assertEqual(adapter.loaded, 0)


if __name__ == '__main__':
    unittest.main()

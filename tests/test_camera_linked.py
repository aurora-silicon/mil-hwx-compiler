#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-program camera packaging guards, using procedural compiler stubs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import camera_linked


class CameraPackaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'selfie'
        self.model.mkdir()
        self.shape = [1, 1, 2, 2]
        (self.model/'model.mil').write_text('''program(1.3) {
          func main<ios18>(tensor<fp16, [1, 1, 2, 2]> x) {
            tensor<fp16, [1, 1, 2, 2]> a = relu(x = x);
            tensor<fp16, [1, 1, 2, 2]> y = relu(x = a);
          } -> (y);
        }''')
        for name in ('model.onnx', 'weights.bin'):
            (self.model/name).write_bytes(b'procedural identity fixture')
        self.metadata = dict(model='selfie', source_sha256='procedural-test-only',
            input=self.shape, output_shapes=[self.shape], output_names=['y'], mil_outputs=['y'],
            artifacts={n:camera_linked.sha(self.model/n) for n in ('model.onnx', 'model.mil', 'weights.bin')})
        (self.model/'model.json').write_text(json.dumps(self.metadata))
        self.compiler = self.root/'compiler'
        self.compiler.write_text('#!/bin/sh\necho "refused procedural test graph"\nexit 65\n')
        self.compiler.chmod(0o755)

    def test_compiler_refusal_never_publishes_package(self):
        output = self.root/'failed'
        result = camera_linked.build(self.model, output, self.compiler)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['compiler_rc'], 65)
        self.assertEqual(result['operation_counts'], {'relu':2})
        self.assertIn('refused', (output/'compiler.log').read_text())
        self.assertFalse((output/'package/camera-linked.json').exists())
        self.assertEqual(json.loads((output/'build.json').read_text()), result)
        with self.assertRaises(FileExistsError):
            camera_linked.build(self.model, output, self.compiler)

    def test_export_identity_is_checked(self):
        (self.model/'weights.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            camera_linked.build(self.model, self.root/'tampered', self.compiler)

    def successful_stub(self, dispatch):
        program = dict(file='program-0.anec', operation='linked', taskDescriptors=2, scratchBytes=16384,
                       inputs=[dict(name='x', shape=self.shape)], outputs=[dict(name='y', shape=self.shape)])
        manifest = dict(programs=[program], dispatchPlan=dispatch)
        self.compiler.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir()
(p/'manifest.json').write_text(json.dumps(MANIFEST))
(p/'program-0.anec').write_bytes(b'procedural-test-only')
'''.replace('MANIFEST', repr(manifest)))
        return manifest

    def test_success_pins_all_generated_artifacts(self):
        manifest = self.successful_stub([0])
        output = self.root/'success'
        with patch.object(camera_linked.inspect_anec, 'load_package', return_value=(manifest, {})):
            result = camera_linked.build(self.model, output, self.compiler)
        self.assertEqual(result['status'], 'pass')
        plan = json.loads((output/'package/camera-linked.json').read_text())
        self.assertEqual(plan['schema'], 'aurora.camera-linked.v1')
        self.assertEqual(plan['onnx_sha256'], camera_linked.sha(self.model/'model.onnx'))
        for name, digest in plan['files'].items():
            self.assertEqual(camera_linked.sha(output/'package'/name), digest)

    def test_repeated_dispatch_cannot_look_like_whole_graph(self):
        manifest = self.successful_stub([0, 0])
        output = self.root/'hybrid'
        with patch.object(camera_linked.inspect_anec, 'load_package', return_value=(manifest, {})):
            result = camera_linked.build(self.model, output, self.compiler)
        self.assertEqual(result['status'], 'fail')
        self.assertFalse((output/'package/camera-linked.json').exists())


if __name__ == '__main__':
    unittest.main()

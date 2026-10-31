#!/usr/bin/env python3
"""CPU semantics, independent small examples, and exact Whisper movement shapes.

No package or device numerical parity is inferred from these tests.
"""
import argparse
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import h13_reference as ref
import whisper_coverage as coverage


def tensor(shape, values, dtype='fp16'):
    return ref.Tensor(dtype, tuple(shape), tuple(ref.fp16(v) if dtype == 'fp16' else v for v in values))


def execute(name, shape, **arguments):
    op = ref.Operation(ref.MILType('fp16', tuple(shape)), 'y', name, {}, {})
    return ref._execute(ref.Operation(op.result_type, op.result, name, arguments, {}), {}, Path('.'))


class ReferenceTests(unittest.TestCase):
    def test_receipt_geometries_and_hashes(self):
        receipt, cases = coverage.load_cases()
        self.assertEqual(len(cases), 126)
        self.assertEqual(sum(coverage.baseline_outcome(r) == 'timeout' for _, _, r in cases), 7)
        self.assertEqual(sum(r['outcome'] == 'compiled' for _, _, r in cases), 5)
        self.assertEqual(sum(r['outcome'] == 'refused' for _, _, r in cases), 114)
        for model, d, h in (('tiny.en', 384, 6), ('base.en', 512, 8), ('small.en', 768, 12)):
            self.assertEqual(receipt['models'][model]['output']['shape'], [1, 1500, d])
            self.assertEqual(d, h*64)

    def test_outcome_contract(self):
        self.assertEqual(coverage.classify(124, '', None), ('timeout', []))
        self.assertEqual(coverage.classify(65, 'error [h13.program-budget]: bounded', None)[0], 'budget-refused')
        self.assertEqual(coverage.classify(0, '', 0)[0], 'error')
        baseline = dict(outcome='timeout', diagnostic_codes=[], program_count=None)
        self.assertEqual(coverage.compare(baseline, 'timeout', [], None), ('unresolved', False))
        self.assertEqual(coverage.compare(baseline, 'budget-refused', ['h13.program-budget'], None), ('bounded', True))
        self.assertEqual(coverage.compare(baseline, 'compiled', [], 1), ('changed-needs-review', False))

    def test_timeout_and_partial_package_are_never_success(self):
        with tempfile.TemporaryDirectory(prefix='whisper-runner-test-') as tmp:
            package = Path(tmp) / 'package'
            run = coverage.run_case([sys.executable, '-c', 'import time; time.sleep(5)'], .05, package)
            self.assertEqual((run['rc'], run['outcome']), (124, 'timeout'))
            script = ("from pathlib import Path; import sys; "
                      "Path(sys.argv[1]).mkdir(); "
                      "print('error [h13.program-budget]: limit', file=sys.stderr); sys.exit(65)")
            run = coverage.run_case([sys.executable, '-c', script, str(package)], 5, package)
            self.assertEqual(run['outcome'], 'error')
            self.assertIn('left a package directory', run['stderr'])

    def test_compile_needs_complete_manifest_and_program_files(self):
        with tempfile.TemporaryDirectory(prefix='whisper-package-test-') as tmp:
            package = Path(tmp) / 'package'
            script = ("from pathlib import Path; import sys,json; "
                      "p=Path(sys.argv[1]);p.mkdir(); "
                      "(p/'manifest.json').write_text(json.dumps({'programs':[{'file':'program-0.anec'}]}))")
            run = coverage.run_case([sys.executable, '-c', script, str(package)], 5, package)
            self.assertEqual(run['outcome'], 'error')
            (package / 'program-0.anec').write_bytes(b'synthetic package test')
            run = coverage.run_case([sys.executable, '-c', 'pass'], 5, package)
            self.assertEqual((run['outcome'], run['program_count']), ('compiled', 1))
            self.assertEqual(len(run['package_programs_sha256']), 64)

    def test_batched_runtime_attention_uses_each_rhs(self):
        left = tensor((1, 2, 2, 2), [1, 2, 3, 4, -1, 2, 2, -3])
        right = tensor((1, 2, 2, 2), [2, 0, 0, 3, 4, 0, 0, -2])
        out = ref._matmul(left, right, False, False, ref.MILType('fp16', (1, 2, 2, 2)))
        self.assertEqual(out.values, (2, 6, 6, 12, -4, -4, 8, 6))
        # transpose_y QK and transpose_x are matrix-local, not batch-flattened.
        qk = ref._matmul(left, left, False, True, ref.MILType('fp16', (1, 2, 2, 2)))
        self.assertEqual(qk.values, (5, 11, 11, 25, 5, -8, -8, 13))
        xt = ref._matmul(left, right, True, False, ref.MILType('fp16', (1, 2, 2, 2)))
        self.assertEqual(xt.values, (2, 9, 4, 12, -4, -4, 8, 6))

    def test_batch_broadcast_is_matrix_local(self):
        left = tensor((2, 1, 2), [1, 2, 3, 4])
        right = tensor((1, 2, 1), [2, -1])
        self.assertEqual(ref._matmul(left, right, False, False,
            ref.MILType('fp16', (2, 1, 1))).values, (0, 2))
        with self.assertRaises(ValueError):
            ref._matmul(left, tensor((3, 2, 1), [1]*6), False, False,
                        ref.MILType('fp16', (2, 1, 1)))

    def test_rank3_conv_custom_padding_stride_and_bias(self):
        x = tensor((1, 1, 5), [1, -2, 3, -4, 5])
        w = tensor((1, 1, 3), [1, 2, 3])
        b = tensor((1,), [.5])
        self.assertEqual(ref._conv1d(x, w, b, (1,), (1,), (1, 1), 'custom', 1,
            ref.MILType('fp16', (1, 1, 5))).values, (-3.5, 6.5, -7.5, 10.5, 6.5))
        self.assertEqual(ref._conv1d(x, w, b, (2,), (1,), (1, 1), 'custom', 1,
            ref.MILType('fp16', (1, 1, 3))).values, (-3.5, -7.5, 6.5))
        groups = ref._conv1d(tensor((1, 2, 3), [1, 2, 3, -1, -2, -3]),
            tensor((2, 1, 1), [2, 3]), None, (1,), (1,), (0, 0), 'custom', 2,
            ref.MILType('fp16', (1, 2, 3)))
        self.assertEqual(groups.values, (2, 4, 6, -3, -6, -9))

    def test_affine_normalization_keeps_actual_epsilon(self):
        eps = ref.fp16(1e-5)
        self.assertEqual(eps, 0.000010013580322265625)
        x = tensor((1, 2, 2), [-1, 1, 2, 2])
        out = ref._layer_norm(x, (-1,), eps, ref.MILType('fp16', x.shape),
                              tensor((2,), [2, 3]), tensor((2,), [.5, -.5]))
        self.assertEqual(out.values, (-1.5, 2.5, .5, -.5))
        with self.assertRaises(ValueError):
            ref._layer_norm(x, (-1,), -eps, ref.MILType('fp16', x.shape))
        with self.assertRaises(ValueError):
            ref._layer_norm(x, (-1,), eps, ref.MILType('fp16', x.shape), tensor((1,), [1]))

    def test_attention_softmax_full_key_extent_sampled_queries(self):
        # Full 1500-key reduction, two queries; deliberately not full attention.
        x = tensor((1, 1, 2, 1500), [0]*1500 + [1000]*1500)
        out = ref._softmax(x, -1, ref.MILType('fp16', x.shape))
        self.assertEqual(set(out.values), {ref.fp16(1/1500)})
        self.assertLess(abs(sum(out.values[:1500])-1), .001)

    def test_exact_gelu_and_bias_rounding(self):
        x = tensor((7,), [-6, -3, -1, 0, 1, 3, 6])
        result = execute('gelu', x.shape, x=x, mode='EXACT')
        for v, actual in zip(x.values, result.values):
            self.assertEqual(actual, ref.fp16(.5*v*math.erfc(-v/math.sqrt(2))))
        with self.assertRaises(ValueError):
            execute('gelu', x.shape, x=x, mode='TANH_APPROXIMATION')
        result = execute('linear', (1, 2), x=tensor((1, 2), [1, -2]),
                         weight=tensor((2, 2), [1, 2, -3, 4]), bias=tensor((2,), [.5, -1]))
        self.assertEqual(result.values, (-2.5, -12))

    def test_layout_and_inferred_views_in_parsed_mil(self):
        source = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2, 3]> x) {
    tensor<int32, [3]> p = const()[name = string("p"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<fp16, [1, 3, 2]> a = transpose(x = x, perm = p)[name = string("a")];
    tensor<int32, [4]> s = const()[name = string("s"), val = tensor<int32, [4]>([1, 3, 1, -1])];
    tensor<fp16, [1, 3, 1, 2]> b = reshape(x = a, shape = s)[name = string("b")];
    tensor<fp16, [1, 3, 1, 2]> y = add(x = b, y = b)[name = string("y")];
  } -> (y);
}'''
        result = ref.evaluate(source, Path('.'), {'x': ref.encode_fp16([1, 2, 3, 4, 5, 6])})
        self.assertEqual(ref.decode_fp16(result['y']), (2, 8, 4, 10, 6, 12))

    def test_positional_and_singleton_tensor_broadcasts(self):
        a = tensor((1, 2, 3), [1, 2, 3, -1, -2, -3])
        b = tensor((2, 3), [1, 1, 1, 2, 2, 2])
        self.assertEqual(ref._binary('add', a, b, ref.MILType('fp16', a.shape)).values,
                         (2, 3, 4, 1, 0, -1))
        scaled = ref._binary('mul', a, tensor((1, 1, 1), [.353515625]), ref.MILType('fp16', a.shape))
        self.assertEqual(scaled.values, tuple(ref.fp16(v*.353515625) for v in a.values))

    def test_invalid_views_and_permutations(self):
        x = tensor((2, 3), range(6))
        for perm in ((0, 0), (0, 2)):
            with self.assertRaises(ValueError):
                ref._transpose(x, perm, ref.MILType('fp16', (2, 3)))
        for shape in ((-1, -1), (4, -1), (0, 6)):
            with self.assertRaises(ValueError):
                ref._shape_op('reshape', x, tensor((2,), shape, 'int32'), ref.MILType('fp16', (2, 3)))


class ExactMovementTests(unittest.TestCase):
    def test_all_exact_layouts_with_unique_index_tags(self):
        for d, h in ((384, 6), (512, 8), (768, 12)):
            # Unique int32 tags detect every moved element, including positions
            # whose fp16 values would otherwise collide after quantization.
            for shape, perm in (((1, d, 1500), (0, 2, 1)),
                                ((1, 1500, h, 64), (0, 2, 1, 3))):
                x = tensor(shape, range(math.prod(shape)), 'int32')
                out_shape = tuple(shape[n] for n in perm)
                y = ref._transpose(x, perm, ref.MILType('int32', out_shape))
                inverse = tuple(perm.index(n) for n in range(len(perm)))
                restored = ref._transpose(y, inverse, ref.MILType('int32', shape))
                self.assertEqual(restored.values, x.values)
                # One-way coordinates catch a pair of symmetric wrong inverses.
                for index in (0, 1, len(y.values)//2+65, len(y.values)-1):
                    remaining, coords = index, []
                    for size in reversed(out_shape):
                        remaining, coordinate = divmod(remaining, size)
                        coords.insert(0, coordinate)
                    source_coords = [coords[perm.index(n)] for n in range(len(shape))]
                    source = 0
                    for size, coordinate in zip(shape, source_coords):
                        source = source*size + coordinate
                    self.assertEqual(y.values[index], source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exact-movement', action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests)
    if args.exact_movement:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ExactMovementTests))
    sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())

#!/usr/bin/env python3
"""Check executable HWX packages (dense and sparse stride-2), linked DRAM chains
and concrete refusal boundaries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from mint_oracles import blob, half_payload, parse_hwx  # noqa: E402
from h13_td import diff_tasks  # noqa: E402

COMPILER = str(Path(sys.argv[1]).resolve())
TARGET = sys.argv[2]
ISA = {'H16G': 'h16', 'H17': 'h17'}[TARGET]


def compile_case(root, name, mil, extra=()):
    source = root / f'{name}.mil'
    source.write_text(mil)
    output = root / name
    result = subprocess.run([COMPILER, '--mil', str(source), '--model-root', str(root),
                             '--target', TARGET, '--format', 'hwx', '--output', str(output),
                             *extra], capture_output=True, text=True, timeout=120)
    return result, output


def main():
    count = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for path in sorted((ROOT / 'research/oracles' / ISA).glob('conv_desc_*.json')):
            oracle = json.loads(path.read_text())
            if len(oracle['task_descriptors']) != 1:
                continue
            (root / 'weights.bin').write_bytes(blob(half_payload(oracle['weights']['payload_bytes'] // 2)))
            if path.stem in ('conv_desc_pw_c1024_n1024_s14', 'conv_desc_pw_c1024_n1024_s28'):
                continue  # output-channel split: two tasks, filtered by the count above
            result, output = compile_case(root, path.stem, oracle['mil'])
            assert result.returncode == 0, (path.stem, result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            assert manifest['target'] == TARGET and manifest['artifactFormat'] == 'hwx'
            assert manifest['deviceValidated'] is False
            assert len(manifest['programs']) == 1
            assert manifest['programs'][0]['encoder'] == 'h16-h17-formula-conv'
            got = parse_hwx((output / 'program-0.hwx').read_bytes(), ISA)
            assert not got['task_decode_errors'], got['task_decode_errors']
            diffs = diff_tasks(got['task_descriptors'][0], oracle['task_descriptors'][0])
            assert all(d['kind'] == 'header' and d['index'] == 1 for d in diffs), (path.name, diffs)
            assert got['kernel_section']['sha256'] == oracle['kernel_section']['sha256'], path.name
            assert got['constant_section']['nonzero_bytes'] == 0, path.name
            assert got['program_descriptor']['task_count'] == 1
            assert got['tensor_descriptors'] == oracle['tensor_descriptors'], path.name
            # Check the kernel relocation's section, width and operand. H17
            # elementwise constants use a different relocation target.
            data = (output / 'program-0.hwx').read_bytes()
            import struct
            cursor = 32
            sections = []
            for _ in range(struct.unpack_from('<I', data, 16)[0]):
                command, length = struct.unpack_from('<II', data, cursor)
                if command == 0x19:
                    for i in range(struct.unpack_from('<I', data, cursor + 64)[0]):
                        at = cursor + 72 + i * 80
                        sections.append((data[at:at+16].split(b'\0')[0], at))
                cursor += length
            text = next(at for name, at in sections if name == b'__text')
            kern_index = next(i+1 for i, (name, _) in enumerate(sections) if name == b'__kern_0')
            reloc, nreloc = struct.unpack_from('<II', data, text + 56)
            assert nreloc == 1
            address, target = struct.unpack_from('<II', data, reloc)
            assert target == 0x07000000 | kern_index, (TARGET, hex(target), kern_index)
            assert address < got['task_descriptors'][0]['size_bytes'] + 16
            count += 1
        # Changing user weights must change __KERN_0, not the descriptor or
        # the zero __const allocation. This detects accidental template replay.
        oracle = json.loads((ROOT / f'research/oracles/{ISA}/conv_desc_cmp_pw_c64_n64_s64.json').read_text())
        changed = bytearray(half_payload(4096))
        changed[0:2] = b'\x00\x34'
        import struct
        valid = bytearray(128 + len(changed))
        struct.pack_into('<II', valid, 0, 1, 2)
        struct.pack_into('<IIQQ', valid, 64, 0xDEADBEEF, 1, len(changed), 128)
        valid[128:] = changed
        (root / 'weights.bin').write_bytes(valid)
        result, output = compile_case(root, 'nonuniform', oracle['mil'])
        assert result.returncode == 0, result.stderr
        got = parse_hwx((output / 'program-0.hwx').read_bytes(), ISA)
        assert got['kernel_section']['sha256'] == hashlib.sha256(changed).hexdigest()
        assert got['constant_section']['nonzero_bytes'] == 0
        # No output directory may appear after any refusal.
        biased = oracle['mil'].replace('    tensor<fp16, [1, 64, 64, 64]> y = conv(',
            '    tensor<fp16, [64]> b = const()[name = string("b"), val = tensor<fp16, [64]>(' + str([0] * 64) + ')];\n'
            '    tensor<fp16, [1, 64, 64, 64]> y = conv(bias = b, ')
        assert biased != oracle['mil']
        result, output = compile_case(root, 'biased', biased)
        assert result.returncode == 0, result.stderr
        assert parse_hwx((output / 'program-0.hwx').read_bytes(), ISA)['task_descriptors'][0]['blocks']['0x04900']['words']['0x04904'] == '0x00103c10'
        neighbours = {
            'valid': (oracle['mil'].replace('string("same")', 'string("valid")'), (), 'SAME'),
            'dilated': (oracle['mil'].replace('tensor<int32, [2]>([1, 1])', 'tensor<int32, [2]>([2, 2])'), (), 'SAME'),
            'linked': (oracle['mil'], ('--schedule', 'linked'), 'two or more convolutions'),
            'linked-l2': (oracle['mil'], ('--schedule', 'linked-l2'), 'descriptor parity only'),
            'anec': (oracle['mil'], ('--format', 'anec'), 'HWX' if TARGET == 'H16G' else 'hwx'),
        }
        for name, (mil, extra, diagnostic) in neighbours.items():
            result, output = compile_case(root, f'refuse-{name}', mil, extra)
            assert result.returncode != 0 and diagnostic in result.stderr, (name, result.stderr)
            assert not output.exists(), name
        linked_chain(root)
    print(f'{TARGET} conv CLI PASS: {count} kernel packages (dense and sparse), bias, nonuniform weights, '
          f'linked DRAM chain, padding/dilation/schedule/format refusals')


def linked_chain(root):
    """A three-layer chain (sparse stride-2 depthwise, then two 1x1) compiles
    per-op and linked; the linked object carries the same kernels back to back,
    one task per layer and a scratch-backed descriptor."""
    import random
    import struct
    rng = random.Random(7)
    shapes = [('W0', [64, 1, 3, 3], 2, 64, 'x', [1, 64, 112, 112], [1, 64, 56, 56]),
              ('W1', [128, 64, 1, 1], 1, 1, 'r0', None, [1, 128, 56, 56]),
              ('W2', [128, 128, 1, 1], 1, 1, 'r1', None, [1, 128, 56, 56])]
    blob_bytes = bytearray(struct.pack('<II', 1, 2) + bytes(56))
    body = []
    for i, (w, shape, stride, groups, src, _, out) in enumerate(shapes):
        n = shape[0] * shape[1] * shape[2] * shape[3]
        values = b''.join(struct.pack('<e', rng.choice([0.0, -0.0, 0.25, -0.5, 0.125]) if i == 0
                                      else rng.randint(-8, 8) / 64) for _ in range(n))
        meta = len(blob_bytes)
        blob_bytes += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(values), meta + 64) + bytes(40) + values
        blob_bytes += bytes((-len(blob_bytes)) % 64)
        t = 'tensor<fp16, [' + ', '.join(map(str, shape)) + ']>'
        o = 'tensor<fp16, [' + ', '.join(map(str, out)) + ']>'
        body += [f'    string p{i} = const()[name = string("p{i}"), val = string("same")];',
                 f'    tensor<int32, [2]> s{i} = const()[name = string("s{i}"), val = tensor<int32, [2]>([{stride}, {stride}])];',
                 f'    tensor<int32, [4]> d{i} = const()[name = string("d{i}"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
                 f'    tensor<int32, [2]> l{i} = const()[name = string("l{i}"), val = tensor<int32, [2]>([1, 1])];',
                 f'    int32 g{i} = const()[name = string("g{i}"), val = int32({groups})];',
                 f'    {t} {w} = const()[name = string("{w}"), val = {t}(BLOBFILE(path = string("@model_path/chain.bin"), offset = uint64({meta})))];',
                 f'    {o} c{i} = conv(dilations = l{i}, groups = g{i}, pad = d{i}, pad_type = p{i}, strides = s{i}, weight = {w}, x = {src})[name = string("c{i}")];',
                 f'    {o} r{i} = relu(x = c{i})[name = string("r{i}")];']
    (root / 'chain.bin').write_bytes(blob_bytes)
    mil = ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
           '  func main<ios18>(tensor<fp16, [1, 64, 112, 112]> x) {\n' + '\n'.join(body) +
           '\n  } -> (r2);\n}\n')
    result, perop = compile_case(root, 'chain-perop', mil)
    assert result.returncode == 0, result.stderr
    result, linked = compile_case(root, 'chain-linked', mil, ('--schedule', 'linked'))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((linked / 'manifest.json').read_text())
    assert manifest['schedule'] == 'linked' and len(manifest['programs']) == 1
    assert manifest['programs'][0]['taskDescriptors'] == 3
    assert manifest['scratch']['bytes'] > 0 and manifest['scratch']['channel'] == 3
    got = parse_hwx((linked / 'program-0.hwx').read_bytes(), ISA)
    assert not got['task_decode_errors'], got['task_decode_errors']
    assert got['program_descriptor']['task_count'] == 3
    kernels = b''
    for i in range(3):
        data = (perop / f'program-{i}.hwx').read_bytes()
        section = parse_hwx(data, ISA)['kernel_section']
        kernels += kernel_bytes(data)
        assert len(kernel_bytes(data)) >= section['size']
    assert hashlib.sha256(kernel_bytes((linked / 'program-0.hwx').read_bytes())).hexdigest() == \
        hashlib.sha256(kernels).hexdigest()
    tasks = got['task_descriptors']
    # Sparse stride-2 kernel on the first task; DRAM edges between tasks.
    assert tasks[0]['blocks']['0x04900']['words']['0x04900'] == ('0x10000182' if TARGET == 'H17' else '0x00000182')
    for i in (1, 2):
        assert int(tasks[i]['blocks']['0x04d00']['words']['0x04d00'], 16) == 0x20000021 | i << 16
        assert int(tasks[i - 1]['blocks']['0x05100']['words']['0x05100'], 16) & 0xff == \
            (0xe1 if TARGET == 'H17' else 0xf1)


def kernel_bytes(data):
    """The __KERN_0/__kern_0 section bytes of an HWX object."""
    import struct
    cursor = 32
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        command, length = struct.unpack_from('<II', data, cursor)
        if command == 0x19:
            for i in range(struct.unpack_from('<I', data, cursor + 64)[0]):
                at = cursor + 72 + i * 80
                if data[at:at + 16].split(b'\0')[0] == b'__kern_0':
                    size, offset = struct.unpack_from('<QI', data, at + 40)
                    return data[offset:offset + size]
        cursor += length
    raise AssertionError('no __kern_0 section')


if __name__ == '__main__':
    main()

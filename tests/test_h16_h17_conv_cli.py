#!/usr/bin/env python3
"""Check executable stride-1 HWX packages and concrete refusal boundaries."""
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
            result, output = compile_case(root, path.stem, oracle['mil'])
            if oracle['parameters']['stride'] == 2:
                assert result.returncode != 0 and 'sparse kernel byte grammar' in result.stderr, result.stderr
                assert not output.exists()
                continue
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
        neighbours = {
            'bias': (biased, (), 'bias packing'),
            'valid': (oracle['mil'].replace('string("same")', 'string("valid")'), (), 'SAME'),
            'dilated': (oracle['mil'].replace('tensor<int32, [2]>([1, 1])', 'tensor<int32, [2]>([2, 2])'), (), 'SAME'),
            'linked': (oracle['mil'], ('--schedule', 'linked'), 'DRAM dependency protocol'),
            'linked-l2': (oracle['mil'], ('--schedule', 'linked-l2'), 'sparse kernel packing'),
            'anec': (oracle['mil'], ('--format', 'anec'), 'HWX' if TARGET == 'H16G' else 'hwx'),
        }
        for name, (mil, extra, diagnostic) in neighbours.items():
            result, output = compile_case(root, f'refuse-{name}', mil, extra)
            assert result.returncode != 0 and diagnostic in result.stderr, (name, result.stderr)
            assert not output.exists(), name
    print(f'{TARGET} conv CLI PASS: {count} kernel packages, nonuniform weights, sparse/bias/padding/dilation/schedule/format refusals')


if __name__ == '__main__':
    main()

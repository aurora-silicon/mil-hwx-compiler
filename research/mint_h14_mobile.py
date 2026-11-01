#!/usr/bin/env python3
"""Compile-only H14 MobileNet probes on macOS; retain decoded facts and hashes.

Uses the installed compiler through its C API, never loads or executes a model.
All compiler output lives in TemporaryDirectory and is removed after decoding.
"""
import argparse
import ctypes as C
import hashlib
import json
import platform
import struct
import tempfile
import threading
from pathlib import Path

import mint_conv_probes as conv
import mint_oracles as om


def compile_mil(mil, weights):
    cf = C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    lib = C.CDLL('/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler')
    system = C.CDLL('/usr/lib/libSystem.B.dylib')
    ptr = C.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ptr, C.c_char_p, C.c_uint32]
    cf.CFStringCreateWithCString.restype = ptr
    cf.CFDictionaryCreateMutableCopy.argtypes = [ptr, C.c_long, ptr]
    cf.CFDictionaryCreateMutableCopy.restype = ptr
    cf.CFDictionarySetValue.argtypes = [ptr, ptr, ptr]
    cf.CFDictionaryCreateMutable.argtypes = [ptr, C.c_long, ptr, ptr]
    cf.CFDictionaryCreateMutable.restype = ptr
    cf.CFRelease.argtypes = [ptr]
    lib.ANECCreateCompilerInputDictionary.argtypes = [C.c_char_p]
    lib.ANECCreateCompilerInputDictionary.restype = ptr
    lib.ANECCompile.argtypes = [ptr, ptr, ptr]
    strings = []
    def string(s):
        p = cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)
        strings.append(p)
        return p
    class Descriptor(C.Structure):
        _fields_ = [('reserved', C.c_ulong), ('size', C.c_ulong)]
    Callback = C.CFUNCTYPE(None, ptr, C.c_int, ptr)
    class Block(C.Structure):
        _fields_ = [('isa', ptr), ('flags', C.c_int), ('reserved', C.c_int),
                    ('invoke', Callback), ('descriptor', C.POINTER(Descriptor))]
    done = threading.Event()
    status = []
    @Callback
    def completed(block, value, result):
        status.append(value)
        if value and result:
            cf.CFCopyDescription.argtypes = [ptr]
            cf.CFCopyDescription.restype = ptr
            cf.CFStringGetCString.argtypes = [ptr, ptr, C.c_long, C.c_uint32]
            desc = cf.CFCopyDescription(result)
            buf = C.create_string_buffer(16384)
            cf.CFStringGetCString(desc, buf, len(buf), 0x08000100)
            print(buf.value.decode(), flush=True)
            cf.CFRelease(desc)
        done.set()
    desc = Descriptor(0, C.sizeof(Block))
    block = Block(C.addressof(C.c_char.in_dll(system, '_NSConcreteGlobalBlock')),
                  1 << 28, 0, completed, C.pointer(desc))
    with tempfile.TemporaryDirectory(prefix='h14-compile-only-') as tmp:
        root = Path(tmp)
        (root / 'model.mil').write_text(mil)
        (root / 'weights.bin').write_bytes(weights)
        raw = lib.ANECCreateCompilerInputDictionary(str(root / 'model.mil').encode())
        if not raw:
            raise RuntimeError('ANECCreateCompilerInputDictionary failed')
        inp = cf.CFDictionaryCreateMutableCopy(None, 0, raw)
        options = cf.CFDictionaryCreateMutable(None, 0,
            C.addressof(C.c_char.in_dll(cf, 'kCFTypeDictionaryKeyCallBacks')),
            C.addressof(C.c_char.in_dll(cf, 'kCFTypeDictionaryValueCallBacks')))
        for key, value in [('OutputFileName', 'model'), ('OutputFilePath', tmp + '/')]:
            cf.CFDictionarySetValue(inp, string(key), string(value))
        cf.CFDictionarySetValue(options, string('TargetArchitecture'), string('H14'))
        lib.ANECCompile(inp, options, C.byref(block))
        # Keep callback storage live until completion; do not return into a freed block.
        if not done.wait(180):
            raise RuntimeError('compile-only timeout')
        try:
            if status != [0]:
                raise RuntimeError(f'ANECompiler status {status}')
            images = [p for p in root.rglob('*') if p.is_file() and
                      p.read_bytes()[:4] == b'\xce\xfa\xef\xbe']
            if len(images) != 1:
                raise RuntimeError(f'expected one HWX, found {len(images)}')
            return om.parse_hwx(images[0].read_bytes(), 'h14')
        finally:
            cf.CFRelease(options)
            cf.CFRelease(inp)
            cf.CFRelease(raw)
            for s in strings:
                cf.CFRelease(s)


def campaign():
    # All distinct shapes of tools/h13-mobilenet, plus unbiased and relu pairs.
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]
    shapes = {(3, 16, 32, 112, 1, 1)}
    ci, spatial = 32, 112
    for co, stride in cfg:
        shapes.add((3, ci, ci, spatial, 1, ci))
        shapes.add((1, ci, co, spatial, stride, 1))
        ci, spatial = co, (spatial + stride - 1) // stride
    shapes.add((1, 1024, 1024, 7, 1, 1))
    for k, ci, co, spatial, stride, groups in sorted(shapes):
        for bias in (False, True):
            item = conv.conv_case(k, ci, co, spatial, stride, groups, bias,
                                  name_prefix='h14_mobile')
            yield item
            relu = dict(item)
            relu['name'] += '_relu'
            output = om.tensor_type(tuple(item['parameters']['output_shape']))
            relu['mil'] = item['mil'].replace('  } -> (y);',
                f'    {output} r = relu(x = y)[name = string("r")];\n  }} -> (r);')
            relu['parameters'] = dict(item['parameters'], relu=True)
            yield relu


def chain_campaign():
    import re
    # Large surfaces force an observable DRAM hand-off rather than L2 aliasing.
    for ci, spatial in ((128, 112), (512, 56), (1024, 28)):
        for count in (2, 3):
            body = []
            current = 'x'
            max_weights = 0
            for i in range(count):
                k = 3 if i == 1 else 1
                groups = ci if i == 1 else 1
                item = conv.conv_case(k, ci, ci, spatial, 1, groups, True)
                text = item['mil'].split(') {\n', 1)[1].rsplit('  } ->', 1)[0]
                for name in ('pt', 'st', 'pd', 'dl', 'gp', 'w', 'b', 'y'):
                    text = re.sub(r'\b' + name + r'\b', name + str(i), text)
                text = text.replace('x = x)', 'x = ' + current + ')')
                body.extend(text.strip().splitlines())
                current = 'y' + str(i)
                max_weights = max(max_weights, item['weights_description']['payload_bytes'])
            name = f'h14_mobile_chain{count}_c{ci}_s{spatial}'
            yield om.case(name, 'h14_mobile_chain',
                          dict(channels=ci, spatial=spatial, layers=count),
                          om.program(f'{om.tensor_type((1,ci,spatial,spatial))} x', body, current),
                          om.blob(om.half_payload(max_weights // 2)),
                          dict(storage='BLOBFILE', value='fp16(0x1p-1)', payload_bytes=max_weights))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'oracles/h14-mobile')
    ap.add_argument('--case')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for item in list(campaign()) + list(chain_campaign()):
        if args.case and args.case not in item['name']:
            continue
        out = args.output / (item['name'] + '.json')
        if out.exists() and not args.force:
            continue
        record = dict(schema_version=1, case=item['name'], family=item['family'],
                      target='h14', source_commit=om.source_commit(),
                      parameters=item['parameters'], mil=item['mil'],
                      weights=item['weights_description'], compile_only=True,
                      compiler={'platform': platform.platform(), 'target': 'H14',
                                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        try:
            record.update(compile_mil(item['mil'], item['weights']))
            record['error'] = None
        except Exception as exc:
            record['error'] = str(exc)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        print(item['name'], record['error'] or 'decoded', flush=True)

if __name__ == '__main__':
    main()

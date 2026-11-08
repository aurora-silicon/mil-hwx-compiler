import json, math, pathlib, shutil, subprocess, sys
import numpy as np
comp, kit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); kit.mkdir(exist_ok=True)
rng = np.random.default_rng(1110); H='program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
erf = np.vectorize(math.erf)
def t(s): return 'tensor<fp16, [' + ', '.join(map(str, s)) + ']>'
cases = []
for name, op, shape in (('add_bert', 'add', (1,128,768)), ('gelu_bert', 'gelu', (1,128,3072)),
                        ('gelu_wtiny', 'gelu', (1,1500,1536)), ('add_wtiny', 'add', (1,1500,384))):
    T = t(shape)
    if op == 'add':
        mil = f'{H}  func main<ios18>({T} x, {T} y) {{\n    {T} z = add(x = x, y = y)[name = string("z")];\n  }} -> (z);\n}}\n'
    else:
        mil = f'{H}  func main<ios18>({T} x) {{\n    {T} z = gelu(x = x, mode = string("EXACT"))[name = string("z")];\n  }} -> (z);\n}}\n'
    (kit / f'{name}.mil').write_text(mil); shutil.rmtree(kit / name, ignore_errors=True)
    subprocess.run([str(comp), '--mil', str(kit / f'{name}.mil'), '--model-root', '/tmp', '--output', str(kit / name)], check=True)
    a = np.float16(rng.standard_normal(shape) * 2); (kit / name / 'in0.bin').write_bytes(a.tobytes()); ins = ['in0.bin']
    if op == 'add':
        b = np.float16(rng.standard_normal(shape) * 2); (kit / name / 'in1.bin').write_bytes(b.tobytes()); ins.append('in1.bin')
        ref = a.astype(np.float64) + b.astype(np.float64)
    else:
        x = a.astype(np.float64); ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    (kit / name / 'ref.bin').write_bytes(np.float16(ref).tobytes())
    cases.append({'name': name, 'programs': ['program-0.hwx'], 'inputs': [ins], 'refs': ['ref.bin']})
(kit / 'cases.json').write_text(json.dumps(cases))
print([c['name'] for c in cases])

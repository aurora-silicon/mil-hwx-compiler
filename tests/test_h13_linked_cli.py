#!/usr/bin/env python3
"""Host-side checks for `--schedule linked` (docs/ane/linked-programs.md).

A straight-line conv(+relu) chain compiles to ONE program whose tasks are the
layers' own formula-built tasks, relinked:

1. Stream: one program, every layer's tasks in order, task ids, first/last
   flags, next pointers and NextSize words consistent, end-of-stream bits only
   on the last task, TBE on every task with a successor. Each layer's last
   task carries TSE+TSR and the next layer's first task waits on it (input
   DMA DependencyMode 7): without that the M1 Pro read stale scratch on cold
   runs. Device equality is checked cold with tools/h13-mobilenet/coldcheck.py.
2. Routing: the first layer reads the program input (channel 5), the last
   writes the program output (channel 4); every intermediate goes through the
   scratch channel (3) at its own tile-aligned region, the producer's write
   base and the consumer's read base agree, and no two regions overlap.
3. Constants: every task's kernel-DMA offsets point at its own layer's
   section in the shared constant section (same bytes as the per-op build).
4. The stride-2 1x1 layers above 128 outputs split into 128-channel tasks.
5. Refusals: graphs that are not a straight conv chain.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from timeouts import scaled_timeout  # noqa: E402

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
HEADER = 0x1000


def halves(count, seed):
    values = bytearray()
    state = seed * 2654435761 % (1 << 32) or 1
    for _ in range(count):
        state = (state * 1103515245 + 12345) % (1 << 31)
        mantissa = state % 1024
        exponent = 9 + (state >> 10) % 5
        sign = (state >> 16) & 1
        values += struct.pack("<H", (sign << 15) | (exponent << 10) | mantissa)
    return bytes(values)


def chain_source(cin, h, w, layers, returns_intermediate=False, extra_add=False):
    """layers: (cout, k, stride, groups, bias, relu). Returns (body, results, weights)."""
    blob = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    lines, cur, c = [], "x", cin
    for i, (cout, k, s, g, bias, relu) in enumerate(layers):
        p = (k - 1) // 2
        ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        offs = []
        for payload in [halves(cout * (c // g) * k * k, 31 * i + 7)] + ([halves(cout, i + 3)] if bias else []):
            meta = len(blob)
            blob += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), meta + 64) + bytes(40) + payload
            blob += bytes((-len(blob)) % 64)
            offs.append(meta)
        pt = "custom" if p else "valid"
        lines += [
            f'string pt{i} = const()[name = string("pt{i}"), val = string("{pt}")];',
            f'tensor<int32, [2]> st{i} = const()[name = string("st{i}"), val = tensor<int32, [2]>([{s}, {s}])];',
            f'tensor<int32, [4]> pd{i} = const()[name = string("pd{i}"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];',
            f'tensor<int32, [2]> dl{i} = const()[name = string("dl{i}"), val = tensor<int32, [2]>([1, 1])];',
            f'int32 gp{i} = const()[name = string("gp{i}"), val = int32({g})];',
            f'tensor<fp16, [{cout}, {c // g}, {k}, {k}]> w{i} = const()[name = string("w{i}"), val = tensor<fp16, [{cout}, {c // g}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offs[0]})))];',
        ]
        bias_arg = ""
        if bias:
            lines.append(f'tensor<fp16, [{cout}]> b{i} = const()[name = string("b{i}"), val = tensor<fp16, [{cout}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offs[1]})))];')
            bias_arg = f"bias = b{i}, "
        lines.append(f'tensor<fp16, [1, {cout}, {ho}, {wo}]> c{i} = conv({bias_arg}dilations = dl{i}, groups = gp{i}, pad = pd{i}, pad_type = pt{i}, strides = st{i}, weight = w{i}, x = {cur})[name = string("conv{i}")];')
        cur = f"c{i}"
        if relu:
            lines.append(f'tensor<fp16, [1, {cout}, {ho}, {wo}]> r{i} = relu(x = {cur})[name = string("relu{i}")];')
            cur = f"r{i}"
        c, h, w = cout, ho, wo
    results = cur
    if extra_add:
        lines.append(f'tensor<fp16, [1, {c}, {h}, {w}]> s = add(x = {cur}, y = {cur})[name = string("add")];')
        results = "s"
    if returns_intermediate:
        results = f"r0, {cur}"
    body = "\n    ".join(lines)
    return body, results, bytes(blob)


def write_case(root, name, cin, h, w, layers, **kw):
    body, results, blob = chain_source(cin, h, w, layers, **kw)
    mil = f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {cin}, {h}, {w}]> x) {{
    {body}
  }} -> ({results});
}}
"""
    case = root / name
    case.mkdir()
    (case / "model.mil").write_text(mil)
    (case / "weights.bin").write_bytes(blob)
    return case


def compile_case(case, schedule="linked", success=True, diagnostic=None):
    out = case / ("out-" + (schedule or "per-op"))
    cmd = [compiler, "--target", "H13", "--mil", str(case / "model.mil"),
           "--model-root", str(case), "--output", str(out)]
    if schedule:
        cmd += ["--schedule", schedule]
    env = dict(os.environ)
    env["MILHWXC_H13_CONV_GENERIC"] = "1"
    run = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=scaled_timeout(120))
    if not success:
        assert run.returncode != 0, (case.name, run.stdout)
        if diagnostic:
            assert diagnostic in run.stderr, (case.name, run.stderr)
        return None, None
    assert run.returncode == 0, (case.name, run.stderr)
    return json.loads((out / "manifest.json").read_text()), out


def words(anec, offset, count):
    return struct.unpack_from(f"<{count}I", anec, HEADER + offset)


def registers(anec, offset, size):
    w = words(anec, offset, size // 4)
    cursor, regs = 10 + (1 if w[9] & 3 == 3 else 0), {}
    while cursor < len(w):
        count = (w[cursor] >> 26) + 1
        base = w[cursor] & 0x03FFFFFF
        for i in range(count):
            if cursor + 1 + i < len(w):
                regs[base + 4 * i] = w[cursor + 1 + i]
        cursor += 1 + count
    return regs


def stream(anec):
    td_size, td_count = struct.unpack_from("<II", anec, 8)
    offset, size, out = 0, td_size, []
    for index in range(td_count):
        h = words(anec, offset, 10)
        out.append((offset, size, h, registers(anec, offset, size)))
        if index + 1 < td_count:
            size = (((h[1] >> 16) & 0x1FF) + 1) * 4
            offset = h[7]
    return out


def task_slices(anec):
    """[(offset, size)] of every task, by walking the links."""
    return [(o, s) for o, s, _, _ in stream(anec)]


CHAINS = {
    # (Cin, H, W, layers, expected tasks per layer)
    "pw-pw": (64, 16, 16, [(64, 1, 1, 1, True, True), (64, 1, 1, 1, True, True)], [1, 1]),
    "dw-pw-s2": (128, 56, 56, [(128, 3, 1, 128, True, True), (256, 1, 2, 1, True, True),
                               (256, 3, 1, 256, True, True), (256, 1, 1, 1, True, False)], [1, 2, 1, 1]),
    "tail": (512, 14, 14, [(512, 3, 1, 512, True, True), (1024, 1, 2, 1, True, True),
                           (1024, 3, 1, 1024, True, True), (1024, 1, 1, 1, True, True)], [1, 8, 1, 4]),
}

with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-linked-") as tmp:
    root = Path(tmp)
    for name, (cin, h, w, layers, per_layer) in CHAINS.items():
        case = write_case(root, name, cin, h, w, layers)
        manifest, out = compile_case(case)
        perop, perop_out = compile_case(case, schedule=None)
        assert manifest["schedule"] == "linked", name
        assert len(manifest["programs"]) == 1, name
        program = manifest["programs"][0]
        assert program["encoder"] == "h13-linked-conv", name
        assert program["taskDescriptors"] == sum(per_layer), (name, program["taskDescriptors"])
        assert [L["taskDescriptors"] for L in program["layers"]] == per_layer, name
        assert len(perop["programs"]) == len(layers), name
        anec = (out / program["file"]).read_bytes()
        tasks = stream(anec)
        assert len(tasks) == sum(per_layer), name

        # 1. stream headers
        for index, (offset, size, h, regs) in enumerate(tasks):
            last = index + 1 == len(tasks)
            assert h[0] & 0xFFFF == index, (name, index)
            assert bool(h[0] & (1 << 24)) == (index > 0), (name, index)
            assert bool(h[0] & (1 << 25)) == last, (name, index)
            assert (h[2] & 0x22 == 0x22) == last and h[2] & 0x408 == 0x408, (name, index)
            assert h[6] & 0x30000800 == 0x30000800, (name, index)
            assert bool(h[8] & (1 << 23)) == (not last), (name, index)
            assert (h[7] == 0) == last, (name, index)
            if not last:
                assert h[7] % 0x100 == 0 and h[7] >= offset + size, (name, index)
        # 1b. inter-layer dependency: TSE+TSR on each layer's last task, and
        # the first task of every consumer layer waits on it (input DMA
        # DependencyMode 7); every other task keeps the program-input mode 3.
        firsts, ends, task_index = set(), set(), 0
        for n in per_layer:
            firsts.add(task_index); ends.add(task_index + n - 1); task_index += n
        for index, (offset, size, h, regs) in enumerate(tasks):
            assert (h[6] & 0x9000 == 0x9000) == (index in ends), (name, index, hex(h[6]))
            li = sum(1 for f in sorted(firsts) if f <= index) - 1
            if li == 0:
                assert regs[0x13800] == 0x33881, (name, index, hex(regs[0x13800]))
            else:
                want = 0x7BB81 if index in firsts else 0x3BB81
                assert regs[0x13800] == want, (name, index, hex(regs[0x13800]))
            last_layer = li == len(per_layer) - 1
            assert regs[0x17800] == (0x40000C1 if last_layer else 0x40003C1), (name, index, hex(regs[0x17800]))
        # 2. routing through the scratch channel
        regions = manifest["scratch"]["regions"]
        assert manifest["scratch"]["channel"] == 3, name
        spans = sorted((r["offset"], r["offset"] + r["bytes"]) for r in regions.values())
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0, (name, spans)
        assert all(o % 0x4000 == 0 for o, _ in spans), (name, spans)
        assert spans[-1][1] <= manifest["scratch"]["bytes"] == program["scratchBytes"], name
        layer_of = []
        for li, n in enumerate(per_layer):
            layer_of += [li] * n
        results = [L["result"] for L in program["layers"]]
        for index, (offset, size, h, regs) in enumerate(tasks):
            li = layer_of[index]
            r0, wsel = h[8] & 31, (h[8] >> 12) & 31
            if li == 0:
                assert r0 == 5 and regs[0x13808] == 0, (name, index)
            else:
                region = regions[results[li - 1]]
                assert r0 == 3 and regs[0x13808] == region["offset"], (name, index)
                assert regs[0x13814] <= region["bytes"], (name, index)
            if li + 1 == len(layers):
                assert wsel == 4, (name, index)
            else:
                region = regions[results[li]]
                assert wsel == 3, (name, index)
                assert region["offset"] <= regs[0x17804], (name, index)
                assert regs[0x17804] + regs[0x17814] <= region["offset"] + region["bytes"], (name, index)
        # 3. constants: each layer's kernel table points at its own section bytes
        constants = anec[HEADER + program["constantOffset"]:]
        cursor = 0
        for li, record in enumerate(perop["programs"]):
            panec = (perop_out / record["file"]).read_bytes()
            pconst = panec[HEADER + record["constantOffset"]:]
            ptasks = stream(panec)
            ltasks = [t for t, l in zip(tasks, layer_of) if l == li]
            assert len(ptasks) == len(ltasks), (name, li)
            for (_, _, _, pregs), (_, _, _, lregs) in zip(ptasks, ltasks):
                for chunk in range(16):
                    po, ps = pregs[0x1F848 + 4 * chunk], pregs[0x1F888 + 4 * chunk]
                    lo, ls = lregs[0x1F848 + 4 * chunk], lregs[0x1F888 + 4 * chunk]
                    assert ps == ls and lo == po + cursor, (name, li, chunk)
                    assert constants[lo:lo + ls] == pconst[po:po + ps], (name, li, chunk)
            cursor += (record["constantBytes"] + 63) // 64 * 64
        assert manifest["tensors"].keys() >= {"x"}, name
        assert not manifest["intermediates"], (name, manifest["intermediates"])

    # 4. stride-2 1x1 above 128 outputs: 128-channel tasks, 8 lanes each.
    case = write_case(root, "s2", 128, 56, 56, [(256, 1, 2, 1, True, True)])
    manifest, out = compile_case(case, schedule=None)
    program = manifest["programs"][0]
    assert program["taskDescriptors"] == 2, program["taskDescriptors"]
    anec = (out / program["file"]).read_bytes()
    for offset, size, h, regs in stream(anec):
        assert regs[0x0C800] == 0x182 and regs[0x10] == 128, hex(regs[0x10])
        assert (regs[0x1C] >> 10) & 7 == 3, hex(regs[0x1C])   # OCG: 8 lanes

    # 5. refusals
    case = write_case(root, "one", 64, 16, 16, [(64, 1, 1, 1, True, True)])
    compile_case(case, success=False, diagnostic="h13.linked-outside-envelope")
    case = write_case(root, "returns-mid", 64, 16, 16, [(64, 1, 1, 1, True, True), (64, 1, 1, 1, True, True)],
                      returns_intermediate=True)
    compile_case(case, success=False, diagnostic="h13.linked-outside-envelope")
    case = write_case(root, "with-add", 64, 16, 16, [(64, 1, 1, 1, True, True), (64, 1, 1, 1, True, True)],
                      extra_add=True)
    compile_case(case, success=False, diagnostic="h13.linked-outside-envelope")

print(f"h13 linked cli: PASS ({len(CHAINS)} chains, stride-2 split, 3 refusals)")

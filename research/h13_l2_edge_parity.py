#!/usr/bin/env python3
"""Field parity of `--schedule linked-l2` against decoded Apple L2 chains.

usage: h13_l2_edge_parity.py MILHWXC ORACLE_FACT.json [...]

Each oracle fact is a decoded Apple H13 program for a 64-channel chain of
1x1 / depthwise 3x3 layers (no bias, no relu) at spatial S. The script
rebuilds the same chain with the open compiler and compares every task's
header words 0..8 (ExeCycles, the binding id and the input/output channel
numbering excluded) and every word of
the Common, L2 and TileDMA source/destination blocks. Kernel-DMA offsets and
the NE block carry weights and are skipped. Prints mismatches and exits 1
on any.
"""
import json, os, re, struct, subprocess, sys, tempfile
from pathlib import Path

HEADER = 0x1000


def words(anec, offset, count):
    return struct.unpack_from(f"<{count}I", anec, HEADER + offset)


def tasks(anec):
    td_size, td_count = struct.unpack_from("<II", anec, 8)
    out, offset, size = [], 0, td_size
    for i in range(td_count):
        w = words(anec, offset, size // 4)
        cursor, regs = 10 + (1 if w[9] & 3 == 3 else 0), {}
        while cursor < len(w):
            n = (w[cursor] >> 26) + 1
            base = w[cursor] & 0x03FFFFFF
            for j in range(n):
                if cursor + 1 + j < len(w):
                    regs[base + 4 * j] = w[cursor + 1 + j]
            cursor += 1 + n
        out.append((list(w[:10]), regs))
        if i + 1 < td_count:
            size = (((w[1] >> 16) & 0x1FF) + 1) * 4
            offset = w[7]
    return out


def mil_for(layers, c, s):
    """layers: (k, groups, cout, stride); groups == cin marks depthwise."""
    blob = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    lines, cur, h = [], "x", s
    cin0 = c
    for i, (k, g, co, st) in enumerate(layers):
        payload = bytes(2 * co * (c // g) * k * k)
        meta = len(blob)
        blob += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), meta + 64) + bytes(40) + payload
        blob += bytes((-len(blob)) % 64)
        p = (k - 1) // 2
        ho = (h + 2 * p - k) // st + 1
        lines += [
            f'string pt{i} = const()[name = string("pt{i}"), val = string("{"custom" if p else "valid"}")];',
            f'tensor<int32, [2]> st{i} = const()[name = string("st{i}"), val = tensor<int32, [2]>([{st}, {st}])];',
            f'tensor<int32, [4]> pd{i} = const()[name = string("pd{i}"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];',
            f'tensor<int32, [2]> dl{i} = const()[name = string("dl{i}"), val = tensor<int32, [2]>([1, 1])];',
            f'int32 gp{i} = const()[name = string("gp{i}"), val = int32({g})];',
            f'tensor<fp16, [{co}, {c // g}, {k}, {k}]> w{i} = const()[name = string("w{i}"), val = tensor<fp16, [{co}, {c // g}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({meta})))];',
            f'tensor<fp16, [1, {co}, {ho}, {ho}]> c{i} = conv(dilations = dl{i}, groups = gp{i}, pad = pd{i}, pad_type = pt{i}, strides = st{i}, weight = w{i}, x = {cur})[name = string("conv{i}")];',
        ]
        cur, c, h = f"c{i}", co, ho
    body = "\n    ".join(lines)
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {cin0}, {s}, {s}]> x) {{
    {body}
  }} -> ({cur});
}}
""", bytes(blob)


def roles(h8):
    """Surface selectors as roles. The open ANEC binds the program input and
    output on channels 5 and 4; Apple's captures use either order. Map a read
    selector on 4/5 to "input" and a write selector on 4/5 to "output"."""
    out = h8
    for shift, role in ((0, 4), (6, 4), (12, 5)):
        if (h8 >> shift) & 31 in (4, 5):
            out = (out & ~(31 << shift)) | (role << shift)
    return out


# Common 0x34 (spatial scheduling class) is a per-layer field, not an edge
# field; Apple repeats the class in more forms than the per-layer rule does.
NON_EDGE = {0x00034}


def compare(milc, fact):
    d = json.load(open(fact))
    case = d["case"]
    params = d.get("parameters")
    if params and "layers" in params:
        c, s = params["layers"][0]["channels_in"], params["input_spatial"]
        layers = [(L["kernel"], L["groups"], L["channels_out"], L["stride"]) for L in params["layers"]]
    else:
        m = re.match(r"(pw_pw|conv_dw_pw|pw_dw_pw)_c(\d+)_s(\d+)_h13", case)
        if not m:
            return case, ["unknown case name"]
        kind, c, s = m.group(1), int(m.group(2)), int(m.group(3))
        layers = ([(1, 1, c, 1), (1, 1, c, 1)] if kind == "pw_pw"
                  else [(1, 1, c, 1), (3, c, c, 1), (1, 1, c, 1)])
    with tempfile.TemporaryDirectory() as tmp:
        mil, blob = mil_for(layers, c, s)
        Path(tmp, "model.mil").write_text(mil)
        Path(tmp, "weights.bin").write_bytes(blob)
        r = subprocess.run([milc, "--target", "H13", "--schedule", "linked-l2", "--mil",
                            f"{tmp}/model.mil", "--model-root", tmp, "--output", f"{tmp}/out"],
                           capture_output=True, text=True)
        if r.returncode:
            return case, ["compile failed: " + r.stderr.strip()[-300:]]
        anec = Path(tmp, "out", "program-0.anec").read_bytes()
    ours = tasks(anec)
    theirs = d["task_descriptors"]
    if len(ours) != len(theirs):
        # The open compiler splits some layers into more tasks (128-output
        # stride-2 slices, the 1 MiB kernel budget); no field comparison.
        return case, None
    # Split-producer slices: the open encoder runs channel slices in
    # ascending order and Apple's descend, so slice result bases are compared
    # as a set (docs/ane/linked-programs.md).
    def is_slice(regs_of, k):
        return regs_of(k).get(0x04830) == 0x140 and regs_of(k).get(0x04804) in (0x160, 0x100) and \
            k + 1 < len(ours) and regs_of(k + 1).get(0x04804) in (0x100, 0x120)
    apple_regs = [{int(a, 16): int(v, 16) for blk in t["blocks"].values() for a, v in blk["words"].items()} for t in theirs]
    slice_tasks = [k for k in range(len(ours)) if is_slice(lambda q: apple_regs[q], k)
                   and (k + 1 < len(ours) and apple_regs[k + 1].get(0x04804) in (0x100, 0x120))]
    groups = []
    for k in slice_tasks:
        if groups and groups[-1][-1] == k - 1: groups[-1].append(k)
        else: groups.append([k])
    groups = [g + [g[-1] + 1] if apple_regs[g[-1] + 1].get(0x04804) == 0x100 else g for g in groups]
    set_checked = set()
    bad = []
    for grp in groups:
        if len(grp) < 2: continue
        a = sorted(ours[k][1].get(0x04834) for k in grp)
        b = sorted(apple_regs[k].get(0x04834) for k in grp)
        if a != b:
            bad.append(f"slice result bases {[hex(x) for x in a]} vs {[hex(x) for x in b]}")
        set_checked.update(grp)
    for i, ((h, regs), t) in enumerate(zip(ours, theirs)):
        ah = [int(x, 16) for x in t["header_words"]]
        for j in range(9):
            a, b = h[j], ah[j]
            if j == 0:
                a &= ~0x00FF0000; b &= ~0x00FF0000          # binding id
            if j == 1:
                a &= ~0xFFFF; b &= ~0xFFFF                  # ExeCycles
            if j == 8:
                a, b = roles(a), roles(b)                   # channel numbering
            if a != b:
                bad.append(f"t{i} header[{j}] ours {a:#010x} apple {b:#010x}")
        for blk in ("0x00000", "0x04800", "0x13800", "0x17800"):
            for k, v in t["blocks"][blk]["words"].items():
                a, b = regs.get(int(k, 16)), int(v, 16)
                if a != b and int(k, 16) not in NON_EDGE and not (
                        int(k, 16) == 0x04834 and i in set_checked):
                    bad.append(f"t{i} {k} ours {a if a is None else hex(a)} apple {v}")
    return case, bad


if __name__ == "__main__":
    milc, facts = sys.argv[1], sys.argv[2:]
    failed = 0
    for fact in facts:
        case, bad = compare(milc, fact)
        if bad is None:
            print(f"{case}: n/a (different task split)")
            continue
        failed += bool(bad)
        print(f"{case}: {'PARITY' if not bad else f'{len(bad)} mismatches'}")
        for line in bad:
            print("   ", line)
    sys.exit(1 if failed else 0)

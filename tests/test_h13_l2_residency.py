#!/usr/bin/env python3
"""Decoded field parity, full-span refusals and host L2 transport parity.

This models address transport between per-layer float64 convolutions. It
does not simulate ANE arithmetic or establish device execution safety.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "tools/h13-mobilenet")]
from h13_l2_edge_parity import compare, tasks  # noqa: E402
from chain import conv_mil, compile_chain, chain_ref, conv_ref, f16_ties_away  # noqa: E402

COMPILER = str(Path(sys.argv[1]).resolve())
RING = 0x200000
FACTS = ROOT / "research/oracles/h13-l2-r3"


def transport(tensor, regs, memory, source, base_channel=0):
    """Logical CHW <-> decoded L2 C/R strides, including circular wrap."""
    base, channel, row = (regs[a] for a in
                         ((0x4808, 0x480c, 0x4810) if source else
                          (0x4834, 0x4838, 0x483c)))
    c, h, w = tensor.shape
    assert channel >= w * 2 and row >= channel * (c + base_channel)
    result = np.empty_like(tensor)
    # Work a channel row at a time, including rows crossing the ring end.
    for y in range(h):
        for z in range(c):
            addresses = (base + y * row + z * channel + np.arange(w) * 2) % RING
            if source:
                result[z, y] = memory[addresses // 2].view(np.float16)
            else:
                memory[addresses // 2] = tensor[z, y].view(np.uint16)
    return result if source else None


def case(root, name, channels, spatial, specs):
    layers = [dict(cout=co, k=k, s=s, g=g, bias=True, relu=True)
              for k, g, co, s in specs]
    directory = root / name
    mil, params, _ = conv_mil(layers, (channels, spatial, spatial),
                              str(directory / "src"), seed=91)
    lm = compile_chain(COMPILER, mil, str(directory / "src"),
                       str(directory / "linked"), "linked-l2")
    pm = compile_chain(COMPILER, mil, str(directory / "src"),
                       str(directory / "perop"), None)
    program = lm["programs"][0]
    anec = (directory / "linked" / program["file"]).read_bytes()
    stream = tasks(anec)
    # Relinking must preserve the numerical task body and every kernel byte.
    constants = anec[0x1000 + program["constantOffset"]:]
    const_base = 0
    for layer, record in zip(program["layers"], pm["programs"]):
        per = (directory / "perop" / record["file"]).read_bytes()
        assert constants[const_base:const_base + record["constantBytes"]] == \
            per[0x1000 + record["constantOffset"]:]
        for index, (_, regs) in enumerate(tasks(per)):
            linked = stream[layer["firstTask"] + index][1]
            for address in regs:
                if 0xc800 <= address < 0xd000 or 0x1f800 <= address < 0x1f900:
                    expected = regs[address] + (const_base if 0x1f848 <= address < 0x1f888 else 0)
                    assert linked[address] == expected, (name, address)
        const_base += (record["constantBytes"] + 63) // 64 * 64

    x = np.random.default_rng(7).uniform(-0.5, 0.5, (channels, spatial, spatial)).astype(np.float16)
    reference = chain_ref(x, layers, params)
    memory = np.full(RING // 2, 0x7e00, np.uint16)
    current = x
    for i, (layer, spec, params_i) in enumerate(zip(program["layers"], layers, params)):
        first = layer["firstTask"]
        if layer["inputEdge"] != "dram":
            current = transport(current, stream[first][1], memory, True)
            assert np.array_equal(current.view(np.uint16), reference[i - 1].view(np.uint16)), (name, i)
            # Every later output-channel task rereads the same assembled input.
            for t in range(1, layer["taskDescriptors"]):
                header, regs = stream[first + t]
                assert regs[0x4804] == 0x100 and regs[0x13800] == 0x8880
                assert not header[6] & 0x2800 and not header[8] & 0x3f
                reread = transport(current, regs, memory, True)
                assert np.array_equal(reread.view(np.uint16), current.view(np.uint16))
        current = f16_ties_away(conv_ref(current, params_i['w'], params_i['b'],
                                        spec['s'], spec['g'], (spec['k'] - 1) // 2, True))
        assert np.array_equal(current.view(np.uint16), reference[i].view(np.uint16)), (name, i)
        if layer["outputEdge"] != "dram":
            done = 0
            for t in range(layer["taskDescriptors"]):
                regs = stream[first + t][1]
                count = regs[0x10]
                transport(current[done:done + count], regs, memory, False, done)
                done += count
            assert done == current.shape[0]
    return program, stream


for channels in (128,):
    name, bad = compare(COMPILER, FACTS / f"pw_dw_pw_c{channels}_s56_h13.json")
    assert bad == [], (name, bad)  # Never accept a different-task-count skip.
    print(name + ": decoded field PARITY")

name, bad = compare(COMPILER, FACTS / "mobile_tail6_c1024_s7_h13.json")
assert bad == [], (name, bad)

# Apple's forced 4/8-task streams prove the handoff protocol independently
# of this encoder's smaller kernel slices. Do not count a task-count skip
# from the general parity script as coverage of these fields.
for co, count in ((2048, 4), (4096, 8)):
    fact = json.loads((FACTS / f"mobile_dw_pw_dw_c1024_n{co}_s16_st1_h13.json").read_text())
    decoded = [{int(a, 16): int(v, 16) for b in task['blocks'].values()
                for a, v in b['words'].items()} for task in fact['task_descriptors']]
    assert len(decoded) == count + 2
    for j, regs in enumerate(decoded[1:-1]):
        assert regs[0x4804] == (0x160 if j == 0 else 0x100)
        assert regs[0x13800] == (0x48880 if j == 0 else 0x8880)
        assert [regs[a] for a in (0x4808, 0x480c, 0x4810, 0x4814, 0x4818)] == \
            [decoded[1][a] for a in (0x4808, 0x480c, 0x4810, 0x4814, 0x4818)]
        if co == 2048:
            assert regs[0x4830] == 0x140 and regs[0x4834] == (count - 1 - j) * 512 * 0x20
            assert regs[0x483c] == 0x10030 and regs[0x4840] == 0x10000
        else:
            assert regs[0x17800] & 1  # 0x200300-byte aggregate spills.

with tempfile.TemporaryDirectory(prefix="h13-l2-residency-") as tmp:
    root = Path(tmp)
    for c in (128,):
        p, stream = case(root, f"wide{c}", c, 56,
                         [(1, 1, c, 1), (3, c, c, 1), (1, 1, c, 1)])
        assert p["scratchBytes"] == 0
        assert [l["outputEdge"] for l in p["layers"]] == ["l2-alias", "l2-dependent-wide", "dram"]
        assert stream[-1][1][0x4830] == 0x142 and stream[-1][1][0x17800] == 0xc1
    # Long-chain wrap and ascending output-channel slice assembly.
    case(root, "wrap", 256, 28, [(3, 256, 256, 1), (1, 1, 256, 1)] * 3)
    p, _ = case(root, "split", 1024, 16, [(3, 1024, 1024, 1), (1, 1, 1024, 1), (3, 1024, 1024, 1)])
    assert p['layers'][1]['taskDescriptors'] > 1 and p['layers'][1]['outputEdge'] == 'l2-dependent-wide'
    # c176 is the first decoded spill. A compute TileHeight must never be
    # substituted for the complete 56-row Source tuple.
    for c in (128, 160, 176, 192, 256):
        fact = json.loads((FACTS / f"pw_dw_pw_c{c}_s56_h13.json").read_text())
        apple = fact['task_descriptors'][1]['blocks']['0x04800']['words']
        fits = 2 * 56 * int(apple['0x04810'], 16) <= RING
        assert fits == (c <= 160)
        assert int(apple['0x04830'], 16) == (0x140 if fits else 0x84a)
        if c in (160, 176, 192):
            # Their mixed-width kernel lane chunks have no generic form;
            # decoded residency evidence must not widen that separate gate.
            try:
                case(root, f"unsupported{c}", c, 56,
                     [(1, 1, c, 1), (3, c, c, 1), (1, 1, c, 1)])
            except RuntimeError as error:
                assert 'different widths' in str(error), error
            else:
                raise AssertionError('mixed-width kernel unexpectedly accepted')
    p, stream = case(root, "spill256", 256, 56,
                     [(1, 1, 256, 1), (3, 256, 256, 1), (1, 1, 256, 1)])
    edge = p['layers'][1]
    assert edge['outputEdge'] == 'dram' and edge['l2SpillReason']
    assert edge['candidateL2Bytes'] == 56 * stream[2][1][0x4810]
    # The wide terminal form may not silently become a scratch writer.
    p, _ = case(root, "wide-successor", 128, 56,
                [(1, 1, 128, 1), (3, 128, 128, 1), (1, 1, 128, 1), (3, 128, 128, 1)])
    assert p['layers'][1]['outputEdge'] == 'dram'
    p, _ = case(root, "early112", 32, 112,
                [(3, 32, 32, 1), (1, 1, 64, 1), (3, 64, 64, 1)])
    assert any(l.get('l2SpillReason') for l in p['layers'][:-1])

print("h13 L2 residency: PASS (56-wide fields, wrap, split, full-span refusals, per-layer transport)")

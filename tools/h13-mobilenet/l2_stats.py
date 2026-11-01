"""Count descriptor-requested DRAM bytes; no bandwidth or latency fit."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
from h13_l2_edge_parity import tasks  # noqa: E402


def package_stats(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    records = []
    for program in manifest["programs"]:
        stream = tasks((directory / program["file"]).read_bytes())
        layers = []
        layout = program.get("layers", [dict(firstTask=0, taskDescriptors=len(stream))])
        for layer in layout:
            selected = stream[layer["firstTask"]:layer["firstTask"] + layer["taskDescriptors"]]
            incoming = sum(r[0x13814] for _, r in selected if r[0x13800] & 1)
            outgoing = sum(r[0x17814] for _, r in selected if r[0x17800] & 1)
            kernels = sum(sum(r[0x1f888 + 4 * chunk] for chunk in range(16)) for _, r in selected)
            layers.append(dict(layer, inputDmaBytes=incoming, outputDmaBytes=outgoing,
                               kernelDmaBytes=kernels, predictedBytesMoved=incoming + outgoing + kernels))
        records.append(dict(tasks=len(stream), scratchBytes=program["scratchBytes"],
                            residentEdges=sum(l.get("outputEdge", "dram") != "dram" for l in layers),
                            layers=layers, predictedBytesMoved=sum(l["predictedBytesMoved"] for l in layers),
                            inputDmaBytes=sum(l["inputDmaBytes"] for l in layers),
                            outputDmaBytes=sum(l["outputDmaBytes"] for l in layers),
                            kernelDmaBytes=sum(l["kernelDmaBytes"] for l in layers)))
    return dict(model="enabled TileDMA extents plus per-task kernel-DMA chunk lengths; excludes cache/DRAM transaction effects",
                programs=records, tasks=sum(p["tasks"] for p in records),
                scratchBytes=sum(p["scratchBytes"] for p in records),
                predictedBytesMoved=sum(p["predictedBytesMoved"] for p in records))


if __name__ == "__main__":
    print(json.dumps(package_stats(sys.argv[1]), indent=2))

#!/usr/bin/env python3
import importlib.util
import os
from pathlib import Path


path = Path(__file__).resolve().parents[1] / 'tools/h13-resnet50/reference.py'
spec = importlib.util.spec_from_file_location('h13_resnet_reference', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    assert module.tf_same_pad(224, 7, 2) == (2, 3)
    assert module.tf_same_pad(56, 3, 2) == (0, 1)
    assert module.tf_same_pad(14, 3, 2) == (0, 1)
    assert module.tf_same_pad(7, 3, 1) == (1, 1)
    # Always execute a float64 residual prefix, including TF-SAME stride 2,
    # even on minimal hosts where NumPy is unavailable.
    sample = [[[float(y * 3 + x + 1) for x in range(3)] for y in range(3)]]
    identity = [[[[0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0]]]]
    down = module.conv2d_python(sample, identity, [0.0], stride=2)
    assert down == [[[1.0, 3.0], [7.0, 9.0]]]
    branch = module.conv2d_python(down, [[[[2.0]]]], [0.0])
    fused = [[[max(0.0, branch[0][y][x] + down[0][y][x])
               for x in range(2)] for y in range(2)]]
    assert module.reduce_mean_python(fused) == [15.0]
    model = os.environ.get('MILHWXC_RESNET50_MODEL')
    if model and module.np is not None:
        sample = module.np.zeros((3, 224, 224), dtype=module.np.float16)
        logits, prefixes = module.evaluate(Path(model), sample)
        assert logits.shape == (1000,)
        assert prefixes[1].shape == (256, 56, 56)
        assert prefixes[4].shape == (512, 28, 28)
        assert prefixes[8].shape == (1024, 14, 14)
        assert prefixes[14].shape == (2048, 7, 7)
        assert module.np.isfinite(logits).all()
    print('H13 ResNet-50 float64 reference: TF-SAME stride-2 padding PASS')


if __name__ == '__main__':
    main()

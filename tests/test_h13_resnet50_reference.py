#!/usr/bin/env python3
import importlib.util
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
    print('H13 ResNet-50 float64 reference: TF-SAME stride-2 padding PASS')


if __name__ == '__main__':
    main()

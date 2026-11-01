"""CPU planning geometry for overlap-save convolution, not an ANE encoder.

Coordinates refer to dense NCHW tensors. A future backend must separately
prove its gather/scatter ABI and every emitted kernel geometry.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialTile:
    output_top: int
    output_left: int
    output_height: int
    output_width: int
    input_top: int
    input_left: int
    patch_size: int
    crop_top: int
    crop_left: int


def conv_tiles(height, width, kernel, patch_size=64):
    """Partition unit-stride SAME convolution into fixed patches with halos.

    The patch includes kernel//2 input samples on each side of its output
    core. Outside-image samples are zero. Discard the halo in the tile
    output and store only output_height x output_width samples. Each output
    pixel has exactly one owner, including partial right/bottom tiles.
    """
    if any(type(v) is not int or v <= 0 for v in (height, width, kernel, patch_size)):
        raise ValueError('spatial.invalid-dimensions')
    if kernel not in (1, 3):
        raise ValueError('spatial.kernel-outside-reference-envelope')
    halo = kernel // 2
    core = patch_size - 2 * halo
    if core <= 0:
        raise ValueError('spatial.patch-too-small')
    return tuple(SpatialTile(y, x, min(core, height - y), min(core, width - x),
                             y - halo, x - halo, patch_size, halo, halo)
                 for y in range(0, height, core) for x in range(0, width, core))

#!/usr/bin/env python3
"""Validate the generated 28-layer H14 package without opening a device."""
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import split_h14_tasks, decode_task, register_words  # noqa: E402
from h14_libane_overrun import overruns  # noqa: E402
from h14_fields import H14Fields  # noqa: E402


# ANEC layout constants are format boundaries.  Register addresses and packed
# fields are read from the compiler's public H14Fields.h vocabulary.
ANEC_TEXT_OFFSET = 4096
ANEC_TILE_TABLE_OFFSET = 40
TILE_BYTES = 16384
FIELDS = H14Fields(ROOT)
SOURCE_CONFIG = FIELDS.register('sourceConfig')
RESULT_CONFIG = FIELDS.register('resultConfig')
SOURCE_DMA_CONFIG = FIELDS.register('sourceDmaConfig')
SOURCE_DMA_EXTENSION = FIELDS.register('sourceDmaExtension')
SOURCE_DMA_BASE = FIELDS.register('sourceDmaBase')
RESULT_DMA_CONFIG = FIELDS.register('resultDmaConfig')
RESULT_DMA_BASE = FIELDS.register('resultDmaBase')
KERNEL_DMA_CONFIG = FIELDS.register('kernelDmaConfig')
KERNEL_DMA_BASE = FIELDS.register('kernelDmaBase')
KERNEL_LANE_CONFIG = FIELDS.register('kernelLaneConfig')
KERNEL_LANE_OFFSET = FIELDS.register('kernelLaneOffset')
KERNEL_LANE_SIZE = FIELDS.register('kernelLaneSize')
# ABI-2 slots, not hardware field encodings.
PROGRAM_INPUT_BAR = 4
SCRATCH_BAR = 3
PROGRAM_OUTPUT_BAR = 5


def task_header2(first, last):
    """Program-open and program-close/final-command fields."""
    return (FIELDS.field('taskProgramOpen') if first else 0) | (
        (FIELDS.field('taskProgramClose') |
         FIELDS.field('taskFinalCommand')) if last else 0)


def task_header7():
    """TSR, scheduler-class and completion fields."""
    return (FIELDS.field('taskTsr') | FIELDS.field('taskSchedulerClass') |
            FIELDS.field('taskCompletion'))


def assert_l2_config(encoded, role):
    """Assert the named controls implied by an allocator edge role."""
    active = {
        'program-input': {'l2RoleDmaTransfer', 'l2RoleProgramEdge',
                          'l2RoleWideSource', 'l2RoleLinearSurface',
                          'l2ResidentEdge', 'l2ExternalDma',
                          'l2ExternalBinding'},
        'program-input-result': {'l2RoleProgramEdge', 'l2RoleWideSource',
                                 'l2RoleLinearSurface', 'l2ResidentEdge',
                                 'l2ExternalDma', 'l2ExternalBinding'},
        'resident-wide': {'l2RoleWideSource', 'l2ResidentEdge'},
        'resident-compact': {'l2RoleCompactSource', 'l2ResidentEdge'},
        'resident-result': {'l2RoleLinearSurface', 'l2ResidentEdge'},
        'scratch-source': {'l2RoleDmaTransfer', 'l2RoleLinearSurface',
                           'l2Scratch'},
        'scratch-result': {'l2RoleDmaTransfer',
                           'l2RoleInterleavedResult',
                           'l2RoleLinearSurface', 'l2Scratch'},
        'direct-result': {'l2RoleDmaTransfer',
                          'l2RoleInterleavedResult',
                          'l2RoleLinearSurface', 'l2ResidentEdge'},
    }[role]
    controls = (
        'l2RoleReservedLow', 'l2RoleDmaTransfer', 'l2RoleCompactSource',
        'l2RoleInterleavedResult', 'l2RoleProgramEdge',
        'l2RoleWideSource', 'l2RoleLinearSurface', 'l2RoleReservedHigh',
        'l2ResidentEdge', 'l2Scratch', 'l2ExternalDma',
        'l2ExternalBinding')
    assert all(bool(FIELDS.get(name, encoded)) == (name in active)
               for name in controls)


def assert_source_dma_config(encoded, task, scratch):
    """Check source TileDMA from named role and extension controls."""
    assert FIELDS.get('dmaEnable', encoded)
    assert FIELDS.get('dmaRoleResultWrite', encoded) == 0
    assert bool(FIELDS.get('dmaRoleLinearSurface', encoded)) != scratch
    assert bool(FIELDS.get('dmaRoleScratchSurface', encoded)) == scratch
    assert FIELDS.get('dmaRoleInterleavedResult', encoded) == 0
    assert FIELDS.get('dmaExtensionReserved', encoded) == 0
    assert all(FIELDS.get(name, encoded) for name in
               ('dmaExtensionCoordinates', 'dmaExtensionStrides',
                'dmaExtensionTask'))
    assert FIELDS.get('dmaTaskTag', encoded) == task
    assert bool(FIELDS.get('dmaScratchDependency', encoded)) == scratch


def assert_source_dma_extension(encoded, program_input):
    assert all(FIELDS.get(name, encoded) for name in
               ('dmaExtensionLowCoordinates', 'dmaExtensionLowStrides',
                'dmaExtensionLowTask', 'dmaExtensionSource'))
    assert bool(FIELDS.get('dmaExtensionProgramInput', encoded)) == program_input


def assert_result_dma_config(encoded, task, scratch):
    """Check result TileDMA direction, surface and task controls."""
    assert FIELDS.get('dmaEnable', encoded)
    assert FIELDS.get('dmaRoleResultWrite', encoded)
    assert FIELDS.get('dmaRoleLinearSurface', encoded)
    assert bool(FIELDS.get('dmaRoleScratchSurface', encoded)) == scratch
    assert bool(FIELDS.get('dmaRoleInterleavedResult', encoded)) == scratch
    assert all(FIELDS.get(name, encoded) for name in
               ('dmaExtensionCoordinates', 'dmaExtensionStrides',
                'dmaExtensionTask', 'dmaExternalSurface'))
    assert FIELDS.get('dmaTaskTag', encoded) == task


def contents(path):
    a = path.read_bytes()
    payload, first, count, text, constants = struct.unpack_from('<QIIQQ', a)
    assert payload + ANEC_TEXT_OFFSET == len(a)
    offset = payload - constants
    assert offset % 64 == 0 and offset >= text
    tasks = split_h14_tasks(a[ANEC_TEXT_OFFSET:ANEC_TEXT_OFFSET + text])
    assert len(tasks) == count and len(tasks[0]) == first
    return a, tasks, a[ANEC_TEXT_OFFSET + offset:]


def main():
    net = Path(sys.argv[1])
    m = json.loads((net / 'linked/manifest.json').read_text())
    assert m['target'] == 'H14' and m['schedule'] == 'linked-ring'
    assert len(m['programs']) == 1 and m['dispatchPlan'] == [0]
    a, tasks, constants = contents(net / 'linked' / m['programs'][0]['file'])
    assert len(tasks) == 28 and len({t['layer'] for t in m['tasks']}) == 28
    assert len(m['scratch']['regions']) == 1
    assert m['scratch']['bytes'] == next(
        iter(m['scratch']['regions'].values()))['bytes']
    assert (struct.unpack_from('<I', a,
            ANEC_TILE_TABLE_OFFSET + 3 * 4)[0] * TILE_BYTES ==
            m['scratch']['bytes'])
    assert not overruns(a)
    constant_bases = []
    for index, (raw, tr) in enumerate(zip(tasks, m['tasks'])):
        task = decode_task(raw, 'h14')
        words = register_words(task)
        header = [int(w, 16) for w in task['header_words']]
        assert FIELDS.get('taskIdentifier', header[0]) == index
        assert FIELDS.get('taskWordCount', header[0]) * 4 == len(raw)
        assert header[2] == task_header2(index == 0, index == 27)
        assert header[7] == task_header7()
        expected_source = ('program-input' if index == 0 else
                           'scratch-source' if index == 2 else
                           'resident-compact'
                           if index == 7 or index >= 10 else
                           'resident-wide')
        assert_l2_config(words[SOURCE_CONFIG], expected_source)
        expected_result = ('program-input-result' if index == 0 else
                           'scratch-result' if index == 1 else
                           'direct-result' if index == 27 else
                           'resident-result')
        assert_l2_config(words[RESULT_CONFIG], expected_result)
        if index not in (0, 2):
            assert not any(SOURCE_DMA_CONFIG <= address < RESULT_DMA_CONFIG
                           for address in words)
        if index not in (1, 27):
            assert RESULT_DMA_CONFIG not in words
        if index == 0:
            assert_source_dma_config(words[SOURCE_DMA_CONFIG], index, False)
            assert_source_dma_extension(words[SOURCE_DMA_EXTENSION], True)
        elif index == 2:
            assert_source_dma_config(words[SOURCE_DMA_CONFIG], index, True)
            assert_source_dma_extension(words[SOURCE_DMA_EXTENSION], False)
        elif index == 1:
            assert_result_dma_config(words[RESULT_DMA_CONFIG], index, True)
        elif index == 27:
            assert_result_dma_config(words[RESULT_DMA_CONFIG], index, False)
        assert FIELDS.get('kernelNextTaskTag',
                          words[KERNEL_DMA_CONFIG]) == index + 1
        constant_bases.append(words[KERNEL_DMA_BASE])
        assert FIELDS.get('dmaTaskTag', words[KERNEL_LANE_CONFIG]) == index
        for i in range(16):
            if KERNEL_LANE_SIZE + 4*i in words:
                assert (words[KERNEL_DMA_BASE] +
                        words.get(KERNEL_LANE_OFFSET + 4*i, 0) +
                        words[KERNEL_LANE_SIZE + 4*i] <= len(constants))
        bars = {int(r['address'], 16):
                FIELDS.get('barSlot', int(r['header'], 16))
                for r in task['records']
                if (int(r['header'], 16) & FIELDS.field('barPresent') and
                    not int(r['header'], 16) & FIELDS.field('scatterPresent'))}
        assert bars[KERNEL_DMA_BASE] == 1
        if index in (0, 2):
            assert bars[SOURCE_DMA_BASE] == (
                PROGRAM_INPUT_BAR if index == 0 else SCRATCH_BAR)
        if index in (1, 27):
            assert bars[RESULT_DMA_BASE] == (
                SCRATCH_BAR if index == 1 else PROGRAM_OUTPUT_BAR)
    assert constant_bases[0] == 0 and constant_bases == sorted(constant_bases)
    last_words = register_words(decode_task(tasks[-1], 'h14'))
    assert max(last_words[KERNEL_DMA_BASE] +
               last_words.get(KERNEL_LANE_OFFSET + 4*i, 0) +
               last_words[KERNEL_LANE_SIZE + 4*i]
               for i in range(16)) == len(constants)
    print(json.dumps(dict(status='PASS', layers=28, tasks=28, scratch=m['scratch']['bytes'],
                          program_bytes=len(a), sha256=hashlib.sha256(a).hexdigest())))

if __name__ == '__main__':
    main()

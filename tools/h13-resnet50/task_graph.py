#!/usr/bin/env python3
"""Audit residual task reads, writes, ranges, and producer dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def end(access: dict, space: str) -> int:
    return access[f'{space}Base'] + access[f'{space}Bytes']


def covers(write: dict, read: dict, space: str = 'surface') -> bool:
    return (write['storage'] == read['storage'] and
            write[f'{space}Base'] <= read[f'{space}Base'] and
            end(write, space) >= end(read, space))


def overlaps(left: dict, right: dict, space: str) -> bool:
    return (left[f'{space}Bytes'] and right[f'{space}Bytes'] and
            left[f'{space}Base'] < end(right, space) and
            right[f'{space}Base'] < end(left, space))


def fmt(access: dict) -> str:
    surface = (f"{access['storage']} "
               f"0x{access['surfaceBase']:x}+0x{access['surfaceBytes']:x}")
    return (surface + f"; L2 0x{access['l2Base']:x}+"
            f"0x{access['l2Bytes']:x}")


def audit(tasks: list[dict]) -> tuple[list[str], list[str]]:
    rows, hazards = [], []
    for index, task in enumerate(tasks):
        producers = []
        for read_index, read in enumerate(task['reads']):
            if read['storage'] == 'external':
                producers.append('external')
                continue
            matches = [producer for producer in range(index)
                       if any(covers(write, read)
                              for write in tasks[producer]['writes'])]
            if not matches:
                hazards.append(
                    f"task {index} read {read_index} has no ordered producer")
                producers.append('MISSING')
            else:
                producers.append(str(matches[-1]))
            if read['storage'] == 'scratch' and read_index == 0:
                reread = any(
                    earlier['role'] == task['role'] and
                    any(covers(old_read, read) and covers(read, old_read)
                        for old_read in earlier['reads'])
                    for earlier in tasks[:index])
                expected = ('tile-dma-mode-3-reread' if reread else
                            'tile-dma-mode-7')
                if task['dependency'] != expected:
                    hazards.append(
                        f"task {index} scratch dependency is "
                        f"{task['dependency']}, expected {expected}")

        for read in task['reads']:
            for write in task['writes']:
                if overlaps(read, write, 'l2'):
                    hazards.append(
                        f"task {index} reuses one L2 staging range for a read and write")
                if read['storage'] != 'external' and covers(write, read):
                    hazards.append(
                        f"task {index} overwrites a live source surface")

        if task['role'] == 'add+relu' and len(task['reads']) != 2:
            hazards.append(f"task {index} residual add does not have two inputs")
        rows.append('| ' + ' | '.join([
            str(index), task['layer'], task['role'],
            '<br>'.join(fmt(item) for item in task['reads']),
            '<br>'.join(fmt(item) for item in task['writes']),
            task['dependency'], ', '.join(producers),
        ]) + ' |')
    return rows, hazards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest['tasks']
    rows, hazards = audit(tasks)
    lines = [
        '# H13 residual task access audit', '',
        '| # | layer | role | reads | writes | dependency | producer(s) |',
        '|---:|---|---|---|---|---|---|', *rows, '',
        '## Findings', '',
    ]
    if hazards:
        lines.extend(f'- HAZARD: {hazard}' for hazard in hazards)
    else:
        lines += [
            '- Every non-external read has an earlier covering producer.',
            '- No task overlaps an L2 read staging range with its write range.',
            '- No task overwrites a live input surface; the residual add reads two initialized branches into a distinct result.',
        ]
    text = '\n'.join(lines) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    if hazards:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

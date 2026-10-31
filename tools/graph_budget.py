#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dispatch-only lower bound for a collection of compiler packages.

This is an admission check under an explicit overhead assumption, never a
latency measurement. A successful bound still needs full-model device timing.
"""
import argparse
import json
import math
from pathlib import Path


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def dispatch_count(manifest):
    if manifest.get('target') not in ('H13', 'H14') or manifest.get('schema') != (
            f"mil-hwxc.{manifest['target'].lower()}-anec-package.v2"):
        raise ValueError('budget.package-schema: expected H13/H14 v2 package')
    programs, dispatch = manifest.get('programs'), manifest.get('dispatchPlan')
    if not isinstance(programs, list) or not programs or not isinstance(dispatch, list) or (
            any(type(i) is not int for i in dispatch) or
            sorted(dispatch) != list(range(len(programs)))):
        raise ValueError('budget.dispatch-plan: every program must be dispatched exactly once')
    return len(dispatch)


def estimate(manifests, overhead_ms, budget_ms=10.0, reserve_ms=0.0, complete=False):
    if not all(math.isfinite(v) for v in (overhead_ms, budget_ms, reserve_ms)) or (
            overhead_ms <= 0 or budget_ms <= 0 or reserve_ms < 0):
        raise ValueError('budget.parameters: finite positive overhead/budget and nonnegative reserve required')
    if not manifests:
        raise ValueError('budget.empty-model: at least one package is required')
    targets = {m.get('target') for m in manifests}
    if len(targets) != 1:
        raise ValueError('budget.mixed-targets: packages must target one generation')
    dispatches = sum(dispatch_count(m) for m in manifests)
    lower_bound = dispatches * overhead_ms + reserve_ms
    return {'schema': 'mil-hwxc.dispatch-budget.v1', 'target': manifests[0]['target'],
            'packages': len(manifests), 'dispatches': dispatches,
            'assumed_dispatch_overhead_ms': overhead_ms, 'reserved_ms': reserve_ms,
            'budget_ms': budget_ms, 'lower_bound_ms': lower_bound,
            'remaining_ms': budget_ms - lower_bound,
            'lower_bound_fits': lower_bound < budget_ms,
            'complete_model_coverage': bool(complete),
            'admissible_under_assumption': bool(complete) and lower_bound < budget_ms,
            'device_budget_proven': False,
            'excludes': ['device computation unless reserved', 'packing and synchronization unless reserved',
                         'DSP unless reserved', 'scheduler jitter', 'uncompiled model regions']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, action='append', required=True,
                   help='repeat for every package executed once per hop')
    p.add_argument('--overhead-ms', type=positive, required=True)
    p.add_argument('--budget-ms', type=positive, default=10.0)
    p.add_argument('--reserve-ms', type=float, default=0.0)
    p.add_argument('--coverage', type=Path,
                   help='optional complete model coverage receipt; fragments never imply completeness')
    p.add_argument('--require-fit', action='store_true')
    a = p.parse_args()
    try:
        manifests = [json.loads((path / 'manifest.json').read_text()) for path in a.package]
        # Only a receipt explicitly establishing all three contracts can mark
        # completeness. Current fragment audits deliberately set these false.
        coverage = json.loads(a.coverage.read_text()) if a.coverage else {}
        complete = all(coverage.get(field) is True for field in (
            'complete_native_graph', 'streaming_state_complete', 'full_model_package_validated'))
        receipt = estimate(manifests, a.overhead_ms, a.budget_ms, a.reserve_ms, complete)
    except (ValueError, OSError, TypeError, KeyError) as error:
        p.error(str(error))
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0 if not a.require_fit or receipt['admissible_under_assumption'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

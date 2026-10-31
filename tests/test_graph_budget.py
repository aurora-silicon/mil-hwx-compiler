#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from graph_budget import dispatch_count, estimate


def package(count, target='H14'):
    return {'schema': f'mil-hwxc.{target.lower()}-anec-package.v2', 'target': target,
            'programs': [{} for _ in range(count)], 'dispatchPlan': list(range(count))}


class BudgetTest(unittest.TestCase):
    def test_recurrent_bounds(self):
        # Three ReLU-candidate GRUs lose one dispatch each; five tanh GRUs do not.
        rnnoise = estimate([package(23)] * 3, .25)
        self.assertEqual(rnnoise['dispatches'], 69)
        self.assertEqual(rnnoise['lower_bound_ms'], 17.25)
        self.assertFalse(rnnoise['admissible_under_assumption'])
        self.assertEqual(estimate([package(24)] * 5, .25)['lower_bound_ms'], 30.0)
        h13 = estimate([package(24, 'H13')] * 5, .04)
        self.assertAlmostEqual(h13['lower_bound_ms'], 4.8)
        self.assertTrue(h13['lower_bound_fits'])
        self.assertFalse(h13['admissible_under_assumption'])
        self.assertFalse(h13['device_budget_proven'])

    def test_deadline_and_reserve(self):
        self.assertFalse(estimate([package(40)], .25, complete=True)['lower_bound_fits'])
        self.assertFalse(estimate([package(36)], .25, reserve_ms=1.0, complete=True)['lower_bound_fits'])
        self.assertTrue(estimate([package(35)], .25, reserve_ms=1.0, complete=True)['admissible_under_assumption'])

    def test_invalid_contracts(self):
        for dispatch in ([True], [0, 0], [1], [], [0, 1.0]):
            with self.assertRaises(ValueError):
                dispatch_count({**package(1), 'dispatchPlan': dispatch})
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError): estimate([package(1)], value)
        with self.assertRaises(ValueError): estimate([], .25)
        with self.assertRaises(ValueError): estimate([package(1), package(1, 'H13')], .25)
        with self.assertRaises(ValueError): estimate([package(1)], .25, reserve_ms=-1)
        with self.assertRaises(ValueError): dispatch_count({**package(1), 'schema': 'legacy'})


if __name__ == '__main__':
    unittest.main()

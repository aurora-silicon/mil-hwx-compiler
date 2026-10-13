"""The one definition of which H14 unary programs this receipt builds, runs and judges.

A case is an oracle record research/oracles/h14/unary_<op>_c<C>.json that holds Apple-compiled output (error null). relu is left out:
it already ran on a device. build-unary.py builds exactly these cases, and compare-unary.py requires exactly these cases, so the two
cannot drift apart."""
import json
import re
from pathlib import Path

CASE_NAME = re.compile(r"unary_([a-z_]+?)_c(\d+)")


def expected_cases(oracle_dir):
    """Return the sorted case names (for example unary_silu_c512) for the oracle records in oracle_dir."""
    cases = []
    for oracle in sorted(Path(oracle_dir).glob("unary_*_c*.json")):
        m = CASE_NAME.fullmatch(oracle.stem)
        if m and m.group(1) != "relu" and json.loads(oracle.read_text()).get("error") is None:
            cases.append(oracle.stem)
    return cases

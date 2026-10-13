#!/usr/bin/env python3
"""Tests of compare-unary.py's exit status, on made-up outputs and made-up oracle lists. No device.

Usage: test-compare-unary.py [path-to-compare-unary.py]
Part 1 runs the checks against the script. Part 2 builds copies of the script with one known fault each and requires the checks to catch
every copy. A gate whose tests cannot fail is not a gate.
The gate requires exactly the case list of unary_cases.expected_cases, so each check writes a small oracle directory of its own and passes
it with --oracles."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "compare-unary.py"
STRIDE = 32
COUNT = 64


def mil(op, params=""):
    arg = f", {params}" if params else ""
    return (f"program(1.3)\n[buildInfo = dict<string, string>({{}})]\n{{\n  func main<ios18>(tensor<fp16, [1, {COUNT}, 1, 1]> x) {{\n"
            f"    tensor<fp16, [1, {COUNT}, 1, 1]> y = {op}(x = x{arg})[name = string(\"y\")];\n  }} -> (y);\n}}\n")


def surface(values, tiles=1):
    s = np.zeros(tiles * 16384 // 2, dtype=np.float16)
    s[::STRIDE][:len(values)] = values
    return s


def run(script, set_dir, out_dir, oracle_dir):
    cmd = [sys.executable, str(script), str(set_dir), str(out_dir), "--oracles", str(oracle_dir)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


class Fixture:
    """Temporary directories: oracle lists, case sets and output sets."""

    def __init__(self, root):
        self.root = root

    def oracles(self, name, cases, error_cases=()):
        d = self.root / name
        d.mkdir()
        for case in cases:
            (d / f"{case}.json").write_text(json.dumps({"error": None, "mil": "x"}))
        for case in error_cases:
            (d / f"{case}.json").write_text(json.dumps({"error": "callback_status=1", "mil": "x"}))
        return d

    def case_set(self, name, cases):
        """cases: {case_name: (x_values, mil_text)}"""
        d = self.root / name
        for case, (x, text) in cases.items():
            (d / case).mkdir(parents=True)
            surface(x).tofile(d / case / "x.f16")
            (d / case / "op.mil").write_text(text)
        d.mkdir(exist_ok=True)
        return d

    def outputs(self, name, per_case):
        """per_case: {case_name: (cal_surface, long_surface)}"""
        d = self.root / name
        d.mkdir()
        for case, (cal, long) in per_case.items():
            cal.tofile(d / f"{case}.cal.y.f16")
            long.tofile(d / f"{case}.long.y.f16")
        return d


def checks(script):
    """Return a list of (name, passed) for one script."""
    out = []

    def check(name, ok):
        out.append((name, bool(ok)))

    with tempfile.TemporaryDirectory() as tmp:
        f = Fixture(Path(tmp))
        rng = np.random.default_rng(3)
        x = rng.uniform(-8, 8, COUNT).astype(np.float16)
        xd = x.astype(np.float64)
        silu = surface((xd / (1 + np.exp(-xd))).astype(np.float16))
        tanh = surface(np.tanh(xd).astype(np.float16))
        S, T = "unary_silu_c64", "unary_tanh_c64"
        oracle_s = f.oracles("or_s", [S])
        set_s = f.case_set("set_s", {S: (x, mil("silu"))})

        r = run(script, set_s, f.outputs("good", {S: (silu, silu)}), oracle_s)
        check("a valid output exits 0 and prints PASS", r.returncode == 0 and "PASS" in r.stdout)
        check("a wrong output (all zeros) exits nonzero", run(script, set_s, f.outputs("zeros", {S: (np.zeros_like(silu),) * 2}), oracle_s).returncode != 0)
        r = run(script, set_s, f.outputs("none", {}), oracle_s)
        check("a missing output exits nonzero", r.returncode != 0 and "MISSING" in r.stdout)
        d = f.root / "half"
        d.mkdir()
        silu.tofile(d / f"{S}.cal.y.f16")
        check("a missing long-loop output exits nonzero", run(script, set_s, d, oracle_s).returncode != 0)
        padded = silu.copy()
        padded[COUNT * STRIDE + 7] = np.float16(1.0)
        check("a nonzero padding value exits nonzero", run(script, set_s, f.outputs("padded", {S: (padded, padded)}), oracle_s).returncode != 0)
        nan = silu.copy()
        nan[5 * STRIDE] = np.float16("nan")
        check("a NaN lane exits nonzero", run(script, set_s, f.outputs("nan", {S: (nan, nan)}), oracle_s).returncode != 0)
        off = silu.copy()
        off[9 * STRIDE] = np.float16(float(off[9 * STRIDE]) + 0.5)
        r = run(script, set_s, f.outputs("off", {S: (off, off)}), oracle_s)
        check("one lane outside the bound exits nonzero", r.returncode != 0 and "outside the bound" in r.stdout)
        check("calibration and long-loop outputs that differ exit nonzero", run(script, set_s, f.outputs("differ", {S: (silu, off)}), oracle_s).returncode != 0)
        r = run(script, set_s, f.root / "good", oracle_s)
        check("the input surface's sha256 prefix is printed", "input sha256" in r.stdout)

        # the expected case list
        oracle_empty = f.oracles("or_empty", [], error_cases=["unary_clip_c64", "unary_log_c64"])
        set_none = f.case_set("set_none", {})  # no case directories either, so only the empty expected list can decide
        check("an expected list with no cases exits nonzero", run(script, set_none, f.root / "none", oracle_empty).returncode != 0)
        oracle_two = f.oracles("or_two", [S, T])
        r = run(script, set_s, f.root / "good", oracle_two)
        check("one valid case when two are expected exits nonzero", r.returncode != 0 and "MISSING case directory" in r.stdout)
        set_two = f.case_set("set_two", {S: (x, mil("silu")), T: (x, mil("tanh"))})
        r = run(script, set_two, f.outputs("good_two", {S: (silu, silu), T: (tanh, tanh)}), oracle_s)
        check("an unexpected extra case directory exits nonzero", r.returncode != 0 and "UNEXPECTED" in r.stdout)
        r = run(script, set_two, f.root / "good_two", oracle_two)
        check("both expected cases valid exits 0", r.returncode == 0)
        oracle_err = f.oracles("or_err", [S], error_cases=["unary_clip_c64"])
        check("an oracle record with an error is not expected (clip and log are out of scope)", run(script, set_s, f.root / "good", oracle_err).returncode == 0)
        oracle_frob = f.oracles("or_frob", ["unary_frob_c64"])
        set_frob = f.case_set("set_frob", {"unary_frob_c64": (x, mil("frob"))})
        r = run(script, set_frob, f.root / "good", oracle_frob)
        check("an unknown operation exits nonzero", r.returncode != 0 and "UNKNOWN" in r.stdout)

        # abs is bit-exact: -0.0 where +0.0 is correct differs in bits and must fail
        A = "unary_abs_c64"
        ax = rng.uniform(-8, 8, COUNT).astype(np.float16)
        ax[0] = np.float16(0.0)
        abs_good = surface(np.abs(ax.astype(np.float64)).astype(np.float16))
        negzero = abs_good.copy()
        negzero[0] = np.float16(-0.0)
        oracle_a = f.oracles("or_a", [A])
        set_a = f.case_set("set_a", {A: (ax, mil("abs"))})
        check("a correct abs output exits 0", run(script, set_a, f.outputs("abs_good", {A: (abs_good, abs_good)}), oracle_a).returncode == 0)
        check("an abs output with -0.0 for +0.0 exits nonzero", run(script, set_a, f.outputs("abs_neg", {A: (negzero, negzero)}), oracle_a).returncode != 0)

        # parameters come from the MIL: the rsqrt epsilon and the leaky_relu slope
        R = "unary_rsqrt_c64"
        rx = rng.uniform(0.0625, 64, COUNT).astype(np.float16)
        rxd = rx.astype(np.float64)
        eps = 0.5  # large on purpose: a gate that ignores it gives a clearly different reference
        with_eps = surface((1.0 / np.sqrt(rxd + eps)).astype(np.float16))
        no_eps = surface((1.0 / np.sqrt(rxd)).astype(np.float16))
        oracle_r = f.oracles("or_r", [R])
        set_r = f.case_set("set_r", {R: (rx, mil("rsqrt", f"epsilon = fp32({eps})"))})
        check("rsqrt output computed with the MIL epsilon exits 0", run(script, set_r, f.outputs("r_eps", {R: (with_eps, with_eps)}), oracle_r).returncode == 0)
        check("rsqrt output computed without the MIL epsilon exits nonzero", run(script, set_r, f.outputs("r_noeps", {R: (no_eps, no_eps)}), oracle_r).returncode != 0)
        L = "unary_leaky_relu_c64"
        lx = rng.uniform(-8, 8, COUNT).astype(np.float16)
        lxd = lx.astype(np.float64)
        slope_half = surface(np.where(lxd >= 0, lxd, 0.5 * lxd).astype(np.float16))
        slope_eighth = surface(np.where(lxd >= 0, lxd, 0.125 * lxd).astype(np.float16))
        oracle_l = f.oracles("or_l", [L])
        set_l = f.case_set("set_l", {L: (lx, mil("leaky_relu", "alpha = fp32(0.5)"))})
        check("leaky_relu output with the MIL slope exits 0", run(script, set_l, f.outputs("l_ok", {L: (slope_half, slope_half)}), oracle_l).returncode == 0)
        check("leaky_relu output with a different slope exits nonzero", run(script, set_l, f.outputs("l_bad", {L: (slope_eighth, slope_eighth)}), oracle_l).returncode != 0)
        # gelu's MIL carries mode = string("EXACT"); the reference is the erf form, so only EXACT is accepted
        G = "unary_gelu_c64"
        gelu_out = surface((0.5 * xd * (1.0 + np.vectorize(__import__("math").erf)(xd / 2 ** 0.5))).astype(np.float16))
        oracle_g = f.oracles("or_g", [G])
        set_g = f.case_set("set_g", {G: (x, mil("gelu", 'mode = string("EXACT")'))})
        check("gelu with mode EXACT and a correct output exits 0", run(script, set_g, f.outputs("g_ok", {G: (gelu_out, gelu_out)}), oracle_g).returncode == 0)
        set_g2 = f.case_set("set_g2", {G: (x, mil("gelu", 'mode = string("TANH_APPROXIMATION")'))})
        check("gelu with a mode other than EXACT exits nonzero", run(script, set_g2, f.root / "g_ok", oracle_g).returncode != 0)
        set_s2 = f.case_set("set_s2", {S: (x, mil("silu", 'mode = string("EXACT")'))})
        check("a string argument on an operation that takes none exits nonzero", run(script, set_s2, f.root / "good", oracle_s).returncode != 0)
        set_unknown = f.case_set("set_unk", {S: (x, mil("silu", "bogus = fp32(1.0)"))})
        check("a MIL call with an unknown parameter exits nonzero", run(script, set_unknown, f.root / "good", oracle_s).returncode != 0)
        set_noparam = f.case_set("set_np", {R: (rx, mil("rsqrt"))})
        check("a MIL call missing a required parameter exits nonzero", run(script, set_noparam, f.root / "r_eps", oracle_r).returncode != 0)
    return out


MUTANTS = {
    "failures do not count": ("return 1 if (failed or not expected) else 0", "return 0"),
    "an empty expected list passes": ("return 1 if (failed or not expected) else 0", "return 1 if failed else 0"),
    "missing output is skipped": ('return False, f"{case_dir.name}: MISSING output file(s) {\', \'.join(gone)}"', 'return True, "skipped"'),
    "padding is ignored": ("    pad = padding_nonzero(y_cal, count)\n", "    pad = 0\n"),
    "the numerical bound is ignored": ("        out_of_band = int((err > tol).sum())\n", "        out_of_band = 0\n"),
    "the long-loop output is never compared with the calibration output":
        ("    if y_cal.tobytes() != y_long.tobytes():\n", "    if False:\n"),
    "missing case directories are ignored":
        ('        print(f"{name}: MISSING case directory in SETDIR")\n        failed += 1\n', '        print(f"{name}: MISSING case directory in SETDIR")\n'),
    "unexpected case directories are ignored":
        ('        print(f"{name}: UNEXPECTED case directory (not in the expected case list)")\n        failed += 1\n',
         '        print(f"{name}: UNEXPECTED case directory (not in the expected case list)")\n'),
    "abs is compared numerically, not bitwise": ('"abs": ("bits", 0.0),', '"abs": ("abs", 0.0),'),
    "the rsqrt epsilon is ignored": ('return 1.0 / np.sqrt(x + params["epsilon"])', "return 1.0 / np.sqrt(x)"),
    "the leaky_relu slope is fixed at 0.125": ('params["alpha"] * x)', "0.125 * x)"),
    "unknown MIL parameters are accepted": ('    if set(params) != REQUIRED_PARAMS.get(op, set()):\n', "    if False:\n"),
    "any gelu mode is accepted": ("    if strings != ALLOWED_STRINGS.get(op, {}):\n", "    if False:\n"),
}


def main():
    bad = 0
    print(f"== checks against {SCRIPT.name}")
    for name, ok in checks(SCRIPT):
        print(("PASS " if ok else "FAIL ") + name)
        bad += not ok
    print("== mutants: each broken copy of the script must be caught by at least one check")
    text = SCRIPT.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "unary_cases.py").write_text((HERE / "unary_cases.py").read_text())
        for label, (old, new) in MUTANTS.items():
            if text.count(old) != 1:
                print(f"FAIL mutant '{label}': the text to break was not found exactly once ({text.count(old)} times)")
                bad += 1
                continue
            path = Path(tmp) / "mutant.py"
            path.write_text(text.replace(old, new, 1))
            caught = [n for n, ok in checks(path) if not ok]
            print(("PASS " if caught else "FAIL ") + f"mutant '{label}' " + (f"caught by: {caught[0]}" if caught else "was NOT caught"))
            bad += not caught
    print(f"failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

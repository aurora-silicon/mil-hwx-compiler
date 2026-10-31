# H13 L2-residency round-2 decoded oracles

These records contain decoded task-descriptor fields and hashes only. The
Apple-generated HWX objects existed only in temporary directories and were
destroyed after decoding. No Apple object bytes are retained here.

`research/mint_h13_l2_residency_r2.py` reproduces the campaign on a macOS
host with a locally licensed compiler tool. All 16 records were decoded with
compiler-tool SHA-256
`9c843c1b9323096f333573f6f915f64f80b2d3522bbdd16413a1b08d7f7f9183`
from public compiler source commit
`e03ca24a025b1020e2e6abf6ee2542f7864f9329`.

The campaign covers:

- exact MobileNet-style `dw -> pw -> dw` triplets at 56, 28, 14 and 7;
- forced 2-, 4- and 8-task pointwise consumers with an L2 input;
- six-layer alternating chains at 28, 14 and 7;
- a single-task 7x7 `pw -> dw -> pw` triplet; and
- the 56x56 c160-resident / c176-spill boundary, plus surrounding points.

Run `python3 research/verify_h13_l2_residency_r2.py` to re-derive one
four-task consumer and all three six-layer chains from the rules and compare
707 header, Common, L2 and TileDMA fields.

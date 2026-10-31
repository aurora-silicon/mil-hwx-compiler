# H13 conv device bench

Compiles single conv layers with `mil-hwxc`, runs them on an M1-family ANE
through libane, and compares the output with a float64 CPU reference rounded
to fp16 with ties away from zero, as the hardware rounds.

On the build host (needs numpy):

```sh
python3 tools/h13-conv-bench/mkcase.py cases pw128:128,56,56,128,1,1,1,1,1 dw128:128,56,56,128,3,1,128,1,1
```

Each spec is `name:Cin,H,W,Cout,k,stride,groups,bias,relu[,same|valid]`.
`cases/NAME/pkg` holds the package and `case.npz` (weights, bias, geometry).

On the M1 (libane from omarchy-ane, user in the `accel` group):

```sh
gcc -O2 -I$LIBANE -o tools/h13-conv-bench/anebench tools/h13-conv-bench/anebench.c $LIBANE/libane.a -ldrm
python3 tools/h13-conv-bench/run_layer.py cases/pw128/pkg 300
```

`run_layer.py` prints one JSON line: exact and within-1-ulp percentages, max ulp,
relative L2 error, p50 wall time per `ane_exec`, engine busy time per exec from
`ane_stats`, and an fp32 numpy time for the same layer.

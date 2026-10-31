// SPDX-License-Identifier: MIT
// Run an open-compiled H14 ANEC through libane's ABI-2 section/port API.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ane.h"
#include "ane_m2.h"

static uint32_t le32(const unsigned char *p) {
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static double us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s program.anec input.bin output.bin [repeat=1] [device=0]\n", argv[0]);
        return 64;
    }
    char *end;
    unsigned long repeat = argc > 4 ? strtoul(argv[4], &end, 10) : 1;
    if (!repeat || repeat > 100000 || (argc > 4 && *end)) return 64;
    unsigned long device = argc > 5 ? strtoul(argv[5], &end, 10) : 0;
    if (device > 63 || (argc > 5 && *end)) return 64;
    uint32_t abi = 0, family = 0;
    if (ane_query_caps((int)device, &abi, &family) || abi != 2) {
        fprintf(stderr, "requires an ABI-2 ANE device; abi=%u family=%u\n", abi, family);
        return 1;
    }
    unsigned char header[0x1000];
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(header, 1, sizeof(header), f) != sizeof(header)) {
        if (f) fclose(f);
        fprintf(stderr, "cannot read ANEC header\n"); return 1;
    }
    fclose(f);
    if (le32(header + 0x20) != 1 || le32(header + 0x24) != 1) return 1;
    const uint64_t in = (uint64_t)le32(header + 0x28 + 5 * 4) << 14;
    const uint64_t out = (uint64_t)le32(header + 0x28 + 4 * 4) << 14;
    const uint64_t scratch = (uint64_t)le32(header + 0x28 + 3 * 4) << 14;
    if (!in || !out || in > SIZE_MAX || out > SIZE_MAX) return 1;
    struct ane_m2_port_spec ports[] = {
        {"input", 0, 5, 4, in}, {"output", 1, 4, 5, out},
        {"scratch", 2, 0x40, 3, scratch}
    };
    unsigned char *x = calloc(1, (size_t)in), *y = calloc(1, (size_t)out);
    if (!x || !y) { free(x); free(y); return 1; }
    f = fopen(argv[2], "rb");
    if (!f || fread(x, 1, (size_t)in, f) != in || fgetc(f) != EOF) {
        if (f) fclose(f);
        free(x); free(y);
        fprintf(stderr, "input must equal the allocated input surface size\n"); return 1;
    }
    fclose(f);
    // Explicit ports bind BARs consistently across all tasks; no inferred
    // cross-task scratch sizing, op-ref overrides, or firmware byte replay.
    struct ane_nn *nn = ane_m2_init_ports(argv[1], ports, scratch ? 3 : 2, (int)device);
    if (!nn) { free(x); free(y); return 1; }
    int rc = ane_m2_send(nn, x, 0);
    double start = us();
    for (unsigned long i = 0; !rc && i < repeat; ++i) rc = ane_exec(nn);
    double elapsed = us() - start;
    if (!rc) rc = ane_m2_read(nn, y, 0);
    if (!rc) {
        f = fopen(argv[3], "wb");
        if (!f) rc = -EIO;
        else { if (fwrite(y, 1, (size_t)out, f) != out) rc = -EIO;
               if (fclose(f)) rc = -EIO; }
    }
    ane_free(nn); free(x); free(y);
    if (rc) { fprintf(stderr, "ABI-2 run failed: %d\n", rc); return 1; }
    printf("{\"abi\":2,\"family\":%u,\"device\":%lu,\"repeat\":%lu,"
           "\"warmup\":0,\"exec_us_per_pass\":%.3f}\n", family, device, repeat, elapsed / repeat);
    return 0;
}

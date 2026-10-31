// anebench: run a sequence of H13 ANEC programs through libane with host
// copies between them, time each exec, and dump surfaces.
// usage: anebench -r REPEAT [-i idx:file]... [-o idx:file]... prog.anec
//    or: anebench -r REPEAT -s steps.txt   (each line: anec in0file[,in1file] outfile)
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ane.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmpd(const void *a, const void *b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
static void *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n ? *n : 1); if (fread(b, 1, *n, f) != *n) { perror("read"); exit(2); } fclose(f); return b;
}
static unsigned long long busy_ns(void) {
    FILE *f = fopen("/sys/class/accel/accel0/device/ane_stats", "r"); char k[64]; unsigned long long v, r = 0;
    if (!f) return 0; while (fscanf(f, "%63s %llu", k, &v) == 2) if (!strcmp(k, "busy_ns")) r = v; fclose(f); return r;
}
int main(int argc, char **argv) {
    int repeat = 100; const char *ins[8] = {0}; const char *outs[8] = {0}; const char *prog = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-r")) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i")) { char *s = argv[++i]; ins[atoi(s)] = strchr(s, ':') + 1; }
        else if (!strcmp(argv[i], "-o")) { char *s = argv[++i]; outs[atoi(s)] = strchr(s, ':') + 1; }
        else prog = argv[i];
    }
    struct ane_nn *nn = ane_init(prog);
    if (!nn) { fprintf(stderr, "ane_init failed %s\n", prog); return 1; }
    uint32_t ns = ane_src_count(nn), nd = ane_dst_count(nn);
    for (uint32_t s = 0; s < ns; s++) {
        uint64_t sz = __ane_src_size(nn, s); void *b = calloc(1, sz);
        if (ins[s]) { size_t n; void *d = slurp(ins[s], &n); memcpy(b, d, n < sz ? n : sz); if (n != sz) fprintf(stderr, "note: src%u file %zu bytes, surface %llu\n", s, n, (unsigned long long)sz); free(d); }
        __ane_send(nn, b, s); free(b);
    }
    int r = ane_exec(nn); if (r < 0) { fprintf(stderr, "ane_exec failed %d\n", r); return 1; }
    double *t = calloc(repeat, sizeof(double));
    unsigned long long b0 = busy_ns();
    double w0 = now_us();
    for (int i = 0; i < repeat; i++) { double a = now_us(); r = ane_exec(nn); t[i] = now_us() - a; if (r < 0) { fprintf(stderr, "exec %d failed %d\n", i, r); return 1; } }
    double wall = now_us() - w0;
    unsigned long long b1 = busy_ns();
    qsort(t, repeat, sizeof(double), cmpd);
    for (uint32_t d = 0; d < nd; d++) {
        uint64_t sz = __ane_dst_size(nn, d); void *b = malloc(sz); __ane_read(nn, b, d);
        if (outs[d]) { FILE *f = fopen(outs[d], "wb"); fwrite(b, 1, sz, f); fclose(f); }
        free(b);
    }
    printf("{\"prog\":\"%s\",\"repeat\":%d,\"min_us\":%.1f,\"p50_us\":%.1f,\"p99_us\":%.1f,\"mean_wall_us\":%.1f,\"busy_us_per_exec\":%.1f}\n",
        prog, repeat, t[0], t[repeat/2], t[(int)(repeat*0.99)], wall/repeat, (b1-b0)/1e3/repeat);
    ane_free(nn); return 0;
}

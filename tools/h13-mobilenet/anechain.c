// anechain: run one H13 program, or a sequence of programs with a host copy of
// each output surface into the next program's input, and time whole passes.
//
// usage: anechain [-r REPEAT] [-w WARMUP] [-n] [-i input.bin] [-o output.bin] prog.anec [prog.anec ...]
//
// -n times ane_exec alone for a single program (input sent once, output read
// once after the loop); without it every pass includes the input and output copies.
//
// With one program a pass is one ane_exec: a linked program does the whole
// chain in one submit. With several, a pass runs them in order and copies
// each raw output surface into the next program's raw input surface, which is
// what a per-layer runtime has to do. Prints one JSON line.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ane.h"

#define MAX_PROGRAMS 64

static double now_us(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static void *slurp(const char *path, size_t *n)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); exit(2); }
	fseek(f, 0, SEEK_END);
	*n = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *b = malloc(*n ? *n : 1);
	if (fread(b, 1, *n, f) != *n) { perror("read"); exit(2); }
	fclose(f);
	return b;
}

static unsigned long long busy_ns(void)
{
	FILE *f = fopen("/sys/class/accel/accel0/device/ane_stats", "r");
	char k[64];
	unsigned long long v, r = 0;
	if (!f) return 0;
	while (fscanf(f, "%63s %llu", k, &v) == 2)
		if (!strcmp(k, "busy_ns")) r = v;
	fclose(f);
	return r;
}

int main(int argc, char **argv)
{
	int repeat = 200, warmup = 20, count = 0, exec_only = 0;
	const char *in = NULL, *out = NULL, *progs[MAX_PROGRAMS];
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-r")) repeat = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-w")) warmup = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-i")) in = argv[++i];
		else if (!strcmp(argv[i], "-o")) out = argv[++i];
		else if (!strcmp(argv[i], "-n")) exec_only = 1;
		else if (count < MAX_PROGRAMS) progs[count++] = argv[i];
	}
	if (!count || repeat < 1) {
		fprintf(stderr, "usage: anechain [-r N] [-w N] [-i in] [-o out] prog.anec...\n");
		return 64;
	}
	struct ane_nn *nn[MAX_PROGRAMS];
	void *bounce[MAX_PROGRAMS];
	for (int p = 0; p < count; p++) {
		nn[p] = ane_init(progs[p]);
		if (!nn[p]) { fprintf(stderr, "ane_init failed %s\n", progs[p]); return 1; }
		if (ane_src_count(nn[p]) != 1 || ane_dst_count(nn[p]) != 1) {
			fprintf(stderr, "%s: needs one input and one output\n", progs[p]);
			return 1;
		}
		bounce[p] = calloc(1, __ane_dst_size(nn[p], 0));
		if (p && __ane_src_size(nn[p], 0) < __ane_dst_size(nn[p - 1], 0)) {
			fprintf(stderr, "%s: input smaller than the previous output\n", progs[p]);
			return 1;
		}
	}
	uint64_t insz = __ane_src_size(nn[0], 0);
	void *x = calloc(1, insz);
	if (in) {
		size_t n;
		void *d = slurp(in, &n);
		memcpy(x, d, n < insz ? n : insz);
		free(d);
	}
	double *t = calloc(repeat, sizeof(double));
	unsigned long long b0 = 0;
	double w0 = 0;
	for (int i = -warmup; i < repeat; i++) {
		if (i == 0) { b0 = busy_ns(); w0 = now_us(); }
		double a = now_us();
		if (!exec_only || i == -warmup) __ane_send(nn[0], x, 0);
		for (int p = 0; p < count; p++) {
			int r = ane_exec(nn[p]);
			if (r < 0) { fprintf(stderr, "exec %d prog %d failed %d\n", i, p, r); return 1; }
			if (exec_only && count == 1) continue;
			__ane_read(nn[p], bounce[p], 0);
			if (p + 1 < count) __ane_send(nn[p + 1], bounce[p], 0);
		}
		if (i >= 0) t[i] = now_us() - a;
	}
	double wall = now_us() - w0;
	unsigned long long b1 = busy_ns();
	if (exec_only && count == 1) __ane_read(nn[0], bounce[0], 0);
	if (out) {
		FILE *f = fopen(out, "wb");
		fwrite(bounce[count - 1], 1, __ane_dst_size(nn[count - 1], 0), f);
		fclose(f);
	}
	qsort(t, repeat, sizeof(double), cmpd);
	printf("{\"programs\":%d,\"repeat\":%d,\"min_us\":%.1f,\"p50_us\":%.1f,\"p99_us\":%.1f,"
	       "\"mean_wall_us\":%.1f,\"inf_per_s\":%.1f,\"busy_us_per_pass\":%.1f}\n",
	       count, repeat, t[0], t[repeat / 2], t[(int)(repeat * 0.99)],
	       wall / repeat, 1e6 * repeat / wall, (b1 - b0) / 1e3 / repeat);
	for (int p = 0; p < count; p++) ane_free(nn[p]);
	return 0;
}

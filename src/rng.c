#include "rng.h"
#include <math.h>

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void rng_seed(Rng *r, uint64_t seed) {
    uint64_t s = seed ? seed : 0xCAFEBABE12345678ULL;
    r->s[0] = splitmix64(&s);
    r->s[1] = splitmix64(&s);
}

uint32_t rng_u32(Rng *r) {
    uint64_t s1 = r->s[0];
    uint64_t s0 = r->s[1];
    uint64_t res = s0 + s1;
    r->s[0] = s0;
    s1 ^= s1 << 23;
    r->s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return (uint32_t)(res >> 32);
}

int rng_range(Rng *r, int n) {
    return (int)(rng_u32(r) % (uint32_t)n);
}

float rng_unit(Rng *r) {
    /* 24-bit mantissa precision, strictly in [0, 1). */
    return (float)(rng_u32(r) >> 8) * (1.0f / 16777216.0f);
}

float rng_normal(Rng *r) {
    float u1, u2;
    do { u1 = rng_unit(r); } while (u1 < 1e-7f);
    u2 = rng_unit(r);
    float mag = sqrtf(-2.0f * logf(u1));
    return mag * cosf(2.0f * 3.14159265358979f * u2);
}

#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* Seedable PRNG (xorshift128+) for reproducible training runs.
 * We don't use rand() because it's low-quality and non-portable. */
typedef struct { uint64_t s[2]; } Rng;

void     rng_seed(Rng *r, uint64_t seed);
uint32_t rng_u32(Rng *r);
int      rng_range(Rng *r, int n);  /* uniform in [0, n) */
float    rng_unit(Rng *r);          /* uniform in [0, 1) */
float    rng_normal(Rng *r);        /* standard normal via Box-Muller */

#endif

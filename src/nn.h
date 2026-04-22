#ifndef NN_H
#define NN_H

#include "rng.h"

/* Two-layer MLP: input (nx) -> hidden (nh, ReLU) -> output (ny) -> softmax.
 * Forward caches h, logits, probs inside the struct so that the backward
 * pass for a given step can use them directly. For a multi-step REINFORCE
 * update, the caller re-runs forward on each step's x before accumulating
 * that step's gradient. */
typedef struct {
    int nx, nh, ny;

    /* Parameters */
    float *W1, *b1;  /* W1: [nh * nx] row-major (each row = one hidden unit) */
    float *W2, *b2;  /* W2: [ny * nh] row-major (each row = one output unit) */

    /* Accumulated gradients (same shape as params) */
    float *dW1, *db1, *dW2, *db2;

    /* Forward activation cache (for the most recent forward) */
    float *h;        /* [nh] post-ReLU hidden */
    float *logits;   /* [ny] */
    float *probs;    /* [ny] */

    /* Backward scratch */
    float *dh;
    float *dlogits;
} MLP;

MLP  *mlp_new(int nx, int nh, int ny, Rng *rng);
void  mlp_free(MLP *m);

void  mlp_forward(MLP *m, const float *x);
int   mlp_sample_action(const MLP *m, Rng *rng);

void  mlp_zero_grad(MLP *m);
/* Accumulates policy-gradient contribution for one (x, action, advantage)
 * tuple. Uses m->h and m->probs as cached by the most recent mlp_forward(x). */
void  mlp_accum_policy_grad(MLP *m, const float *x, int action, float advantage);
void  mlp_apply_sgd(MLP *m, float lr);

#endif

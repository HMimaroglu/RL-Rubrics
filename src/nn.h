#ifndef NN_H
#define NN_H

#include "rng.h"

/* Two-layer MLP policy network:
 *   x -> W1 -> ReLU -> W2 -> softmax -> action distribution
 * Forward caches h, logits, probs inside the struct so the backward pass
 * can reuse them. */
typedef struct {
    int nx, nh, ny;

    /* Parameters */
    float *W1, *b1;  /* W1: [nh * nx] row-major */
    float *W2, *b2;  /* W2: [ny * nh] row-major */

    /* Accumulated gradients */
    float *dW1, *db1;
    float *dW2, *db2;

    /* Forward activation cache */
    float *h;        /* [nh] post-ReLU hidden */
    float *logits;   /* [ny] */
    float *probs;    /* [ny] */

    /* Backward scratch */
    float *dh;
    float *dlogits;
} MLP;

MLP  *mlp_new(int nx, int nh, int ny, Rng *rng);
void  mlp_free(MLP *m);

/* Forward pass with optional face mask for action masking.
 * forbidden_face in {0..5} blocks all 3 action variants on that face by
 * setting their logits to -infinity before softmax. Pass -1 for no mask.
 * Fills m->h, m->logits, m->probs. */
void  mlp_forward(MLP *m, const float *x, int forbidden_face);
int   mlp_sample_action(const MLP *m, Rng *rng);

void  mlp_zero_grad(MLP *m);

/* REINFORCE + entropy regularizer gradient accumulation. Caller passes
 * the h and probs cached from the forward that produced `action` (so
 * the same masked distribution is used for sampling and backward).
 *   d/dz_j [-adv * log pi(a|x)] = adv * (probs_j - 1[j==a])
 *   d/dz_j [-beta * H(pi)]      = beta * probs_j * (log probs_j + H)
 * Pass entropy_beta = 0 to disable the regularizer. */
void  mlp_accum_policy_grad(MLP *m,
                             const float *x,
                             const float *h,
                             const float *probs,
                             int action,
                             float advantage,
                             float entropy_beta);
void  mlp_apply_sgd(MLP *m, float lr);

#endif

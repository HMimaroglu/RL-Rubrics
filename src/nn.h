#ifndef NN_H
#define NN_H

#include "rng.h"

/* Actor-critic MLP with a shared trunk:
 *   x -> W1 -> ReLU (nh) -> { policy head (W2, b2, softmax over ny)
 *                           , value  head (Wv, bv, scalar V(s))      }
 *
 * The policy head uses advantage = G - V(s) as its coefficient, which
 * gives it a per-state baseline and cuts gradient variance hugely vs. a
 * single global baseline. The value head is trained by MSE(V, G) and its
 * gradients flow back through the shared trunk to improve the hidden
 * representation for both heads.
 *
 * Value head weights are initialized to zero so V=0 at start. This means
 * advantage = G - 0 = G initially, and early training behaves exactly
 * like vanilla REINFORCE while V catches up. */
typedef struct {
    int nx, nh, ny;

    /* Trunk */
    float *W1, *b1;
    float *dW1, *db1;

    /* Policy head */
    float *W2, *b2;
    float *dW2, *db2;

    /* Value head */
    float *Wv, *bv;
    float *dWv, *dbv;

    /* Forward activation cache */
    float *h;
    float *logits;
    float *probs;
    float  value;

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

/* Pure REINFORCE + entropy. Used by train_viz (visualization) and any
 * caller that doesn't want to train a value head. */
void  mlp_accum_policy_grad(MLP *m,
                             const float *x,
                             const float *h,
                             const float *probs,
                             int action,
                             float advantage,
                             float entropy_beta);

/* Actor-critic gradient: policy uses advantage = G - value_cached, value
 * head trained with MSE(V, G), entropy regularizer on the policy.
 *   policy:  d/dz_j = (G - V) * (probs_j - 1[j==a]) + beta * p_j (log p_j + H)
 *   value:   d/dV   = 2 * value_coef * (V - G)
 * Both heads' gradients flow into the shared trunk. The advantage is
 * detached: we use V as a numerical baseline only, not as a target the
 * policy loss differentiates through. */
void  mlp_accum_ac_grad(MLP *m,
                         const float *x,
                         const float *h,
                         const float *probs,
                         float value_cached,
                         int action,
                         float G,
                         float entropy_beta,
                         float value_coef);

void  mlp_apply_sgd(MLP *m, float lr);

#endif

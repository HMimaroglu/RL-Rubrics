#include "nn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *zeros(int n) {
    return (float *)calloc((size_t)n, sizeof(float));
}

MLP *mlp_new(int nx, int nh, int ny, Rng *rng) {
    MLP *m = (MLP *)calloc(1, sizeof(MLP));
    m->nx = nx; m->nh = nh; m->ny = ny;

    m->W1 = zeros(nh * nx); m->b1 = zeros(nh);
    m->W2 = zeros(ny * nh); m->b2 = zeros(ny);
    m->dW1 = zeros(nh * nx); m->db1 = zeros(nh);
    m->dW2 = zeros(ny * nh); m->db2 = zeros(ny);
    m->h = zeros(nh);
    m->logits = zeros(ny);
    m->probs  = zeros(ny);
    m->dh = zeros(nh);
    m->dlogits = zeros(ny);

    /* He init for ReLU hidden layer */
    float s1 = sqrtf(2.0f / (float)nx);
    for (int i = 0; i < nh * nx; i++) m->W1[i] = rng_normal(rng) * s1;
    /* Smaller init for policy head so initial policy is near-uniform */
    float s2 = sqrtf(1.0f / (float)nh) * 0.1f;
    for (int i = 0; i < ny * nh; i++) m->W2[i] = rng_normal(rng) * s2;

    return m;
}

void mlp_free(MLP *m) {
    if (!m) return;
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->dW1); free(m->db1); free(m->dW2); free(m->db2);
    free(m->h); free(m->logits); free(m->probs);
    free(m->dh); free(m->dlogits);
    free(m);
}

void mlp_forward(MLP *m, const float *x, int forbidden_face) {
    /* Hidden layer: h = ReLU(W1 x + b1) */
    for (int j = 0; j < m->nh; j++) {
        const float *w = m->W1 + (size_t)j * m->nx;
        float s = m->b1[j];
        for (int i = 0; i < m->nx; i++) s += w[i] * x[i];
        m->h[j] = (s > 0.0f) ? s : 0.0f;
    }
    /* Output layer: logits = W2 h + b2 */
    for (int j = 0; j < m->ny; j++) {
        const float *w = m->W2 + (size_t)j * m->nh;
        float s = m->b2[j];
        for (int i = 0; i < m->nh; i++) s += w[i] * m->h[i];
        m->logits[j] = s;
    }
    /* Action mask: block all 3 variants on forbidden_face by setting their
     * logits far below anything reachable. After softmax the resulting probs
     * are effectively zero and those actions can't be sampled. */
    if (forbidden_face >= 0 && forbidden_face < 6) {
        m->logits[forbidden_face * 3 + 0] = -1e20f;
        m->logits[forbidden_face * 3 + 1] = -1e20f;
        m->logits[forbidden_face * 3 + 2] = -1e20f;
    }
    /* Softmax (subtract max for numerical stability) */
    float mx = m->logits[0];
    for (int j = 1; j < m->ny; j++) if (m->logits[j] > mx) mx = m->logits[j];
    float sum = 0.0f;
    for (int j = 0; j < m->ny; j++) {
        m->probs[j] = expf(m->logits[j] - mx);
        sum += m->probs[j];
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < m->ny; j++) m->probs[j] *= inv;
}

int mlp_sample_action(const MLP *m, Rng *rng) {
    float u = rng_unit(rng);
    float c = 0.0f;
    for (int j = 0; j < m->ny; j++) {
        c += m->probs[j];
        if (u < c) return j;
    }
    return m->ny - 1;
}

void mlp_zero_grad(MLP *m) {
    memset(m->dW1, 0, sizeof(float) * (size_t)m->nh * m->nx);
    memset(m->db1, 0, sizeof(float) * (size_t)m->nh);
    memset(m->dW2, 0, sizeof(float) * (size_t)m->ny * m->nh);
    memset(m->db2, 0, sizeof(float) * (size_t)m->ny);
}

/* Gradient for one step, combining REINFORCE and an optional entropy
 * regularizer. Caller provides cached h (from the forward pass that
 * produced this action) and cached probs (post-mask, post-softmax).
 *
 *   REINFORCE:   d/dz_j = advantage * (probs_j - 1[j == action])
 *   Entropy:     d/dz_j = beta * probs_j * (log probs_j + H)
 *
 * where H = -sum_k probs_k log probs_k (masked actions contribute ~0).
 * Maximizing entropy keeps the softmax from collapsing to one action,
 * which is the usual cause of plateaus under sparse reward.
 */
void mlp_accum_policy_grad(MLP *m,
                            const float *x,
                            const float *h,
                            const float *probs,
                            int action,
                            float advantage,
                            float entropy_beta) {
    for (int j = 0; j < m->ny; j++) {
        m->dlogits[j] = advantage * (probs[j] - (j == action ? 1.0f : 0.0f));
    }
    if (entropy_beta > 0.0f) {
        float H = 0.0f;
        for (int j = 0; j < m->ny; j++) {
            float p = probs[j];
            if (p > 1e-12f) H -= p * logf(p);
        }
        for (int j = 0; j < m->ny; j++) {
            float p = probs[j];
            if (p > 1e-12f) {
                m->dlogits[j] += entropy_beta * p * (logf(p) + H);
            }
        }
    }
    /* Output layer grads */
    for (int j = 0; j < m->ny; j++) m->db2[j] += m->dlogits[j];
    for (int j = 0; j < m->ny; j++) {
        float dz = m->dlogits[j];
        if (dz == 0.0f) continue;
        float *w = m->dW2 + (size_t)j * m->nh;
        for (int i = 0; i < m->nh; i++) w[i] += dz * h[i];
    }
    /* Propagate through output: dh = W2^T dlogits, then ReLU mask */
    for (int i = 0; i < m->nh; i++) {
        float s = 0.0f;
        for (int j = 0; j < m->ny; j++) s += m->W2[(size_t)j * m->nh + i] * m->dlogits[j];
        m->dh[i] = (h[i] > 0.0f) ? s : 0.0f;
    }
    /* Hidden layer grads */
    for (int i = 0; i < m->nh; i++) m->db1[i] += m->dh[i];
    for (int i = 0; i < m->nh; i++) {
        float d = m->dh[i];
        if (d == 0.0f) continue;
        float *w = m->dW1 + (size_t)i * m->nx;
        for (int k = 0; k < m->nx; k++) w[k] += d * x[k];
    }
}

void mlp_apply_sgd(MLP *m, float lr) {
    int n;
    n = m->nh * m->nx; for (int i = 0; i < n; i++) m->W1[i] -= lr * m->dW1[i];
    n = m->nh;         for (int i = 0; i < n; i++) m->b1[i] -= lr * m->db1[i];
    n = m->ny * m->nh; for (int i = 0; i < n; i++) m->W2[i] -= lr * m->dW2[i];
    n = m->ny;         for (int i = 0; i < n; i++) m->b2[i] -= lr * m->db2[i];
}

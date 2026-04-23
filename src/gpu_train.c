/* Batched A2C trainer with GPU forward pass.
 *
 * Runs B (batch) cube episodes in lockstep:
 *   - Encode all B states, run a single batched GPU forward.
 *   - Sample actions from the returned probs (CPU).
 *   - Apply actions to all B cubes (CPU).
 *   - Repeat until all episodes terminate or hit the step budget.
 * Backward + SGD still run on CPU (a single accumulator over all
 * B*T steps per batch).
 *
 * GPU forward is currently the hot spot for sequential training so we
 * get a real speedup here. Backward will move to GPU later.
 */

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gl_loader.h"
#include "cube.h"
#include "nn.h"
#include "gpu_nn.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NX (54 * 6)
#define NH 128
#define NY MV_COUNT

#define LR 0.01f
#define GAMMA 0.99f
#define ENTROPY_BETA 0.05f
#define VALUE_COEF   0.25f
#define WIN_WINDOW        200
#define ADVANCE_THRESHOLD 0.95f
#define MAX_SCRAMBLE      30
#define MAX_EP_STEPS      64

/* BATCH > 1 dispatches multiple parallel cube rollouts per GPU forward,
 * which would saturate the GPU. But naive A2C with sparse reward suffers
 * from severe gradient cancellation when batched: 256 episodes computed
 * from the same frozen weights produce 256 noisy gradient estimates that
 * partially cancel. Net result: GPU runs faster but the policy learns
 * far slower per episode.
 *
 * BATCH=1 mirrors the CPU trainer exactly (correctness verified) and is
 * the safe default. Larger batches need a batched-friendly algorithm
 * (PPO with importance-weighting + multiple epochs, or DQN with experience
 * replay) — the SSBO/compute-shader infrastructure here will plug into
 * those when we add them. */
#define BATCH 1

typedef struct {
    float x[NX];
    float h[NH];
    float probs[NY];
    float value;
    int   action;
    int   forbidden_face;
} Step;

typedef struct {
    Cube cube;
    Step traj[MAX_EP_STEPS];
    int  T;
    int  solved;
    int  done;             /* solved or T == budget */
    int  forbidden_face;
} Episode;

static Episode episodes[BATCH];
static float   batch_inputs[BATCH * NX];
static int     batch_masks[BATCH];
static float   batch_h    [BATCH * NH];
static float   batch_probs[BATCH * NY];
static float   batch_vals [BATCH];

static void encode(const Cube *c, float *out) {
    for (int i = 0; i < NX; i++) out[i] = 0.0f;
    for (int i = 0; i < 54; i++) out[(size_t)i * 6 + c->stickers[i]] = 1.0f;
}

static void scramble(Cube *c, int depth, Rng *rng) {
    cube_reset(c);
    int prev_face = -1;
    for (int i = 0; i < depth; i++) {
        Move m;
        int face;
        do {
            m = (Move)rng_range(rng, MV_COUNT);
            face = (int)m / 3;
        } while (face == prev_face);
        cube_apply_move(c, m);
        prev_face = face;
    }
}

static int sample_from_probs(const float *p, int n, Rng *rng) {
    float u = rng_unit(rng);
    float c = 0.0f;
    for (int j = 0; j < n; j++) {
        c += p[j];
        if (u < c) return j;
    }
    return n - 1;
}

int main(int argc, char **argv) {
    uint64_t seed = (uint64_t)time(NULL);
    int max_scramble = MAX_SCRAMBLE;
    long max_episodes = -1;
    if (argc > 1) seed = strtoull(argv[1], NULL, 10);
    if (argc > 2) max_scramble = atoi(argv[2]);
    if (argc > 3) max_episodes = strtol(argv[3], NULL, 10);
    if (max_scramble < 1) max_scramble = 1;
    if (max_scramble > MAX_SCRAMBLE) max_scramble = MAX_SCRAMBLE;

    /* GLFW + hidden GL 4.3 context */
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "gpu_train", NULL, NULL);
    if (!win) { fprintf(stderr, "no GL 4.3 context\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gl_load()) { fprintf(stderr, "gl_load failed\n"); return 1; }

    printf("seed=%llu  max_depth=%d  max_eps=%ld  arch=%d->%d->%d  lr=%g  batch=%d\n",
           (unsigned long long)seed, max_scramble, max_episodes, NX, NH, NY,
           (double)LR, BATCH);

    Rng rng; rng_seed(&rng, seed);
    MLP    *net = mlp_new(NX, NH, NY, &rng);
    GpuMlp *gpu = gpu_mlp_new(NX, NH, NY, BATCH);
    if (!gpu) { fprintf(stderr, "gpu_mlp_new failed\n"); return 1; }

    int wins[WIN_WINDOW] = {0};
    int widx = 0, wcount = 0, wsum = 0;

    int depth = 1;
    long total_eps = 0, stage_eps = 0;
    double t_start = glfwGetTime();
    double t_last_log = t_start;

    while (depth <= max_scramble) {
        if (max_episodes > 0 && total_eps >= max_episodes) {
            printf("Hit episode limit (%ld), stopping.\n", max_episodes);
            break;
        }

        /* Push current weights to GPU */
        gpu_mlp_upload_params(gpu,
                               net->W1, net->b1,
                               net->W2, net->b2,
                               net->Wv, net->bv);

        /* Initialize all B episodes */
        for (int b = 0; b < BATCH; b++) {
            scramble(&episodes[b].cube, depth, &rng);
            episodes[b].T = 0;
            episodes[b].solved = 0;
            episodes[b].done = 0;
            episodes[b].forbidden_face = -1;
        }

        int budget = depth + 5;
        if (budget > MAX_EP_STEPS) budget = MAX_EP_STEPS;

        /* Lockstep rollout */
        for (int step = 0; step < budget; step++) {
            int any_active = 0;
            for (int b = 0; b < BATCH; b++) {
                if (!episodes[b].done) {
                    any_active = 1;
                    encode(&episodes[b].cube, &batch_inputs[b * NX]);
                    batch_masks[b] = episodes[b].forbidden_face;
                } else {
                    /* Pad slot with zeros so the GPU still has valid data */
                    memset(&batch_inputs[b * NX], 0, NX * sizeof(float));
                    batch_masks[b] = -1;
                }
            }
            if (!any_active) break;

            gpu_mlp_forward(gpu, BATCH, batch_inputs, batch_masks,
                             batch_h, batch_probs, batch_vals);

            for (int b = 0; b < BATCH; b++) {
                if (episodes[b].done) continue;
                Episode *e = &episodes[b];
                int t = e->T;
                memcpy(e->traj[t].x,     &batch_inputs[b * NX], NX * sizeof(float));
                memcpy(e->traj[t].h,     &batch_h     [b * NH], NH * sizeof(float));
                memcpy(e->traj[t].probs, &batch_probs [b * NY], NY * sizeof(float));
                e->traj[t].value = batch_vals[b];
                int a = sample_from_probs(&batch_probs[b * NY], NY, &rng);
                e->traj[t].action = a;
                e->traj[t].forbidden_face = e->forbidden_face;
                e->forbidden_face = a / 3;
                e->T++;
                cube_apply_move(&e->cube, (Move)a);
                if (cube_is_solved(&e->cube)) {
                    e->solved = 1;
                    e->done = 1;
                } else if (e->T >= budget) {
                    e->done = 1;
                }
            }
        }

        /* Single CPU backward over all B episodes, all their steps. */
        mlp_zero_grad(net);
        int batch_solves = 0;
        for (int b = 0; b < BATCH; b++) {
            Episode *e = &episodes[b];
            int T = e->T;
            int solved = e->solved;
            if (solved) batch_solves++;
            for (int t = 0; t < T; t++) {
                float G = solved ? powf(GAMMA, (float)(T - 1 - t)) : 0.0f;
                mlp_accum_ac_grad(net, e->traj[t].x, e->traj[t].h, e->traj[t].probs,
                                  e->traj[t].value, e->traj[t].action, G,
                                  ENTROPY_BETA, VALUE_COEF);
            }
        }
        /* Average gradient across batch so each episode contributes
         * the same amount as a sequential CPU update. */
        mlp_apply_sgd(net, LR / (float)BATCH);

        /* Window update over BATCH episode outcomes */
        for (int b = 0; b < BATCH; b++) {
            int outcome = episodes[b].solved ? 1 : 0;
            if (wcount < WIN_WINDOW) {
                wins[widx] = outcome; wsum += outcome; wcount++;
            } else {
                wsum += outcome - wins[widx];
                wins[widx] = outcome;
            }
            widx = (widx + 1) % WIN_WINDOW;
        }
        total_eps += BATCH;
        stage_eps += BATCH;

        /* Periodic log */
        double now = glfwGetTime();
        if (now - t_last_log > 1.0) {
            float wr = (wcount > 0) ? (float)wsum / (float)wcount : 0.0f;
            double eps_per_sec = (double)total_eps / (now - t_start);
            printf("ep=%ld  depth=%d  stage_ep=%ld  win_rate=%.3f  batch_solves=%d/%d  speed=%.0f eps/s\n",
                   total_eps, depth, stage_eps, wr,
                   batch_solves, BATCH, eps_per_sec);
            fflush(stdout);
            t_last_log = now;
        }

        if (wcount >= WIN_WINDOW &&
            (float)wsum / (float)wcount >= ADVANCE_THRESHOLD) {
            float wr = (float)wsum / (float)wcount;
            printf("*** advance: depth %d cleared at win_rate=%.3f after %ld stage eps (total %ld) ***\n",
                   depth, wr, stage_eps, total_eps);
            depth++;
            stage_eps = 0;
            wsum = 0; wcount = 0; widx = 0;
            memset(wins, 0, sizeof(wins));
        }
    }

    printf("Curriculum complete after %ld total episodes.\n", total_eps);
    gpu_mlp_free(gpu);
    mlp_free(net);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

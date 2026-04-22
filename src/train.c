/* Curriculum REINFORCE trainer for the Rubik's cube.
 *
 *   - State: one-hot encoding of cube->stickers (54 * 6 = 324 features)
 *   - Policy: MLP (324 -> 128 -> 18), softmax over 18 quarter/half turns
 *   - Reward: 1 if solved within the step budget, else 0 (sparse)
 *   - Curriculum: start at scramble depth 1, advance when the agent wins
 *     >= 95% of the last WIN_WINDOW episodes. Stop after depth 30.
 */

#include "cube.h"
#include "nn.h"
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
#define WIN_WINDOW        200
#define ADVANCE_THRESHOLD 0.95f
#define MAX_SCRAMBLE      30
#define MAX_EPISODE_STEPS 64

static void encode(const Cube *c, float *out) {
    for (int i = 0; i < NX; i++) out[i] = 0.0f;
    for (int i = 0; i < 54; i++) out[(size_t)i * 6 + c->stickers[i]] = 1.0f;
}

typedef struct {
    float x[NX];
    int   action;
} Step;

/* Scramble by applying `depth` random moves, but avoid consecutive moves on
 * the same face (they can cancel to an effectively shallower scramble). */
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

int main(int argc, char **argv) {
    uint64_t seed = (uint64_t)time(NULL);
    int max_scramble = MAX_SCRAMBLE;
    long max_episodes = -1;  /* -1 = unbounded */
    if (argc > 1) seed = strtoull(argv[1], NULL, 10);
    if (argc > 2) max_scramble = atoi(argv[2]);
    if (argc > 3) max_episodes = strtol(argv[3], NULL, 10);
    if (max_scramble < 1) max_scramble = 1;
    if (max_scramble > MAX_SCRAMBLE) max_scramble = MAX_SCRAMBLE;
    printf("seed=%llu  max_depth=%d  max_eps=%ld  arch=%d->%d->%d  lr=%g  win_window=%d  threshold=%.0f%%\n",
           (unsigned long long)seed, max_scramble, max_episodes, NX, NH, NY,
           (double)LR, WIN_WINDOW, ADVANCE_THRESHOLD * 100.0f);

    Rng rng;
    rng_seed(&rng, seed);
    MLP *net = mlp_new(NX, NH, NY, &rng);

    Cube cube;
    Step *traj = (Step *)malloc(sizeof(Step) * MAX_EPISODE_STEPS);

    /* Rolling window of recent outcomes (1 = solved, 0 = not) */
    int wins[WIN_WINDOW] = {0};
    int widx = 0, wcount = 0, wsum = 0;

    int depth = 1;
    long total_eps = 0, stage_eps = 0, stage_solved = 0;
    double stage_steps_sum = 0.0;

    while (depth <= max_scramble) {
        if (max_episodes > 0 && total_eps >= max_episodes) {
            printf("Hit episode limit (%ld), stopping.\n", max_episodes);
            break;
        }
        scramble(&cube, depth, &rng);

        int budget = depth + 5;
        if (budget > MAX_EPISODE_STEPS) budget = MAX_EPISODE_STEPS;

        int T = 0;
        int solved = 0;
        for (int step = 0; step < budget; step++) {
            encode(&cube, traj[T].x);
            mlp_forward(net, traj[T].x);
            int action = mlp_sample_action(net, &rng);
            traj[T].action = action;
            T++;
            cube_apply_move(&cube, (Move)action);
            if (cube_is_solved(&cube)) { solved = 1; break; }
        }

        /* REINFORCE update: only when we got signal (reward 1). */
        if (solved) {
            mlp_zero_grad(net);
            for (int t = 0; t < T; t++) {
                float G = powf(GAMMA, (float)(T - 1 - t));
                mlp_forward(net, traj[t].x);
                mlp_accum_policy_grad(net, traj[t].x, traj[t].action, G);
            }
            mlp_apply_sgd(net, LR);
        }

        /* Update rolling window */
        int outcome = solved ? 1 : 0;
        if (wcount < WIN_WINDOW) {
            wins[widx] = outcome; wsum += outcome; wcount++;
        } else {
            wsum += outcome - wins[widx];
            wins[widx] = outcome;
        }
        widx = (widx + 1) % WIN_WINDOW;

        total_eps++;
        stage_eps++;
        if (solved) { stage_solved++; stage_steps_sum += (double)T; }

        if (total_eps % 200 == 0) {
            float wr = (wcount > 0) ? (float)wsum / (float)wcount : 0.0f;
            double avg_len = (stage_solved > 0) ? stage_steps_sum / (double)stage_solved : 0.0;
            printf("ep=%ld  depth=%d  stage_ep=%ld  win_rate=%.3f  avg_solve_steps=%.2f\n",
                   total_eps, depth, stage_eps, wr, avg_len);
            fflush(stdout);
        }

        if (wcount >= WIN_WINDOW &&
            (float)wsum / (float)wcount >= ADVANCE_THRESHOLD) {
            float wr = (float)wsum / (float)wcount;
            printf("*** advance: depth %d cleared at win_rate=%.3f after %ld stage eps (total %ld) ***\n",
                   depth, wr, stage_eps, total_eps);
            depth++;
            stage_eps = 0; stage_solved = 0; stage_steps_sum = 0.0;
            wsum = 0; wcount = 0; widx = 0;
            memset(wins, 0, sizeof(wins));
        }
    }

    printf("Curriculum complete after %ld total episodes.\n", total_eps);
    free(traj);
    mlp_free(net);
    return 0;
}

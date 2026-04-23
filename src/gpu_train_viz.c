/* gpu_train_viz: real-time visualization of A2C training that actually
 * uses the GPU. B parallel cube episodes run on GPU compute (batched
 * forward), CPU does action sampling, cube simulation, and the SGD step.
 * Episode 0 is rendered each frame so the user can watch one cube
 * progress through scramble -> agent moves -> outcome.
 *
 * Note: under vanilla A2C this won't out-train the CPU sequential trainer
 * (gradient cancellation across the batch), but it does run the forward
 * pass on the GPU and demonstrates the full pipeline working end-to-end. */

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gl_loader.h"
#include "cube.h"
#include "camera.h"
#include "renderer.h"
#include "hud.h"
#include "nn.h"
#include "gpu_nn.h"
#include "rng.h"
#include "linmath.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846f

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

#define BATCH 64

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
    int  done;
    int  forbidden_face;
} Episode;

typedef struct {
    Camera   camera;
    MLP     *net;
    GpuMlp  *gpu;
    Rng      rng;

    Episode  eps[BATCH];
    float    buf_inputs[BATCH * NX];
    int      buf_masks [BATCH];
    float    buf_h     [BATCH * NH];
    float    buf_probs [BATCH * NY];
    float    buf_vals  [BATCH];

    int      wins[WIN_WINDOW];
    int      widx, wcount, wsum;

    int      depth;
    long     total_eps, stage_eps;

    int      paused;
    int      ticks_per_frame;     /* lockstep micro-steps per frame */
    int      fb_w, fb_h;
    int      dragging;
    double   last_mx, last_my;
} App;

static App app;

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

/* Initialize all B episodes with fresh scrambles for the current depth. */
static void start_new_batch(void) {
    for (int b = 0; b < BATCH; b++) {
        scramble(&app.eps[b].cube, app.depth, &app.rng);
        app.eps[b].T = 0;
        app.eps[b].solved = 0;
        app.eps[b].done = 0;
        app.eps[b].forbidden_face = -1;
    }
    /* Push current weights to GPU (network changes each SGD step) */
    gpu_mlp_upload_params(app.gpu,
                           app.net->W1, app.net->b1,
                           app.net->W2, app.net->b2,
                           app.net->Wv, app.net->bv);
}

/* One lockstep step across all active episodes. Returns 1 if any episode
 * was still active (work done), 0 if all are done. */
static int batch_step(int budget) {
    int any_active = 0;
    for (int b = 0; b < BATCH; b++) {
        if (!app.eps[b].done) {
            any_active = 1;
            encode(&app.eps[b].cube, &app.buf_inputs[b * NX]);
            app.buf_masks[b] = app.eps[b].forbidden_face;
        } else {
            memset(&app.buf_inputs[b * NX], 0, NX * sizeof(float));
            app.buf_masks[b] = -1;
        }
    }
    if (!any_active) return 0;

    gpu_mlp_forward(app.gpu, BATCH, app.buf_inputs, app.buf_masks,
                     app.buf_h, app.buf_probs, app.buf_vals);

    for (int b = 0; b < BATCH; b++) {
        if (app.eps[b].done) continue;
        Episode *e = &app.eps[b];
        int t = e->T;
        memcpy(e->traj[t].x,     &app.buf_inputs[b * NX], NX * sizeof(float));
        memcpy(e->traj[t].h,     &app.buf_h     [b * NH], NH * sizeof(float));
        memcpy(e->traj[t].probs, &app.buf_probs [b * NY], NY * sizeof(float));
        e->traj[t].value = app.buf_vals[b];
        int a = sample_from_probs(&app.buf_probs[b * NY], NY, &app.rng);
        e->traj[t].action = a;
        e->traj[t].forbidden_face = e->forbidden_face;
        e->forbidden_face = a / 3;
        e->T++;
        cube_apply_move(&e->cube, (Move)a);
        if (cube_is_solved(&e->cube)) { e->solved = 1; e->done = 1; }
        else if (e->T >= budget)      { e->done = 1; }
    }
    return 1;
}

/* After all episodes finish: backward + SGD update + window bookkeeping. */
static void finish_batch(void) {
    mlp_zero_grad(app.net);
    for (int b = 0; b < BATCH; b++) {
        Episode *e = &app.eps[b];
        int T = e->T;
        int solved = e->solved;
        for (int t = 0; t < T; t++) {
            float G = solved ? powf(GAMMA, (float)(T - 1 - t)) : 0.0f;
            mlp_accum_ac_grad(app.net, e->traj[t].x, e->traj[t].h, e->traj[t].probs,
                              e->traj[t].value, e->traj[t].action, G,
                              ENTROPY_BETA, VALUE_COEF);
        }
    }
    /* Average across batch (equivalent CPU per-episode update magnitude). */
    mlp_apply_sgd(app.net, LR / (float)BATCH);

    for (int b = 0; b < BATCH; b++) {
        int outcome = app.eps[b].solved ? 1 : 0;
        if (app.wcount < WIN_WINDOW) {
            app.wins[app.widx] = outcome; app.wsum += outcome; app.wcount++;
        } else {
            app.wsum += outcome - app.wins[app.widx];
            app.wins[app.widx] = outcome;
        }
        app.widx = (app.widx + 1) % WIN_WINDOW;
    }
    app.total_eps += BATCH;
    app.stage_eps += BATCH;

    if (app.wcount >= WIN_WINDOW &&
        (float)app.wsum / (float)app.wcount >= ADVANCE_THRESHOLD &&
        app.depth < MAX_SCRAMBLE) {
        printf("*** advance: depth %d cleared after %ld stage eps (total %ld) ***\n",
               app.depth, app.stage_eps, app.total_eps);
        app.depth++;
        app.stage_eps = 0;
        app.wsum = 0; app.wcount = 0; app.widx = 0;
        memset(app.wins, 0, sizeof(app.wins));
    }
}

/* GLFW callbacks */

static void error_callback(int code, const char *desc) {
    fprintf(stderr, "GLFW error %d: %s\n", code, desc);
}
static void fb_size_callback(GLFWwindow *win, int w, int h) {
    (void)win; app.fb_w = w; app.fb_h = h; glViewport(0, 0, w, h);
}
static void key_callback(GLFWwindow *win, int key, int sc, int action, int mods) {
    (void)sc; (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, 1); break;
        case GLFW_KEY_SPACE:  app.paused = !app.paused; break;
        case GLFW_KEY_F:
            app.ticks_per_frame *= 2;
            if (app.ticks_per_frame > 256) app.ticks_per_frame = 256;
            break;
        case GLFW_KEY_S:
            app.ticks_per_frame /= 2;
            if (app.ticks_per_frame < 1) app.ticks_per_frame = 1;
            break;
    }
}
static void mouse_button_callback(GLFWwindow *win, int btn, int act, int mods) {
    (void)mods;
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        if (act == GLFW_PRESS) { app.dragging = 1; glfwGetCursorPos(win, &app.last_mx, &app.last_my); }
        else                   { app.dragging = 0; }
    }
}
static void cursor_pos_callback(GLFWwindow *win, double x, double y) {
    (void)win;
    if (!app.dragging) return;
    double dx = x - app.last_mx, dy = y - app.last_my;
    app.last_mx = x; app.last_my = y;
    camera_orbit(&app.camera, (float)(-dx * 0.008), (float)(dy * 0.008));
}
static void scroll_callback(GLFWwindow *win, double xo, double yo) {
    (void)win; (void)xo;
    camera_zoom(&app.camera, (yo > 0) ? 0.9f : 1.1f);
}

static void draw_hud(void) {
    hud_begin_frame();
    float bar_x0 = -0.96f, bar_x1 = 0.96f;
    float wr_y0 = 0.86f, wr_y1 = 0.94f;
    hud_quad(bar_x0, wr_y0, bar_x1, wr_y1, 0.18f, 0.18f, 0.20f);
    float wr = (app.wcount > 0) ? (float)app.wsum / (float)app.wcount : 0.0f;
    float fill_x = bar_x0 + (bar_x1 - bar_x0) * wr;
    float r, g, b;
    if (wr >= ADVANCE_THRESHOLD)      { r = 0.10f; g = 0.85f; b = 0.20f; }
    else if (wr >= 0.5f)              { r = 0.95f; g = 0.80f; b = 0.10f; }
    else                              { r = 0.85f; g = 0.30f; b = 0.10f; }
    hud_quad(bar_x0, wr_y0, fill_x, wr_y1, r, g, b);
    float thr_x = bar_x0 + (bar_x1 - bar_x0) * ADVANCE_THRESHOLD;
    hud_quad(thr_x - 0.003f, wr_y0 - 0.005f, thr_x + 0.003f, wr_y1 + 0.005f,
             0.90f, 0.10f, 0.10f);

    float d_y0 = 0.78f, d_y1 = 0.83f;
    hud_quad(bar_x0, d_y0, bar_x1, d_y1, 0.18f, 0.18f, 0.20f);
    float depth_frac = (float)app.depth / (float)MAX_SCRAMBLE;
    float d_fill_x = bar_x0 + (bar_x1 - bar_x0) * depth_frac;
    hud_quad(bar_x0, d_y0, d_fill_x, d_y1, 0.30f, 0.55f, 0.95f);
    for (int i = 1; i <= MAX_SCRAMBLE; i++) {
        float tx = bar_x0 + (bar_x1 - bar_x0) * ((float)i / (float)MAX_SCRAMBLE);
        hud_quad(tx - 0.0015f, d_y0, tx + 0.0015f, d_y1, 0.05f, 0.05f, 0.07f);
    }

    /* Small green square = GPU active indicator */
    hud_quad(0.92f, 0.95f, 0.99f, 0.99f, 0.10f, 0.85f, 0.20f);
    if (app.paused) {
        hud_quad(-0.99f, 0.95f, -0.92f, 0.99f, 0.95f, 0.10f, 0.10f);
    }
    hud_end_frame();
}

static void update_title(GLFWwindow *win) {
    static double last = 0.0;
    double now = glfwGetTime();
    if (now - last < 0.1) return;
    last = now;
    float wr = (app.wcount > 0) ? (float)app.wsum / (float)app.wcount : 0.0f;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "RL-Rubrics GPU train | depth=%d/%d | win=%.1f%% (%d/%d) | ep=%ld | batch=%d | ticks/frame=%d%s",
             app.depth, MAX_SCRAMBLE, wr * 100.0f, app.wsum, app.wcount,
             app.total_eps, BATCH, app.ticks_per_frame,
             app.paused ? " | PAUSED" : "");
    glfwSetWindowTitle(win, buf);
}

int main(int argc, char **argv) {
    uint64_t seed = (uint64_t)time(NULL);
    if (argc > 1) seed = strtoull(argv[1], NULL, 10);
    rng_seed(&app.rng, seed);
    camera_init(&app.camera);
    app.depth = 1;
    app.ticks_per_frame = 4;
    app.paused = 0;

    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow *win = glfwCreateWindow(1100, 800, "RL-Rubrics GPU train", NULL, NULL);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    if (!gl_load()) { glfwDestroyWindow(win); glfwTerminate(); return 1; }
    glfwGetFramebufferSize(win, &app.fb_w, &app.fb_h);
    glViewport(0, 0, app.fb_w, app.fb_h);
    glfwSetFramebufferSizeCallback(win, fb_size_callback);
    glfwSetKeyCallback(win, key_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetCursorPosCallback(win, cursor_pos_callback);
    glfwSetScrollCallback(win, scroll_callback);

    if (!renderer_init()) return 1;
    if (!hud_init())      return 1;

    app.net = mlp_new(NX, NH, NY, &app.rng);
    app.gpu = gpu_mlp_new(NX, NH, NY, BATCH);
    if (!app.gpu) { fprintf(stderr, "gpu_mlp_new failed\n"); return 1; }

    glClearColor(0.10f, 0.11f, 0.14f, 1.0f);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    printf("RL-Rubrics GPU training visualizer\n"
           "  Space  pause/resume   F/S faster/slower   left-drag orbit   wheel zoom   Esc quit\n"
           "  BATCH=%d episodes per backward step (rendering episode 0)\n", BATCH);

    /* Prime first batch */
    start_new_batch();
    int budget = app.depth + 5;

    while (!glfwWindowShouldClose(win)) {
        if (!app.paused) {
            for (int t = 0; t < app.ticks_per_frame; t++) {
                if (!batch_step(budget)) {
                    finish_batch();
                    start_new_batch();
                    budget = app.depth + 5;
                    if (budget > MAX_EP_STEPS) budget = MAX_EP_STEPS;
                }
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        float aspect = (app.fb_h > 0) ? (float)app.fb_w / (float)app.fb_h : 1.0f;
        Mat4 proj = mat4_perspective(PI / 4.0f, aspect, 0.1f, 100.0f);
        Mat4 view = camera_view(&app.camera);
        renderer_draw(&app.eps[0].cube, view, proj);
        glDisable(GL_DEPTH_TEST);
        draw_hud();

        update_title(win);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    gpu_mlp_free(app.gpu);
    mlp_free(app.net);
    hud_shutdown();
    renderer_shutdown();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

/* train_viz: Real-time visualization of REINFORCE curriculum training.
 *
 * Same training algorithm as train.c, but the agent's environment cube is
 * rendered live and a HUD shows scramble depth + win-rate progress. */

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gl_loader.h"
#include "cube.h"
#include "camera.h"
#include "renderer.h"
#include "hud.h"
#include "nn.h"
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
#define ENTROPY_BETA 0.01f
#define WIN_WINDOW        200
#define ADVANCE_THRESHOLD 0.95f
#define MAX_SCRAMBLE      30
#define MAX_EP_STEPS      64

typedef struct {
    float x[NX];
    float h[NH];
    float probs[NY];
    int   action;
    int   forbidden_face;
} Step;

typedef struct {
    Cube     cube;
    Camera   camera;
    MLP     *net;
    Rng      rng;
    Step     traj[MAX_EP_STEPS];

    int      wins[WIN_WINDOW];
    int      widx, wcount, wsum;

    int      depth;
    long     total_eps, stage_eps, stage_solved;

    /* UI state */
    int      paused;
    int      eps_per_frame;   /* training speed */
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

/* Run one episode, update policy if solved, update window. */
static void run_one_episode(void) {
    scramble(&app.cube, app.depth, &app.rng);

    int budget = app.depth + 5;
    if (budget > MAX_EP_STEPS) budget = MAX_EP_STEPS;

    int T = 0;
    int solved = 0;
    int prev_face = -1;
    for (int s = 0; s < budget; s++) {
        encode(&app.cube, app.traj[T].x);
        mlp_forward(app.net, app.traj[T].x, prev_face);
        memcpy(app.traj[T].h,     app.net->h,     sizeof(float) * NH);
        memcpy(app.traj[T].probs, app.net->probs, sizeof(float) * NY);
        int a = mlp_sample_action(app.net, &app.rng);
        app.traj[T].action = a;
        app.traj[T].forbidden_face = prev_face;
        prev_face = a / 3;
        T++;
        cube_apply_move(&app.cube, (Move)a);
        if (cube_is_solved(&app.cube)) { solved = 1; break; }
    }

    mlp_zero_grad(app.net);
    for (int t = 0; t < T; t++) {
        float G = solved ? powf(GAMMA, (float)(T - 1 - t)) : 0.0f;
        mlp_accum_policy_grad(app.net, app.traj[t].x, app.traj[t].h,
                              app.traj[t].probs, app.traj[t].action,
                              G, ENTROPY_BETA);
    }
    mlp_apply_sgd(app.net, LR);

    int outcome = solved ? 1 : 0;
    if (app.wcount < WIN_WINDOW) {
        app.wins[app.widx] = outcome; app.wsum += outcome; app.wcount++;
    } else {
        app.wsum += outcome - app.wins[app.widx];
        app.wins[app.widx] = outcome;
    }
    app.widx = (app.widx + 1) % WIN_WINDOW;
    app.total_eps++;
    app.stage_eps++;
    if (solved) app.stage_solved++;

    if (app.wcount >= WIN_WINDOW &&
        (float)app.wsum / (float)app.wcount >= ADVANCE_THRESHOLD &&
        app.depth < MAX_SCRAMBLE) {
        printf("*** advance: depth %d cleared after %ld stage eps (total %ld) ***\n",
               app.depth, app.stage_eps, app.total_eps);
        app.depth++;
        app.stage_eps = 0; app.stage_solved = 0;
        app.wsum = 0; app.wcount = 0; app.widx = 0;
        memset(app.wins, 0, sizeof(app.wins));
    }
}

/* ---------- UI / GLFW callbacks ---------- */

static void error_callback(int code, const char *desc) {
    fprintf(stderr, "GLFW error %d: %s\n", code, desc);
}

static void fb_size_callback(GLFWwindow *win, int w, int h) {
    (void)win;
    app.fb_w = w; app.fb_h = h;
    glViewport(0, 0, w, h);
}

static void key_callback(GLFWwindow *win, int key, int sc, int action, int mods) {
    (void)sc; (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, 1); break;
        case GLFW_KEY_SPACE:  app.paused = !app.paused; break;
        case GLFW_KEY_F:
            app.eps_per_frame *= 2;
            if (app.eps_per_frame > 1000) app.eps_per_frame = 1000;
            break;
        case GLFW_KEY_S:
            app.eps_per_frame /= 2;
            if (app.eps_per_frame < 1) app.eps_per_frame = 1;
            break;
    }
}

static void mouse_button_callback(GLFWwindow *win, int btn, int action, int mods) {
    (void)mods;
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            app.dragging = 1;
            glfwGetCursorPos(win, &app.last_mx, &app.last_my);
        } else {
            app.dragging = 0;
        }
    }
}

static void cursor_pos_callback(GLFWwindow *win, double x, double y) {
    (void)win;
    if (!app.dragging) return;
    double dx = x - app.last_mx;
    double dy = y - app.last_my;
    app.last_mx = x; app.last_my = y;
    const float sens = 0.008f;
    camera_orbit(&app.camera, (float)(-dx * sens), (float)(dy * sens));
}

static void scroll_callback(GLFWwindow *win, double xo, double yo) {
    (void)win; (void)xo;
    camera_zoom(&app.camera, (yo > 0) ? 0.9f : 1.1f);
}

/* ---------- HUD ---------- */

static void draw_hud(void) {
    hud_begin_frame();

    /* Win rate bar background */
    float wr_y0 = 0.86f, wr_y1 = 0.94f;
    float bar_x0 = -0.96f, bar_x1 = 0.96f;
    hud_quad(bar_x0, wr_y0, bar_x1, wr_y1, 0.18f, 0.18f, 0.20f);

    /* Win rate fill */
    float wr = (app.wcount > 0) ? (float)app.wsum / (float)app.wcount : 0.0f;
    float fill_x = bar_x0 + (bar_x1 - bar_x0) * wr;
    /* Color: red below 0.5, yellow between, green above 0.95 */
    float r, g, b;
    if (wr >= ADVANCE_THRESHOLD)      { r = 0.10f; g = 0.85f; b = 0.20f; }
    else if (wr >= 0.5f)              { r = 0.95f; g = 0.80f; b = 0.10f; }
    else                              { r = 0.85f; g = 0.30f; b = 0.10f; }
    hud_quad(bar_x0, wr_y0, fill_x, wr_y1, r, g, b);

    /* 95% threshold marker (thin red line) */
    float thr_x = bar_x0 + (bar_x1 - bar_x0) * ADVANCE_THRESHOLD;
    hud_quad(thr_x - 0.003f, wr_y0 - 0.005f, thr_x + 0.003f, wr_y1 + 0.005f,
             0.90f, 0.10f, 0.10f);

    /* Depth bar background */
    float d_y0 = 0.78f, d_y1 = 0.83f;
    hud_quad(bar_x0, d_y0, bar_x1, d_y1, 0.18f, 0.18f, 0.20f);
    /* Depth fill (1..30 -> 0..100% across the bar) */
    float depth_frac = (float)app.depth / (float)MAX_SCRAMBLE;
    float d_fill_x = bar_x0 + (bar_x1 - bar_x0) * depth_frac;
    hud_quad(bar_x0, d_y0, d_fill_x, d_y1, 0.30f, 0.55f, 0.95f);
    /* Tick marks for each depth level */
    for (int i = 1; i <= MAX_SCRAMBLE; i++) {
        float tx = bar_x0 + (bar_x1 - bar_x0) * ((float)i / (float)MAX_SCRAMBLE);
        hud_quad(tx - 0.0015f, d_y0, tx + 0.0015f, d_y1,
                 0.05f, 0.05f, 0.07f);
    }

    /* Pause indicator: small red square in corner if paused */
    if (app.paused) {
        hud_quad(-0.99f, 0.95f, -0.92f, 0.99f, 0.95f, 0.10f, 0.10f);
    }

    hud_end_frame();
}

static void update_title(GLFWwindow *win) {
    static double last_update = 0.0;
    double now = glfwGetTime();
    if (now - last_update < 0.1) return;
    last_update = now;

    float wr = (app.wcount > 0) ? (float)app.wsum / (float)app.wcount : 0.0f;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "RL-Rubrics train | depth=%d/%d | win=%.1f%% (%d/%d) | ep=%ld | speed=%d/frame%s",
             app.depth, MAX_SCRAMBLE, wr * 100.0f,
             app.wsum, app.wcount, app.total_eps,
             app.eps_per_frame, app.paused ? " | PAUSED" : "");
    glfwSetWindowTitle(win, buf);
}

/* ---------- main ---------- */

static void print_help(void) {
    printf(
        "RL-Rubrics training visualization\n"
        "  Space       pause / resume\n"
        "  F           faster (more episodes per frame)\n"
        "  S           slower\n"
        "  Left-drag   orbit camera\n"
        "  Wheel       zoom\n"
        "  Esc         quit\n"
    );
}

int main(int argc, char **argv) {
    uint64_t seed = (uint64_t)time(NULL);
    if (argc > 1) seed = strtoull(argv[1], NULL, 10);
    rng_seed(&app.rng, seed);

    cube_reset(&app.cube);
    camera_init(&app.camera);
    app.depth = 1;
    app.eps_per_frame = 50;
    app.paused = 0;

    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow *win = glfwCreateWindow(1100, 800, "RL-Rubrics train", NULL, NULL);
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

    if (!renderer_init()) { glfwDestroyWindow(win); glfwTerminate(); return 1; }
    if (!hud_init())      { renderer_shutdown(); glfwDestroyWindow(win); glfwTerminate(); return 1; }

    app.net = mlp_new(NX, NH, NY, &app.rng);

    glClearColor(0.10f, 0.11f, 0.14f, 1.0f);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    print_help();
    printf("seed=%llu\n", (unsigned long long)seed);

    while (!glfwWindowShouldClose(win)) {
        if (!app.paused) {
            for (int i = 0; i < app.eps_per_frame; i++) {
                run_one_episode();
                if (app.depth > MAX_SCRAMBLE) break;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* 3D cube pass */
        glEnable(GL_DEPTH_TEST);
        float aspect = (app.fb_h > 0) ? (float)app.fb_w / (float)app.fb_h : 1.0f;
        Mat4 proj = mat4_perspective(PI / 4.0f, aspect, 0.1f, 100.0f);
        Mat4 view = camera_view(&app.camera);
        renderer_draw(&app.cube, view, proj);

        /* HUD pass (no depth, on top) */
        glDisable(GL_DEPTH_TEST);
        draw_hud();

        update_title(win);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    mlp_free(app.net);
    hud_shutdown();
    renderer_shutdown();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

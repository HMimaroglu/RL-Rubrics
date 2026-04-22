#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "gl_loader.h"
#include "cube.h"
#include "camera.h"
#include "renderer.h"
#include "linmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PI 3.14159265358979323846f

typedef struct {
    Cube    cube;
    Camera  camera;
    int     fb_w, fb_h;
    int     dragging;
    double  last_mx, last_my;
} App;

static App app;

static void error_callback(int code, const char *desc) {
    fprintf(stderr, "GLFW error %d: %s\n", code, desc);
}

static void fb_size_callback(GLFWwindow *win, int w, int h) {
    (void)win;
    app.fb_w = w;
    app.fb_h = h;
    glViewport(0, 0, w, h);
}

static int face_index_for_key(int key) {
    switch (key) {
        case GLFW_KEY_U: return 0;
        case GLFW_KEY_D: return 1;
        case GLFW_KEY_F: return 2;
        case GLFW_KEY_B: return 3;
        case GLFW_KEY_L: return 4;
        case GLFW_KEY_R: return 5;
        default: return -1;
    }
}

static void key_callback(GLFWwindow *win, int key, int scancode, int action, int mods) {
    (void)scancode;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, 1);
        return;
    }
    if (key == GLFW_KEY_TAB) {
        cube_reset(&app.cube);
        printf("[reset]\n");
        return;
    }
    if (key == GLFW_KEY_SPACE) {
        cube_scramble(&app.cube, 25);
        printf("[scrambled 25 moves]\n");
        return;
    }

    int face = face_index_for_key(key);
    if (face < 0) return;

    int variant;
    if (mods & GLFW_MOD_ALT)        variant = 2;  /* 180  */
    else if (mods & GLFW_MOD_SHIFT) variant = 1;  /* prime */
    else                            variant = 0;  /* CW    */

    Move m = (Move)(face * 3 + variant);
    cube_apply_move(&app.cube, m);

    static const char face_letters[6] = { 'U', 'D', 'F', 'B', 'L', 'R' };
    const char *suffix = (variant == 0) ? "" : (variant == 1) ? "'" : "2";
    printf("[move] %c%s  solved=%d\n",
           face_letters[face], suffix, cube_is_solved(&app.cube));
}

static void mouse_button_callback(GLFWwindow *win, int button, int action, int mods) {
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            app.dragging = 1;
            glfwGetCursorPos(win, &app.last_mx, &app.last_my);
        } else if (action == GLFW_RELEASE) {
            app.dragging = 0;
        }
    }
}

static void cursor_pos_callback(GLFWwindow *win, double x, double y) {
    (void)win;
    if (!app.dragging) return;
    double dx = x - app.last_mx;
    double dy = y - app.last_my;
    app.last_mx = x;
    app.last_my = y;
    const float sens = 0.008f;
    camera_orbit(&app.camera, (float)(-dx * sens), (float)(dy * sens));
}

static void scroll_callback(GLFWwindow *win, double xoff, double yoff) {
    (void)win; (void)xoff;
    float factor = (yoff > 0) ? 0.9f : 1.1f;
    camera_zoom(&app.camera, factor);
}

static void print_help(void) {
    printf(
        "RL-Rubrics frontend\n"
        "  U D F B L R     quarter turn of that face (clockwise)\n"
        "  Shift + letter  counter-clockwise turn (X')\n"
        "  Alt + letter    180 degree turn (X2)\n"
        "  Space           scramble (25 random moves)\n"
        "  Tab             reset to solved\n"
        "  Left-drag       orbit camera\n"
        "  Mouse wheel     zoom\n"
        "  Esc             quit\n"
    );
}

int main(void) {
    srand((unsigned)time(NULL));
    cube_reset(&app.cube);
    camera_init(&app.camera);

    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES,               4);

    GLFWwindow *win = glfwCreateWindow(1024, 768, "RL-Rubrics", NULL, NULL);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gl_load()) {
        fprintf(stderr, "gl_load failed\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }

    glfwGetFramebufferSize(win, &app.fb_w, &app.fb_h);
    glViewport(0, 0, app.fb_w, app.fb_h);

    glfwSetFramebufferSizeCallback(win, fb_size_callback);
    glfwSetKeyCallback(win,              key_callback);
    glfwSetMouseButtonCallback(win,      mouse_button_callback);
    glfwSetCursorPosCallback(win,        cursor_pos_callback);
    glfwSetScrollCallback(win,           scroll_callback);

    if (!renderer_init()) {
        fprintf(stderr, "renderer_init failed\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glClearColor(0.12f, 0.13f, 0.16f, 1.0f);

    print_help();

    while (!glfwWindowShouldClose(win)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (app.fb_h > 0) ? (float)app.fb_w / (float)app.fb_h : 1.0f;
        Mat4 proj = mat4_perspective(PI / 4.0f, aspect, 0.1f, 100.0f);
        Mat4 view = camera_view(&app.camera);

        renderer_draw(&app.cube, view, proj);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    renderer_shutdown();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

/* GPU compute shader sanity check.
 *
 * Creates a hidden GL 4.3 context, compiles a minimal compute shader
 * that does y[i] = a*x[i] + b for an array of 4096 floats, dispatches
 * it, reads back the result, and verifies it matches the expected
 * answer. If this works on your machine, the same pipeline scales up
 * to the MLP forward / backward shaders for the trainer. */

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gl_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 4096

static const char *CS_SRC =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) readonly  buffer InBuf  { float x[]; };\n"
    "layout(std430, binding = 1) writeonly buffer OutBuf { float y[]; };\n"
    "uniform float u_a;\n"
    "uniform float u_b;\n"
    "void main() {\n"
    "    uint i = gl_GlobalInvocationID.x;\n"
    "    if (i >= x.length()) return;\n"
    "    y[i] = u_a * x[i] + u_b;\n"
    "}\n";

static GLuint compile_compute(const char *src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(sh, sizeof(log), &n, log);
        fprintf(stderr, "Compute shader compile error:\n%.*s\n", (int)n, log);
        glDeleteShader(sh);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetProgramInfoLog(prog, sizeof(log), &n, log);
        fprintf(stderr, "Compute program link error:\n%.*s\n", (int)n, log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

int main(void) {
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow *win = glfwCreateWindow(64, 64, "gpu_check", NULL, NULL);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed (need GL 4.3 context)\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    if (!gl_load()) {
        fprintf(stderr, "gl_load failed\n");
        glfwDestroyWindow(win); glfwTerminate();
        return 1;
    }

    const GLubyte *vendor   = glGetString(GL_VENDOR);
    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *version  = glGetString(GL_VERSION);
    const GLubyte *glsl_ver = glGetString(GL_SHADING_LANGUAGE_VERSION);
    printf("GL_VENDOR    : %s\n", vendor);
    printf("GL_RENDERER  : %s\n", renderer);
    printf("GL_VERSION   : %s\n", version);
    printf("GL_SLVERSION : %s\n", glsl_ver);

    GLuint prog = compile_compute(CS_SRC);
    if (!prog) { glfwDestroyWindow(win); glfwTerminate(); return 1; }

    /* Inputs */
    float xs[N];
    for (int i = 0; i < N; i++) xs[i] = (float)i * 0.001f;
    const float a = 3.0f, b = -2.0f;

    /* Buffers */
    GLuint in_buf, out_buf;
    glGenBuffers(1, &in_buf);
    glGenBuffers(1, &out_buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, in_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(xs), xs, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(xs), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in_buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out_buf);

    /* Dispatch */
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_a"), a);
    glUniform1f(glGetUniformLocation(prog, "u_b"), b);
    GLuint groups = (N + 63) / 64;
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Readback + verify */
    float ys[N];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_buf);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(ys), ys);

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = a * xs[i] + b;
        float err = fabsf(ys[i] - expected);
        if (err > 1e-5f) {
            if (errors < 5) {
                fprintf(stderr, "mismatch i=%d got=%g expected=%g err=%g\n",
                        i, ys[i], expected, err);
            }
            errors++;
        }
        if (err > max_err) max_err = err;
    }
    if (errors == 0) {
        printf("OK: %d elements processed on GPU, max abs err = %g\n", N, max_err);
    } else {
        printf("FAIL: %d mismatches, max err = %g\n", errors, max_err);
    }

    /* Need 1f loader (we didn't list it in gl_loader); use glUniform1i path. */
    glDeleteBuffers(1, &in_buf);
    glDeleteBuffers(1, &out_buf);
    glDeleteProgram(prog);
    glfwDestroyWindow(win);
    glfwTerminate();
    return errors == 0 ? 0 : 1;
}

/* Verify GPU MLP forward matches the CPU reference within float precision.
 *
 * Builds a CPU MLP with random weights, copies them to a GpuMlp, runs the
 * same batch of inputs through both, and compares h, probs, and value
 * outputs element-by-element.
 *
 * Also benchmarks raw forward throughput at batch sizes 1 and 256. */

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gl_loader.h"
#include "gpu_nn.h"
#include "nn.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NX (54 * 6)
#define NH 128
#define NY 18
#define BATCH 256

static double now_seconds(void) {
    return glfwGetTime();
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(void) {
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "gpu_nn_test", NULL, NULL);
    if (!win) { fprintf(stderr, "no GL 4.3 context\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gl_load()) { fprintf(stderr, "gl_load failed\n"); return 1; }

    Rng rng; rng_seed(&rng, 12345);

    /* Build CPU MLP with random weights (and a value head, since we want
     * to test that path too). */
    MLP *cpu = mlp_new(NX, NH, NY, &rng);

    /* GPU MLP, upload CPU weights */
    GpuMlp *gpu = gpu_mlp_new(NX, NH, NY, BATCH);
    if (!gpu) { fprintf(stderr, "gpu_mlp_new failed\n"); return 1; }
    gpu_mlp_upload_params(gpu,
                           cpu->W1, cpu->b1,
                           cpu->W2, cpu->b2,
                           cpu->Wv, cpu->bv);

    /* Random batch of inputs (one-hot 54-sticker style: 54 of the 324 are 1) */
    float *inputs = (float *)calloc(BATCH * NX, sizeof(float));
    int   *masks  = (int   *)calloc(BATCH, sizeof(int));
    for (int b = 0; b < BATCH; b++) {
        for (int s = 0; s < 54; s++) {
            int color = rng_range(&rng, 6);
            inputs[b * NX + s * 6 + color] = 1.0f;
        }
        masks[b] = (rng_range(&rng, 4) == 0) ? -1 : (int)rng_range(&rng, 6);
    }

    /* GPU forward */
    float *gpu_h     = (float *)malloc(BATCH * NH * sizeof(float));
    float *gpu_probs = (float *)malloc(BATCH * NY * sizeof(float));
    float *gpu_vals  = (float *)malloc(BATCH * sizeof(float));
    gpu_mlp_forward(gpu, BATCH, inputs, masks, gpu_h, gpu_probs, gpu_vals);

    /* CPU forward, one element at a time, into matching arrays */
    float *cpu_h     = (float *)malloc(BATCH * NH * sizeof(float));
    float *cpu_probs = (float *)malloc(BATCH * NY * sizeof(float));
    float *cpu_vals  = (float *)malloc(BATCH * sizeof(float));
    for (int b = 0; b < BATCH; b++) {
        mlp_forward(cpu, &inputs[b * NX], masks[b]);
        memcpy(&cpu_h[b * NH],     cpu->h,     NH * sizeof(float));
        memcpy(&cpu_probs[b * NY], cpu->probs, NY * sizeof(float));
        cpu_vals[b] = cpu->value;
    }

    float dh = max_abs_diff(cpu_h,     gpu_h,     BATCH * NH);
    float dp = max_abs_diff(cpu_probs, gpu_probs, BATCH * NY);
    float dv = max_abs_diff(cpu_vals,  gpu_vals,  BATCH);
    printf("max abs diff h     : %g\n", dh);
    printf("max abs diff probs : %g\n", dp);
    printf("max abs diff value : %g\n", dv);
    int ok = (dh < 1e-3f) && (dp < 1e-5f) && (dv < 1e-3f);
    printf("correctness        : %s\n", ok ? "OK" : "FAIL");

    /* Throughput benchmark: many forwards back to back */
    const int reps = 1000;
    double t0, t1;

    /* CPU sequential, batch size 1 (matches train.exe) */
    t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        for (int b = 0; b < BATCH; b++) mlp_forward(cpu, &inputs[b * NX], masks[b]);
    }
    t1 = now_seconds();
    double cpu_per_fw = (t1 - t0) / (double)(reps * BATCH);
    printf("CPU seq forward    : %g us/elem  (%.1f kHz)\n",
           cpu_per_fw * 1e6, 1e-3 / cpu_per_fw);

    /* GPU batched, with readback (the trainer's hot path) */
    t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        gpu_mlp_forward(gpu, BATCH, inputs, masks, gpu_h, gpu_probs, gpu_vals);
    }
    t1 = now_seconds();
    double gpu_per_fw = (t1 - t0) / (double)(reps * BATCH);
    printf("GPU batched fwd    : %g us/elem  (%.1f kHz)\n",
           gpu_per_fw * 1e6, 1e-3 / gpu_per_fw);
    printf("GPU speedup        : %.1fx\n", cpu_per_fw / gpu_per_fw);

    free(inputs); free(masks);
    free(gpu_h); free(gpu_probs); free(gpu_vals);
    free(cpu_h); free(cpu_probs); free(cpu_vals);
    gpu_mlp_free(gpu);
    mlp_free(cpu);
    glfwDestroyWindow(win);
    glfwTerminate();
    return ok ? 0 : 1;
}

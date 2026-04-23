#include "gpu_nn.h"
#include "gl_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_SIZE 64
#define MAX_NY 64    /* upper bound for the per-thread logit scratch array */

/* Compute shader: one thread per batch element, full trunk + policy + value.
 * The std430 layout matches plain row-major float arrays on the C side. */
static const char *FORWARD_CS_SRC =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0)  readonly  buffer W1Buf { float W1[]; };\n"
    "layout(std430, binding = 1)  readonly  buffer B1Buf { float B1[]; };\n"
    "layout(std430, binding = 2)  readonly  buffer W2Buf { float W2[]; };\n"
    "layout(std430, binding = 3)  readonly  buffer B2Buf { float B2[]; };\n"
    "layout(std430, binding = 4)  readonly  buffer WvBuf { float Wv[]; };\n"
    "layout(std430, binding = 5)  readonly  buffer BvBuf { float Bv[]; };\n"
    "layout(std430, binding = 6)  readonly  buffer XBuf  { float X[];  };\n"
    "layout(std430, binding = 7)  readonly  buffer MBuf  { int   Mask[]; };\n"
    "layout(std430, binding = 8)            buffer HBuf  { float H[];  };\n"
    "layout(std430, binding = 9)  writeonly buffer PBuf  { float P[];  };\n"
    "layout(std430, binding = 10) writeonly buffer VBuf  { float V[];  };\n"
    "uniform int u_nx;\n"
    "uniform int u_nh;\n"
    "uniform int u_ny;\n"
    "uniform int u_batch;\n"
    "uniform int u_use_value;\n"
    "void main() {\n"
    "    uint b = gl_GlobalInvocationID.x;\n"
    "    if (b >= uint(u_batch)) return;\n"
    "    uint xb  = b * uint(u_nx);\n"
    "    uint hb  = b * uint(u_nh);\n"
    "    uint pb  = b * uint(u_ny);\n"
    "    /* Hidden: h = ReLU(W1 . x + b1) */\n"
    "    for (int j = 0; j < u_nh; ++j) {\n"
    "        float s = B1[j];\n"
    "        uint  rowoff = uint(j) * uint(u_nx);\n"
    "        for (int i = 0; i < u_nx; ++i) {\n"
    "            s += W1[rowoff + uint(i)] * X[xb + uint(i)];\n"
    "        }\n"
    "        H[hb + uint(j)] = max(0.0, s);\n"
    "    }\n"
    "    /* Logits in registers (capped at 64; we only need 18) */\n"
    "    float logits[64];\n"
    "    for (int j = 0; j < u_ny; ++j) {\n"
    "        float s = B2[j];\n"
    "        uint  rowoff = uint(j) * uint(u_nh);\n"
    "        for (int i = 0; i < u_nh; ++i) {\n"
    "            s += W2[rowoff + uint(i)] * H[hb + uint(i)];\n"
    "        }\n"
    "        logits[j] = s;\n"
    "    }\n"
    "    /* Action mask: 3 logits per face */\n"
    "    int fb = Mask[b];\n"
    "    if (fb >= 0 && fb < 6) {\n"
    "        logits[fb * 3 + 0] = -1e20;\n"
    "        logits[fb * 3 + 1] = -1e20;\n"
    "        logits[fb * 3 + 2] = -1e20;\n"
    "    }\n"
    "    /* Softmax */\n"
    "    float mx = logits[0];\n"
    "    for (int j = 1; j < u_ny; ++j) mx = max(mx, logits[j]);\n"
    "    float sum = 0.0;\n"
    "    for (int j = 0; j < u_ny; ++j) {\n"
    "        logits[j] = exp(logits[j] - mx);\n"
    "        sum += logits[j];\n"
    "    }\n"
    "    float inv = 1.0 / sum;\n"
    "    for (int j = 0; j < u_ny; ++j) {\n"
    "        P[pb + uint(j)] = logits[j] * inv;\n"
    "    }\n"
    "    /* Value head */\n"
    "    if (u_use_value != 0) {\n"
    "        float v = Bv[0];\n"
    "        for (int i = 0; i < u_nh; ++i) v += Wv[i] * H[hb + uint(i)];\n"
    "        V[b] = v;\n"
    "    } else {\n"
    "        V[b] = 0.0;\n"
    "    }\n"
    "}\n";

struct GpuMlp {
    int nx, nh, ny, max_batch;
    GLuint program;
    GLint  u_nx, u_nh, u_ny, u_batch, u_use_value;
    /* SSBOs */
    GLuint b_W1, b_b1, b_W2, b_b2, b_Wv, b_bv;
    GLuint b_X, b_M, b_H, b_P, b_V;
    int has_value;
};

static GLuint compile_compute(const char *src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; GLsizei n = 0;
        glGetShaderInfoLog(sh, sizeof(log), &n, log);
        fprintf(stderr, "Forward CS compile error:\n%.*s\n", (int)n, log);
        glDeleteShader(sh);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, sh);
    glLinkProgram(p);
    glDeleteShader(sh);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        fprintf(stderr, "Forward CS link error:\n%.*s\n", (int)n, log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static GLuint make_ssbo(GLsizeiptr bytes, GLenum usage) {
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, NULL, usage);
    return b;
}

GpuMlp *gpu_mlp_new(int nx, int nh, int ny, int max_batch) {
    if (ny > MAX_NY) {
        fprintf(stderr, "gpu_mlp_new: ny=%d exceeds MAX_NY=%d\n", ny, MAX_NY);
        return NULL;
    }
    GpuMlp *m = (GpuMlp *)calloc(1, sizeof(GpuMlp));
    m->nx = nx; m->nh = nh; m->ny = ny; m->max_batch = max_batch;
    m->program = compile_compute(FORWARD_CS_SRC);
    if (!m->program) { free(m); return NULL; }
    m->u_nx        = glGetUniformLocation(m->program, "u_nx");
    m->u_nh        = glGetUniformLocation(m->program, "u_nh");
    m->u_ny        = glGetUniformLocation(m->program, "u_ny");
    m->u_batch     = glGetUniformLocation(m->program, "u_batch");
    m->u_use_value = glGetUniformLocation(m->program, "u_use_value");

    m->b_W1 = make_ssbo((GLsizeiptr)nh * nx * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_b1 = make_ssbo((GLsizeiptr)nh        * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_W2 = make_ssbo((GLsizeiptr)ny * nh * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_b2 = make_ssbo((GLsizeiptr)ny        * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_Wv = make_ssbo((GLsizeiptr)nh        * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_bv = make_ssbo((GLsizeiptr)1         * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_X  = make_ssbo((GLsizeiptr)max_batch * nx * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_M  = make_ssbo((GLsizeiptr)max_batch * sizeof(int),   GL_DYNAMIC_DRAW);
    m->b_H  = make_ssbo((GLsizeiptr)max_batch * nh * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_P  = make_ssbo((GLsizeiptr)max_batch * ny * sizeof(float), GL_DYNAMIC_DRAW);
    m->b_V  = make_ssbo((GLsizeiptr)max_batch * sizeof(float), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return m;
}

void gpu_mlp_free(GpuMlp *m) {
    if (!m) return;
    glDeleteBuffers(1, &m->b_W1); glDeleteBuffers(1, &m->b_b1);
    glDeleteBuffers(1, &m->b_W2); glDeleteBuffers(1, &m->b_b2);
    glDeleteBuffers(1, &m->b_Wv); glDeleteBuffers(1, &m->b_bv);
    glDeleteBuffers(1, &m->b_X);  glDeleteBuffers(1, &m->b_M);
    glDeleteBuffers(1, &m->b_H);  glDeleteBuffers(1, &m->b_P);
    glDeleteBuffers(1, &m->b_V);
    glDeleteProgram(m->program);
    free(m);
}

static void upload(GLuint buf, GLsizeiptr bytes, const void *data) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
}

void gpu_mlp_upload_params(GpuMlp *m,
                            const float *W1, const float *b1,
                            const float *W2, const float *b2,
                            const float *Wv, const float *bv) {
    upload(m->b_W1, (GLsizeiptr)m->nh * m->nx * sizeof(float), W1);
    upload(m->b_b1, (GLsizeiptr)m->nh        * sizeof(float), b1);
    upload(m->b_W2, (GLsizeiptr)m->ny * m->nh * sizeof(float), W2);
    upload(m->b_b2, (GLsizeiptr)m->ny        * sizeof(float), b2);
    if (Wv && bv) {
        upload(m->b_Wv, (GLsizeiptr)m->nh * sizeof(float), Wv);
        upload(m->b_bv, (GLsizeiptr)1     * sizeof(float), bv);
        m->has_value = 1;
    } else {
        m->has_value = 0;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void gpu_mlp_forward(GpuMlp *m,
                      int batch_size,
                      const float *inputs,
                      const int   *forbidden_face,
                      float       *out_h,
                      float       *out_probs,
                      float       *out_values) {
    if (batch_size > m->max_batch) {
        fprintf(stderr, "gpu_mlp_forward: batch_size=%d > max_batch=%d\n",
                batch_size, m->max_batch);
        return;
    }

    /* Upload inputs and mask */
    upload(m->b_X, (GLsizeiptr)batch_size * m->nx * sizeof(float), inputs);
    upload(m->b_M, (GLsizeiptr)batch_size * sizeof(int), forbidden_face);

    /* Bind all SSBOs to their slots */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,  m->b_W1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1,  m->b_b1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2,  m->b_W2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3,  m->b_b2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,  m->b_Wv);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5,  m->b_bv);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6,  m->b_X);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7,  m->b_M);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8,  m->b_H);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9,  m->b_P);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m->b_V);

    glUseProgram(m->program);
    glUniform1i(m->u_nx,        m->nx);
    glUniform1i(m->u_nh,        m->nh);
    glUniform1i(m->u_ny,        m->ny);
    glUniform1i(m->u_batch,     batch_size);
    glUniform1i(m->u_use_value, m->has_value);

    GLuint groups = (GLuint)((batch_size + LOCAL_SIZE - 1) / LOCAL_SIZE);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (out_h) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m->b_H);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           (GLsizeiptr)batch_size * m->nh * sizeof(float), out_h);
    }
    if (out_probs) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m->b_P);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           (GLsizeiptr)batch_size * m->ny * sizeof(float), out_probs);
    }
    if (out_values) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m->b_V);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           (GLsizeiptr)batch_size * sizeof(float), out_values);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);
}

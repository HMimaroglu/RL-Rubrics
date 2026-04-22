#include "renderer.h"
#include "gl_loader.h"
#include "cube.h"
#include "linmath.h"
#include <stdio.h>
#include <stdlib.h>

static const char *VS_SRC =
    "#version 330 core\n"
    "layout(location=0) in vec3 a_pos;\n"
    "layout(location=1) in vec3 a_normal;\n"
    "layout(location=2) in float a_sticker_idx;\n"
    "uniform mat4 u_view;\n"
    "uniform mat4 u_proj;\n"
    "out vec3 v_normal;\n"
    "flat out int v_idx;\n"
    "void main() {\n"
    "    v_normal = a_normal;\n"
    "    v_idx = int(a_sticker_idx + 0.5);\n"
    "    gl_Position = u_proj * u_view * vec4(a_pos, 1.0);\n"
    "}\n";

static const char *FS_SRC =
    "#version 330 core\n"
    "in vec3 v_normal;\n"
    "flat in int v_idx;\n"
    "uniform int u_mode;\n"
    "uniform vec3 u_palette[6];\n"
    "uniform int u_state[54];\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    vec3 base;\n"
    "    if (u_mode == 0) {\n"
    "        base = vec3(0.08, 0.08, 0.09);\n"
    "    } else {\n"
    "        int color_idx = u_state[v_idx];\n"
    "        base = u_palette[color_idx];\n"
    "    }\n"
    "    vec3 L = normalize(vec3(0.6, 1.0, 0.5));\n"
    "    float d = max(dot(normalize(v_normal), L), 0.0);\n"
    "    vec3 lit = base * (0.35 + 0.65 * d);\n"
    "    frag_color = vec4(lit, 1.0);\n"
    "}\n";

/* Matches cube.h face conventions:
 *   Outward normal, viewer's "up" direction, viewer's "right" direction. */
static const Vec3 face_N[6] = {
    { 0,  1,  0}, /* U */
    { 0, -1,  0}, /* D */
    { 0,  0,  1}, /* F */
    { 0,  0, -1}, /* B */
    {-1,  0,  0}, /* L */
    { 1,  0,  0}, /* R */
};
static const Vec3 face_Up[6] = {
    { 0,  0, -1}, /* U */
    { 0,  0,  1}, /* D */
    { 0,  1,  0}, /* F */
    { 0,  1,  0}, /* B */
    { 0,  1,  0}, /* L */
    { 0,  1,  0}, /* R */
};
static const Vec3 face_Rt[6] = {
    { 1,  0,  0}, /* U */
    { 1,  0,  0}, /* D */
    { 1,  0,  0}, /* F */
    {-1,  0,  0}, /* B */
    { 0,  0,  1}, /* L */
    { 0,  0, -1}, /* R */
};

static const float PALETTE[6][3] = {
    {1.00f, 1.00f, 1.00f},  /* U white  */
    {1.00f, 0.85f, 0.12f},  /* D yellow */
    {0.05f, 0.72f, 0.18f},  /* F green  */
    {0.05f, 0.30f, 0.90f},  /* B blue   */
    {1.00f, 0.50f, 0.00f},  /* L orange */
    {0.85f, 0.10f, 0.10f},  /* R red    */
};

typedef struct {
    GLuint program;
    GLint  u_view, u_proj, u_mode, u_palette, u_state;
    GLuint body_vao, body_vbo;
    GLuint stk_vao,  stk_vbo;
    GLsizei body_vcount;
    GLsizei stk_vcount;
} Renderer;

static Renderer G;

/* ---------- shader helpers ---------- */

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei n = 0;
        glGetShaderInfoLog(sh, (GLsizei)sizeof(log), &n, log);
        fprintf(stderr, "Shader compile error:\n%.*s\n", (int)n, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei n = 0;
        glGetProgramInfoLog(p, (GLsizei)sizeof(log), &n, log);
        fprintf(stderr, "Program link error:\n%.*s\n", (int)n, log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

/* ---------- mesh builders ---------- */

/* Build a single 3x3x3 dark cube (36 verts: 6 faces x 2 tris x 3 verts).
 * Layout per vertex: pos (3f) + normal (3f) = 6 floats. */
static void build_body_mesh(float **out, int *out_vcount) {
    const float h = 1.5f;
    const int V = 36;
    float *data = (float *)malloc((size_t)V * 6 * sizeof(float));
    int w = 0;

    for (int axis = 0; axis < 3; axis++) {
        int uax = (axis + 1) % 3;
        int vax = (axis + 2) % 3;
        for (int sign = -1; sign <= 1; sign += 2) {
            /* Build basis vectors for this face */
            Vec3 n  = {0, 0, 0};
            Vec3 ua = {0, 0, 0};
            Vec3 va = {0, 0, 0};
            float *np = (axis == 0) ? &n.x : (axis == 1) ? &n.y : &n.z;
            float *up = (uax  == 0) ? &ua.x : (uax  == 1) ? &ua.y : &ua.z;
            float *vp = (vax  == 0) ? &va.x : (vax  == 1) ? &va.y : &va.z;
            *np = (float)sign; *up = 1.0f; *vp = 1.0f;

            /* Ensure (ua x va) points outward (= n). If not, swap. */
            if (v3_dot(v3_cross(ua, va), n) < 0.0f) {
                Vec3 t = ua; ua = va; va = t;
            }

            Vec3 fc = v3_scale(n, h);
            Vec3 c0 = v3_add(fc, v3_add(v3_scale(ua, -h), v3_scale(va, -h)));
            Vec3 c1 = v3_add(fc, v3_add(v3_scale(ua,  h), v3_scale(va, -h)));
            Vec3 c2 = v3_add(fc, v3_add(v3_scale(ua,  h), v3_scale(va,  h)));
            Vec3 c3 = v3_add(fc, v3_add(v3_scale(ua, -h), v3_scale(va,  h)));
            Vec3 tri[6] = { c0, c1, c2, c0, c2, c3 };
            for (int t = 0; t < 6; t++) {
                data[w++] = tri[t].x; data[w++] = tri[t].y; data[w++] = tri[t].z;
                data[w++] = n.x;      data[w++] = n.y;      data[w++] = n.z;
            }
        }
    }
    *out = data;
    *out_vcount = V;
}

/* Build 54 sticker quads (324 verts).
 * Layout per vertex: pos (3f) + normal (3f) + sticker_idx (1f) = 7 floats. */
static void build_sticker_mesh(float **out, int *out_vcount) {
    const float OUTER = 1.502f;  /* just past the body surface to avoid z-fighting */
    const float HS    = 0.425f;  /* sticker half-size; leaves ~0.15 dark border  */
    const int V = 54 * 6;
    float *data = (float *)malloc((size_t)V * 7 * sizeof(float));
    int w = 0;

    for (int f = 0; f < 6; f++) {
        Vec3 N  = face_N[f];
        Vec3 Uv = face_Up[f];
        Vec3 Rv = face_Rt[f];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                Vec3 sc = v3_add(v3_scale(N, OUTER),
                                 v3_add(v3_scale(Rv, (float)(c - 1)),
                                        v3_scale(Uv, (float)(1 - r))));
                Vec3 c0 = v3_sub(v3_sub(sc, v3_scale(Rv, HS)), v3_scale(Uv, HS));
                Vec3 c1 = v3_sub(v3_add(sc, v3_scale(Rv, HS)), v3_scale(Uv, HS));
                Vec3 c2 = v3_add(v3_add(sc, v3_scale(Rv, HS)), v3_scale(Uv, HS));
                Vec3 c3 = v3_add(v3_sub(sc, v3_scale(Rv, HS)), v3_scale(Uv, HS));
                Vec3 tri[6] = { c0, c1, c2, c0, c2, c3 };
                float idx = (float)(f * 9 + r * 3 + c);
                for (int t = 0; t < 6; t++) {
                    data[w++] = tri[t].x; data[w++] = tri[t].y; data[w++] = tri[t].z;
                    data[w++] = N.x;      data[w++] = N.y;      data[w++] = N.z;
                    data[w++] = idx;
                }
            }
        }
    }
    *out = data;
    *out_vcount = V;
}

/* ---------- public API ---------- */

int renderer_init(void) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER,   VS_SRC);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_SRC);
    if (!vs || !fs) return 0;
    G.program = link_program(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!G.program) return 0;

    G.u_view    = glGetUniformLocation(G.program, "u_view");
    G.u_proj    = glGetUniformLocation(G.program, "u_proj");
    G.u_mode    = glGetUniformLocation(G.program, "u_mode");
    G.u_palette = glGetUniformLocation(G.program, "u_palette");
    G.u_state   = glGetUniformLocation(G.program, "u_state");

    /* Body */
    {
        float *data = NULL;
        int vcount = 0;
        build_body_mesh(&data, &vcount);
        G.body_vcount = vcount;
        glGenVertexArrays(1, &G.body_vao);
        glGenBuffers(1, &G.body_vbo);
        glBindVertexArray(G.body_vao);
        glBindBuffer(GL_ARRAY_BUFFER, G.body_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vcount * 6 * sizeof(float), data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
        free(data);
    }

    /* Stickers */
    {
        float *data = NULL;
        int vcount = 0;
        build_sticker_mesh(&data, &vcount);
        G.stk_vcount = vcount;
        glGenVertexArrays(1, &G.stk_vao);
        glGenBuffers(1, &G.stk_vbo);
        glBindVertexArray(G.stk_vao);
        glBindBuffer(GL_ARRAY_BUFFER, G.stk_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vcount * 7 * sizeof(float), data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(6 * sizeof(float)));
        free(data);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    /* Palette is constant for the lifetime of the program */
    glUseProgram(G.program);
    glUniform3fv(G.u_palette, 6, (const float *)PALETTE);
    glUseProgram(0);

    return 1;
}

void renderer_shutdown(void) {
    if (G.body_vbo) glDeleteBuffers(1, &G.body_vbo);
    if (G.body_vao) glDeleteVertexArrays(1, &G.body_vao);
    if (G.stk_vbo)  glDeleteBuffers(1, &G.stk_vbo);
    if (G.stk_vao)  glDeleteVertexArrays(1, &G.stk_vao);
    if (G.program)  glDeleteProgram(G.program);
    G = (Renderer){0};
}

void renderer_draw(const Cube *cube, Mat4 view, Mat4 proj) {
    glUseProgram(G.program);
    glUniformMatrix4fv(G.u_view, 1, GL_FALSE, view.m);
    glUniformMatrix4fv(G.u_proj, 1, GL_FALSE, proj.m);

    /* Mirror the backend state into the u_state uniform each frame.
     * Keeps the frontend a pure read-only view of the Cube struct. */
    GLint state[54];
    for (int i = 0; i < 54; i++) state[i] = (GLint)cube->stickers[i];
    glUniform1iv(G.u_state, 54, state);

    /* Body pass */
    glUniform1i(G.u_mode, 0);
    glBindVertexArray(G.body_vao);
    glDrawArrays(GL_TRIANGLES, 0, G.body_vcount);

    /* Sticker pass */
    glUniform1i(G.u_mode, 1);
    glBindVertexArray(G.stk_vao);
    glDrawArrays(GL_TRIANGLES, 0, G.stk_vcount);

    glBindVertexArray(0);
    glUseProgram(0);
}

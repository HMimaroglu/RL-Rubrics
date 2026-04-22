#include "hud.h"
#include "gl_loader.h"
#include <stdio.h>

#define MAX_QUADS  256
#define VERTS_PER_QUAD 6
#define FLOATS_PER_VERT 5  /* x, y, r, g, b */

static const char *VS_SRC =
    "#version 330 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=1) in vec3 a_color;\n"
    "out vec3 v_color;\n"
    "void main() {\n"
    "    v_color = a_color;\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char *FS_SRC =
    "#version 330 core\n"
    "in vec3 v_color;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    frag_color = vec4(v_color, 1.0);\n"
    "}\n";

typedef struct {
    GLuint program, vao, vbo;
    float buf[MAX_QUADS * VERTS_PER_QUAD * FLOATS_PER_VERT];
    int n_quads;
} Hud;

static Hud H;

static GLuint compile(GLenum type, const char *src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n = 0;
        glGetShaderInfoLog(sh, sizeof(log), &n, log);
        fprintf(stderr, "HUD shader compile error:\n%.*s\n", (int)n, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

int hud_init(void) {
    GLuint vs = compile(GL_VERTEX_SHADER, VS_SRC);
    GLuint fs = compile(GL_FRAGMENT_SHADER, FS_SRC);
    if (!vs || !fs) return 0;
    H.program = glCreateProgram();
    glAttachShader(H.program, vs);
    glAttachShader(H.program, fs);
    glLinkProgram(H.program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(H.program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n = 0;
        glGetProgramInfoLog(H.program, sizeof(log), &n, log);
        fprintf(stderr, "HUD link error:\n%.*s\n", (int)n, log);
        return 0;
    }

    glGenVertexArrays(1, &H.vao);
    glGenBuffers(1, &H.vbo);
    glBindVertexArray(H.vao);
    glBindBuffer(GL_ARRAY_BUFFER, H.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(H.buf), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          FLOATS_PER_VERT * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          FLOATS_PER_VERT * sizeof(float),
                          (void *)(2 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return 1;
}

void hud_shutdown(void) {
    if (H.vbo)     glDeleteBuffers(1, &H.vbo);
    if (H.vao)     glDeleteVertexArrays(1, &H.vao);
    if (H.program) glDeleteProgram(H.program);
    H = (Hud){0};
}

void hud_begin_frame(void) {
    H.n_quads = 0;
}

void hud_quad(float x0, float y0, float x1, float y1,
              float r, float g, float b) {
    if (H.n_quads >= MAX_QUADS) return;
    float verts[6][5] = {
        { x0, y0, r, g, b },
        { x1, y0, r, g, b },
        { x1, y1, r, g, b },
        { x0, y0, r, g, b },
        { x1, y1, r, g, b },
        { x0, y1, r, g, b },
    };
    float *dst = &H.buf[H.n_quads * VERTS_PER_QUAD * FLOATS_PER_VERT];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 5; j++)
            dst[i * 5 + j] = verts[i][j];
    H.n_quads++;
}

void hud_end_frame(void) {
    if (H.n_quads == 0) return;
    int floats = H.n_quads * VERTS_PER_QUAD * FLOATS_PER_VERT;
    glUseProgram(H.program);
    glBindVertexArray(H.vao);
    glBindBuffer(GL_ARRAY_BUFFER, H.vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    (GLsizeiptr)floats * sizeof(float), H.buf);
    glDrawArrays(GL_TRIANGLES, 0, H.n_quads * VERTS_PER_QUAD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

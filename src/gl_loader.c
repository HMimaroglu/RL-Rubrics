#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "gl_loader.h"
#include <stdio.h>

PFNGLGENBUFFERSPROC              glGenBuffers              = NULL;
PFNGLBINDBUFFERPROC              glBindBuffer              = NULL;
PFNGLBUFFERDATAPROC              glBufferData              = NULL;
PFNGLBUFFERSUBDATAPROC           glBufferSubData           = NULL;
PFNGLDELETEBUFFERSPROC           glDeleteBuffers           = NULL;
PFNGLGENVERTEXARRAYSPROC         glGenVertexArrays         = NULL;
PFNGLBINDVERTEXARRAYPROC         glBindVertexArray         = NULL;
PFNGLDELETEVERTEXARRAYSPROC      glDeleteVertexArrays      = NULL;
PFNGLVERTEXATTRIBPOINTERPROC     glVertexAttribPointer     = NULL;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = NULL;
PFNGLCREATESHADERPROC            glCreateShader            = NULL;
PFNGLSHADERSOURCEPROC            glShaderSource            = NULL;
PFNGLCOMPILESHADERPROC           glCompileShader           = NULL;
PFNGLGETSHADERIVPROC             glGetShaderiv             = NULL;
PFNGLGETSHADERINFOLOGPROC        glGetShaderInfoLog        = NULL;
PFNGLDELETESHADERPROC            glDeleteShader            = NULL;
PFNGLCREATEPROGRAMPROC           glCreateProgram           = NULL;
PFNGLATTACHSHADERPROC            glAttachShader            = NULL;
PFNGLLINKPROGRAMPROC             glLinkProgram             = NULL;
PFNGLGETPROGRAMIVPROC            glGetProgramiv            = NULL;
PFNGLGETPROGRAMINFOLOGPROC       glGetProgramInfoLog       = NULL;
PFNGLUSEPROGRAMPROC              glUseProgram              = NULL;
PFNGLDELETEPROGRAMPROC           glDeleteProgram           = NULL;
PFNGLGETUNIFORMLOCATIONPROC      glGetUniformLocation      = NULL;
PFNGLUNIFORMMATRIX4FVPROC        glUniformMatrix4fv        = NULL;
PFNGLUNIFORM1FPROC               glUniform1f               = NULL;
PFNGLUNIFORM3FPROC               glUniform3f               = NULL;
PFNGLUNIFORM3FVPROC              glUniform3fv              = NULL;
PFNGLUNIFORM1IPROC               glUniform1i               = NULL;
PFNGLUNIFORM1IVPROC              glUniform1iv              = NULL;

PFNGLDISPATCHCOMPUTEPROC         glDispatchCompute         = NULL;
PFNGLMEMORYBARRIERPROC           glMemoryBarrier           = NULL;
PFNGLBINDBUFFERBASEPROC          glBindBufferBase          = NULL;
PFNGLGETBUFFERSUBDATAPROC        glGetBufferSubData        = NULL;
PFNGLMAPBUFFERRANGEPROC          glMapBufferRange          = NULL;
PFNGLUNMAPBUFFERPROC             glUnmapBuffer             = NULL;

int gl_load(void) {
    int ok = 1;
    #define L(name, T) do { \
        name = (T)(void(*)(void))glfwGetProcAddress(#name); \
        if (!name) { fprintf(stderr, "gl_load: missing %s\n", #name); ok = 0; } \
    } while (0)

    L(glGenBuffers,              PFNGLGENBUFFERSPROC);
    L(glBindBuffer,              PFNGLBINDBUFFERPROC);
    L(glBufferData,              PFNGLBUFFERDATAPROC);
    L(glBufferSubData,           PFNGLBUFFERSUBDATAPROC);
    L(glDeleteBuffers,           PFNGLDELETEBUFFERSPROC);
    L(glGenVertexArrays,         PFNGLGENVERTEXARRAYSPROC);
    L(glBindVertexArray,         PFNGLBINDVERTEXARRAYPROC);
    L(glDeleteVertexArrays,      PFNGLDELETEVERTEXARRAYSPROC);
    L(glVertexAttribPointer,     PFNGLVERTEXATTRIBPOINTERPROC);
    L(glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC);
    L(glCreateShader,            PFNGLCREATESHADERPROC);
    L(glShaderSource,            PFNGLSHADERSOURCEPROC);
    L(glCompileShader,           PFNGLCOMPILESHADERPROC);
    L(glGetShaderiv,             PFNGLGETSHADERIVPROC);
    L(glGetShaderInfoLog,        PFNGLGETSHADERINFOLOGPROC);
    L(glDeleteShader,            PFNGLDELETESHADERPROC);
    L(glCreateProgram,           PFNGLCREATEPROGRAMPROC);
    L(glAttachShader,            PFNGLATTACHSHADERPROC);
    L(glLinkProgram,             PFNGLLINKPROGRAMPROC);
    L(glGetProgramiv,            PFNGLGETPROGRAMIVPROC);
    L(glGetProgramInfoLog,       PFNGLGETPROGRAMINFOLOGPROC);
    L(glUseProgram,              PFNGLUSEPROGRAMPROC);
    L(glDeleteProgram,           PFNGLDELETEPROGRAMPROC);
    L(glGetUniformLocation,      PFNGLGETUNIFORMLOCATIONPROC);
    L(glUniformMatrix4fv,        PFNGLUNIFORMMATRIX4FVPROC);
    L(glUniform1f,               PFNGLUNIFORM1FPROC);
    L(glUniform3f,               PFNGLUNIFORM3FPROC);
    L(glUniform3fv,              PFNGLUNIFORM3FVPROC);
    L(glUniform1i,               PFNGLUNIFORM1IPROC);
    L(glUniform1iv,              PFNGLUNIFORM1IVPROC);

    /* Compute / SSBO functions. These will fail to load on a context
     * older than GL 4.3; the caller can detect that via the return value. */
    L(glDispatchCompute,         PFNGLDISPATCHCOMPUTEPROC);
    L(glMemoryBarrier,           PFNGLMEMORYBARRIERPROC);
    L(glBindBufferBase,          PFNGLBINDBUFFERBASEPROC);
    L(glGetBufferSubData,        PFNGLGETBUFFERSUBDATAPROC);
    L(glMapBufferRange,          PFNGLMAPBUFFERRANGEPROC);
    L(glUnmapBuffer,             PFNGLUNMAPBUFFERPROC);

    #undef L
    return ok;
}

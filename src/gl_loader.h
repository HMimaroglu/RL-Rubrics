#ifndef GL_LOADER_H
#define GL_LOADER_H

/* We pull modern OpenGL function pointer typedefs from <GL/glext.h>
 * and resolve them at runtime via glfwGetProcAddress. We keep GLFW
 * from pulling in its own gl headers, and we include them ourselves
 * so the symbols are declared in a controlled order. */
#include <GL/gl.h>
#include <GL/glext.h>

extern PFNGLGENBUFFERSPROC              glGenBuffers;
extern PFNGLBINDBUFFERPROC              glBindBuffer;
extern PFNGLBUFFERDATAPROC              glBufferData;
extern PFNGLBUFFERSUBDATAPROC           glBufferSubData;
extern PFNGLDELETEBUFFERSPROC           glDeleteBuffers;
extern PFNGLGENVERTEXARRAYSPROC         glGenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC         glBindVertexArray;
extern PFNGLDELETEVERTEXARRAYSPROC      glDeleteVertexArrays;
extern PFNGLVERTEXATTRIBPOINTERPROC     glVertexAttribPointer;
extern PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
extern PFNGLCREATESHADERPROC            glCreateShader;
extern PFNGLSHADERSOURCEPROC            glShaderSource;
extern PFNGLCOMPILESHADERPROC           glCompileShader;
extern PFNGLGETSHADERIVPROC             glGetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC        glGetShaderInfoLog;
extern PFNGLDELETESHADERPROC            glDeleteShader;
extern PFNGLCREATEPROGRAMPROC           glCreateProgram;
extern PFNGLATTACHSHADERPROC            glAttachShader;
extern PFNGLLINKPROGRAMPROC             glLinkProgram;
extern PFNGLGETPROGRAMIVPROC            glGetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC       glGetProgramInfoLog;
extern PFNGLUSEPROGRAMPROC              glUseProgram;
extern PFNGLDELETEPROGRAMPROC           glDeleteProgram;
extern PFNGLGETUNIFORMLOCATIONPROC      glGetUniformLocation;
extern PFNGLUNIFORMMATRIX4FVPROC        glUniformMatrix4fv;
extern PFNGLUNIFORM1FPROC               glUniform1f;
extern PFNGLUNIFORM3FPROC               glUniform3f;
extern PFNGLUNIFORM3FVPROC              glUniform3fv;
extern PFNGLUNIFORM1IPROC               glUniform1i;
extern PFNGLUNIFORM1IVPROC              glUniform1iv;

/* Compute shader / SSBO functions (require GL 4.3+). */
extern PFNGLDISPATCHCOMPUTEPROC         glDispatchCompute;
extern PFNGLMEMORYBARRIERPROC           glMemoryBarrier;
extern PFNGLBINDBUFFERBASEPROC          glBindBufferBase;
extern PFNGLGETBUFFERSUBDATAPROC        glGetBufferSubData;
extern PFNGLMAPBUFFERRANGEPROC          glMapBufferRange;
extern PFNGLUNMAPBUFFERPROC             glUnmapBuffer;

/* Returns 1 on success, 0 if any function failed to resolve.
 * Must be called after a GL context is current. */
int gl_load(void);

#endif

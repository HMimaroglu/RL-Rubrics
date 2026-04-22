#ifndef RENDERER_H
#define RENDERER_H

#include "cube.h"
#include "linmath.h"

/* Init/shutdown need a current GL context. Returns 1 on success. */
int  renderer_init(void);
void renderer_shutdown(void);

/* Draw the cube using its current sticker state. */
void renderer_draw(const Cube *cube, Mat4 view, Mat4 proj);

#endif

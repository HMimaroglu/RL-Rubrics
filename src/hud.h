#ifndef HUD_H
#define HUD_H

/* Tiny 2D HUD overlay: solid-color quads in NDC space.
 * Coordinates are clip space ([-1, 1] x [-1, 1]).
 * y = +1 is top of window. */

int  hud_init(void);
void hud_shutdown(void);

void hud_begin_frame(void);
void hud_quad(float x0, float y0, float x1, float y1,
              float r, float g, float b);
void hud_end_frame(void);

#endif

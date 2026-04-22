#ifndef CUBE_H
#define CUBE_H

#include <stdint.h>

/* Face indices, also used as color indices into the renderer's palette. */
enum {
    FACE_U = 0,  /* White  (top)    */
    FACE_D = 1,  /* Yellow (bottom) */
    FACE_F = 2,  /* Green  (front)  */
    FACE_B = 3,  /* Blue   (back)   */
    FACE_L = 4,  /* Orange (left)   */
    FACE_R = 5,  /* Red    (right)  */
    FACE_COUNT = 6
};

/* 18 quarter/half turns.
 * For face X in {U,D,F,B,L,R}, encoded as 3*face + variant:
 *   variant 0 = X  (90 degrees clockwise looking at the face from outside)
 *   variant 1 = X' (90 degrees counter-clockwise)
 *   variant 2 = X2 (180 degrees)
 */
typedef enum {
    MV_U = 0,  MV_Up = 1,  MV_U2 = 2,
    MV_D = 3,  MV_Dp = 4,  MV_D2 = 5,
    MV_F = 6,  MV_Fp = 7,  MV_F2 = 8,
    MV_B = 9,  MV_Bp = 10, MV_B2 = 11,
    MV_L = 12, MV_Lp = 13, MV_L2 = 14,
    MV_R = 15, MV_Rp = 16, MV_R2 = 17,
    MV_COUNT = 18
} Move;

/* Cube state: 54 stickers, laid out as state[face*9 + pos].
 *
 * For each face, pos is 0..8 in reading order from the face's OWN outside viewer:
 *     0 1 2
 *     3 4 5
 *     6 7 8
 *
 * Face viewer conventions (outward normal N, "up" U, "right" R of the viewer):
 *   U:  N=+y  U=-z  R=+x     (F face is at the bottom of U's view)
 *   D:  N=-y  U=+z  R=+x     (F face is at the top of D's view)
 *   F:  N=+z  U=+y  R=+x
 *   B:  N=-z  U=+y  R=-x
 *   L:  N=-x  U=+y  R=+z
 *   R:  N=+x  U=+y  R=-z
 *
 * Solved state: state[face*9 + i] == face for all i.
 */
typedef struct {
    uint8_t stickers[54];
} Cube;

void cube_reset(Cube *c);
void cube_apply_move(Cube *c, Move m);
int  cube_is_solved(const Cube *c);
void cube_scramble(Cube *c, int n_moves);

#endif

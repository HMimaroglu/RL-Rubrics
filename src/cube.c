#include "cube.h"
#include <stdlib.h>

/* Cycle values: value at a -> b, b -> c, c -> d, d -> a. */
static void cycle4(uint8_t *s, int a, int b, int c, int d) {
    uint8_t t = s[d];
    s[d] = s[c];
    s[c] = s[b];
    s[b] = s[a];
    s[a] = t;
}

/* Each base move is a 90 deg CW turn of one face.
 * CCW (prime) and 180 are built by repeating the CW turn.
 *
 * Permutations derived geometrically from the face-viewer conventions
 * documented in cube.h. Verified by: U U' is identity, R R R R is identity,
 * and the classic "sexy move" (R U R' U') has order 6.
 */

static void move_U(uint8_t *s) {
    /* U face itself */
    cycle4(s, 0*9+0, 0*9+2, 0*9+8, 0*9+6);
    cycle4(s, 0*9+1, 0*9+5, 0*9+7, 0*9+3);
    /* Top row of F -> L -> B -> R -> F */
    cycle4(s, 2*9+0, 4*9+0, 3*9+0, 5*9+0);
    cycle4(s, 2*9+1, 4*9+1, 3*9+1, 5*9+1);
    cycle4(s, 2*9+2, 4*9+2, 3*9+2, 5*9+2);
}

static void move_D(uint8_t *s) {
    cycle4(s, 1*9+0, 1*9+2, 1*9+8, 1*9+6);
    cycle4(s, 1*9+1, 1*9+5, 1*9+7, 1*9+3);
    /* Bottom row of F -> R -> B -> L -> F */
    cycle4(s, 2*9+6, 5*9+6, 3*9+6, 4*9+6);
    cycle4(s, 2*9+7, 5*9+7, 3*9+7, 4*9+7);
    cycle4(s, 2*9+8, 5*9+8, 3*9+8, 4*9+8);
}

static void move_F(uint8_t *s) {
    cycle4(s, 2*9+0, 2*9+2, 2*9+8, 2*9+6);
    cycle4(s, 2*9+1, 2*9+5, 2*9+7, 2*9+3);
    /* Ring: U bottom -> R left -> D top (reversed) -> L right (reversed) -> U bottom */
    cycle4(s, 0*9+6, 5*9+0, 1*9+2, 4*9+8);
    cycle4(s, 0*9+7, 5*9+3, 1*9+1, 4*9+5);
    cycle4(s, 0*9+8, 5*9+6, 1*9+0, 4*9+2);
}

static void move_B(uint8_t *s) {
    cycle4(s, 3*9+0, 3*9+2, 3*9+8, 3*9+6);
    cycle4(s, 3*9+1, 3*9+5, 3*9+7, 3*9+3);
    /* Ring: U top -> L left (reversed) -> D bottom (reversed) -> R right -> U top */
    cycle4(s, 0*9+0, 4*9+6, 1*9+8, 5*9+2);
    cycle4(s, 0*9+1, 4*9+3, 1*9+7, 5*9+5);
    cycle4(s, 0*9+2, 4*9+0, 1*9+6, 5*9+8);
}

static void move_L(uint8_t *s) {
    cycle4(s, 4*9+0, 4*9+2, 4*9+8, 4*9+6);
    cycle4(s, 4*9+1, 4*9+5, 4*9+7, 4*9+3);
    /* Ring: U left -> F left -> D left -> B right (reversed) -> U left */
    cycle4(s, 0*9+0, 2*9+0, 1*9+0, 3*9+8);
    cycle4(s, 0*9+3, 2*9+3, 1*9+3, 3*9+5);
    cycle4(s, 0*9+6, 2*9+6, 1*9+6, 3*9+2);
}

static void move_R(uint8_t *s) {
    cycle4(s, 5*9+0, 5*9+2, 5*9+8, 5*9+6);
    cycle4(s, 5*9+1, 5*9+5, 5*9+7, 5*9+3);
    /* Ring: U right -> B left (reversed) -> D right -> F right -> U right */
    cycle4(s, 0*9+8, 3*9+0, 1*9+8, 2*9+8);
    cycle4(s, 0*9+5, 3*9+3, 1*9+5, 2*9+5);
    cycle4(s, 0*9+2, 3*9+6, 1*9+2, 2*9+2);
}

typedef void (*MoveFn)(uint8_t *);
static const MoveFn base_moves[6] = {
    move_U, move_D, move_F, move_B, move_L, move_R
};

void cube_reset(Cube *c) {
    for (int f = 0; f < FACE_COUNT; f++) {
        for (int i = 0; i < 9; i++) {
            c->stickers[f * 9 + i] = (uint8_t)f;
        }
    }
}

void cube_apply_move(Cube *c, Move m) {
    int face = (int)m / 3;
    int variant = (int)m % 3;
    int turns = (variant == 0) ? 1 : (variant == 1) ? 3 : 2;
    for (int i = 0; i < turns; i++) {
        base_moves[face](c->stickers);
    }
}

int cube_is_solved(const Cube *c) {
    for (int f = 0; f < FACE_COUNT; f++) {
        uint8_t color = c->stickers[f * 9];
        for (int i = 1; i < 9; i++) {
            if (c->stickers[f * 9 + i] != color) return 0;
        }
    }
    return 1;
}

void cube_scramble(Cube *c, int n_moves) {
    for (int i = 0; i < n_moves; i++) {
        cube_apply_move(c, (Move)(rand() % MV_COUNT));
    }
}

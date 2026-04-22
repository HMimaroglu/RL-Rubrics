#ifndef LINMATH_H
#define LINMATH_H

#include <math.h>

typedef struct { float x, y, z; } Vec3;
typedef struct { float m[16]; } Mat4;  /* column-major, OpenGL convention */

static inline Vec3 v3(float x, float y, float z) { Vec3 v = {x, y, z}; return v; }
static inline Vec3 v3_add(Vec3 a, Vec3 b)   { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 v3_sub(Vec3 a, Vec3 b)   { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 v3_scale(Vec3 a, float s){ return v3(a.x*s, a.y*s, a.z*s); }
static inline float v3_dot(Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 v3_cross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x);
}
static inline Vec3 v3_normalize(Vec3 a) {
    float l2 = v3_dot(a, a);
    if (l2 < 1e-12f) return v3(0, 0, 0);
    return v3_scale(a, 1.0f / sqrtf(l2));
}

static inline Mat4 mat4_identity(void) {
    Mat4 m = {{0}};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

/* r = a * b, both column-major. */
static inline Mat4 mat4_mul(Mat4 a, Mat4 b) {
    Mat4 r = {{0}};
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a.m[row + k*4] * b.m[k + col*4];
            r.m[row + col*4] = s;
        }
    }
    return r;
}

/* Right-handed perspective. fovy in radians. Maps z to NDC [-1, 1]. */
static inline Mat4 mat4_perspective(float fovy, float aspect, float znear, float zfar) {
    Mat4 m = {{0}};
    float f = 1.0f / tanf(fovy * 0.5f);
    m.m[0]  = f / aspect;
    m.m[5]  = f;
    m.m[10] = (zfar + znear) / (znear - zfar);
    m.m[11] = -1.0f;
    m.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return m;
}

/* Right-handed look-at (gluLookAt equivalent). */
static inline Mat4 mat4_lookat(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = v3_normalize(v3_sub(target, eye));
    Vec3 s = v3_normalize(v3_cross(f, up));
    Vec3 u = v3_cross(s, f);

    Mat4 m = mat4_identity();
    m.m[0]  =  s.x; m.m[4]  =  s.y; m.m[8]  =  s.z;
    m.m[1]  =  u.x; m.m[5]  =  u.y; m.m[9]  =  u.z;
    m.m[2]  = -f.x; m.m[6]  = -f.y; m.m[10] = -f.z;
    m.m[12] = -v3_dot(s, eye);
    m.m[13] = -v3_dot(u, eye);
    m.m[14] =  v3_dot(f, eye);
    return m;
}

#endif

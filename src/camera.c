#include "camera.h"
#include <math.h>

#define PI 3.14159265358979323846f

void camera_init(Camera *c) {
    c->azimuth   = PI / 5.0f;   /* a slight rotation so F and R faces are both visible */
    c->elevation = PI / 6.0f;   /* looking down ~30 deg */
    c->distance  = 8.0f;
}

Vec3 camera_position(const Camera *c) {
    float ce = cosf(c->elevation);
    return v3(c->distance * ce * sinf(c->azimuth),
              c->distance * sinf(c->elevation),
              c->distance * ce * cosf(c->azimuth));
}

Mat4 camera_view(const Camera *c) {
    return mat4_lookat(camera_position(c), v3(0, 0, 0), v3(0, 1, 0));
}

void camera_orbit(Camera *c, float d_azimuth, float d_elevation) {
    c->azimuth += d_azimuth;
    c->elevation += d_elevation;
    float lim = (PI * 0.5f) - 0.01f;
    if (c->elevation >  lim) c->elevation =  lim;
    if (c->elevation < -lim) c->elevation = -lim;
}

void camera_zoom(Camera *c, float factor) {
    c->distance *= factor;
    if (c->distance < 3.5f)  c->distance = 3.5f;
    if (c->distance > 40.0f) c->distance = 40.0f;
}

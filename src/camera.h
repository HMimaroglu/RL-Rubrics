#ifndef CAMERA_H
#define CAMERA_H

#include "linmath.h"

/* Orbit camera around the origin. */
typedef struct {
    float azimuth;   /* radians, 0 = looking down +z */
    float elevation; /* radians, clamped to [-pi/2 + eps, pi/2 - eps] */
    float distance;
} Camera;

void  camera_init(Camera *c);
Vec3  camera_position(const Camera *c);
Mat4  camera_view(const Camera *c);
void  camera_orbit(Camera *c, float d_azimuth, float d_elevation);
void  camera_zoom(Camera *c, float factor);

#endif

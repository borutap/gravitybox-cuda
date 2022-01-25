#ifndef PARTICLECOMMON_H
#define PARTICLECOMMON_H
#include "particle.h"
#include "glm/glm.hpp"

__device__ __host__ glm::vec2 calculate_gravity_force();
__device__ __host__ glm::vec2 calculate_storm_force(Particle &particle);
__device__ __host__ glm::vec2 calculate_outward_force(Particle &particle);
__device__ __host__ glm::vec2 calculate_circular_force(Particle &particle);
__device__ __host__ void bounce_off_ground(Particle &particle, float bounce_factor);
__device__ __host__ void collision_besides_ground(Particle &particle, float walls_ceiling_margin, float turn_factor);
__device__ __host__ void limit_speed(Particle &particle, float speed_limit);

enum Force { gravity, storm, circular, outward };

#endif
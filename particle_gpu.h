#ifndef PARTICLEGPU_H
#define PARTICLEGPU_H

#include "particle.h"
#include "particle_common.h"
#include "glm/glm.hpp"


__global__ void kernel_update(Particle *particles, glm::mat4 *trans, int n, float dt, float t,                                
                              Force selected_force, float speed_limit, float bounce_factor,
                              float walls_ceiling_margin, float turn_factor);

// utils
void copy_particle_structure_to_device(Particle **particles, Particle **d_pointer, int n);
void copy_trans_matrix_to_device(glm::mat4 **mat, glm::mat4 **d_mat, int n);
void copy_trans_matrix_to_host(glm::mat4 **mat, glm::mat4 **d_mat, int n);

#endif
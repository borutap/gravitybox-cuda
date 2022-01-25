#ifndef PARTICLECPU_H
#define PARTICLECPU_H

#include "particle.h"
#include "glm/glm.hpp"
#include "particle_common.h"

namespace cpu
{
    void update(Particle *particles, glm::mat4 *trans, int n, float dt, float t,                                
                Force selected_force, float speed_limit, float bounce_factor,
                float walls_ceiling_margin, float turn_factor);
}

#endif
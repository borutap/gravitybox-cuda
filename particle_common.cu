#include "particle_common.h"

// sztorm
__device__ __host__ glm::vec2 calculate_storm_force(Particle &particle)
{
    return glm::vec2(50*particle.y/(particle.x*particle.x+particle.y*particle.y),
                    -50*particle.x/(particle.x*particle.x+particle.y*particle.y));
}

// do zewnatrz
__device__ __host__ glm::vec2 calculate_outward_force(Particle &particle)
{
    return glm::vec2(4*particle.x*particle.y*particle.y, 4*particle.x*particle.x*particle.y);
}

// rotacyjne
__device__ __host__ glm::vec2 calculate_circular_force(Particle &particle)
{
    return glm::vec2(particle.y / 2, -particle.x  / 2);
    //return glm::vec2(particle.x - particle.x*particle.x*particle.x, -particle.x  / 2);
}

__device__ __host__ glm::vec2 calculate_gravity_force()
{
    // -masa * grawitacja
    return glm::vec2(0, -1.0f * 9.81f);
}

__device__ __host__ glm::vec2 calculate_oscillator_force(Particle &particle)
{
    float k = 30.0f;
    return glm::vec2(-k * particle.x, -k * particle.y);
}

__device__ __host__ void limit_speed(Particle &particle, float speed_limit)
{
    float speed = glm::sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
    if (speed <= speed_limit)
        return;
    particle.vx = (particle.vx / speed) * speed_limit;
    particle.vy = (particle.vy / speed) * speed_limit;
}

__device__ __host__ void bounce_off_ground(Particle &particle, float bounce_factor)
{
    if (particle.y >= -1.0f || particle.vy >= 0.0f)
        return;

    //particle.y = -1; // wyrownuje do jednej linii odbite
    particle.vy *= -bounce_factor;
    //apply_damping_forces(particle, 0.9f);
}

__device__ __host__ void collision_besides_ground(Particle &particle, float walls_ceiling_margin, float turn_factor)
{
    if (particle.x < -1.0f + walls_ceiling_margin && particle.vx < 0.0f)
    {
        particle.vx *= -turn_factor;
    }

    if (particle.x > 1.0f - walls_ceiling_margin && particle.vx > 0.0f)
    {
        particle.vx *= -turn_factor;
    }

    if (particle.y > 1.0f - walls_ceiling_margin && particle.vy > 0.0f)
    {
        particle.vy *= -turn_factor;
    }
}
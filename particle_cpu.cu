#include "particle.h"
#include "particle_cpu.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <iostream>

void keep_within_xy(Particle &particle)
{
    float turn_factor = 0.1f;
    float margin = 0.03f;
    if (particle.x < -1.0f + margin)
        particle.vx += turn_factor;

    if (particle.x > 1.0f - margin)
        particle.vx -= turn_factor;

    if (particle.y < -1.0f + margin)
        particle.vy += turn_factor;

    if (particle.y > 1.0f - margin)
        particle.vy -= turn_factor;
}

glm::vec2 calculate_oscillator_force(Particle &particle)
{
    glm::vec2 r0(0.0f, 0.0f);
    float k = 1.0f;
    return glm::vec2(-k * (particle.x - r0.x), -k * (particle.y - r0.y));
}

// zalozenie k = 1
float oscilator_particle_energy(Particle &particle)
{
    float energy = particle.vx * particle.vx / 2;
    energy += particle.vy * particle.vy / 2;        
    energy += (particle.x * particle.x + particle.y * particle.y) / 2;
    return energy;
}

glm::vec2 calculate_test_force(Particle &particle, float t)
{
    return glm::vec2(-(particle.x-4)*(particle.x+3), -5*particle.y+5*t*t+2*t);
}

void cpu::update(Particle *particles, glm::mat4 *trans, int n, float dt, float t,                                
                 Force selected_force, float speed_limit, float bounce_factor,
                 float walls_ceiling_margin, float turn_factor)
{
    float mass = 1.0f;
    for (int i = 0; i < n; i++)
    {
        Particle &particle = particles[i];

        glm::vec2 force;
        if (selected_force == Force::gravity)
        {
            force = calculate_gravity_force();
        }
        else if (selected_force == Force::storm)
        {
            force = calculate_storm_force(particle);
        }
        else if (selected_force == Force::circular)
        {
            force = calculate_circular_force(particle);
        }
        else if (selected_force == Force::outward)
        {
            force = calculate_outward_force(particle);
        }

        particle.x += dt * particle.vx;
        particle.y += dt * particle.vy;

        particle.vx += dt / mass * force.x;
        particle.vy += dt / mass * force.y;

        if (selected_force != Force::gravity)
        {
            limit_speed(particle, speed_limit);        
        }
        bounce_off_ground(particle, bounce_factor);
        collision_besides_ground(particle, walls_ceiling_margin, turn_factor); 

        trans[i] = glm::translate(glm::mat4(1.0f), glm::vec3(particle.x, particle.y, 0.0f));
    }
}

void apply_damping_forces(Particle &particle, float damping_factor)
{
    if (particle.vy >= 0)
        return;

    particle.vx *= damping_factor;
    particle.vy *= damping_factor;
}
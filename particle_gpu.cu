#include "particle_gpu.h"
#include "glm/gtc/matrix_transform.hpp"

__global__ void kernel_update(Particle *particles, glm::mat4 *trans, int n, float dt, float t,                                
                              Force selected_force, float speed_limit, float bounce_factor,
                              float walls_ceiling_margin, float turn_factor)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }
    Particle &particle = particles[index];
    float mass = 1.0f;
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

    trans[index] = glm::translate(glm::mat4(1.0f), glm::vec3(particle.x, particle.y, 0.0f));
}

void copy_particle_structure_to_device(Particle **particles, Particle **d_pointer, int n)
{
    size_t size = sizeof(Particle);
    cudaMalloc(d_pointer, n * size);
    cudaMemcpy(*d_pointer, *particles, n * size, cudaMemcpyHostToDevice);    
}

void copy_trans_matrix_to_device(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    size_t size = sizeof(glm::mat4);
    cudaMalloc(d_mat, n * size);
    cudaMemcpy(*d_mat, *mat, n * size, cudaMemcpyHostToDevice);
}

void copy_trans_matrix_to_host(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    cudaMemcpy(*mat, *d_mat, n *  sizeof(glm::mat4), cudaMemcpyDeviceToHost);
}
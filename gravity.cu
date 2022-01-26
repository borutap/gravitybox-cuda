#include <GL/glew.h>
#include <GLFW/glfw3.h>

/* Using SDL2 for the base window and OpenGL context init */
#include <SDL2/SDL.h>

#include "learnopengl/shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <iostream>
#include "particle.h"
#include "particle_cpu.h"
#include "particle_gpu.h"
#include "particle_common.h"
#include "parameters.h"
#include "utils.h"

using namespace std;

// settings
unsigned int SCR_WIDTH = 1024;
unsigned int SCR_HEIGHT = 768;
const int N = 100000;
const bool RUN_CPU = false;
const bool RUN_LOGGER = true;

int VERTICES_IN_PARTICLE;

// float vertexData[] = {
//     // positions  // colors
//     0.005f, 0.0f, 1.0f, 1.0f, 0.0f,
//     0.0038242109364224424f, 0.003221088436188455f, 1.0f, 1.0f, 0.0f,
//     0.0008498357145012052f, 0.004927248649942301f, 1.0f, 1.0f, 0.0f,
//     -0.0025242305229992855f, 0.00431604683324437f, 1.0f, 1.0f, 0.0f,
//     -0.00471111170334329f, 0.0016749407507795256f, 1.0f, 1.0f, 0.0f,
//     -0.004682283436453982f, -0.0017539161384480992f, 1.0f, 1.0f, 0.0f,
//     -0.0024513041067034972f, -0.004357878862067941f, 1.0f, 1.0f, 0.0f,
//     0.0009325618471128788f, -0.0049122630631216625f, 1.0f, 1.0f, 0.0f,
//     0.003877829392551251f, -0.0031563331893616044f, 1.0f, 1.0f, 0.0f,
// };
float *vertexData;

GLuint particleVAO, particleVBO;
GLuint transformationVBO;
glm::vec2 *start_translations;
// nowe
glm::vec2 *start_speeds;
//
glm::mat4 *trans_matrices;

Particle *d_particles;
glm::mat4 *d_trans;

Logger *logger = nullptr;

void init_transform_resources();
void render(SDL_Window* window, Shader* shader);

void init_vertex_data()
{    
    vector<float> circle_points = get_circle_points(0.00125f, 0.7f);
    int all_points = circle_points.size();
    VERTICES_IN_PARTICLE = all_points / 2;
    cout << "VERTICES_IN_PARTICLE: " << VERTICES_IN_PARTICLE << endl;
    vertexData = new float[all_points + VERTICES_IN_PARTICLE * 3];
    int it = 0;
    for (int i = 0; i < circle_points.size(); i++)
    {
        vertexData[it] = circle_points[i];
        i++;
        vertexData[it + 1] = circle_points[i];
        vertexData[it + 2] = 1.0f;
        vertexData[it + 3] = 1.0f;
        vertexData[it + 4] = 0.0f;
        it += 5;
    }
}

Shader* init_resources()
{
    init_vertex_data();
    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader* shader = new Shader("particle.vs", "particle.fs");
    trans_matrices = new glm::mat4[N];
    start_translations = new glm::vec2[N];
    if (RUN_LOGGER)
    {
        logger = new Logger();   
        logger->start_timed_measurement("generating starting translations");
    }
    for (int i = 0; i < N; i++)
    {                
        glm::vec3 translation;
        translation.x = glm::linearRand(-1.0f, 1.0f);
        translation.y = glm::linearRand(-1.0f, 1.0f);
        trans_matrices[i] = glm::translate(glm::mat4(1.0f), translation);
        start_translations[i] = translation;
    }
    if (RUN_LOGGER)
    {
        logger->end_timed_measurement();
    }
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * VERTICES_IN_PARTICLE * 5, vertexData, GL_STATIC_DRAW);
    // wspolne dane
    glEnableVertexAttribArray(0);
    // pozycja
    glVertexAttribPointer(
        0,                  // nr atrybutu - zgodny z layoutem w .vs
        2,                  // rozmiar
        GL_FLOAT,           // typ
        GL_FALSE,           // czy znormalizowane
        5 * sizeof(float),  // stride - odstep do kolejnej wartosci
        (void*)0            // offset jezeli jest we wspolnej tablicy
    );
    glEnableVertexAttribArray(1);
    // kolor
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    
    init_transform_resources();

    return shader;
}

void init_transform_resources()
{    
    // kazda czastka ma swoja macierz transformacji
    glGenBuffers(1, &transformationVBO);
    glBindBuffer(GL_ARRAY_BUFFER, transformationVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrices[0], GL_DYNAMIC_DRAW);

    glBindVertexArray(particleVAO);
    // set attribute pointers for matrix (4 times vec4)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
    glVertexAttribDivisor(6, 1);

    glBindVertexArray(0);
}

void set_initial_particle_position(Particle &particle, float *vertexData, glm::vec2 &translation)
{
    float origin_x = 0.0f;
    //float origin_y = (vertexData[11] + vertexData[1]) / 2;
    float origin_y = 0.0f;
    particle.x = origin_x + translation.x;
    particle.y = origin_y + translation.y;    
}

Particle *init_particle_structure(int n, float *vertexData, glm::vec2 *start_translations)
{
    if (RUN_LOGGER)
    {
        logger->start_timed_measurement("settings particle structures");
    }
    Particle *particles = new Particle[n];
    start_speeds = new glm::vec2[n];
    for (int i = 0; i < n; i++)
    {
        particles[i].mass = glm::linearRand(1.0f, 10.0f);
        particles[i].vx = glm::linearRand(-0.5f, 0.5f);
        particles[i].vy = glm::linearRand(-0.5f, 0.5f);
        // nowe
        start_speeds[i] = glm::vec2(particles[i].vx, particles[i].vy);
        //
        set_initial_particle_position(particles[i], vertexData, start_translations[i]);        
    }
    if (RUN_LOGGER)
    {
        logger->end_timed_measurement();
    }

    return particles;
}

void keep_within_bounds(Particle *particles, int index,
                        float margin, float turn_factor)
{
    Particle &particle = particles[index];

    if (particle.x < -1.0f + margin)
        particle.vx += turn_factor;    

    if (particle.x > 1.0f - margin)
        particle.vx -= turn_factor;

    if (particle.y < -1.0f + margin)
        particle.vy += turn_factor;    

    if (particle.y > 1.0f - margin)
        particle.vy -= turn_factor;
}

void limit_speed(Particle *particles, int index, float speed_limit)
{
    Particle &particle = particles[index];

    float speed = glm::sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
    if (speed <= speed_limit)
        return;
    particle.vx = (particle.vx / speed) * speed_limit;
    particle.vy = (particle.vy / speed) * speed_limit;
}

void move(Particle *particles, glm::mat4 *trans, int n, float margin, float turn_factor)
{
    for (int i = 0; i < n; i++)
    {
        limit_speed(particles, i, 0.005f);
        keep_within_bounds(particles, i, margin, turn_factor);
        Particle &particle = particles[i];
        particle.x += particle.vx;
        particle.y += particle.vy;

        // float angle = glm::atan(particle.dy / particle.dx);
        // float pi = glm::pi<float>();
        // if (particle.dx <= 0)
        // {
        //     angle += pi / 2;
        // }        
        // else
        // {
        //     angle -= pi / 2;
        // }

        auto transformation = glm::translate(glm::mat4(1.0f), glm::vec3(particle.x, particle.y, 0.0f));
        // transformation = glm::rotate(transformation, angle, glm::vec3(0.0f, 0.0f, 1.0f));
        trans[i] = transformation;
    }
}

void main_loop(SDL_Window* window, Shader* shader)
{
    Particle *particles = init_particle_structure(N, vertexData, start_translations);    

    char title[50];
    int frame_time;
    float time = 1.0f / 144.0f;
    float dt = 1.0f / 240.0f;
    bool played = false;    
    if (!RUN_CPU)
    {
        if (RUN_LOGGER)
        {
            logger->start_timed_measurement("copying particle structures to device");
        }
        copy_particle_structure_to_device(&particles, &d_particles, N);
        if (RUN_LOGGER)
        {
            logger->end_timed_measurement();
            logger->start_timed_measurement("copying translation matrices to device");
        }
        copy_trans_matrix_to_device(&trans_matrices, &d_trans, N);
        if (RUN_LOGGER)
        {
            logger->end_timed_measurement();
        }
    }
    dim3 num_threads(1024);
    dim3 num_blocks(N / 1024 + 1);   

    Parameters p;
    p.set_default();
    p.print_values();  

    while (true)
    {
        Uint32 frame_start = SDL_GetTicks();

        SDL_Event ev;        
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_QUIT)
            {    
                delete[] particles;        
                return;
            }	
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_p)
            {
                played = !played;                
            }
            else if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_r)
            {
                for (int i = 0; i < N; i++)
                {
                    set_initial_particle_position(particles[i], vertexData, start_translations[i]);
                    particles[i].vx = start_speeds[i].x;
                    particles[i].vy = start_speeds[i].y; 
                }            
                if (!RUN_CPU)
                {
                    copy_particle_structure_to_device(&particles, &d_particles, N);
                }
            }
            else if (ev.type == SDL_KEYDOWN && p.handle_keyboard(ev))
            {
                p.print_values();
            }            
            if (ev.type == SDL_WINDOWEVENT &&
                    ev.window.event == SDL_WINDOWEVENT_RESIZED)
            {
                SCR_WIDTH = ev.window.data1;
                SCR_HEIGHT = ev.window.data2;
                glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
            }        
        }
        if (!played)
        {
            render(window, shader);
            continue;
        }
        if (RUN_CPU)
        {
            cpu::update(particles, trans_matrices, N, dt, time,
                 p.selected_force, p.speed_limit, p.bounce_factor, p.walls_ceiling_margin, p.turn_factor);
        }
        else
        {
            kernel_update<<<num_blocks, num_threads>>>(d_particles, d_trans, N, dt, time,
                 p.selected_force, p.speed_limit, p.bounce_factor, p.walls_ceiling_margin, p.turn_factor);
            copy_trans_matrix_to_host(&trans_matrices, &d_trans, N);
        }
        

        glBindBuffer(GL_ARRAY_BUFFER, transformationVBO);
        glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrices[0], GL_DYNAMIC_DRAW);
        // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 

        render(window, shader);
        time += 1.0f / 144.0f;
        
        frame_time = SDL_GetTicks() - frame_start;

        // przy wylaczonym v-sync moze byc 0
        frame_time = frame_time == 0 ? 1 : frame_time;
        snprintf(title, 50, "Gravitation box (%.2f FPS)", 1000.0/frame_time);
        SDL_SetWindowTitle(window, title);
    }
    
}

void render(SDL_Window* window, Shader* shader)
{
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
    // draw N instanced particles 
    (*shader).use();
    glBindVertexArray(particleVAO);
    // N czasteczek po VERTEX_IN_PARTICLE wierzcholkow
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, VERTICES_IN_PARTICLE, N);
    glBindVertexArray(0); // zrywa binding

    SDL_GL_SwapWindow(window);
}

void free_resources(SDL_Window* window, Shader *shader)
{
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleVBO);
    glDeleteBuffers(1, &transformationVBO);
    delete shader;
    delete[] trans_matrices;
    delete[] start_translations;
    // nowe
    delete[] start_speeds;
    //
    if (RUN_LOGGER)
    {
        logger->close_file();
        delete logger;
    }
    delete[] vertexData;    
    cudaFree(d_particles);
    cudaFree(d_trans);

    
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    cout << "Wyczyszczono" << endl;
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* window = SDL_CreateWindow("Gravitation box",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		SCR_WIDTH, SCR_HEIGHT,
		SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);

    if (window == NULL)
    {
        cerr << "Error: can't create window: " << SDL_GetError() << endl;
        return EXIT_FAILURE;
    }
    SDL_GL_CreateContext(window);

    /* Extension wrangler initialising */
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK)
    {
        cerr << "Error: glewInit: " << glewGetErrorString(glew_status) << endl;
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 1);
    //SDL_GL_SetSwapInterval(0); // wylacza vsync
    if (SDL_GL_CreateContext(window) == NULL)
    {
        cerr << "Error: SDL_GL_CreateContext: " << SDL_GetError() << endl;
        return EXIT_FAILURE;
    }

    Shader* shader = init_resources();

    main_loop(window, shader);

    free_resources(window, shader);
    return EXIT_SUCCESS;
}
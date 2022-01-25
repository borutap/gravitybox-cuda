#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <SDL2/SDL.h>
#include "particle_common.h"

class Parameters
{
public:    
    Force selected_force;
    float speed_limit;
    float bounce_factor;
    float walls_ceiling_margin;
    float turn_factor;

    void set_default();
    void print_values();
    bool handle_keyboard(SDL_Event &ev);

private:
    char is_force_selected(Force force);
};

#endif
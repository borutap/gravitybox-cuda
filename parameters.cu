#include "parameters.h"
#include <iostream>

using namespace std;

void Parameters::set_default()
{
    selected_force = Force::storm;
    speed_limit = 2.0f;
    bounce_factor = 0.9f;
    walls_ceiling_margin = 0.02f;
    turn_factor = 0.35f;
}

char Parameters::is_force_selected(Force force)
{
    return force == selected_force ? 'x' : '\0';
}

void Parameters::print_values()
{    
    cout << endl;
    cout << "Pause - p  Restart - r" << endl;
    cout << "Set force field:" << endl;
    cout << "1. Gravity [" << is_force_selected(Force::gravity) << "]" << endl;
    cout << "2. Storm [" << is_force_selected(Force::storm) << "]" << endl;
    cout << "3. Outward [" << is_force_selected(Force::outward) << "]" << endl;    
    cout << "4. Circular [" << is_force_selected(Force::circular) << "]" << endl;   
    cout << "5. Oscillator [" << is_force_selected(Force::oscillator) << "]" << endl;  
    cout << "Speed limit = " << speed_limit << " (-q +w) doesn't apply to gravity" <<  endl;
    cout << "Bounce factor = " << bounce_factor << " (-a +s)" <<  endl;
    cout << "Walls and ceiling margin = " << walls_ceiling_margin << " (-z +x)" <<  endl;
    cout << "Turn factor on wall = " << turn_factor << " (-d +f)" <<  endl;
    cout << "Set defaults (k)" << endl;
}

bool Parameters::handle_keyboard(SDL_Event &ev)
{
    switch (ev.key.keysym.sym)
    {
        case SDLK_1:
            selected_force = Force::gravity;
        break;

        case SDLK_2:
            selected_force = Force::storm;
        break;

        case SDLK_3:
            selected_force = Force::outward;
        break;

        case SDLK_4:
            selected_force = Force::circular;
        break;

        case SDLK_5:
            selected_force = Force::oscillator;
        break;

        case SDLK_q:
            speed_limit -= 0.1f;
        break;

        case SDLK_w:
            speed_limit += 0.1f;
        break;

        case SDLK_a:
            bounce_factor -= 0.05f;
        break;

        case SDLK_s:
            bounce_factor += 0.05f;
        break;

        case SDLK_z:
            walls_ceiling_margin -= 0.01f;
        break;

        case SDLK_x:
            walls_ceiling_margin += 0.01f;
        break;

        case SDLK_d:
            turn_factor -= 0.05f;
        break;

        case SDLK_f:
            turn_factor += 0.05f;
        break;

        case SDLK_k:
            set_default();
        break;

        default:
            return false;      
    }
    return true;
}
#include "utils.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"

// zwracamy kolejne x,y punktow tworzacych okrag
std::vector<float> get_circle_points(float radius, float step_size)
{
    std::vector<float> ret;
    float t = 0;
    while (t < 2 * glm::pi<float>())
    {
        ret.push_back(radius * glm::cos(t));
        ret.push_back(radius * glm::sin(t));
        t += step_size;
    }
    return ret;
}
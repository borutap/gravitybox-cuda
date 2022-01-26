#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "particle.h"
#include <string>
#include <fstream>
#include <chrono>
#include "glm/glm.hpp"

class Logger
{
public:
    Logger();
    void start_timed_measurement(std::string title);
    void end_timed_measurement();
    void close_file();
private:
    std::ofstream file_handle;
    std::chrono::_V2::system_clock::time_point start;
    std::string format_time(
        std::chrono::_V2::system_clock::time_point &t);
};

std::vector<float> get_circle_points(float radius, float step_size);
float calc_radius(int k, int n, int b);
void generate_start_translations_circle(glm::mat4 *in_trans_matrices, glm::vec2 *in_start_translations,
                                        float radius, int n);
void generate_start_translations_random(glm::mat4 *in_trans_matrices, glm::vec2 *in_start_translations, int n);

#endif
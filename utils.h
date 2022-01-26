#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "particle.h"
#include <string>
#include <fstream>
#include <chrono>

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

#endif
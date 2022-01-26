#include "utils.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include <iomanip>
#include <ctime>
#include <sstream>
#include <chrono>
#include <iostream>

Logger::Logger()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << "debug_" << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S") << ".csv";
    auto title = oss.str();
    file_handle.open(title);
    file_handle << "Measurement,Time (s)\n";
}

void Logger::start_timed_measurement(std::string title)
{
    file_handle << title << ",";
    start = std::chrono::high_resolution_clock::now();
}

void Logger::end_timed_measurement()
{
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    file_handle << elapsed_seconds.count() << "\n";
}

void Logger::close_file()
{
    file_handle.close();
}

std::string Logger::format_time(std::chrono::_V2::system_clock::time_point &t)
{
    auto in_time_t = std::chrono::system_clock::to_time_t(t);

    return std::ctime(&in_time_t);
}

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
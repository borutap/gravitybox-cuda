#include "utils.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/random.hpp"
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

// returning x, y coordinates making a circle
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

float calc_radius(int k, int n, int b)
{
    if (k > n - b)
    {
        return 1.0f;
    }
    return sqrt(k - 0.5f) / sqrt(n - (b+1) / 2.0f);
}

void generate_start_translations_circle(glm::mat4 *in_trans_matrices, glm::vec2 *in_start_translations,
                                        float radius, int n)
{
    // sunflower generating function
    float alpha = 2.0f;
    int b = glm::round(alpha * glm::sqrt(n));
    float phi = (glm::sqrt(5)+1) / 2.0f; // golden ratio
    for (int k = 1; k <= n; k++)
    {
        float r = calc_radius(k, n, b);
        float theta = 2 * glm::pi<float>() * k / glm::pow(phi, 2);
        glm::vec3 translation;
        translation.x = r * glm::cos(theta);
        translation.y = r * glm::sin(theta);
        in_trans_matrices[k - 1] = glm::translate(glm::mat4(1.0f), translation);
        in_start_translations[k - 1] = translation;
    }
}

void generate_start_translations_random(glm::mat4 *in_trans_matrices, glm::vec2 *in_start_translations, int n)
{
    for (int i = 0; i < n; i++)
    {                
        glm::vec3 translation;
        translation.x = glm::linearRand(-1.0f, 1.0f);
        translation.y = glm::linearRand(-1.0f, 1.0f);
        in_trans_matrices[i] = glm::translate(glm::mat4(1.0f), translation);
        in_start_translations[i] = translation;
    }
}
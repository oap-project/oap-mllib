#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Profiler {
  public:
    Profiler(std::string s) : subject(s) {}

    void startProfile(std::string s = "") {
        action = s;
        std::cout << subject << " (native): start " << action << std::endl;
        startTime = std::chrono::high_resolution_clock::now();
    }

    void endProfile() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - startTime)
                            .count();
        std::cout << subject << " (native): " << action << " took " << (float)duration / 1000
                  << " secs" << std::endl;
    }

    void println(std::string msg) {
        std::cout << subject << " (native): " << msg << std::endl;
    }

  private:
    std::string subject;
    std::string action;
    std::chrono::system_clock::time_point startTime;
};

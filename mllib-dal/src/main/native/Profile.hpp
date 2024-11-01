#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include "Logger.h"

class Profiler {
  public:
    Profiler(const std::string &s) : subject(s) {}

    void startProfile(std::string s = "") {
        action = s;
        logger::println(logger::INFO, "%s (native): start %s", subject.c_str(),
                        action.c_str());
        startTime = std::chrono::high_resolution_clock::now();
    }

    void endProfile() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - startTime)
                            .count();
        logger::println(logger::INFO, "%s (native): start %s took %f secs",
                        subject.c_str(), action.c_str(),
                        (float)duration / 1000);
    }

    void println(std::string msg) {
        logger::println(logger::INFO, "%s (native): %s", subject.c_str(),
                        msg.c_str());
    }

  private:
    std::string subject;
    std::string action;
    std::chrono::system_clock::time_point startTime;
};

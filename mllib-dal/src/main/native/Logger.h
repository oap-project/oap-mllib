#pragma once

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace logger {
std::ofstream logFile;
std::string name;

public:
static Logger &getInstance(std::string name) {
    static std::once_flag flag;
    static Logger instance(name);
    std::call_once(flag, [&name] { instance = Logger(name); });
    return instance;
}

void printLogToFile(const char *format, ...);
void closeFile();

private:
Logger(std::string name) {
    char *path = std::getenv("SPARKJOB_CONFIG_DIR");
    if (path != nullptr) {
        std::cout << "SPARKJOB_CONFIG_DIR Directory: " << path << std::endl;
    } else {
        std::cout << "SPARKJOB_CONFIG_DIR environment variable not found."
                  << std::endl;
    }
    auto filePath = fs::path(path) / fs::path(name);
    std::cout << "file path: " << filePath << std::endl;
    logFile.open(filePath, std::ios::out | std::ios::app);
}
}; // namespace logger

// message type for print functions
enum MessageType {
    DEBUG = 0,
    ASSERT = 1,
    INFO = 2,
    NONE = 3,
    WARN = 4,
    ERROR = 5
};

int print(MessageType message_type, const std::string &msg);
int print(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const std::string &msg);

int printerr(MessageType message_type, const std::string &msg);
int printerr(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const std::string &msg);
}
; // namespace logger

#pragma once

#include "iostream"
#include <cstdarg>
#include <string>

// message type for print functions
enum MessageType {
    NONE = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    DEBUG = 4,
    ASSERT = 5
};

int print(MessageType message_type, const std::string &msg);
int print(MessageType message_type, const char *format, ...);

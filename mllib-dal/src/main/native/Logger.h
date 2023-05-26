#pragma once

#include <cstdarg>
#include <cstring>
#include <string>

namespace logger {
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
}; // namespace logger
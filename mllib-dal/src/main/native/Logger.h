#pragma once

#include <cstdarg>
#include <string>

namespace logger {
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
int println(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const std::string &msg);

int printerr(MessageType message_type, const std::string &msg);
int printerr(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const std::string &msg);
}; // namespace logger

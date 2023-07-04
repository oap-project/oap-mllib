#pragma once

#include <cstdarg>
#include <string>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/common.hpp"

namespace logger{
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
int print(MessageType message_type, const oneapi::dal::table &table);
int println(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const std::string &msg);
};

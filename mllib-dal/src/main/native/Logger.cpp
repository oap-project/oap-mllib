#include <cstdlib>
#include <iomanip>
#include <cstdlib>
#include <iomanip>
#include <tuple>

#include "Logger.h"

namespace logger {

class LoggerLevel {
  public:
    int level;
    LoggerLevel() {
        level = 2;
        if (const char *env_p = std::getenv("OAP_MLLIB_LOGGER_CPP_LEVEL")) {
            level = atoi(env_p);
        }
        if (level > 5 || level < 0 || level == 3) {
            level = 2;
        }
    }
    int get_level() { return level; }
} logger_level;

std::tuple<std::string, bool> get_prefix(MessageType message_type) {
    std::string prefix;
    bool isLoggerEnabled = false;
    if (message_type >= logger_level.get_level()) {
        isLoggerEnabled = true;
    }
bool isLoggerEnabled = true;

std::tuple<std::string, bool> get_prefix(MessageType message_type) {
    std::string prefix;
    bool isLoggerEnabled = false;
    if (message_type >= logger_level.get_level()) {
        isLoggerEnabled = true;
    }
    switch (message_type) {
    case NONE:
        break;
    case INFO:
        prefix = "[INFO]";
        break;
    case WARN:
        prefix = "[WARNING]";
        break;
    case ERROR:
        prefix = "[ERROR]";
        break;
    case DEBUG:
        prefix = "[DEBUG]";
        break;
    case ASSERT:
        prefix = "[ASSERT]";
        break;
    default:
        break;
    }

    return {prefix + " ", isLoggerEnabled};
}

int print2streamFromArgs(MessageType message_type, FILE *stream,
                         const char *format, va_list args) {
    // print prefix
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    fprintf(stream, "%s", prefix.c_str());

    // print message
    int ret = vfprintf(stream, format, args);
    fflush(stream);

    return ret;
}

int print2streamFromArgsln(MessageType message_type, FILE *stream,
                           const char *format, va_list args) {
    // print prefix
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    fprintf(stream, "%s", prefix.c_str());

    // print message
    int ret = vfprintf(stream, format, args);
    fflush(stream);
    fprintf(stream, "\n");
    fflush(stream);

    return ret;
}

int print2stream(MessageType message_type, FILE *stream, const char *format,
                 ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stream, format, args);
    va_end(args);

    return ret;
}

int print2streamln(MessageType message_type, FILE *stream, const char *format,
                   ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stream, format, args);
    va_end(args);

    return ret;
}

int print(MessageType message_type, const std::string &msg) {
    int ret = print2stream(message_type, stdout, msg.c_str());
    return ret;
}

int print(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stdout, format, args);
    va_end(args);
    return ret;
}

int println(MessageType message_type, const std::string &msg) {
    int ret = print2streamln(message_type, stdout, msg.c_str());
    return ret;
}

int println(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stdout, format, args);
    va_end(args);
    return ret;
}

int printerr(MessageType message_type, const std::string &msg) {
    int ret = print2stream(message_type, stderr, msg.c_str());
    return ret;
}

int printerr(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stderr, format, args);
    va_end(args);
    return ret;
}

int printerrln(MessageType message_type, const std::string &msg) {
    int ret = print2streamln(message_type, stderr, msg.c_str());
    return ret;
}

int printerrln(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stderr, format, args);
    va_end(args);
    return ret;
}
}; // namespace logger

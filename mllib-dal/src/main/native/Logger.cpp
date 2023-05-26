#include "Logger.h"
#include <tuple>

std::tuple<std::string, bool> get_prefix(MessageType message_type) {
    std::string prefix;
    bool enable{true};
    switch (message_type) {
        case NONE:
            break;
        case INFO:
            prefix = "[INFO   ]";
            break;
        case WARN:
            prefix = "[WARNING]";
            break;
        case ERROR:
            prefix = "[ERROR  ]";
            break;
        case DEBUG:
            prefix = "[DEBUG  ]";
            break;
        case ASSERT:
            prefix = "[ASSERT ]";
            break;
        default:
            break;
    }
    return {prefix + " ", enable};
}

int print2streamFromArgs(MessageType message_type, FILE *stream, const char *format, va_list args) {
    // print prefix
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    fprintf(stream, "%s", prefix.c_str());

    // print message
    int ret = vfprintf(stream, format, args);

    return ret;
}

int print2stream(MessageType message_type, FILE *stream, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stream, format, args);
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

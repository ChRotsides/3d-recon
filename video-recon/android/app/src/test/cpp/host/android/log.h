#pragma once
#include <cstdarg>
#include <cstdio>
constexpr int ANDROID_LOG_INFO = 4;
inline int __android_log_print(int, const char*, const char* format, ...) {
    va_list args; va_start(args, format);
    const int result = std::vfprintf(stderr, format, args);
    va_end(args); std::fputc('\n', stderr);
    return result;
}

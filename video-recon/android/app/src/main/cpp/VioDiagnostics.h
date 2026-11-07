#pragma once
#include <array>
#include <limits>
#include "StationaryDetector.h"

// Per-thread observer only: no changes to OpenVINS state layout or filter math.
namespace live_diagnostics {
struct Frame {
    int tracks = 0;
    int msckfAttempted = 0, msckfAccepted = 0;
    int slamAttempted = 0, slamAccepted = 0;
    bool stationary = false;
    double previousCameraTime = NAN;
    std::vector<StationaryDetector::Point> witness;
    std::array<double, 60> stationaryDetails = [] {
        std::array<double, 60> values;
        values.fill(std::numeric_limits<double>::quiet_NaN());
        for (int i=33; i<=36; ++i) values[i]=0;
        for (int i=40; i<=55; ++i) values[i]=0;
        values[58]=values[59]=0;
        return values;
    }();
};
inline thread_local Frame current;
inline thread_local StationaryDetector *detector = nullptr;
}

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>
#include "StartupMotionCheck.h"
#include "SpatialMotion.h"

// Sensor/image evidence only. Estimated velocity is deliberately NOT evidence of
// physical motion or rest.
//
// Handheld policy (2026-09-07): thresholds admit an ordinary hand-held hold, not
// only a supported phone. Flow limits are pixels at 960 px width; with a ~750 px
// focal length a 1 px/s trend is ~2.5 mm/s at 2 m, so the zero-velocity update
// (sigma 0.03 m/s) costs a few centimetres per hold, far inside the accepted
// budget, while a walking translation (tens of px/s) is still rejected. Window
// requirements are in seconds so a decimated 10-15 Hz camera feed still qualifies.
class StationaryDetector {
public:
    static constexpr double kWindow = 2.7;      // longest history compared against
    static constexpr double kMinWindow = 1.0;   // shortest history that can prove a hold
    static constexpr double kMaxGap = 0.35;     // tolerated camera gap inside the window (10 Hz feed + one miss)
    struct Point { size_t id; double x, y; };
    struct Result {
        bool stationary = false, velocityStationary = false;
        double longMotion = NAN, shortMotion = NAN, gyroRms = NAN, accelStd = NAN;
        double trend = NAN, trendP75 = NAN, gyroMean = NAN;
        int rejection = 1; // bits: history, IMU, coverage, image excursion, trend, inertial motion
        int matched = 0, coverage = 0;
    };
    void imu(double t, const std::array<double, 3> &gyro, const std::array<double, 3> &accel) {
        if (!std::isfinite(t) || (!samples.empty() && t <= samples.back().t)) return;
        for (int i = 0; i < 3; ++i) if (!std::isfinite(gyro[i]) || !std::isfinite(accel[i])) return;
        samples.push_back({t, gyro, accel});
        while (samples.size() > 1200 || (!samples.empty() && t - samples.front().t > 2.0)) samples.pop_front();
    }
    void image(double t, const std::vector<Point> &points, int width, int height) {
        if (width <= 0 || height <= 0 || !std::isfinite(t) || (!images.empty() && t <= images.back().t)) return;
        Image f; f.t = t;
        // Coordinates normalized to a 960-wide image for resolution-independent thresholds.
        for (const auto &p : points) if (std::isfinite(p.x) && std::isfinite(p.y))
            f.points[p.id] = {p.x * 960.0 / width, p.y * 960.0 / width};
        f.height = height * 960.0 / width;
        images.push_back(std::move(f));
        while (images.size() > 100 || (!images.empty() && t - images.front().t > 2.8)) images.pop_front();
    }
    Result evaluate(double cameraTime, double imuTime) {
        Result r;
        if (images.size() < 2 || std::abs(images.back().t - cameraTime) > 1e-6) { anchorValid = false; return r; }
        const auto &now = images.back(); const auto &previous = images[images.size() - 2];
        // Oldest frame inside the window: judged over kMinWindow..kWindow seconds.
        const Image *old = nullptr;
        for (const auto &f : images) if (cameraTime - f.t <= kWindow) { old = &f; break; }
        if (!old || cameraTime - old->t < kMinWindow || now.t - previous.t > kMaxGap) { anchorValid = false; return r; }
        double lastImage=old->t;
        for (const auto &f : images) if (f.t >= old->t) {
            if (f.t-lastImage > kMaxGap) { anchorValid=false; return r; }
            lastImage=f.t;
        }
        const auto motionTrend = trendFlow(old->t, now);
        if (anchorValid) old = &anchor;
        const auto longFlow = flow(*old, now); const auto shortFlow = flow(previous, now);
        r.longMotion = longFlow.median; r.shortMotion = shortFlow.median;
        r.matched = longFlow.count; r.coverage = longFlow.cells;
        std::vector<StartupMotionCheck::Sample> recent;
        for (const auto &s : samples) {
            if (s.t >= imuTime - 1.5 && s.t <= imuTime) recent.push_back({s.t,s.g,s.a});
        }
        const auto inertial = StartupMotionCheck::stats(recent, 1.45);
        if (!inertial.valid || imuTime-recent.back().t > .03) { anchorValid = false; r.rejection = 2; return r; }
        r.gyroRms = inertial.gyroRms; r.accelStd = inertial.accelStd; r.gyroMean = inertial.gyroMean;
        r.trend = motionTrend.median; r.trendP75 = motionTrend.p75;
        r.rejection = 0;
        if (longFlow.count < 16 || shortFlow.count < 16 || longFlow.cells < 3 || motionTrend.count < 16 || motionTrend.cells < 3) r.rejection |= 4;
        if (longFlow.median >= 5.0 || longFlow.p75 >= 8.0 || shortFlow.median >= 1.5 || shortFlow.p75 >= 3.0) r.rejection |= 8;
        if (r.trend >= 1.5 || r.trendP75 >= 2.5) r.rejection |= 16;
        // Hand tremor: gyro RMS of a few hundredths rad/s and accelerometer scatter of
        // a few tenths m/s^2 are normal for a held phone and must not veto a hold.
        if (r.gyroRms >= .08 || r.gyroMean >= .012 || r.accelStd >= .35 || std::abs(inertial.accelNorm-9.81) >= .6) r.rejection |= 32;
        r.stationary = r.rejection == 0;
        // Gentle oscillation can support bias/gravity correction without asserting
        // zero translation. Stronger long-term evidence is needed for velocity.
        r.velocityStationary = r.stationary && r.trend < 1.0 && r.trendP75 < 1.8 &&
            r.gyroMean < .008 && longFlow.median < 3.5 && longFlow.p75 < 6.0;
        if (r.stationary && !anchorValid) { anchor = *old; anchorValid = true; }
        if (!r.stationary) anchorValid = false;
        return r;
    }
    bool velocityUpdateDue(double t) {
        if (t - lastVelocityUpdate < 0.20) return false;
        lastVelocityUpdate = t; return true;
    }
private:
    struct Sample { double t; std::array<double, 3> g, a; };
    struct Image { double t = 0, height = 0; std::unordered_map<size_t, std::array<double, 2>> points; };
    struct Flow { double median = INFINITY, p75 = INFINITY; int count = 0, cells = 0; };
    // Keep signed slopes, then fit their spatial field with uncertainty. Taking
    // each noisy slope's magnitude first creates a positive rest-motion bias.
    Flow trendFlow(double start, const Image &now) const {
        Flow out; std::vector<SpatialMotion::Track> tracks; std::array<int,9> cells{};
        for (const auto &p : now.points) {
            double n=0, st=0, stt=0, sx=0, sy=0, stx=0, sty=0, first=NAN, last=NAN;
            for (const auto &f : images) {
                if (f.t < start) continue;
                const auto it = f.points.find(p.first); if (it == f.points.end()) continue;
                const double t=f.t-start, x=it->second[0], y=it->second[1];
                if (!n) first=f.t; last=f.t;
                ++n; st+=t; stt+=t*t; sx+=x; sy+=y; stx+=t*x; sty+=t*y;
            }
            // Tracks must span most of the window; the sample count follows the frame rate.
            if (n < 8 || last-first < 0.7*(now.t-start)) continue;
            const double denominator=n*stt-st*st;
            if (denominator <= 0) continue;
            tracks.push_back({(p.second[0]-480)/480,2*p.second[1]/now.height-1,
                (n*stx-st*sx)/denominator,(n*sty-st*sy)/denominator});
            const int x=std::clamp(int(p.second[0]/320.0),0,2);
            const int y=std::clamp(int(3*p.second[1]/now.height),0,2); cells[y*3+x]++;
        }
        out.count=tracks.size();
        out.cells=std::count_if(cells.begin(),cells.end(),[](int n){return n>=3;});
        if(out.cells<4) return out;
        const SpatialMotion motion(tracks);
        if(motion.valid) { out.median=motion.median; out.p75=motion.p75; }
        return out;
    }
    static Flow flow(const Image &before, const Image &after) {
        Flow out; std::vector<double> distances; std::array<bool, 9> cells{};
        for (const auto &p : after.points) {
            const auto it = before.points.find(p.first); if (it == before.points.end()) continue;
            distances.push_back(std::hypot(p.second[0] - it->second[0], p.second[1] - it->second[1]));
            const int x = std::clamp(int(p.second[0] / 320.0), 0, 2);
            const int y = std::clamp(int(3.0 * p.second[1] / after.height), 0, 2);
            cells[y * 3 + x] = true;
        }
        if (distances.empty()) return out;
        std::sort(distances.begin(), distances.end()); out.count = distances.size();
        out.median = distances[distances.size() / 2]; out.p75 = distances[3 * distances.size() / 4];
        out.cells = std::count(cells.begin(), cells.end(), true); return out;
    }
    std::deque<Sample> samples;
    std::deque<Image> images;
    Image anchor;
    bool anchorValid = false;
    double lastVelocityUpdate = -INFINITY;
};

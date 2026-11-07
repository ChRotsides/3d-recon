#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

// Check the exact two windows consumed by StaticInitializer, not a later window.
// Android supplies calibrated gyro rates. A persistent mean is treated as motion
// until proven otherwise; never silently learn a large rate as startup bias.
struct StartupMotionCheck {
    struct Prior { bool valid=false; std::array<double,3> bias{}, variance{}; double expires=-INFINITY; };
    struct Sample { double t; std::array<double,3> gyro, accel; };
    struct Stats {
        bool valid = false;
        std::array<double,3> meanGyro{};
        double gyroMean = NAN, gyroRms = NAN, accelStd = NAN, accelNorm = NAN;
    };
    static Stats stats(const std::vector<Sample> &samples, double requiredSpan) {
        Stats s;
        if (samples.size() < 40) return s;
        std::array<double,3> a{}, a2{}; double g2 = 0;
        for (size_t j = 0; j < samples.size(); ++j) {
            const auto &v = samples[j];
            if (!std::isfinite(v.t) || (j && (v.t <= samples[j-1].t || v.t-samples[j-1].t > .03))) return s;
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(v.gyro[i]) || !std::isfinite(v.accel[i])) return s;
                s.meanGyro[i] += v.gyro[i]; g2 += v.gyro[i]*v.gyro[i];
                a[i] += v.accel[i]; a2[i] += v.accel[i]*v.accel[i];
            }
        }
        double mean2 = 0, variance = 0, aNorm2 = 0;
        for (int i = 0; i < 3; ++i) {
            s.meanGyro[i] /= samples.size(); mean2 += s.meanGyro[i]*s.meanGyro[i];
            a[i] /= samples.size(); aNorm2 += a[i]*a[i];
            variance += std::max(0.0, a2[i]/samples.size()-a[i]*a[i]);
        }
        s.gyroMean = std::sqrt(mean2); s.gyroRms = std::sqrt(g2/samples.size());
        s.accelStd = std::sqrt(variance); s.accelNorm = std::sqrt(aNorm2);
        s.valid = samples.back().t-samples.front().t >= requiredSpan;
        return s;
    }
    Stats older, newer;
    double meanChange = NAN;
    std::array<double,3> bias{}, biasVariance{};
    bool accepted = false;
    StartupMotionCheck(const std::vector<Sample> &oldSamples, const std::vector<Sample> &newSamples,
                       double halfWindow, double gravity = 9.81, const Prior* prior = nullptr) {
        older = stats(oldSamples, halfWindow-.03); newer = stats(newSamples, halfWindow-.03);
        if (!older.valid || !newer.valid || newSamples.front().t-oldSamples.back().t > .03 ||
            newSamples.front().t <= oldSamples.back().t) return;
        double delta2 = 0;
        for (int i = 0; i < 3; ++i) delta2 += std::pow(older.meanGyro[i]-newer.meanGyro[i], 2);
        meanChange = std::sqrt(delta2);
        const auto quiet = [gravity](const Stats &s) {
            return s.gyroMean < .005 && s.gyroRms < .04 && s.accelStd < .20 && std::abs(s.accelNorm-gravity) < .4;
        };
        // Warm recovery reuses a recent *measured* bias, not the rate during the
        // current settling motion. Keep IMU/image stillness checks and covariance.
        if(halfWindow>=1.45 && quiet(older) && quiet(newer) && prior && prior->valid && newSamples.back().t<=prior->expires) {
            bool valid=true;
            for(int k=0;k<3;++k) valid &= std::isfinite(prior->bias[k]) && std::abs(prior->bias[k])<=.005 &&
                std::isfinite(prior->variance[k]) && prior->variance[k]>=1e-6 && prior->variance[k]<=1e-4;
            if(valid) { bias=prior->bias; biasVariance=prior->variance; accepted=true; return; }
        }
        // A quiet total mean can hide the phone settling during the older half.
        // Require three seconds and agreement of non-overlapping half-second
        // blocks before interpreting the rate as sensor bias.
        if (halfWindow < 1.45 || !quiet(older) || !quiet(newer) || meanChange >= .0005) return;
        std::vector<Sample> all=oldSamples;
        all.insert(all.end(),newSamples.begin(),newSamples.end());
        const auto whole=stats(all,2*halfWindow-.03);
        bias=whole.meanGyro;
        std::vector<std::array<double,3>> means;
        for(int block=0;block<6;++block) {
            std::vector<Sample> part;
            const double begin=all.front().t+block*(2*halfWindow/6);
            const double end=begin+2*halfWindow/6;
            for(const auto &s:all) if(s.t>=begin && s.t<end) part.push_back(s);
            const auto local=stats(part,2*halfWindow/6-.03);
            if(!local.valid) return;
            double delta=0;
            for(int k=0;k<3;++k) delta+=std::pow(local.meanGyro[k]-bias[k],2);
            if(std::sqrt(delta)>=.00075) return;
            means.push_back(local.meanGyro);
        }
        // Conservative block dispersion (not sample-count-scaled white-noise
        // precision): retain a 0.001 rad/s uncertainty floor at initialization.
        for(int k=0;k<3;++k) {
            double variance=0;
            for(const auto &m:means) variance+=std::pow(m[k]-bias[k],2);
            biasVariance[k]=std::max(1e-6,variance/(means.size()-1));
        }
        accepted=true;
    }
};

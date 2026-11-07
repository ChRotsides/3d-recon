#pragma once
#include "VioDiagnostics.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/sensor_data.h"
#include "ZeroVelocityMeasurement.h"

namespace live_diagnostics {
inline thread_local std::array<double,3> startupBias{}, startupBiasVariance{};
inline thread_local StartupMotionCheck::Prior startupPrior;
inline bool startupMotion(const std::vector<ov_core::ImuData> &older,
                          const std::vector<ov_core::ImuData> &newer, double halfWindow, double gravity) {
    const auto convert = [](const std::vector<ov_core::ImuData> &input) {
        std::vector<StartupMotionCheck::Sample> result;
        for (const auto &s : input) result.push_back({s.timestamp,{s.wm(0),s.wm(1),s.wm(2)},{s.am(0),s.am(1),s.am(2)}});
        return result;
    };
    const StartupMotionCheck check(convert(older),convert(newer),halfWindow,gravity,&startupPrior);
    auto &d=current.stationaryDetails;
    d[29]=check.accepted; d[30]=check.older.gyroMean; d[31]=check.newer.gyroMean;
    d[32]=std::max(check.older.accelStd,check.newer.accelStd);
    d[37]=std::max(check.older.gyroRms,check.newer.gyroRms); d[38]=check.meanChange;
    if(check.accepted) { startupBias=check.bias; startupBiasVariance=check.biasVariance; }
    return check.accepted;
}
inline bool stationaryEvidence(double cameraTime, double imuTime) {
    if (!detector) return false;
    const auto evidence = detector->evaluate(cameraTime, imuTime);
    auto &d = current.stationaryDetails;
    d[14] = evidence.stationary; d[15] = 0;
    d[16] = evidence.longMotion; d[17] = evidence.shortMotion;
    d[18] = evidence.gyroRms; d[19] = evidence.accelStd;
    d[20] = evidence.matched; d[21] = evidence.coverage;
    d[25] = evidence.trend; d[26] = evidence.trendP75; d[27] = evidence.gyroMean;
    d[28] = evidence.velocityStationary; d[39] = evidence.rejection;
    return evidence.stationary;
}

// Linear measurement z=0 of world-frame velocity, H=I on the velocity block.
// Use the full-state EKF machinery, including all cross-covariances. A large
// innovation is reported/rejected, never repaired with a hard state assignment.
inline void zeroVelocityMeasurement(const std::shared_ptr<ov_msckf::State> &state, double t) {
    for (int i = 0; i < 3; ++i) current.stationaryDetails[22 + i] = state->_imu->vel()(i);
    if (!detector || current.stationaryDetails[28] != 1 || !detector->velocityUpdateDue(t)) return;
    const Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d P = ov_msckf::StateHelper::get_marginal_covariance(state, {state->_imu->v()});
    const ZeroVelocityMeasurement measurement(state->_imu->vel(), P);
    if (!measurement.accepted) { current.stationaryDetails[15] = -1; return; }
    ov_msckf::StateHelper::EKFUpdate(state, {state->_imu->v()}, H, measurement.residual, measurement.noise);
    current.stationaryDetails[15] = 1;
}

// Same calibration and gravity convention as Propagator. These are diagnostic
// averages over the update's IMU interval, not a separate acceleration estimator.
inline void stationarySnapshot(const std::shared_ptr<ov_msckf::State> &state,
        const std::vector<ov_core::ImuData> &samples, const Eigen::Vector3d &gravity, bool after) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero(); double duration = 0;
    const auto Da = ov_msckf::State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        const double dt = samples[i + 1].timestamp - samples[i].timestamp;
        mean += dt * (state->_calib_imu_ACCtoIMU->Rot() * Da * (samples[i].am - state->_imu->bias_a()));
        duration += dt;
    }
    if (duration <= 0) return;
    const Eigen::Vector3d acceleration = state->_imu->Rot().transpose() * (mean / duration) - gravity;
    const auto velocity = state->_imu->vel();
    const auto covariance = ov_msckf::StateHelper::get_marginal_covariance(state, {state->_imu->v()});
    for (int i = 0; i < 3; ++i) {
        current.stationaryDetails[(after ? 3 : 0) + i] = velocity(i);
        current.stationaryDetails[(after ? 9 : 6) + i] = acceleration(i);
    }
    current.stationaryDetails[after ? 13 : 12] = covariance.trace();
}
}

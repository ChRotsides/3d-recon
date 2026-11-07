#pragma once
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cmath>

struct ZeroVelocityMeasurement {
    Eigen::Vector3d residual;
    Eigen::Matrix3d noise;
    double innovation = NAN;
    bool accepted = false;
    ZeroVelocityMeasurement(const Eigen::Vector3d &velocity, const Eigen::Matrix3d &covariance)
        : residual(-velocity), noise(Eigen::Matrix3d::Identity() * (0.03 * 0.03)) {
        if (!velocity.allFinite() || !covariance.allFinite() || covariance.diagonal().minCoeff() < 0) return;
        const Eigen::Matrix3d S = covariance + noise;
        const Eigen::LLT<Eigen::Matrix3d> solver(S);
        if (solver.info() != Eigen::Success) return;
        innovation = residual.dot(solver.solve(residual));
        accepted = std::isfinite(innovation) && innovation <= 16.27;
    }
};

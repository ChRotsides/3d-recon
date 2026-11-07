#pragma once
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// Fit a spatial motion field, not the mean magnitude of noisy feature slopes.
// Affine terms preserve translation, rotation and expansion; a plain global
// vector average would incorrectly cancel rotation/forward motion. Bounds grow
// when tracks are few/noisy. This is an image gate, not an estimator measurement.
struct SpatialMotion {
    struct Track { double x,y,vx,vy; }; // x/y scaled to roughly [-1,1]
    double median=INFINITY, p75=INFINITY, scatter=INFINITY;
    bool valid=false;
    explicit SpatialMotion(const std::vector<Track> &tracks) {
        if(tracks.size()<12) return;
        const int n=tracks.size();
        Eigen::MatrixXd x(n,3),y(n,2);
        for(int i=0;i<n;++i) {
            x.row(i)<<1,tracks[i].x,tracks[i].y;
            y.row(i)<<tracks[i].vx,tracks[i].vy;
        }
        if(!x.allFinite()||!y.allFinite()) return;
        Eigen::VectorXd weights=Eigen::VectorXd::Ones(n);
        Eigen::Matrix<double,3,2> beta;
        Eigen::Matrix3d normal;
        for(int iteration=0;iteration<4;++iteration) {
            normal=x.transpose()*weights.asDiagonal()*x;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(normal);
            if(eig.info()!=Eigen::Success || eig.eigenvalues()(0)<=.01 || eig.eigenvalues()(2)/eig.eigenvalues()(0)>1000) return;
            beta=normal.ldlt().solve(x.transpose()*weights.asDiagonal()*y);
            if(!beta.allFinite()) return;
            if(iteration==3) break;
            std::vector<double> errors;
            for(int i=0;i<n;++i) errors.push_back((y.row(i)-x.row(i)*beta).norm());
            auto sorted=errors; std::sort(sorted.begin(),sorted.end());
            const double cap=std::max(.1,2.5*sorted[n/2]);
            for(int i=0;i<n;++i) weights(i)=std::min(1.0,cap/std::max(1e-12,errors[i]));
        }
        if(weights.sum()<12) return;
        // Include even downweighted errors in the uncertainty estimate: do not
        // make a contradictory flow field look stationary by discarding it.
        const double variance=(y-x*beta).squaredNorm()/(2*n-6);
        scatter=std::sqrt(variance);
        if(!std::isfinite(scatter)||scatter>1.0) return;
        const Eigen::Matrix3d inv=normal.ldlt().solve(Eigen::Matrix3d::Identity());
        std::vector<double> bounds;
        for(int i=0;i<n;++i) {
            const double leverage=(x.row(i)*inv*x.row(i).transpose())(0);
            bounds.push_back((x.row(i)*beta).norm()+2*std::sqrt(std::max(0.0,variance*leverage)));
        }
        std::sort(bounds.begin(),bounds.end());
        median=bounds[n/2]; p75=bounds[3*n/4]; valid=true;
    }
};

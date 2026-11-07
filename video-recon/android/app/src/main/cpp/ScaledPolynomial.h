#pragma once
#include <Eigen/Core>
#include <cmath>

// lambda = scale * mu balances a monic polynomial's companion matrix. Its
// numerical rank is not an observability test: a legitimate zero root makes it
// singular. The initializer still validates gravity, parallax and covariance.
struct ScaledPolynomial {
    double scale = 1e-12;
    Eigen::Matrix<double,6,6> companion = Eigen::Matrix<double,6,6>::Zero();
    bool valid = false;
    explicit ScaledPolynomial(const Eigen::Matrix<double,7,1>& coefficients) {
        if (!coefficients.allFinite() || coefficients[0] != 1.0) return;
        for (int i=1;i<=6;++i)
            scale=std::max(scale,std::pow(std::abs(coefficients[i]),1.0/i));
        if (!std::isfinite(scale)) return;
        companion.diagonal(-1).setOnes();
        for (int i=1;i<=6;++i) {
            double value=-coefficients[i];
            for(int j=0;j<i;++j) value/=scale;
            companion(6-i,5)=value;
        }
        valid=companion.allFinite();
    }
};

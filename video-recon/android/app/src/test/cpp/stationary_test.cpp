#include "StationaryDetector.h"
#include "ZeroVelocityMeasurement.h"
#include <Eigen/Eigenvalues>
#include <cassert>
#include <iostream>
#include <random>

std::vector<StationaryDetector::Point> points(double shift, bool clustered = false) {
    std::vector<StationaryDetector::Point> p;
    for (size_t i = 0; i < 90; ++i) p.push_back({i,
        (clustered ? 10.0 : 50.0 + (i % 10) * 90.0) + shift,
        clustered ? 10.0 : 30.0 + (i / 10) * 55.0});
    return p;
}
StationaryDetector::Result tick(StationaryDetector &d, int step, double shift = 0,
        double gyro = 0.001, double acceleration = 9.81, bool clustered = false) {
    const double t = step * .005;
    d.imu(t, {gyro, 0, 0}, {0, acceleration, 0});
    if (step % 7 != 0) return {};
    d.image(t, points(shift, clustered), 960, 540);
    return d.evaluate(t, t);
}

int main() {
    int tests = 0;
    { StationaryDetector d; StationaryDetector::Result r;
      for (int i = 1; i <= 602; ++i) { auto v = tick(d, i); if (i % 7 == 0) r = v; }
      assert(r.stationary && r.matched == 90 && r.coverage >= 4); ++tests; }
    { StationaryDetector d;
      for (int i = 1; i <= 602; ++i) { auto r = tick(d, i, i * .005 * 2.0); assert(!r.stationary); }
      ++tests; } // slow visual translation, perfectly quiet IMU, must not be classified as rest
    { StationaryDetector d;
      for (int i = 1; i <= 602; ++i) assert(!tick(d, i, 0, .1).stationary);
      ++tests; } // tilt/rotation even with artificially unchanged images
    { StationaryDetector d;
      for (int i = 1; i <= 602; ++i) assert(!tick(d, i, 0, .001, 9.81 + (i % 2 ? .5 : -.5)).stationary);
      ++tests; } // a shaking hand (0.5 m/s^2 scatter) is not a hold; ordinary tremor (0.15) is
    { StationaryDetector d;
      for (int i = 1; i <= 602; ++i) assert(!tick(d, i, 0, .001, 9.81, true).stationary);
      ++tests; } // insufficient spatial coverage
    { StationaryDetector d; StationaryDetector::Result r;
      for (int i = 1; i <= 1008; ++i) { auto v = tick(d, i, std::min(i * .005, 2.0) * 4.0); if (i % 7 == 0) r = v; }
      assert(r.stationary); ++tests; } // translation then a genuine stop
    { StationaryDetector d; bool rejectedAfterAnchor = false, acquired = false;
      for (int i = 1; i <= 7000; ++i) if (i % 7 == 0) {
          // Supply dense IMU history independently of image cadence.
          for (int j = i - 6; j <= i; ++j) d.imu(j * .005, {0,0,0}, {0,9.81,0});
          const double t = i * .005; d.image(t, points(t * .5), 960, 540);
          const auto r = d.evaluate(t,t); acquired |= r.stationary;
          if (acquired && t > 29 && !r.stationary) rejectedAfterAnchor = true;
      }
      assert(acquired && rejectedAfterAnchor); ++tests; } // fixed anchor prevents endless slow creep
    { StationaryDetector d;
      for (int i = 1; i <= 602; ++i) tick(d, i);
      d.image(3.5, points(0), 960, 540);
      assert(!d.evaluate(3.5,3.5).stationary); ++tests; } // missing IMU/camera updates
    { StationaryDetector d; assert(d.velocityUpdateDue(1.0)); assert(!d.velocityUpdateDue(1.1)); assert(d.velocityUpdateDue(1.21)); ++tests; }
    { StationaryDetector d; int eligible=0, frames=0;
      for (int i=1; i<=1204; ++i) {
          const double t=i*.005, jitter=std::sin(2*3.141592653589793*4*t);
          const auto r=tick(d,i,.6*jitter,.001+.025*jitter,9.81+.15*jitter);
          if (i%7==0 && t>2) { ++frames; eligible+=r.velocityStationary; }
      }
      assert(eligible > frames*.7); ++tests; } // observed-scale zero-mean tremor, not perfectly quiet synthetic IMU
    { StationaryDetector d;
      for (int i=1; i<=1204; ++i) {
          const double t=i*.005, jitter=std::sin(2*3.141592653589793*4*t);
          const auto r=tick(d,i,2.0*t+.6*jitter,.001+.025*jitter,9.81+.15*jitter);
          assert(!r.stationary && !r.velocityStationary);
      }
      ++tests; } // slow translation (2 px/s) underneath tremor must not receive zero velocity
    { StationaryDetector d;
      for (int i=1; i<=602; ++i) assert(!tick(d,i,0,.015).stationary);
      ++tests; } // low but persistent rotation is not tremor
    { StationaryDetector d; StationaryDetector::Result r;
      for (int i=1;i<=602;++i) { const auto s=tick(d,i,i*.005*1.2); if(i%7==0) r=s; }
      assert(r.stationary && !r.velocityStationary); ++tests;
    } // declining the stronger velocity constraint need not disable bias/gravity support
    { StationaryDetector d;
      for(int i=1;i<=602;++i) tick(d,i);
      for(int i=603;i<=680;++i) d.imu(i*.005,{0,0,0},{0,9.81,0});
      for(int i=681;i<=756;++i) tick(d,i);
      d.image(3.79,points(0),960,540);
      assert(!d.evaluate(3.79,3.78).stationary); ++tests;
    } // an internal camera gap must not disappear after one fresh frame
    const auto startupSamples=[](double begin, double rate, bool tremor=false) {
        std::vector<StartupMotionCheck::Sample> s;
        for (int i=1;i<=300;++i) {
            const double jitter=tremor ? (i%2 ? .02 : -.02) : 0;
            s.push_back({begin+i*.005,{rate+jitter,0,0},{0,9.81+(tremor ? 5*jitter:0),0}});
        }
        return s;
    };
    { const StartupMotionCheck s(startupSamples(0,.001,true),startupSamples(1.5,.001,true),1.5);
      assert(s.accepted); ++tests; }
    { const StartupMotionCheck s(startupSamples(0,.03),startupSamples(1.5,.001),1.5);
      assert(!s.accepted); ++tests; } // original failure: startup rate learned as gyro bias
    { const StartupMotionCheck s(startupSamples(0,.01),startupSamples(1.5,.01),1.5);
      assert(!s.accepted); ++tests; } // sustained gentle rotation
    { auto a=startupSamples(0,.001); a[10].gyro[0]=NAN;
      assert(!StartupMotionCheck(a,startupSamples(1.5,.001),1.5).accepted); ++tests; }
    { auto a=startupSamples(0,.001); a.erase(a.begin()+20,a.begin()+40);
      assert(!StartupMotionCheck(a,startupSamples(1.5,.001),1.5).accepted); ++tests; }
    { auto a=startupSamples(0,.001); for(int i=0;i<30;++i) a[i].gyro[0]=.15;
      assert(!StartupMotionCheck(a,startupSamples(1.5,.001),1.5).accepted); ++tests; } // tap transient
    { const StartupMotionCheck s(startupSamples(0,.0024),startupSamples(1.5,.00054),1.5);
      assert(!s.accepted); ++tests; } // observed settling-error scale, formerly accepted
    { auto a=startupSamples(0,.0005);
      for(int i=0;i<100;++i) a[i].gyro[0]+=.0015;
      for(int i=100;i<200;++i) a[i].gyro[0]-=.0015;
      assert(!StartupMotionCheck(a,startupSamples(1.5,.0005),1.5).accepted); ++tests;
    } // equal half-window means cannot hide opposing settling transients
    { const StartupMotionCheck s(startupSamples(0,.0004),startupSamples(1.5,.0007),1.5);
      assert(s.accepted && std::abs(s.bias[0]-.00055)<1e-9 && s.biasVariance[0]>=1e-6); ++tests;
    } // use both halves, with a conservative covariance floor
    { auto a=startupSamples(0,.001), b=startupSamples(.75,.001);
      a.resize(150); b.resize(150);
      assert(!StartupMotionCheck(a,b,.75).accepted); ++tests;
    } // short windows no longer learn an under-averaged bias
    { StartupMotionCheck::Prior p; p.valid=true; p.bias={.0004,0,0}; p.variance={1e-6,1e-6,1e-6};p.expires=120;
      const StartupMotionCheck warm(startupSamples(0,.0024),startupSamples(1.5,.00054),1.5,9.81,&p);
      assert(warm.accepted && warm.bias==p.bias && warm.biasVariance==p.variance); ++tests;
      assert(!StartupMotionCheck(startupSamples(0,.03),startupSamples(1.5,.03),1.5,9.81,&p).accepted); ++tests;
      p.expires=2;
      assert(!StartupMotionCheck(startupSamples(0,.0024),startupSamples(1.5,.00054),1.5,9.81,&p).accepted); ++tests;
      p.expires=120;p.variance[0]=NAN;
      assert(!StartupMotionCheck(startupSamples(0,.0024),startupSamples(1.5,.00054),1.5,9.81,&p).accepted); ++tests;
    } // warm recovery retains validated bias, still rejects movement/expired/malformed priors
    { std::vector<SpatialMotion::Track> tracks;
      for(const auto &p:points(0)) tracks.push_back({(p.x-480)/480,(p.y-270)/270,0,0});
      for(int kind=0;kind<4;++kind) {
          auto moving=tracks;
          for(auto &p:moving) {
              if(kind==0) p.vx=.5;
              if(kind==1) p.vy=.5;
              if(kind==2) { p.vx=-2*p.y; p.vy=2*p.x; }
              if(kind==3) { p.vx=2*p.x; p.vy=2*p.y; }
          }
          const SpatialMotion m(moving);
          assert(m.valid && m.median>.35); ++tests;
      }
      assert(SpatialMotion(tracks).valid && SpatialMotion(tracks).p75<1e-8); ++tests;
      tracks.resize(11); assert(!SpatialMotion(tracks).valid); ++tests;
    } // slow translation, rotation and expansion cannot cancel into false rest
    for(double drift : {0.0,2.0}) { StationaryDetector d; std::mt19937 rng(42); std::normal_distribution<double> noise(0,.12);
      std::array<std::array<double,2>,90> walk{}; int stationary=0, velocity=0,frames=0;
      for(int i=1;i<=1600;++i) {
          const double t=i*.005; d.imu(t,{.0004,0,0},{0,9.81,0});
          if(i%7) continue;
          auto p=points(drift*t);
          for(int j=0;j<90;++j) { walk[j][0]+=noise(rng); walk[j][1]+=noise(rng); p[j].x+=walk[j][0];p[j].y+=walk[j][1]; }
          d.image(t,p,960,540);const auto r=d.evaluate(t,t);
          if(t>3) {frames++;stationary+=r.stationary;velocity+=r.velocityStationary;}
      }
      std::cerr<<"noisy independent tracks drift="<<drift<<" stationary/velocity="<<stationary<<"/"<<velocity<<" of "<<frames<<"\n";
      if(drift==0) assert(stationary>frames*.5);
      else assert(stationary==0 && velocity==0);
      ++tests;
    }

    { StationaryDetector d; int eligible=0, frames=0;
      for (int i=1; i<=1204; ++i) {
          const double t=i*.005, jitter=std::sin(2*3.141592653589793*4*t);
          d.imu(t, {.001+.025*jitter, 0, 0}, {0, 9.81+.15*jitter, 0});
          if (i % 17 != 0) continue;
          d.image(t, points(.6*jitter), 960, 540);
          const auto r = d.evaluate(t, t);
          if (t>2) { ++frames; eligible+=r.velocityStationary; }
      }
      assert(eligible > frames*.7); ++tests; } // 12 Hz decimated feed with tremor still supports a hold

    // Mechanism: no direct velocity columns and zero inertial residual leaves false velocity intact.
    Eigen::Matrix<double,6,6> P = Eigen::Matrix<double,6,6>::Identity() * .04;
    Eigen::Matrix<double,6,1> x; x << .4,0,0,0,0,0;
    Eigen::Matrix<double,3,6> legacy = Eigen::Matrix<double,3,6>::Zero(); legacy.rightCols<3>().setIdentity();
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * .0009;
    auto legacyK = (P * legacy.transpose() * (legacy * P * legacy.transpose() + R).inverse()).eval();
    const auto unchanged = (x + legacyK * Eigen::Vector3d::Zero()).eval();
    assert(std::abs(unchanged(0) - .4) < 1e-12); ++tests;
    const ZeroVelocityMeasurement m(x.head<3>(), P.topLeftCorner<3,3>());
    assert(m.accepted); Eigen::Matrix<double,3,6> H = Eigen::Matrix<double,3,6>::Zero(); H.leftCols<3>().setIdentity();
    const auto K = (P * H.transpose() * (H * P * H.transpose() + m.noise).inverse()).eval();
    const auto corrected = (x + K * m.residual).eval();
    const auto I = Eigen::Matrix<double,6,6>::Identity();
    const Eigen::Matrix<double,6,6> posterior = (I-K*H)*P*(I-K*H).transpose()+K*m.noise*K.transpose();
    assert(corrected(0) > 0 && corrected(0) < .01);
    assert(posterior.isApprox(posterior.transpose(), 1e-12));
    assert((Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>>(posterior).eigenvalues().minCoeff() > 0)); ++tests;
    // Cross covariance participates in the full-state update; no velocity-only hard assignment.
    P(0,3)=P(3,0)=.01;
    const auto crossK = (P * H.transpose() * (H * P * H.transpose() + m.noise).inverse()).eval();
    assert(std::abs((crossK * m.residual)(3)) > .01); ++tests;
    assert(!ZeroVelocityMeasurement(Eigen::Vector3d(5,0,0), Eigen::Matrix3d::Identity()*.001).accepted); ++tests;
    assert(!ZeroVelocityMeasurement(Eigen::Vector3d(NAN,0,0), Eigen::Matrix3d::Identity()).accepted); ++tests;
    std::cout << tests << " native stationary/covariance tests passed; corrected synthetic velocity=" << corrected(0) << " m/s\n";
}

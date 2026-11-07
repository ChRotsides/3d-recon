#include "init/InertialInitializer.h"
#include "init/InertialInitializerOptions.h"
#include "feat/FeatureDatabase.h"
#include "cam/CamRadtan.h"
#include "types/IMU.h"
#include "utils/sensor_data.h"
#include "utils/print.h"
#include "core/VioManager.h"
#include "state/State.h"
#include "FeatureDiagnostics.h"
#include "ScaledPolynomial.h"
#include <opencv2/imgproc.hpp>
#include <Eigen/Geometry>
#include <chrono>
#include <cassert>
#include <iostream>
#include <random>

struct Motion {
    static Eigen::Vector3d p(double t) { return {.22*std::sin(1.7*t),.12*std::sin(2.3*t),.08*std::cos(1.3*t)}; }
    static Eigen::Vector3d v(double t) { return {.22*1.7*std::cos(1.7*t),.12*2.3*std::cos(2.3*t),-.08*1.3*std::sin(1.3*t)}; }
    static Eigen::Vector3d a(double t) { return {-.22*1.7*1.7*std::sin(1.7*t),-.12*2.3*2.3*std::sin(2.3*t),-.08*1.3*1.3*std::cos(1.3*t)}; }
    static Eigen::Matrix3d r(double t) {
        return (Eigen::AngleAxisd(-.12*std::sin(1.9*t),Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(-.10*std::sin(1.7*t),Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(-.08*std::sin(1.3*t),Eigen::Vector3d::UnitZ())).toRotationMatrix();
    }
    static Eigen::Vector3d w(double t) {
        const double eps=1e-5;
        const Eigen::Matrix3d skew=-(r(t+eps)-r(t-eps))/(2*eps)*r(t).transpose();
        return {skew(2,1),skew(0,2),skew(1,0)};
    }
};

int main(int argc,char **argv) {
    std::cout.setf(std::ios::unitbuf);
    { // Exact zero roots are valid; coefficient scale must not destroy other roots.
      Eigen::Matrix<double,7,1> coeff; coeff<<1,-15000,85000000,-225000000000.,274000000000000.,-120000000000000000.,0;
      ScaledPolynomial poly(coeff); assert(poly.valid && poly.scale>1000);
      Eigen::EigenSolver<Eigen::Matrix<double,6,6>> solver(poly.companion,false);
      assert(solver.info()==Eigen::Success);
      std::vector<double> roots;
      for(int i=0;i<6;++i) {assert(std::abs(solver.eigenvalues()[i].imag())<1e-8);roots.push_back(solver.eigenvalues()[i].real()*poly.scale);}
      std::sort(roots.begin(),roots.end());
      for(int i=0;i<6;++i) assert(std::abs(roots[i]-1000*i)<1e-3);
    }
    ov_core::Printer::setPrintLevel("DEBUG");
    for(int scenario=0;scenario<6;++scenario) {
        std::mt19937 random(1234 + scenario);
        std::normal_distribution<double> noise;
        const bool noisy = scenario >= 3;
        ov_init::InertialInitializerOptions params;
        params.init_window_time=3; params.init_dyn_use=true; params.init_imu_thresh=0;
        params.init_max_disparity=2; params.init_max_features=120;
        params.init_dyn_num_pose=6; params.init_dyn_min_deg=10;
        params.init_dyn_mle_max_time=.5; params.init_dyn_mle_max_iter=50; params.init_dyn_mle_max_threads=2;
        params.init_dyn_min_rec_cond=1e-12; params.init_dyn_mle_opt_calib=false;
        params.init_dyn_inflation_orientation=10; params.init_dyn_inflation_velocity=100;
        params.init_dyn_inflation_bias_gyro=10; params.init_dyn_inflation_bias_accel=100;
        params.use_stereo=false;
        if(noisy) { params.sigma_w=.0017; params.sigma_a=.02; params.sigma_wb=.0002; params.sigma_pix=.5; }
        auto camera=std::make_shared<ov_core::CamRadtan>(640,480);
        Eigen::VectorXd intr(8); intr<<400,400,320,240,0,0,0,0; camera->set_value(intr);
        params.camera_intrinsics[0]=camera;
        Eigen::VectorXd ext(7); ext<<0,0,0,1,0,0,0; params.camera_extrinsics[0]=ext;
        auto db=std::make_shared<ov_core::FeatureDatabase>();
        ov_init::InertialInitializer initializer(params,db);
        const Eigen::Vector3d gravity(0,0,9.81), bg(.0004,-.0003,.0002);
        for(int j=0;j<=645;++j) {
            const double t=j/200.;
            if(scenario==2 && t>1 && t<1.5) continue;
            ov_core::ImuData imu; imu.timestamp=100+t;
            imu.wm=Motion::w(t)+bg;
            imu.am=Motion::r(t)*(gravity+(scenario==1?Eigen::Vector3d::Zero():Motion::a(t)));
            if(noisy) for(int axis=0;axis<3;++axis) {
                imu.wm[axis] += params.sigma_w*std::sqrt(200.)*noise(random);
                imu.am[axis] += params.sigma_a*std::sqrt(200.)*noise(random) + (axis==0?.05:0);
            }
            initializer.feed_imu(imu);
        }
        for(int frame=0;frame<=96;++frame) {
            const double t=frame/30.;
            for(int id=0;id<150;++id) {
                const Eigen::Vector3d point(-1.2+.18*(id%15),-.9+.18*(id/15),4.+.27*(id%7));
                const Eigen::Vector3d pc=Motion::r(t)*(point-(scenario==1?Eigen::Vector3d::Zero():Motion::p(t)));
                const double x=pc.x()/pc.z()+(noisy?.5*noise(random)/400:0);
                const double y=pc.y()/pc.z()+(noisy?.5*noise(random)/400:0);
                db->update_feature(id,100+t,0,400*x+320,400*y+240,x,y);
            }
        }
        auto state=std::make_shared<ov_type::IMU>(); const Eigen::MatrixXd before=state->value();
        Eigen::MatrixXd covariance; std::vector<std::shared_ptr<ov_type::Type>> order; double timestamp=-1;
        const auto began=std::chrono::steady_clock::now();
        const bool success=initializer.initialize(timestamp,covariance,order,state,false);
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
        std::cout<<"scenario="<<scenario<<" initialized="<<success<<" sensor_window=3.2s solve="<<seconds<<"s\n";
        if(scenario==0 || noisy) {
            assert(success && covariance.rows()==15 && covariance.allFinite());
            assert(order.size()==1 && order[0]==state);
            const double velocityError=(state->Rot()*state->vel()-Motion::r(3.2)*Motion::v(3.2)).norm();
            const double gravityError=(state->Rot()*gravity-Motion::r(3.2)*gravity).norm();
            std::cout<<"body velocity error="<<velocityError<<" gravity vector error="<<gravityError<<" bg="<<state->bias_g().transpose()<<"\n";
            assert(velocityError<.15 && gravityError<.15);
            assert((state->bias_g()-bg).norm()<(noisy?.01:.003));
            assert(3.2+seconds<5.0);
        } else {
            assert(!success); assert(state->value().isApprox(before,0));
        }
    }
    std::cout<<"6 handheld initialization scenarios passed\n";
    if(argc>1) {
        ov_core::Printer::setPrintLevel("ERROR");
        auto parser=std::make_shared<ov_core::YamlParser>(argv[1]);
        ov_msckf::VioManagerOptions p; p.print_and_load(parser); assert(parser->successful());
        p.use_multi_threading_subs=false;
        p.init_options.init_window_time=3; p.init_options.init_dyn_use=true; p.init_options.init_max_features=120;
        p.init_options.init_dyn_mle_max_time=.5; p.init_options.init_dyn_mle_max_iter=50;
        p.init_options.init_dyn_mle_max_threads=2; p.init_options.init_dyn_min_deg=10;
        p.init_options.init_dyn_num_pose=6; p.init_options.init_dyn_mle_opt_calib=false;
        p.state_options.max_clone_size=31; p.calib_camimu_dt=0; p.init_options.calib_camimu_dt=0;
        auto camera=std::make_shared<ov_core::CamRadtan>(640,480);
        Eigen::VectorXd intr(8); intr<<400,400,320,240,0,0,0,0; camera->set_value(intr);
        Eigen::VectorXd ext(7); ext<<0,0,0,1,0,0,0;
        p.camera_intrinsics[0]=camera; p.camera_extrinsics[0]=ext;
        p.init_options.camera_intrinsics[0]=camera; p.init_options.camera_extrinsics[0]=ext;
        ov_msckf::VioManager manager(p);
        StationaryDetector detector; live_diagnostics::detector=&detector;
        cv::setNumThreads(1); cv::setRNGSeed(91);
        cv::Mat texture(480,640,CV_8UC1); cv::randu(texture,0,255); cv::GaussianBlur(texture,texture,{3,3},.7);
        Eigen::Matrix3d k; k<<400,0,320,0,400,240,0,0,1;
        double imuTime=100,initAt=-1,visualAt=-1,maxCameraSeconds=0;
        int accepted=0;
        for(int frame=0;frame<180;++frame) {
            const double t=frame/30.;
            while(imuTime<=100+t+.01) {
                const double s=imuTime-100; ov_core::ImuData m; m.timestamp=imuTime;
                m.wm=Motion::w(s); m.am=Motion::r(s)*(Eigen::Vector3d(0,0,9.81)+Motion::a(s));
                manager.feed_measurement_imu(m);
                detector.imu(imuTime,{m.wm.x(),m.wm.y(),m.wm.z()},{m.am.x(),m.am.y(),m.am.z()}); imuTime+=.005;
            }
            const Eigen::Matrix3d h=k*Motion::r(t)*(Eigen::Matrix3d::Identity()-Motion::p(t)*Eigen::RowVector3d(0,0,.25))*k.inverse();
            cv::Mat warp(3,3,CV_64F); for(int r=0;r<3;++r) for(int c=0;c<3;++c) warp.at<double>(r,c)=h(r,c);
            cv::Mat image; cv::warpPerspective(texture,image,warp,texture.size());
            ov_core::CameraData data; data.timestamp=100+t; data.sensor_ids={0}; data.images={image}; data.masks={cv::Mat::zeros(image.size(),CV_8UC1)};
            live_diagnostics::current={}; live_diagnostics::current.previousCameraTime=frame?100+(frame-1)/30.:NAN;
            const auto began=std::chrono::steady_clock::now(); manager.feed_measurement_camera(data);
            maxCameraSeconds=std::max(maxCameraSeconds,std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count());
            if(manager.initialized() && initAt<0) initAt=t;
            const auto &d=live_diagnostics::current; accepted+=d.msckfAccepted+d.slamAccepted;
            if(d.msckfAccepted+d.slamAccepted>=6 && visualAt<0) visualAt=t;
            if(frame%30==29) std::cout<<"rendered moving VIO t="<<t<<" initialized="<<manager.initialized()<<" accepted="<<accepted<<" speed="<<manager.get_state()->_imu->vel().norm()<<"\n";
        }
        live_diagnostics::detector=nullptr;
        std::cout<<"rendered moving VIO initialized_at="<<initAt<<"s first_six_visual_updates="<<visualAt<<"s max_camera_compute="<<maxCameraSeconds<<"s\n";
        assert(initAt>=0 && initAt+maxCameraSeconds<5 && visualAt>=0 && visualAt<5 && accepted>20);
        std::cout<<"full rendered moving-start VIO test passed\n";
    }
}

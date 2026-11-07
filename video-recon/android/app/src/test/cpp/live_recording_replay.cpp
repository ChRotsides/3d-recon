// Host-only sensor replay of the same instrumented estimator as Live Scan.
// stdin: I t wx wy wz ax ay az\n or C t width height\n followed by width*height bytes.
// This measures estimator behavior, not Android scheduling or dense-map accuracy.
#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "FeatureDiagnostics.h"
#include "StationaryDiagnostics.h"
#include "utils/print.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    if(argc!=3) return 2;
    const char* logLevel=std::getenv("LIVE_REPLAY_LOG_LEVEL");
    ov_core::Printer::setPrintLevel(logLevel?logLevel:"ERROR");
    auto parser=std::make_shared<ov_core::YamlParser>(argv[1]);
    ov_msckf::VioManagerOptions params; params.print_and_load(parser);
    if(!parser->successful()) return 2;
    params.use_multi_threading_subs=false;
    params.try_zupt=true; params.zupt_only_at_beginning=false;
    ov_msckf::VioManager manager(params);
    StationaryDetector detector; live_diagnostics::detector=&detector;
    cv::setNumThreads(1); cv::setRNGSeed(91);
    std::ofstream out(argv[2]); if(!out) return 2;
    out<<std::setprecision(12)<<"t,initialized,state_t,tracks,attempted,accepted,stationary,speed,position_sigma,px,py,pz,camera_ms,active_points,offset\n";
    std::string line; double previous=NAN;
    while(std::getline(std::cin,line)) {
        if(line.empty()) continue;
        std::istringstream row(line); char kind; row>>kind;
        if(kind=='I') {
            ov_core::ImuData m;
            if(!(row>>m.timestamp>>m.wm.x()>>m.wm.y()>>m.wm.z()>>m.am.x()>>m.am.y()>>m.am.z())) return 3;
            manager.feed_measurement_imu(m);
            detector.imu(m.timestamp,{m.wm.x(),m.wm.y(),m.wm.z()},{m.am.x(),m.am.y(),m.am.z()});
        } else if(kind=='C') {
            double t; int w,h; if(!(row>>t>>w>>h) || w!=960 || h!=540) return 3;
            cv::Mat gray(h,w,CV_8UC1); if(!std::cin.read(reinterpret_cast<char*>(gray.data),w*h)) return 3;
            ov_core::CameraData data; data.timestamp=t; data.sensor_ids={0}; data.images={gray};
            data.masks={cv::Mat::zeros(h,w,CV_8UC1)};
            live_diagnostics::current={}; live_diagnostics::current.previousCameraTime=previous;previous=t;
            const auto start=std::chrono::steady_clock::now(); manager.feed_measurement_camera(data);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            auto state=manager.get_state(); const auto& d=live_diagnostics::current;
            double sigma=NAN;
            if(manager.initialized()) {
                const Eigen::MatrixXd covariance=ov_msckf::StateHelper::get_marginal_covariance(state,{state->_imu->p()});
                sigma=std::sqrt(covariance.diagonal().maxCoeff());
            }
            double activeTime; std::unordered_map<size_t,Eigen::Vector3d> points,pixels;
            manager.get_active_tracks(activeTime,points,pixels);
            const Eigen::Vector3d p=state->_imu->pos();
            out<<t<<','<<manager.initialized()<<','<<state->_timestamp<<','<<d.tracks<<','
                <<d.msckfAttempted+d.slamAttempted<<','<<d.msckfAccepted+d.slamAccepted<<','<<d.stationary<<','
                <<state->_imu->vel().norm()<<','<<sigma<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<ms<<','
                <<(std::abs(activeTime-t)<1e-6?pixels.size():0)<<','<<state->_calib_dt_CAMtoIMU->value()(0)<<'\n';
            out.flush();
        } else return 3;
    }
}

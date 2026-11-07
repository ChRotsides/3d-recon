// JNI bridge: OpenVINS (poses from camera + IMU) and the voxel-hash TSDF, all on the phone.
#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "VioDiagnostics.h"
#include "StationaryDiagnostics.h"
#include "XFeatBridge.h"
#include "MapRelocalizer.h"
#include "cam/CamRadtan.h"
#include "cam/CamEqui.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "tsdf.h"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "livescan", __VA_ARGS__)
using namespace ov_msckf;

namespace {
struct Live {
    std::shared_ptr<VioManager> vio; std::mutex vioMutex;
    StationaryDetector stationaryDetector;
    StartupMotionCheck::Prior startupPrior;
    std::vector<double> featureSamples;
    live_xfeat::Store xfeat;
    live_map::Relocalizer map;
    double firstImu = std::numeric_limits<double>::infinity();
    double lastImu = -std::numeric_limits<double>::infinity();
    double lastCamera = -std::numeric_limits<double>::infinity();
    double lastDiagnostic = -std::numeric_limits<double>::infinity();
    std::unique_ptr<VoxelHashTSDF> tsdf; std::mutex tsdfMutex; std::vector<float> surface;
};
// OpenVINS global frame G has z up; the viewer and the TSDF use Y up like the web page.
const Eigen::Matrix3d ZUP_TO_YUP = (Eigen::Matrix3d() << 1, 0, 0, 0, 0, 1, 0, -1, 0).finished();
}

extern "C" JNIEXPORT jlong JNICALL Java_com_recon_videoscan_live_Native_init(JNIEnv* env, jclass, jstring configPath, jdoubleArray biasPrior) {
    const char* path = env->GetStringUTFChars(configPath, nullptr);
    auto live = new Live();
    if(biasPrior && env->GetArrayLength(biasPrior)==7) {
        double values[7]; env->GetDoubleArrayRegion(biasPrior,0,7,values);
        live->startupPrior.valid=true;
        live->startupPrior.expires=values[6];
        for(int k=0;k<3;++k) { live->startupPrior.bias[k]=values[k]; live->startupPrior.variance[k]=values[k+3]; }
    }
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    ov_core::Printer::setPrintLevel("INFO");
    VioManagerOptions params; params.print_and_load(parser);
    if (!parser->successful()) { LOGI("config parse failed: %s", path); env->ReleaseStringUTFChars(configPath, path); delete live; return 0; }
    // Stationary periods can occur anywhere in a scan. Estimated velocity is not
    // ground truth: the visual stationary override is also needed to arrest drift.
    params.try_zupt = true;
    params.zupt_only_at_beginning = false;
    // The app already has a dedicated VIO thread; keep initialization and diagnostic
    // snapshots synchronous with that thread and the native session lock.
    params.use_multi_threading_subs = false;
    // Existing on-device YAML may predate the startup-motion guard.
    params.init_options.init_max_disparity = 2.0;
    params.init_options.init_window_time = 3.0;
    params.init_options.init_dyn_use = true;
    params.init_options.init_max_features = 120;
    params.init_options.init_dyn_mle_opt_calib = false;
    params.init_options.init_dyn_mle_max_iter = 50;
    params.init_options.init_dyn_mle_max_time = 0.5;
    params.init_options.init_dyn_mle_max_threads = 2;
    // Assets are copied only on first install. Upgrade the legacy window in
    // memory, preserving saved calibration and any explicitly custom window.
    if (params.state_options.max_clone_size == 11) params.state_options.max_clone_size = 31;
    // Handheld policy (2026-09-07), applied in memory because the on-device YAML is
    // only copied on first install: a gentle turn is enough to initialise, SLAM
    // landmarks form after one second of tracking so a pause keeps visual anchors,
    // and the tracker budget is trimmed so the VIO step fits the camera interval.
    params.init_options.init_dyn_min_deg = std::min(params.init_options.init_dyn_min_deg, 3.0);
    params.dt_slam_delay = std::min(params.dt_slam_delay, 1.0);
    params.num_pts = std::min(params.num_pts, 200);
    params.state_options.max_msckf_in_update = std::min(params.state_options.max_msckf_in_update, 30);
    LOGI("handheld policy: init_dyn_min_deg=%.1f dt_slam_delay=%.1f num_pts=%d max_msckf_in_update=%d",
         params.init_options.init_dyn_min_deg, params.dt_slam_delay, params.num_pts, params.state_options.max_msckf_in_update);
    LOGI("startup: handheld dynamic enabled, 3s window, 0.5s optimizer; guarded static fallback");
    LOGI("recovery startup: recent stationary bias prior=%d",live->startupPrior.valid);
    LOGI("feature tracking: bidirectional KLT + geometry gates; clone window=%d", params.state_options.max_clone_size);
    LOGI("live policy: stationary updates available throughout scan (enabled=%d)", params.try_zupt);
    live->vio = std::make_shared<VioManager>(params);
    live->tsdf = std::make_unique<VoxelHashTSDF>(0.03f);
    env->ReleaseStringUTFChars(configPath, path);
    LOGI("OpenVINS ready");
    return reinterpret_cast<jlong>(live);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_recon_videoscan_live_Native_submitXFeat(
    JNIEnv* env, jclass, jlong h, jdouble t, jint w, jint hgt, jfloatArray candidates) {
    auto live = reinterpret_cast<Live*>(h); if (!live || !candidates) return JNI_FALSE;
    const jsize length = env->GetArrayLength(candidates);
    if (length < 0 || length % 3 != 0) return JNI_FALSE;
    std::vector<float> points(static_cast<size_t>(length));
    if (length) env->GetFloatArrayRegion(candidates, 0, length, points.data());
    std::lock_guard<std::mutex> lk(live->vioMutex);
    return live->xfeat.submit(t, w, hgt, points.data(), points.size() / 3) ? JNI_TRUE : JNI_FALSE;
}

// Transfer ownership only while the Activity's write gate has drained old VIO/depth
// jobs. A new estimator starts in a new local frame, with fusion disabled until PnP.
extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_retainMap(JNIEnv*,jclass,jlong oldHandle,jlong newHandle) {
    auto old=reinterpret_cast<Live*>(oldHandle), next=reinterpret_cast<Live*>(newHandle);
    if(!old || !next || old==next) return;
    std::scoped_lock lock(old->vioMutex,old->tsdfMutex,next->vioMutex,next->tsdfMutex);
    next->tsdf=std::move(old->tsdf);next->map=std::move(old->map);
    const bool hasSurface=next->tsdf->hasSurface();
    // A provisional frame allocates voxels without producing visible geometry.
    // With no saved reference there is nothing to relocalize against. Discard
    // those unconfirmed voxels before accepting a new coordinate frame.
    if (!hasSurface && next->map.frames.empty()) next->tsdf->clear();
    next->map.restart(hasSurface);
}

// Returns aligned/keyframes/inliers/confirmations/error. Pose + landmarks are
// transformed together, leaving the complete local EKF state/covariance untouched.
extern "C" JNIEXPORT jdoubleArray JNICALL Java_com_recon_videoscan_live_Native_mapFrame(
    JNIEnv* env,jclass,jlong h,jdouble t,jboolean reliable,jboolean poseValid,jfloatArray pose,
    jfloatArray landmarks,jint count,jfloatArray candidates,jfloatArray descriptorMap,jboolean invalidateAlignment,jdoubleArray calibration) {
    auto live=reinterpret_cast<Live*>(h);double info[5]={0,0,0,0,0};
    auto result=env->NewDoubleArray(5);if(!live) return result;
    std::lock_guard<std::mutex> lock(live->vioMutex);
    // Frame quality and map-frame continuity are separate. A rejected visual
    // update pauses fusion in Kotlin; it does not change the local VIO origin.
    if(invalidateAlignment && live->map.aligned) live->map.lost();
    if(poseValid && env->GetArrayLength(pose)==7 && count>=0 && env->GetArrayLength(landmarks)>=3*count) {
        float p[7];env->GetFloatArrayRegion(pose,0,7,p);
        Eigen::Quaterniond q(p[6],p[3],p[4],p[5]);
        live_map::Pose local{q.normalized().toRotationMatrix()*Eigen::Vector3d(1,-1,-1).asDiagonal(),Eigen::Vector3d(p[0],p[1],p[2])};
        std::vector<float> lm(count*3); if(count) env->GetFloatArrayRegion(landmarks,0,count*3,lm.data());
        std::vector<Eigen::Vector3d> points;for(int i=0;i<count;++i)points.emplace_back(lm[3*i],lm[3*i+1],lm[3*i+2]);
        if(reliable && candidates && descriptorMap && env->GetArrayLength(descriptorMap)==64*24*40) {
            const int nc=std::min(1000,env->GetArrayLength(candidates)/3);std::vector<float> xy(nc*3),field(64*24*40);
            env->GetFloatArrayRegion(candidates,0,nc*3,xy.data());env->GetFloatArrayRegion(descriptorMap,0,field.size(),field.data());
            std::shared_ptr<ov_core::CamBase> camera=live->vio->get_state()->_cam_intrinsics_cameras.at(0);
            if(calibration && env->GetArrayLength(calibration)==11) {
                double values[11]; env->GetDoubleArrayRegion(calibration,0,11,values);
                bool valid=true;for(double value:values) valid=valid&&std::isfinite(value);
                if(!valid || values[0]<=0 || values[1]<=0 || values[8]<=0 || values[9]<=0) {
                    env->SetDoubleArrayRegion(result,0,5,info);return result;
                }
                if(values[10]==1.0) camera=std::make_shared<ov_core::CamEqui>(int(values[8]),int(values[9]));
                else camera=std::make_shared<ov_core::CamRadtan>(int(values[8]),int(values[9]));
                Eigen::VectorXd intrinsics(8);for(int i=0;i<8;++i) intrinsics[i]=values[i];
                camera->set_value(intrinsics);
            }
            // Landmarks need descriptors at their actual projected locations.
            // Requiring a fresh detector maximum within 3px of a KLT track made
            // reference creation depend on an accidental detector/track overlap.
            if(live->map.aligned) for(const auto& point:points) {
                const Eigen::Vector3d pc=local.r.transpose()*(point-local.p);
                if(!pc.allFinite() || pc.z()<.3 || pc.z()>8) continue;
                const auto pixel=camera->distort_cv(cv::Point2f(pc.x()/pc.z(),pc.y()/pc.z()));
                if(std::isfinite(pixel.x) && std::isfinite(pixel.y) && pixel.x>=0 && pixel.y>=0 && pixel.x<camera->w() && pixel.y<camera->h())
                    {xy.push_back(pixel.x);xy.push_back(pixel.y);xy.push_back(1.f);}
            }
            const double scale=std::min(320.0/camera->w(),192.0/camera->h());
            const int rw=int(std::floor(camera->w()*scale)),rh=int(std::floor(camera->h()*scale));
            const int pl=(320-rw)/2,pt=(192-rh)/2;
            // Unit-normalize each feature-map vector, bilinear sample with the
            // same grid_sample coordinates as XFeat reliability, then normalize.
            for(int i=0;i<960;++i) {double norm=0;for(int c=0;c<64;++c)norm+=double(field[c*960+i])*field[c*960+i];
                norm=std::sqrt(norm);for(int c=0;c<64;++c)field[c*960+i]/=std::max(1e-12,norm);}
            cv::Mat desc;std::vector<live_map::Feature> features;
            for(size_t i=0;i<xy.size()/3;++i) {
                const float x=xy[3*i],y=xy[3*i+1];if(!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x>=camera->w()||y>=camera->h())continue;
                const float sx=(((x+.5f)*rw/camera->w()-.5f+pl)*40/319)-.5f;
                const float sy=(((y+.5f)*rh/camera->h()-.5f+pt)*24/191)-.5f;
                int x0=int(std::floor(sx)),y0=int(std::floor(sy));float tx=sx-x0,ty=sy-y0;
                cv::Mat row(1,64,CV_32F,cv::Scalar(0));
                for(int c=0;c<64;++c)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                    int xx=x0+dx,yy=y0+dy;if(xx>=0&&xx<40&&yy>=0&&yy<24)
                        row.at<float>(c)+=field[c*960+yy*40+xx]*(dx?tx:1-tx)*(dy?ty:1-ty);
                }
                const double norm=cv::norm(row);if(!std::isfinite(norm)||norm<1e-6)continue;row/=norm;
                const auto ray=camera->undistort_cv(cv::Point2f(x,y));if(!std::isfinite(ray.x)||!std::isfinite(ray.y))continue;
                desc.push_back(row);features.push_back({cv::Point2f(x,y),ray});
            }
            const double focal=std::max(camera->get_K()(0,0),camera->get_K()(1,1));
            const bool wasAligned=live->map.aligned;
            if(!wasAligned) {
                try { live->map.locate(t,local,features,desc,focal); }
                catch(const cv::Exception& e) { live->map.lost(); LOGI("relocalization geometry rejected: %s",e.what()); }
            }
            // Do not use the recovery frame itself to create new map references.
            if(wasAligned) live->map.remember(t,local,features,desc,points,focal);
        }
        if(live->map.aligned) {
            const auto mapped=live->map.mapped(local);Eigen::Quaterniond qm(mapped.r*Eigen::Vector3d(1,-1,-1).asDiagonal());
            float out[7]={float(mapped.p.x()),float(mapped.p.y()),float(mapped.p.z()),float(qm.x()),float(qm.y()),float(qm.z()),float(qm.w())};
            env->SetFloatArrayRegion(pose,0,7,out);
            for(int i=0;i<count;++i) {const auto v=live->map.mapped(points[i]);for(int k=0;k<3;++k)lm[3*i+k]=float(v[k]);}
            if(count)env->SetFloatArrayRegion(landmarks,0,count*3,lm.data());
        }
    }
    info[0]=live->map.aligned;info[1]=live->map.frames.size();info[2]=live->map.inliers;info[3]=live->map.confirmations;info[4]=live->map.residual;
    env->SetDoubleArrayRegion(result,0,5,info);return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_setXFeatEnabled(
    JNIEnv*, jclass, jlong h, jboolean enabled) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return;
    std::lock_guard<std::mutex> lk(live->vioMutex);
    live->xfeat.setEnabled(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_feedImu(JNIEnv*, jclass, jlong h, jdouble t, jdouble wx, jdouble wy, jdouble wz, jdouble ax, jdouble ay, jdouble az) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return;
    ov_core::ImuData m; m.timestamp = t; m.wm << wx, wy, wz; m.am << ax, ay, az;
    std::lock_guard<std::mutex> lk(live->vioMutex);
    if (!std::isfinite(t) || t <= live->lastImu || !m.wm.allFinite() || !m.am.allFinite()) return;
    live->vio->feed_measurement_imu(m);
    live->stationaryDetector.imu(t, {wx, wy, wz}, {ax, ay, az});
    live->firstImu = std::min(live->firstImu, (double)t); live->lastImu = t;
}

extern "C" JNIEXPORT jdouble JNICALL Java_com_recon_videoscan_live_Native_cameraImuOffset(JNIEnv*, jclass, jlong h) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return std::numeric_limits<double>::quiet_NaN();
    std::lock_guard<std::mutex> lk(live->vioMutex);
    return live->vio->get_state()->_calib_dt_CAMtoIMU->value()(0);
}

// Only returns a pose for this exact camera timestamp, in OpenGL camera axes / Y-up world.
extern "C" JNIEXPORT jint JNICALL Java_com_recon_videoscan_live_Native_feedImage(JNIEnv* env, jclass, jlong h, jdouble t, jint w, jint hgt, jbyteArray gray, jfloatArray pose, jfloatArray landmarks, jdoubleArray calibration, jdoubleArray diagnostics) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return -1;
    if (env->GetArrayLength(diagnostics) != 70) return -1;
    double diagnosticOut[70] = {t, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 10; i < 70; ++i) diagnosticOut[i] = std::numeric_limits<double>::quiet_NaN();
    env->SetDoubleArrayRegion(diagnostics, 0, 70, diagnosticOut);
    std::lock_guard<std::mutex> lk(live->vioMutex);
    live->featureSamples.clear();
    const double offset = live->vio->get_state()->_calib_dt_CAMtoIMU->value()(0);
    const double imuTime = t + offset;
    if (!std::isfinite(imuTime) || t <= live->lastCamera ||
        live->firstImu > imuTime || live->lastImu <= imuTime) return -1;
    const auto camera = live->vio->get_state()->_cam_intrinsics_cameras.at(0);
    if (camera->w() != w || camera->h() != hgt || env->GetArrayLength(calibration) != 11) {
        LOGI("VIO calibration %dx%d does not match frame %dx%d (snapshot length %d)", camera->w(), camera->h(), w, hgt, env->GetArrayLength(calibration)); return -1;
    }
    jbyte* g = env->GetByteArrayElements(gray, nullptr);
    cv::Mat img(hgt, w, CV_8UC1, g);
    ov_core::CameraData cam; cam.timestamp = t; cam.sensor_ids.push_back(0); cam.images.push_back(img.clone()); cam.masks.push_back(cv::Mat::zeros(hgt, w, CV_8UC1));
    env->ReleaseByteArrayElements(gray, g, JNI_ABORT);
    live_diagnostics::current = {};
    live_diagnostics::current.previousCameraTime = live->lastCamera;
    live->lastCamera = t;
    live_diagnostics::detector = &live->stationaryDetector;
    live_diagnostics::startupPrior = live->startupPrior;
    live_xfeat::FeedScope xfeatScope(&live->xfeat);
    live->vio->feed_measurement_camera(cam);
    auto state = live->vio->get_state();
    const auto &d = live_diagnostics::current;
    diagnosticOut[1] = state->_timestamp;
    diagnosticOut[2] = d.tracks;
    diagnosticOut[3] = d.msckfAttempted; diagnosticOut[4] = d.msckfAccepted;
    diagnosticOut[5] = d.slamAttempted; diagnosticOut[6] = d.slamAccepted;
    diagnosticOut[7] = d.stationary ? 1 : 0;
    diagnosticOut[8] = state->_imu->vel().norm();
    if (live->vio->initialized()) {
        const auto cov = StateHelper::get_marginal_covariance(state, {state->_imu->p()});
        diagnosticOut[9] = cov.allFinite() && cov.diagonal().minCoeff() >= 0 ?
            std::sqrt(cov.diagonal().maxCoeff()) : std::numeric_limits<double>::quiet_NaN();
    }
    for (int i = 0; i < 60; ++i) diagnosticOut[10 + i] = d.stationaryDetails[i];
    env->SetDoubleArrayRegion(diagnostics, 0, 70, diagnosticOut);
    // Timestamp-associated normalized rays + the geometry used to interpret IMU
    // motion. No pixels/images are recorded. Getter returns this saved snapshot.
    auto &samples=live->featureSamples; samples.assign(14,0);
    samples[1]=state->_calib_dt_CAMtoIMU->value()(0);
    for(int i=0;i<3;++i) samples[2+i]=state->_imu->bias_g()(i);
    const auto rotation=state->_calib_IMUtoCAM.at(0)->Rot();
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) samples[5+3*r+c]=rotation(r,c);
    const size_t step=std::max(size_t(1),(d.witness.size()+191)/192);
    for(size_t i=0;i<d.witness.size() && samples[0]<192;i+=step) {
        const auto &p=d.witness[i]; const auto n=camera->undistort_cv(cv::Point2f(p.x,p.y));
        if(!std::isfinite(n.x)||!std::isfinite(n.y)) continue;
        samples.push_back(double(p.id)); samples.push_back(n.x); samples.push_back(n.y); samples[0]++;
    }
    if (!live->vio->initialized()) {
        if (t-live->lastDiagnostic >= 1.0) {
            live->lastDiagnostic=t;
            LOGI("startup t=%.3f guard=%.0f gyro_mean=[%.5f,%.5f] gyro_rms=%.5f accel_std=%.5f (NaN: earlier init checks pending)",
                 t,d.stationaryDetails[29],d.stationaryDetails[30],d.stationaryDetails[31],d.stationaryDetails[37],d.stationaryDetails[32]);
        }
        return 0;
    }
    // Initialisation or failed propagation can leave a state from a different frame.
    if (!std::isfinite(state->_timestamp) || std::abs(state->_timestamp - t) > 1e-6) return 0;
    const auto intrinsics = state->_cam_intrinsics.at(0)->value();
    const double model = std::dynamic_pointer_cast<ov_core::CamRadtan>(camera) ? 0.0 :
        (std::dynamic_pointer_cast<ov_core::CamEqui>(camera) ? 1.0 : -1.0);
    if (model < 0 || !intrinsics.allFinite() || intrinsics(0) <= 0 || intrinsics(1) <= 0) return -1;
    double calibrationOut[11];
    for (int i = 0; i < 8; ++i) calibrationOut[i] = intrinsics(i);
    calibrationOut[8] = w; calibrationOut[9] = hgt; calibrationOut[10] = model;
    env->SetDoubleArrayRegion(calibration, 0, 11, calibrationOut);
    Eigen::Matrix3d R_GtoI = state->_imu->Rot(); Eigen::Vector3d p_IinG = state->_imu->pos();
    auto ext = state->_calib_IMUtoCAM.at(0); Eigen::Matrix3d R_ItoC = ext->Rot(); Eigen::Vector3d p_IinC = ext->pos();
    Eigen::Matrix3d R_CtoI = R_ItoC.transpose(); Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;
    Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_CtoI; Eigen::Vector3d p_CinG = p_IinG + R_GtoI.transpose() * p_CinI;
    // world<-camera in the OpenGL camera convention the app uses everywhere (camera looks -Z): R_gl = R_cv * diag(1,-1,-1)
    Eigen::Matrix3d R_w = ZUP_TO_YUP * R_CtoG * Eigen::Vector3d(1, -1, -1).asDiagonal(); Eigen::Vector3d p_w = ZUP_TO_YUP * p_CinG; Eigen::Quaterniond q(R_w);
    float out[7] = {(float)p_w.x(), (float)p_w.y(), (float)p_w.z(), (float)q.x(), (float)q.y(), (float)q.z(), (float)q.w()};
    env->SetFloatArrayRegion(pose, 0, 7, out);
    std::vector<Eigen::Vector3d> pts;
    double activeTime = -1;
    std::unordered_map<size_t,Eigen::Vector3d> activePoints, activePixels;
    live->vio->get_active_tracks(activeTime,activePoints,activePixels);
    // Retired MSCKF tracks are often already outside the image. Use the active
    // triangulations produced by this very update, checked against current pixels.
    if(std::abs(activeTime-t)<1e-6) for(const auto& item:activePixels) {
        const auto found=activePoints.find(item.first); if(found==activePoints.end()) continue;
        const auto& point=found->second;
        const Eigen::Vector3d pc=R_CtoG.transpose()*(point-p_CinG);
        if(!point.allFinite() || !pc.allFinite() || pc.z()<.2 || pc.z()>8) continue;
        const auto uv=camera->distort_cv(cv::Point2f(pc.x()/pc.z(),pc.y()/pc.z()));
        if(std::hypot(uv.x-item.second.x(),uv.y-item.second.y())<=2.5) pts.push_back(point);
    }
    const auto appendUnique=[&](const std::vector<Eigen::Vector3d>& candidates) {
        for(const auto& point:candidates) if(point.allFinite() &&
            std::none_of(pts.begin(),pts.end(),[&](const Eigen::Vector3d& p){return (p-point).squaredNorm()<1e-8;})) pts.push_back(point);
    };
    appendUnique(live->vio->get_features_SLAM()); appendUnique(live->vio->get_good_features_MSCKF());
    const int cap = env->GetArrayLength(landmarks) / 3; int n = 0; std::vector<float> buf;
    for (auto& p : pts) { if (n >= cap) break; Eigen::Vector3d pw = ZUP_TO_YUP * p; buf.push_back((float)pw.x()); buf.push_back((float)pw.y()); buf.push_back((float)pw.z()); n++; }
    if (n) env->SetFloatArrayRegion(landmarks, 0, n * 3, buf.data());
    if (t - live->lastDiagnostic >= 1.0) {
        live->lastDiagnostic = t;
        LOGI("tracking t=%.3f p=[%.5f,%.5f,%.5f] speed=%.5f landmarks=%d", t,
             p_w.x(), p_w.y(), p_w.z(), state->_imu->vel().norm(), n);
        LOGI("features tracks=%d klt=[%.0f,%.0f,%.0f,%.0f] age5=%.0f clones=%.0f visual_accepted=%d/%d tri_condition=%.0f",
             d.tracks,d.stationaryDetails[40],d.stationaryDetails[41],d.stationaryDetails[42],d.stationaryDetails[43],
             d.stationaryDetails[45],d.stationaryDetails[52],d.msckfAccepted+d.slamAccepted,
             d.msckfAttempted+d.slamAttempted,d.stationaryDetails[47]);
        const auto xs = live->xfeat.stats();
        LOGI("xfeat submitted=%zu consumed=%zu selected_total=%zu selected_last=%zu", xs.submitted, xs.consumed, xs.selected, xs.lastSelected);
        LOGI("stationary evidence=%.0f velocity_evidence=%.0f correction=%.0f trend=%.3f gyro_mean=%.5f reject=%.0f msckf_reject=[%.0f,%.0f,%.0f,%.0f]",
             d.stationaryDetails[14],d.stationaryDetails[28],d.stationaryDetails[15],d.stationaryDetails[25],d.stationaryDetails[27],
             d.stationaryDetails[39],d.stationaryDetails[33],d.stationaryDetails[34],d.stationaryDetails[35],d.stationaryDetails[36]);
    }
    return 1 + n;
}

extern "C" JNIEXPORT jdoubleArray JNICALL Java_com_recon_videoscan_live_Native_featureSamples(JNIEnv* env,jclass,jlong h) {
    auto live=reinterpret_cast<Live*>(h); if(!live) return env->NewDoubleArray(0);
    std::lock_guard<std::mutex> lock(live->vioMutex);
    auto result=env->NewDoubleArray(live->featureSamples.size());
    if(result && !live->featureSamples.empty()) env->SetDoubleArrayRegion(result,0,live->featureSamples.size(),live->featureSamples.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_tsdfReset(JNIEnv*, jclass, jlong h, jfloat voxel) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return; std::lock_guard<std::mutex> lk(live->tsdfMutex); live->tsdf = std::make_unique<VoxelHashTSDF>(voxel);
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_recon_videoscan_live_Native_tsdfIntegrate(JNIEnv* env, jclass, jlong h, jshortArray depth, jint w, jint hgt, jfloatArray K, jfloatArray R, jfloatArray c, jbyteArray rgba, jint rw, jint rh, jint stride, jbyteArray confidence) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return nullptr;
    if (!depth || !K || !R || !c || w<=0 || hgt<=0 || w>4096 || hgt>4096 || stride<=0 ||
        env->GetArrayLength(depth)!=int64_t(w)*hgt || env->GetArrayLength(K)!=4 ||
        env->GetArrayLength(R)!=9 || env->GetArrayLength(c)!=3 ||
        (confidence && env->GetArrayLength(confidence)!=int64_t(w)*hgt) ||
        (rgba && (rw<=0 || rh<=0 || env->GetArrayLength(rgba)!=int64_t(rw)*rh*4))) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid TSDF input dimensions");
        return nullptr;
    }
    jshort* d = env->GetShortArrayElements(depth, nullptr); jfloat* k = env->GetFloatArrayElements(K, nullptr); jfloat* r = env->GetFloatArrayElements(R, nullptr); jfloat* cc = env->GetFloatArrayElements(c, nullptr);
    jbyte* rgb = rgba ? env->GetByteArrayElements(rgba, nullptr) : nullptr;
    jbyte* weights = confidence ? env->GetByteArrayElements(confidence, nullptr) : nullptr;
    VoxelHashTSDF::Stats st;
    { std::lock_guard<std::mutex> lk(live->tsdfMutex); st = live->tsdf->integrate(reinterpret_cast<const uint16_t*>(d), w, hgt, k, r, cc, reinterpret_cast<const uint8_t*>(rgb), rw, rh, stride, reinterpret_cast<const uint8_t*>(weights)); }
    env->ReleaseShortArrayElements(depth, d, JNI_ABORT); env->ReleaseFloatArrayElements(K, k, JNI_ABORT); env->ReleaseFloatArrayElements(R, r, JNI_ABORT); env->ReleaseFloatArrayElements(c, cc, JNI_ABORT);
    if (rgb) env->ReleaseByteArrayElements(rgba, rgb, JNI_ABORT);
    if (weights) env->ReleaseByteArrayElements(confidence, weights, JNI_ABORT);
    float out[6] = {(float)st.integrate_ms, (float)st.extract_ms, (float)st.blocks_touched, (float)st.blocks, (float)st.points, (float)st.status};
    jfloatArray res = env->NewFloatArray(6); env->SetFloatArrayRegion(res, 0, 6, out); return res;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_recon_videoscan_live_Native_tsdfSurface(JNIEnv* env, jclass, jlong h, jint maxPoints) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return nullptr;
    std::lock_guard<std::mutex> lk(live->tsdfMutex); const size_t n = live->tsdf->surface(live->surface, (size_t)maxPoints);
    jfloatArray res = env->NewFloatArray((jsize)(n * 6)); if (n) env->SetFloatArrayRegion(res, 0, (jsize)(n * 6), live->surface.data()); return res;
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_destroy(JNIEnv*, jclass, jlong h) { delete reinterpret_cast<Live*>(h); }

extern "C" JNIEXPORT jdoubleArray JNICALL Java_com_recon_videoscan_live_Native_biasPrior(JNIEnv* env,jclass,jlong h) {
    auto live=reinterpret_cast<Live*>(h); if(!live) return env->NewDoubleArray(0);
    std::lock_guard<std::mutex> lock(live->vioMutex);
    auto state=live->vio->get_state();
    const auto p=StateHelper::get_marginal_covariance(state,{state->_imu->bg()});
    double values[6];for(int k=0;k<3;++k) { values[k]=state->_imu->bias_g()(k);values[k+3]=p(k,k); }
    auto out=env->NewDoubleArray(6);env->SetDoubleArrayRegion(out,0,6,values);return out;
}
extern "C" JNIEXPORT jlong JNICALL Java_com_recon_videoscan_live_Native_archiveSurface(JNIEnv* env,jclass,jlong h,jstring path) {
    auto live=reinterpret_cast<Live*>(h); if(!live) return -1;
    const char* name=env->GetStringUTFChars(path,nullptr);
    std::lock_guard<std::mutex> lock(live->tsdfMutex);
    const auto count=live->tsdf->saveSurface(name);env->ReleaseStringUTFChars(path,name);return count;
}

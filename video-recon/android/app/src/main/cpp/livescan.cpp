// JNI bridge: OpenVINS (poses from camera + IMU) and the voxel-hash TSDF, all on the phone.
#include <jni.h>
#include <android/log.h>
#include <memory>
#include <mutex>
#include <vector>
#include <opencv2/core.hpp>
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "tsdf.h"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "livescan", __VA_ARGS__)
using namespace ov_msckf;

namespace {
struct Live {
    std::shared_ptr<VioManager> vio; std::mutex vioMutex;
    std::unique_ptr<VoxelHashTSDF> tsdf; std::mutex tsdfMutex; std::vector<float> surface;
};
// OpenVINS global frame G has z up; the viewer and the TSDF use Y up like the web page.
const Eigen::Matrix3d ZUP_TO_YUP = (Eigen::Matrix3d() << 1, 0, 0, 0, 0, 1, 0, -1, 0).finished();
}

extern "C" JNIEXPORT jlong JNICALL Java_com_recon_videoscan_live_Native_init(JNIEnv* env, jclass, jstring configPath) {
    const char* path = env->GetStringUTFChars(configPath, nullptr);
    auto live = new Live();
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    ov_core::Printer::setPrintLevel("INFO");
    VioManagerOptions params; params.print_and_load(parser);
    if (!parser->successful()) { LOGI("config parse failed: %s", path); env->ReleaseStringUTFChars(configPath, path); delete live; return 0; }
    live->vio = std::make_shared<VioManager>(params);
    live->tsdf = std::make_unique<VoxelHashTSDF>(0.03f);
    env->ReleaseStringUTFChars(configPath, path);
    LOGI("OpenVINS ready");
    return reinterpret_cast<jlong>(live);
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_feedImu(JNIEnv*, jclass, jlong h, jdouble t, jdouble wx, jdouble wy, jdouble wz, jdouble ax, jdouble ay, jdouble az) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return;
    ov_core::ImuData m; m.timestamp = t; m.wm << wx, wy, wz; m.am << ax, ay, az;
    std::lock_guard<std::mutex> lk(live->vioMutex); live->vio->feed_measurement_imu(m);
}

// Feeds one grey frame; when initialised fills pose[7] = camera centre (Y-up world) + quaternion xyzw of
// R_world<-camera(OpenCV axes), and landmarks (x,y,z in Y-up world) of the current SLAM/MSCKF features.
extern "C" JNIEXPORT jint JNICALL Java_com_recon_videoscan_live_Native_feedImage(JNIEnv* env, jclass, jlong h, jdouble t, jint w, jint hgt, jbyteArray gray, jfloatArray pose, jfloatArray landmarks) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return -1;
    jbyte* g = env->GetByteArrayElements(gray, nullptr);
    cv::Mat img(hgt, w, CV_8UC1, g);
    ov_core::CameraData cam; cam.timestamp = t; cam.sensor_ids.push_back(0); cam.images.push_back(img.clone()); cam.masks.push_back(cv::Mat::zeros(hgt, w, CV_8UC1));
    env->ReleaseByteArrayElements(gray, g, JNI_ABORT);
    std::lock_guard<std::mutex> lk(live->vioMutex);
    live->vio->feed_measurement_camera(cam);
    if (!live->vio->initialized()) return 0;
    auto state = live->vio->get_state();
    Eigen::Matrix3d R_GtoI = state->_imu->Rot(); Eigen::Vector3d p_IinG = state->_imu->pos();
    auto ext = state->_calib_IMUtoCAM.at(0); Eigen::Matrix3d R_ItoC = ext->Rot(); Eigen::Vector3d p_IinC = ext->pos();
    Eigen::Matrix3d R_CtoI = R_ItoC.transpose(); Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;
    Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_CtoI; Eigen::Vector3d p_CinG = p_IinG + R_GtoI.transpose() * p_CinI;
    // world<-camera in the OpenGL camera convention the app uses everywhere (camera looks -Z): R_gl = R_cv * diag(1,-1,-1)
    Eigen::Matrix3d R_w = ZUP_TO_YUP * R_CtoG * Eigen::Vector3d(1, -1, -1).asDiagonal(); Eigen::Vector3d p_w = ZUP_TO_YUP * p_CinG; Eigen::Quaterniond q(R_w);
    float out[7] = {(float)p_w.x(), (float)p_w.y(), (float)p_w.z(), (float)q.x(), (float)q.y(), (float)q.z(), (float)q.w()};
    env->SetFloatArrayRegion(pose, 0, 7, out);
    std::vector<Eigen::Vector3d> pts = live->vio->get_features_SLAM(); auto msckf = live->vio->get_good_features_MSCKF(); pts.insert(pts.end(), msckf.begin(), msckf.end());
    const int cap = env->GetArrayLength(landmarks) / 3; int n = 0; std::vector<float> buf;
    for (auto& p : pts) { if (n >= cap) break; Eigen::Vector3d pw = ZUP_TO_YUP * p; buf.push_back((float)pw.x()); buf.push_back((float)pw.y()); buf.push_back((float)pw.z()); n++; }
    if (n) env->SetFloatArrayRegion(landmarks, 0, n * 3, buf.data());
    return 1 + n;
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_tsdfReset(JNIEnv*, jclass, jlong h, jfloat voxel) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return; std::lock_guard<std::mutex> lk(live->tsdfMutex); live->tsdf = std::make_unique<VoxelHashTSDF>(voxel);
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_recon_videoscan_live_Native_tsdfIntegrate(JNIEnv* env, jclass, jlong h, jshortArray depth, jint w, jint hgt, jfloatArray K, jfloatArray R, jfloatArray c, jbyteArray rgba, jint rw, jint rh, jint stride) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return nullptr;
    jshort* d = env->GetShortArrayElements(depth, nullptr); jfloat* k = env->GetFloatArrayElements(K, nullptr); jfloat* r = env->GetFloatArrayElements(R, nullptr); jfloat* cc = env->GetFloatArrayElements(c, nullptr);
    jbyte* rgb = rgba ? env->GetByteArrayElements(rgba, nullptr) : nullptr;
    VoxelHashTSDF::Stats st;
    { std::lock_guard<std::mutex> lk(live->tsdfMutex); st = live->tsdf->integrate(reinterpret_cast<const uint16_t*>(d), w, hgt, k, r, cc, reinterpret_cast<const uint8_t*>(rgb), rw, rh, stride); }
    env->ReleaseShortArrayElements(depth, d, JNI_ABORT); env->ReleaseFloatArrayElements(K, k, JNI_ABORT); env->ReleaseFloatArrayElements(R, r, JNI_ABORT); env->ReleaseFloatArrayElements(c, cc, JNI_ABORT);
    if (rgb) env->ReleaseByteArrayElements(rgba, rgb, JNI_ABORT);
    float out[5] = {(float)st.integrate_ms, (float)st.extract_ms, (float)st.blocks_touched, (float)st.blocks, (float)st.points};
    jfloatArray res = env->NewFloatArray(5); env->SetFloatArrayRegion(res, 0, 5, out); return res;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_recon_videoscan_live_Native_tsdfSurface(JNIEnv* env, jclass, jlong h, jint maxPoints) {
    auto live = reinterpret_cast<Live*>(h); if (!live) return nullptr;
    std::lock_guard<std::mutex> lk(live->tsdfMutex); const size_t n = live->tsdf->surface(live->surface, (size_t)maxPoints);
    jfloatArray res = env->NewFloatArray((jsize)(n * 6)); if (n) env->SetFloatArrayRegion(res, 0, (jsize)(n * 6), live->surface.data()); return res;
}

extern "C" JNIEXPORT void JNICALL Java_com_recon_videoscan_live_Native_destroy(JNIEnv*, jclass, jlong h) { delete reinterpret_cast<Live*>(h); }

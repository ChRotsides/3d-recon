// Offline OpenVINS runner for a videoscan recording converted by convert_recording.py.
// Usage: run_offline <estimator_config.yaml> <data_dir> <out_traj.txt>
// data_dir: imu.csv (t_ns,wx,wy,wz,ax,ay,az), cam0.csv (t_ns per line), cam0/<t_ns>.png
// Output: TUM format "t px py pz qx qy qz qw" (q is Hamilton, IMU->global), one line per
// camera update, plus <out>.calib.txt with the final online-calibrated extrinsics/intrinsics.
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: run_offline <config.yaml> <data_dir> <out_traj.txt>\n");
    return 1;
  }
  std::string config_path = argv[1], data = argv[2], out_path = argv[3];
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  VioManagerOptions params;
  params.print_and_load(parser);
  if (!parser->successful()) {
    PRINT_ERROR("unable to parse all parameters\n");
    return 1;
  }
  auto sys = std::make_shared<VioManager>(params);

  std::vector<ov_core::ImuData> imu;
  {
    std::ifstream f(data + "/imu.csv");
    std::string line;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string tok;
      std::vector<double> v;
      while (std::getline(ss, tok, ','))
        v.push_back(std::stod(tok));
      if (v.size() != 7) continue;
      ov_core::ImuData m;
      m.timestamp = v[0] * 1e-9;
      m.wm << v[1], v[2], v[3];
      m.am << v[4], v[5], v[6];
      imu.push_back(m);
    }
  }
  std::vector<long long> cam_ts;
  {
    std::ifstream f(data + "/cam0.csv");
    std::string line;
    while (std::getline(f, line))
      if (!line.empty()) cam_ts.push_back(std::stoll(line));
  }
  PRINT_INFO("loaded %zu imu samples, %zu frames\n", imu.size(), cam_ts.size());

  std::ofstream out(out_path);
  std::ofstream calib(out_path + ".calib.txt");
  out << std::setprecision(9) << std::fixed;
  size_t i_imu = 0;
  int n_updates = 0;
  for (size_t k = 0; k < cam_ts.size(); k++) {
    double t = cam_ts[k] * 1e-9;
    // feed all IMU up to this frame (OpenVINS buffers and propagates internally)
    while (i_imu < imu.size() && imu[i_imu].timestamp <= t + 0.05) {
      sys->feed_measurement_imu(imu[i_imu]);
      i_imu++;
    }
    ov_core::CameraData cam;
    cam.timestamp = t;
    cam.sensor_ids.push_back(0);
    cv::Mat img = cv::imread(data + "/cam0/" + std::to_string(cam_ts[k]) + ".png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      PRINT_WARNING("missing frame %lld\n", cam_ts[k]);
      continue;
    }
    cam.images.push_back(img);
    cam.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
    sys->feed_measurement_camera(cam);
    if (!sys->initialized()) continue;
    auto state = sys->get_state();
    // OpenVINS stores q_GtoI as a JPL quaternion; the equivalent Hamilton IMU->G quaternion
    // has the same components with the vector part negated? No: JPL q_GtoI == Hamilton q_ItoG
    // component-wise (see docs.openvins.com/namespaceov__core.html). Write R_ItoG explicitly instead.
    Eigen::Matrix3d R_GtoI = state->_imu->Rot();
    Eigen::Matrix3d R_ItoG = R_GtoI.transpose();
    Eigen::Quaterniond q(R_ItoG);
    Eigen::Vector3d p = state->_imu->pos();
    out << state->_timestamp << " " << p.x() << " " << p.y() << " " << p.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    n_updates++;
    if (n_updates % 300 == 0) {
      PRINT_INFO("frame %zu/%zu  pos %.3f %.3f %.3f\n", k, cam_ts.size(), p.x(), p.y(), p.z());
    }
  }
  auto state = sys->get_state();
  if (sys->initialized()) {
    auto ext = state->_calib_IMUtoCAM.at(0);
    Eigen::Matrix3d R_ItoC = ext->Rot();
    Eigen::Vector3d p_IinC = ext->pos();
    calib << std::setprecision(8);
    calib << "R_ItoC\n" << R_ItoC << "\np_IinC\n" << p_IinC.transpose() << "\n";
    calib << "cam_intrinsics\n" << state->_cam_intrinsics.at(0)->value().transpose() << "\n";
    calib << "dt_CAMtoIMU\n" << state->_calib_dt_CAMtoIMU->value()(0) << "\n";
  }
  PRINT_INFO("done: %d pose updates written to %s\n", n_updates, out_path.c_str());
  return 0;
}

// Streaming OpenVINS runner: a line protocol on stdin/stdout so a Python process can drive the
// filter one measurement at a time. Same estimator and same output convention as run_offline.cpp.
//
//   in : "I t_ns wx wy wz ax ay az"      IMU sample (rad/s, m/s^2)
//        "C t_ns /abs/path/frame.png"    grayscale camera frame
//        "Q"                             quit (writes the K line first)
// OpenVINS printf()s straight to stdout, so at startup the real stdout is duplicated for the
// protocol and fd 1 is pointed at stderr: the caller's pipe then carries protocol lines only.
//   out: "P t px py pz qx qy qz qw"      TUM, Hamilton q, IMU->global, after every camera update
//        "N t"                           camera update but not initialised yet
//        "L t n x1 y1 z1 ..."            current SLAM landmarks in global frame
//        "K R_ItoC(9) p_IinC(3) intrinsics(8) dt_CAMtoIMU"   final calibration
// Every camera update flushes, so the caller can read a reply per frame.
#include <cstdio>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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
  if (argc < 2) {
    fprintf(stderr, "usage: vio_stream <config.yaml>\n");
    return 1;
  }
  FILE *out = fdopen(dup(STDOUT_FILENO), "w");
  dup2(STDERR_FILENO, STDOUT_FILENO);   // library chatter must not reach the protocol pipe
  auto parser = std::make_shared<ov_core::YamlParser>(argv[1]);
  std::string verbosity = "WARNING";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  VioManagerOptions params;
  params.print_and_load(parser);
  if (!parser->successful()) {
    PRINT_ERROR("unable to parse all parameters\n");
    return 1;
  }
  auto sys = std::make_shared<VioManager>(params);

  std::ostringstream buf;
  buf << std::setprecision(9) << std::fixed;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const char tag = line[0];
    if (tag == 'Q') break;
    std::istringstream ss(line.substr(1));
    if (tag == 'I') {
      long long t_ns;
      ov_core::ImuData m;
      if (!(ss >> t_ns >> m.wm(0) >> m.wm(1) >> m.wm(2) >> m.am(0) >> m.am(1) >> m.am(2))) continue;
      m.timestamp = t_ns * 1e-9;
      sys->feed_measurement_imu(m);
      continue;
    }
    if (tag != 'C') continue;
    long long t_ns;
    std::string path;
    if (!(ss >> t_ns >> path)) continue;
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      fprintf(out, "N %.9f\n", t_ns * 1e-9);
      fflush(out);
      continue;
    }
    ov_core::CameraData cam;
    cam.timestamp = t_ns * 1e-9;
    cam.sensor_ids.push_back(0);
    cam.images.push_back(img);
    cam.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
    sys->feed_measurement_camera(cam);
    buf.str("");
    if (!sys->initialized()) {
      buf << "N " << (t_ns * 1e-9) << "\n";
    } else {
      auto state = sys->get_state();
      Eigen::Matrix3d R_ItoG = state->_imu->Rot().transpose();
      Eigen::Quaterniond q(R_ItoG);
      Eigen::Vector3d p = state->_imu->pos();
      buf << "P " << state->_timestamp << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z()
          << " " << q.w() << "\n";
      std::vector<Eigen::Vector3d> lm = sys->get_features_SLAM();
      buf << "L " << state->_timestamp << " " << lm.size();
      for (auto &f : lm) buf << " " << f.x() << " " << f.y() << " " << f.z();
      buf << "\n";
    }
    fputs(buf.str().c_str(), out);
    fflush(out);
  }
  if (sys->initialized()) {
    auto state = sys->get_state();
    auto ext = state->_calib_IMUtoCAM.at(0);
    Eigen::Matrix3d R_ItoC = ext->Rot();
    Eigen::Vector3d p_IinC = ext->pos();
    Eigen::VectorXd intr = state->_cam_intrinsics.at(0)->value();
    buf.str("");
    buf << "K";
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++) buf << " " << R_ItoC(r, c);
    for (int i = 0; i < 3; i++) buf << " " << p_IinC(i);
    for (int i = 0; i < intr.size(); i++) buf << " " << intr(i);
    buf << " " << state->_calib_dt_CAMtoIMU->value()(0) << "\n";
    fputs(buf.str().c_str(), out);
    fflush(out);
  }
  return 0;
}

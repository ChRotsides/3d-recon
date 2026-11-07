  // Keep the stationary path when genuinely quiet; normal hand motion may fall
  // through to joint visual/inertial initialization instead of restarting a wait.
  const bool has_jerk = !disparity_detected_moving_2to1 && disparity_detected_moving_1to0;
  const bool is_still = !disparity_detected_moving_2to1 && !disparity_detected_moving_1to0;
  if (((has_jerk && wait_for_jerk) || (is_still && !wait_for_jerk)) && params.init_imu_thresh > 0.0 &&
      init_static->initialize(timestamp, covariance, order, t_imu, wait_for_jerk)) return true;
  if (params.init_dyn_use && init_dynamic) {
    if(imu_data->size()<2) return false;
    for(size_t i=0;i<imu_data->size();++i) {
      const auto &m=imu_data->at(i);
      if(!std::isfinite(m.timestamp) || !m.wm.allFinite() || !m.am.allFinite() ||
          (i && (m.timestamp<=imu_data->at(i-1).timestamp || m.timestamp-imu_data->at(i-1).timestamp>.03))) return false;
    }
    const auto began = std::chrono::steady_clock::now();
    // The upstream solver writes its output state before covariance recovery.
    // Failed attempts must not modify the live IMU or advertise a valid pose.
    auto candidate = std::make_shared<ov_type::IMU>();
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
    std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> features;
    Eigen::MatrixXd candidate_covariance;
    std::vector<std::shared_ptr<ov_type::Type>> candidate_order;
    double candidate_time = -1;
    const bool solved = std::static_pointer_cast<DynamicInitializer>(init_dynamic)->initialize(
        candidate_time, candidate_covariance, candidate_order, candidate, clones, features);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    double parallaxPixels = 0;
    const bool observable = solved && initializationHasParallax(_db, clones, params, parallaxPixels);
    if (observable && candidate->value().allFinite() && std::isfinite(candidate_time) &&
        candidate_covariance.rows()==15 && candidate_covariance.cols()==15 && candidate_covariance.allFinite() &&
        candidate_covariance.diagonal().minCoeff()>=0 && candidate_order.size()==1 && candidate_order[0]==candidate) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(candidate_covariance);
      if(eigen.info()==Eigen::Success && eigen.eigenvalues().minCoeff()>=-1e-10) {
        t_imu->set_value(candidate->value()); t_imu->set_fej(candidate->fej());
        covariance=candidate_covariance; order={t_imu}; timestamp=candidate_time;
        __android_log_print(ANDROID_LOG_INFO,"livescan","dynamic initialization accepted: solve=%.3fs state_t=%.6f speed=%.4f",seconds,timestamp,t_imu->vel().norm());
        PRINT_INFO("[handheld-init]: dynamic accepted solve=%.3fs speed=%.4f bg=%.6f %.6f %.6f\n",
            seconds,t_imu->vel().norm(),t_imu->bias_g()(0),t_imu->bias_g()(1),t_imu->bias_g()(2));
        return true;
      }
    }
    if(seconds>.01) PRINT_INFO("[handheld-init]: dynamic not ready solve=%.3fs; state not committed\n",seconds);
  }

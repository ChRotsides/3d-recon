# Compile app-owned instrumented translation units ahead of the static archive.
# Their original archive objects are not extracted by the linker. Keep the original
# dependency untouched and fail configuration if upstream hook sites change.
function(observe old new expected)
    string(REPLACE "${old}" "" stripped "${source}")
    string(LENGTH "${source}" before)
    string(LENGTH "${stripped}" after)
    string(LENGTH "${old}" length)
    math(EXPR count "(${before} - ${after}) / ${length}")
    if(NOT count EQUAL expected)
        message(FATAL_ERROR "OpenVINS diagnostic hook mismatch (${unit}): expected ${expected}, found ${count}: ${old}")
    endif()
    string(REPLACE "${old}" "${new}" source "${source}")
    set(source "${source}" PARENT_SCOPE)
endfunction()

function(observe_region start_marker end_marker replacement)
    observe("${start_marker}" "${start_marker}" 1)
    observe("${end_marker}" "${end_marker}" 1)
    string(FIND "${source}" "${start_marker}" begin)
    string(FIND "${source}" "${end_marker}" end)
    if(end LESS_EQUAL begin)
        message(FATAL_ERROR "Invalid OpenVINS region in ${unit}")
    endif()
    math(EXPR length "${end}-${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} old)
    observe("${old}" "${replacement}" 1)
    set(source "${source}" PARENT_SCOPE)
endfunction()

set(OBSERVED_OV_SOURCES)
foreach(unit core/VioManager core/VioManagerHelper update/UpdaterMSCKF update/UpdaterSLAM update/UpdaterZeroVelocity static/StaticInitializer track/TrackKLT feat/FeatureInitializer)
    if(unit STREQUAL "static/StaticInitializer")
        set(input "${OV_ROOT}/ov_init/${unit}.cpp")
    elseif(unit STREQUAL "track/TrackKLT" OR unit STREQUAL "feat/FeatureInitializer")
        set(input "${OV_ROOT}/ov_core/${unit}.cpp")
    else()
        set(input "${OV_ROOT}/ov_msckf/${unit}.cpp")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
    file(READ "${input}" source)
    if(unit STREQUAL "core/VioManager")
        observe("trackFEATS->feed_new_camera(message);"
            "trackFEATS->feed_new_camera(message);\n  const auto observations = trackFEATS->get_last_obs();\n  const auto ids = trackFEATS->get_last_ids();\n  std::vector<StationaryDetector::Point> witness;\n  for (const auto &camera : observations) {\n    live_diagnostics::current.tracks += camera.second.size();\n    const auto found = ids.find(camera.first);\n    if (camera.first != 0 || found == ids.end()) continue;\n    for (size_t i = 0; i < std::min(camera.second.size(), found->second.size()); ++i) witness.push_back({found->second[i], camera.second[i].pt.x, camera.second[i].pt.y});\n  }\n  if (live_diagnostics::detector) live_diagnostics::detector->image(message.timestamp, witness, message.images.at(0).cols, message.images.at(0).rows);" 1)
        observe("if (did_zupt_update) {" "if (did_zupt_update) {\n      live_diagnostics::current.stationary = true;" 2)
        observe("assert(state->_timestamp == message.timestamp);"
            "assert(state->_timestamp == message.timestamp);\n      // A valid stationary EKF correction is also a completed state update.\n      if (timelastupdate < 0) timelastupdate = state->_timestamp;" 1)
        observe("if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {"
            "if (retireOldestFeatureTracks(state->_clones_IMU.size(), state->_options.max_clone_size)) {" 1)
        observe("// Return if we where unable to propagate"
            "live_diagnostics::current.stationaryDetails[52] = state->_clones_IMU.size();\n  // Return if we where unable to propagate" 1)
        observe("if (live_diagnostics::detector) live_diagnostics::detector->image(message.timestamp, witness, message.images.at(0).cols, message.images.at(0).rows);"
            "live_diagnostics::trackHistory(trackFEATS->get_feature_database(), witness);\n  if (live_diagnostics::detector) live_diagnostics::detector->image(message.timestamp, witness, message.images.at(0).cols, message.images.at(0).rows);" 1)
        observe("updaterMSCKF->update(state, featsup_MSCKF);"
            "live_diagnostics::current.msckfAttempted += featsup_MSCKF.size();\n  { live_diagnostics::MsckfScope scope; updaterMSCKF->update(state, featsup_MSCKF); }" 1)
        observe("updaterSLAM->update(state, featsup_TEMP);"
            "live_diagnostics::current.slamAttempted += featsup_TEMP.size();\n    updaterSLAM->update(state, featsup_TEMP);" 1)
    elseif(unit STREQUAL "core/VioManagerHelper")
        observe("std::thread thread([&] {" "auto *diagnostic_frame = &live_diagnostics::current;\n  const auto startup_prior=live_diagnostics::startupPrior;\n  std::thread thread([&, diagnostic_frame, startup_prior] {\n    live_diagnostics::startupPrior=startup_prior;" 1)
        observe("thread_init_running = false;"
            "if (!params.use_multi_threading_subs) {\n      for (int i : {29,30,31,32,37,38}) diagnostic_frame->stationaryDetails[i] = live_diagnostics::current.stationaryDetails[i];\n    }\n    thread_init_running = false;" 1)
    elseif(unit STREQUAL "track/TrackKLT")
        observe_region("  // Now do KLT tracking to get the valid new points" "  // Copy back the updated positions"
            "  const double focal = std::max({camera_calib.at(id0)->get_K()(0,0), camera_calib.at(id0)->get_K()(1,1), camera_calib.at(id1)->get_K()(0,0), camera_calib.at(id1)->get_K()(1,1)});\n  const auto e = robustKlt(img0pyr, img1pyr, pts0, pts1, win_size, pyr_levels, focal,\n      [&](const cv::Point2f &p){ return camera_calib.at(id0)->undistort_cv(p); },\n      [&](const cv::Point2f &p){ return camera_calib.at(id1)->undistort_cv(p); }, mask_out);\n  auto &d = live_diagnostics::current.stationaryDetails;\n  d[40]+=e.input; d[41]+=e.forward; d[42]+=e.roundTrip; d[43]+=e.geometric; d[58]+=e.homography;\n\n")
        observe("d[58]+=e.homography;" "d[58]+=e.homography; d[59]+=e.photometric;" 1)
        observe("perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);"
            "live_xfeat::setDetectionTimestamp(message.timestamp);\n    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);\n    for(size_t i=0;i<good_left.size();++i) {\n      const auto p=good_left[i].pt; const auto n=camera_calib.at(cam_id)->undistort_cv(p);\n      database->update_feature(good_ids_left[i],message.timestamp,cam_id,p.x,p.y,n.x,n.y);\n      live_diagnostics::current.stationaryDetails[54]++;\n    }" 1)
        observe("auto ids_left_old = ids_last[cam_id];" "auto ids_left_old = ids_last[cam_id];\n  const auto ids_before_topoff = ids_left_old;" 1)
        observe("perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);"
            "live_xfeat::setDetectionTimestamp(live_diagnostics::current.previousCameraTime);\n  perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);\n  const double previousTime=live_diagnostics::current.previousCameraTime;\n  if(std::isfinite(previousTime) && previousTime < message.timestamp) {\n    for(size_t i=0;i<ids_left_old.size();++i) {\n      if(std::find(ids_before_topoff.begin(),ids_before_topoff.end(),ids_left_old[i])!=ids_before_topoff.end()) continue;\n      const auto p=pts_left_old[i].pt; const auto n=camera_calib.at(cam_id)->undistort_cv(p);\n      database->update_feature(ids_left_old[i],previousTime,cam_id,p.x,p.y,n.x,n.y);\n      live_diagnostics::current.stationaryDetails[54]++;\n    }\n  }" 1)
        observe("  std::vector<cv::KeyPoint> pts0_ext;\n  Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);" "  std::vector<cv::KeyPoint> pts0_ext;\n  live_xfeat::Frame xfeatFrame;\n  const auto xfeatLookup = live_xfeat::lookup(img0pyr.at(0).cols, img0pyr.at(0).rows, xfeatFrame);\n  if (xfeatLookup == live_xfeat::Lookup::Found) {\n    cv::Mat validGrid = cv::Mat::zeros(size_grid, CV_8UC1);\n    for (const auto &loc : valid_locs) validGrid.at<uint8_t>(loc.second, loc.first) = 255;\n    cv::Mat selectionClose = grid_2d_close.clone();\n    for (const auto &candidate : xfeatFrame.candidates) {\n      if ((int)pts0.size() + (int)pts0_ext.size() >= num_features) break;\n      const int x = cvRound(candidate.x), y = cvRound(candidate.y), edge = 10;\n      if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge || mask0_updated.at<uint8_t>(y, x) > 127) continue;\n      const int xClose = (int)(candidate.x / (float)min_px_dist), yClose = (int)(candidate.y / (float)min_px_dist);\n      const int xGrid = (int)std::floor(candidate.x / size_x), yGrid = (int)std::floor(candidate.y / size_y);\n      if (xClose < 0 || xClose >= size_close.width || yClose < 0 || yClose >= size_close.height ||\n          xGrid < 0 || xGrid >= size_grid.width || yGrid < 0 || yGrid >= size_grid.height ||\n          selectionClose.at<uint8_t>(yClose, xClose) > 127 || validGrid.at<uint8_t>(yGrid, xGrid) == 0 ||\n          grid_2d_grid.at<uint8_t>(yGrid, xGrid) >= num_features_grid) continue;\n      pts0_ext.emplace_back(candidate.x, candidate.y, 7.0f, -1.0f, candidate.score);\n      selectionClose.at<uint8_t>(yClose, xClose) = 255;\n      grid_2d_grid.at<uint8_t>(yGrid, xGrid)++;\n    }\n    live_xfeat::selected(pts0_ext.size());\n  }\n  // XFeat is asynchronous and sparse in time. Missing/empty results must never\n  // disable replenishment, including immediately after complete track loss.\n  if ((int)pts0.size() + (int)pts0_ext.size() < num_features) {\n    for (const auto &point : pts0_ext)\n      cv::rectangle(mask0_updated, point.pt - cv::Point2f(min_px_dist,min_px_dist),\n                    point.pt + cv::Point2f(min_px_dist,min_px_dist), cv::Scalar(255), -1);\n    std::vector<cv::KeyPoint> fallback;\n    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, fallback, num_features, grid_x, grid_y, threshold, true);\n    const size_t remaining = std::max(0, num_features - (int)pts0.size() - (int)pts0_ext.size());\n    pts0_ext.insert(pts0_ext.end(), fallback.begin(), fallback.begin() + std::min(remaining,fallback.size()));\n  }" 1)
    elseif(unit STREQUAL "feat/FeatureInitializer")
        observe("if (std::abs(condA) > _options.max_cond_number || p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist ||\n      std::isnan(p_f.norm())) {"
            "if (live_diagnostics::triangulationRejected(condA, p_f, _options.max_cond_number, _options.min_dist, _options.max_dist)) {" 1)
    elseif(unit STREQUAL "update/UpdaterMSCKF")
        observe("(*it0)->clean_old_measurements(clonetimes);"
            "size_t observations_before=0; for(const auto &c:(*it0)->timestamps) observations_before+=c.second.size();\n    (*it0)->clean_old_measurements(clonetimes);\n    size_t observations_after=0; for(const auto &c:(*it0)->timestamps) observations_after+=c.second.size();\n    live_diagnostics::current.stationaryDetails[53]+=observations_before-observations_after;" 1)
        observe("if (ct_meas < 2) {" "if (ct_meas < 2) {\n      live_diagnostics::current.stationaryDetails[33]++;" 1)
        observe("if (!success_tri || !success_refine) {" "if (!success_tri || !success_refine) {\n      if (!success_tri) live_diagnostics::current.stationaryDetails[34]++;\n      else live_diagnostics::current.stationaryDetails[35]++;" 1)
        observe("if (chi2 > _options.chi2_multipler * chi2_check) {" "if (chi2 > _options.chi2_multipler * chi2_check) {\n      live_diagnostics::current.stationaryDetails[36]++;" 1)
        observe("StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);"
            "StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);\n  live_diagnostics::current.msckfAccepted += feature_vec.size();" 1)
    elseif(unit STREQUAL "update/UpdaterSLAM")
        observe("StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);"
            "StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);\n  live_diagnostics::current.slamAccepted += feature_vec.size();" 1)
    elseif(unit STREQUAL "static/StaticInitializer")
        observe("// Get rotation with z axis aligned with -g (z_in_G=0,0,1)"
            "if (!live_diagnostics::startupMotion(window_2to1, window_1to0, 0.5 * params.init_window_time, params.gravity_mag)) return false;\n  // Get rotation with z axis aligned with -g (z_in_G=0,0,1)" 1)
        observe("Eigen::Vector3d bg = w_avg_2to1;"
            "Eigen::Vector3d bg;\n  for(int k=0;k<3;++k) bg(k)=live_diagnostics::startupBias[k];" 1)
        observe("// A VIO system has 4dof unobservable directions"
            "for(int k=0;k<3;++k) covariance(9+k,9+k)=live_diagnostics::startupBiasVariance[k];\n\n  // A VIO system has 4dof unobservable directions" 1)
    else()
        observe("bool integrated_accel_constraint = false;"
            "live_diagnostics::stationarySnapshot(state, imu_recent, _gravity, false);\n  bool integrated_accel_constraint = false;" 1)
        observe("// Finally return\n  last_zupt_state_timestamp = timestamp;"
            "live_diagnostics::zeroVelocityMeasurement(state, timestamp);\n  live_diagnostics::stationarySnapshot(state, imu_recent, _gravity, true);\n  // Finally return\n  last_zupt_state_timestamp = timestamp;" 1)
        observe("if (!disparity_passed && (chi2 > _options.chi2_multipler * chi2_check || state->_imu->vel().norm() > _zupt_max_velocity)) {"
            "if (!live_diagnostics::stationaryEvidence(timestamp, time1)) {" 1)
        observe("PRINT_INFO(CYAN \"[ZUPT]: accepted |v_IinG| = %.3f (chi2 %.3f < %.3f)\\n\" RESET, state->_imu->vel().norm(), chi2,\n             _options.chi2_multipler * chi2_check);"
            "PRINT_DEBUG(CYAN \"[ZUPT]: accepted physical-rest evidence |v_IinG| = %.3f\\n\" RESET, state->_imu->vel().norm());" 1)
        observe("PRINT_INFO(CYAN \"[ZUPT]: passed disparity (%.3f < %.3f, %d features)\\n\" RESET, disp_avg, _zupt_max_disparity, (int)num_features);"
            "PRINT_DEBUG(CYAN \"[ZUPT]: legacy disparity statistic (not acceptance gate) %.3f\\n\" RESET, disp_avg);" 1)
    endif()
    get_filename_component(name "${unit}" NAME)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/observed_${name}.cpp")
    file(CONFIGURE OUTPUT "${output}" CONTENT "#include \"VioDiagnostics.h\"\n#include \"StationaryDiagnostics.h\"\n#include \"FeatureDiagnostics.h\"\n#include \"XFeatBridge.h\"\n${source}")
    list(APPEND OBSERVED_OV_SOURCES "${output}")
endforeach()

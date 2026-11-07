# Restore the existing OpenVINS dynamic initializer without changing the Android
# archive or its class layout (its helper is intentionally shared_ptr<void>).
set(OV_FULL_ROOT "$ENV{HOME}/ai/vio/open_vins" CACHE PATH "Complete matching OpenVINS source tree")
set(Ceres_DIR "$ENV{HOME}/ai/phone/ceres-android-2.1.0/install-arm64/lib/cmake/Ceres" CACHE PATH "Android arm64 Ceres installation")
set(Eigen3_DIR "${THIRD_PARTY_INCLUDE}/../share/eigen3/cmake" CACHE PATH "Matching Eigen headers")
find_package(Ceres 2.1 REQUIRED)
set(DYNAMIC_ROOT "${OV_FULL_ROOT}/ov_init/src")
file(SHA256 "${DYNAMIC_ROOT}/init/InertialInitializerOptions.h" full_options_hash)
file(SHA256 "${OV_ROOT}/ov_init/init/InertialInitializerOptions.h" android_options_hash)
if(NOT full_options_hash STREQUAL android_options_hash)
    message(FATAL_ERROR "Dynamic initializer options do not match Android ABI")
endif()

set(unit "init/InertialInitializer")
file(READ "${OV_ROOT}/ov_init/${unit}.cpp" source)
observe_region("#ifndef __ANDROID__\n// dynamic initializer (Ceres) not built for Android" "#include \"static/StaticInitializer.h\""
    "#include \"dynamic/DynamicInitializer.h\"\n#include \"types/IMU.h\"\n#include <chrono>\n")
observe("#ifndef __ANDROID__\n  PRINT_ERROR(\"dynamic init is not compiled in; set init_dyn_use=false\\n\"); std::exit(EXIT_FAILURE);\n#else\n  init_dynamic = nullptr;\n#endif"
    "if(params.init_dyn_use) init_dynamic = std::make_shared<DynamicInitializer>(params, _db, imu_data);" 1)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/HandheldDispatch.inc" dispatch)
observe_region("  // Use our static initializer!" "  return false;\n}" "${dispatch}\n")
set(output "${CMAKE_CURRENT_BINARY_DIR}/observed_InertialInitializer.cpp")
file(CONFIGURE OUTPUT "${output}" CONTENT "#include <android/log.h>\n#include \"InitializationParallax.h\"\n${source}" @ONLY)
list(APPEND OBSERVED_OV_SOURCES "${output}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${OV_ROOT}/ov_init/${unit}.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/HandheldDispatch.inc")
set(unit "dynamic/DynamicInitializer")
set(input "${DYNAMIC_ROOT}/${unit}.cpp")
file(READ "${input}" source)
observe_region("  // Eigen::MatrixXd A1A1_inv" "  Eigen::Matrix<double, 7, 1> coeff"
    "  // Eliminate feature/velocity columns without squaring their condition number\n  // or allocating the full measurement-space projector.\n  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> feature_solver(A1);\n  feature_solver.setThreshold(1e-10);\n  if (feature_solver.rank() != A1.cols()) return false;\n  const Eigen::MatrixXd A2 = A.rightCols(3);\n  const Eigen::MatrixXd residual_A2 = A2 - A1 * feature_solver.solve(A2);\n  const Eigen::VectorXd residual_b = b - A1 * feature_solver.solve(b);\n  Eigen::MatrixXd D = residual_A2.transpose() * residual_A2;\n  const Eigen::Vector3d d = residual_A2.transpose() * residual_b;\n")
observe("Eigen::VectorXd state_feat_vel = -A1A1_inv * A1.transpose() * A2 * state_grav + A1A1_inv * A1.transpose() * b;"
    "Eigen::VectorXd state_feat_vel = feature_solver.solve(b - A2 * state_grav);" 1)
observe_region("  // Create companion matrix of our polynomial" "  // Find its eigenvalues (can be complex)"
    "  const ScaledPolynomial polynomial(coeff);\n  if (!polynomial.valid) return false;\n  const auto &companion_matrix = polynomial.companion;\n\n")
observe("double lambda = val.real();" "double lambda = val.real() * polynomial.scale;" 1)
observe("if (val.imag() == 0)" "if (std::abs(val.imag()) <= 1e-9 * std::max(1.0, std::abs(val.real())))" 1)
# Only the root with positive-definite D-lambda*I minimizes the constrained
# least-squares problem. Other real roots can satisfy |g| while pointing gravity
# the wrong way; LLT success enforces this condition before comparing residuals.
observe("      Eigen::MatrixXd D_lambdaI_inv = (D - lambda * I_dd).llt().solve(I_dd);\n      Eigen::VectorXd state_grav = D_lambdaI_inv * d;"
    "      Eigen::LLT<Eigen::MatrixXd> gravity_solver(D - lambda * I_dd);\n      if (gravity_solver.info() != Eigen::Success) continue;\n      Eigen::VectorXd state_grav = gravity_solver.solve(d);\n      if (!state_grav.allFinite()) continue;" 1)
observe("if (std::abs(gravity_inI0.norm() - params.gravity_mag) > init_max_grav_difference)"
    "if (!x_hat.allFinite() || std::abs(gravity_inI0.norm() - params.gravity_mag) > init_max_grav_difference)" 1)
set(output "${CMAKE_CURRENT_BINARY_DIR}/observed_DynamicInitializer.cpp")
file(CONFIGURE OUTPUT "${output}" CONTENT "#include \"ScaledPolynomial.h\"\n${source}" @ONLY)
list(APPEND OBSERVED_OV_SOURCES "${output}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
foreach(unit ceres/Factor_GenericPrior ceres/Factor_ImageReprojCalib ceres/Factor_ImuCPIv1 ceres/State_JPLQuatLocal)
    list(APPEND OBSERVED_OV_SOURCES "${DYNAMIC_ROOT}/${unit}.cpp")
endforeach()

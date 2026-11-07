#pragma once
#include "feat/FeatureDatabase.h"
#include "feat/Feature.h"
#include "types/PoseJPL.h"
#include "init/InertialInitializerOptions.h"
#include "utils/quat_ops.h"
#include <algorithm>

// A converged IMU state after pure rotation is not a useful moving scan start.
// Require image displacement remaining after the solved rotation is removed.
// This is an initialization observability gate, never a zero-motion measurement.
inline bool initializationHasParallax(const std::shared_ptr<ov_core::FeatureDatabase> &db,
        const std::map<double,std::shared_ptr<ov_type::PoseJPL>> &clones,
        const ov_init::InertialInitializerOptions &params, double &medianPixels) {
    std::vector<double> distances;
    for(const auto &entry:db->get_internal_data()) {
        const auto &feature=*entry.second;
        for(const auto &camera:feature.timestamps) {
            const auto &times=camera.second;
            size_t first=times.size(),last=0;
            for(size_t i=0;i<times.size();++i) if(clones.count(times[i])) {
                if(first==times.size()) first=i;
                last=i;
            }
            if(first==times.size() || times[last]-times[first]<.8) continue;
            const Eigen::Vector4d q=params.camera_extrinsics.at(camera.first).head<4>();
            const Eigen::Matrix3d Ric=ov_core::quat_2_Rot(q);
            const Eigen::Matrix3d rotation=Ric*clones.at(times[last])->Rot()*clones.at(times[first])->Rot().transpose()*Ric.transpose();
            const auto a=feature.uvs_norm.at(camera.first).at(first).cast<double>().eval();
            const auto b=feature.uvs_norm.at(camera.first).at(last).cast<double>().eval();
            const Eigen::Vector3d ray=rotation*Eigen::Vector3d(a.x(),a.y(),1);
            if(!ray.allFinite() || ray.z()<=0) continue;
            const auto cam=params.camera_intrinsics.at(camera.first);
            const double focal=cam->get_K()(0,0)*960.0/cam->w();
            const double pixels=(ray.head<2>()/ray.z()-b).norm()*focal;
            if(std::isfinite(pixels)) distances.push_back(pixels);
        }
    }
    medianPixels=0;
    if(distances.size()<24) return false;
    std::sort(distances.begin(),distances.end()); medianPixels=distances[distances.size()/2];
    return medianPixels>=1.5;
}

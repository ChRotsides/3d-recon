#pragma once
#include "VioDiagnostics.h"
#include "FeatureTrackPolicy.h"
#include "RobustKlt.h"
#include "feat/FeatureDatabase.h"
#include "feat/Feature.h"
#include <Eigen/Core>

namespace live_diagnostics {
inline thread_local bool measuringMsckf=false;
struct MsckfScope {
    bool previous=measuringMsckf;
    MsckfScope() { measuringMsckf=true; }
    ~MsckfScope() { measuringMsckf=previous; }
};
inline bool triangulationRejected(double condition,const Eigen::Vector3d &point,double maxCondition,double minDistance,double maxDistance) {
    const bool nonfinite=!std::isfinite(condition)||!point.allFinite();
    const bool rejected=nonfinite||std::abs(condition)>maxCondition||point.z()<minDistance||point.z()>maxDistance;
    if(measuringMsckf) {
        auto &d=current.stationaryDetails;
        if(nonfinite) d[51]++;
        else if(std::abs(condition)>maxCondition) d[47]++;
        else if(point.z()<=0) d[48]++;
        else if(point.z()<minDistance) d[49]++;
        else if(point.z()>maxDistance) d[50]++;
        else d[55]++;
        if(std::isfinite(condition)) d[56]=std::isfinite(d[56]) ? std::min(d[56],std::abs(condition)):std::abs(condition);
        if(point.allFinite()) d[57]=std::isfinite(d[57]) ? std::max(d[57],std::abs(point.z())):std::abs(point.z());
    }
    return rejected;
}
inline void trackHistory(const std::shared_ptr<ov_core::FeatureDatabase> &database,const std::vector<StationaryDetector::Point> &points) {
    std::vector<double> lengths;
    for(const auto &p:points) {
        const auto f=database->get_feature(p.id); if(!f) continue;
        const auto it=f->timestamps.find(0); if(it==f->timestamps.end()) continue;
        const auto n=it->second.size(); lengths.push_back(n);
        current.stationaryDetails[44]+=n>=2; current.stationaryDetails[45]+=n>=5;
    }
    if(!lengths.empty()) { std::sort(lengths.begin(),lengths.end()); current.stationaryDetails[46]=lengths[lengths.size()/2]; }
    current.witness=points;
}
}

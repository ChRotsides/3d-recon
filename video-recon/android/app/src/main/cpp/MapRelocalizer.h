#pragma once
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <deque>
#include <set>

// A fixed, gravity-preserving transform relates a healthy local VIO instance to
// the persistent map. Never overwrite an EKF pose without its covariance/biases.
namespace live_map {
struct Pose {
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity(); // OpenCV camera -> world
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};
struct Feature { cv::Point2f pixel, ray; };
struct Keyframe { double t; Pose pose; std::vector<cv::Point3f> points; cv::Mat descriptors; };
class Relocalizer {
public:
    bool aligned = true;
    int inliers = 0, confirmations = 0;
    double residual = 0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::deque<Keyframe> frames;
    Pose mapped(const Pose& local) const { return {rotation * local.r, rotation * local.p + translation}; }
    Eigen::Vector3d mapped(const Eigen::Vector3d& point) const { return rotation * point + translation; }
    void lost() { aligned = false; confirmations = 0; lastMatch = -1; }
    void restart(bool hasGeometry) {
        // With no map or reference there is no old coordinate frame to recover.
        // Waiting for PnP here would prevent the first reference from ever forming.
        if (!hasGeometry && frames.empty()) *this = Relocalizer();
        else lost();
    }

    // Called at most at the model cadence. Input descriptors are unit length,
    // and rays are undistorted with the current frame's exact calibration.
    bool locate(double t, const Pose& local, const std::vector<Feature>& features,
                const cv::Mat& descriptors, double focal) {
        if (aligned || descriptors.rows < 20 || frames.empty()) return aligned;
        inliers = 0;
        Pose best; int bestCount = 0; double bestError = 1e9; size_t bestIndex=0;
        // Reverse order prefers recent overlap. Bound matching work to 4 keyframes
        // per call, rotating through older references if recent ones do not match.
        const size_t attempts = std::min(size_t(4), frames.size());
        for (size_t a = 0; a < attempts; ++a) {
            const auto& key = frames[frames.size()-1-(searchCursor+a)%frames.size()];
            std::vector<std::vector<cv::DMatch>> matches;
            cv::BFMatcher matcher(cv::NORM_L2);
            matcher.knnMatch(key.descriptors, descriptors, matches, 2);
            std::vector<cv::DMatch> reverse; matcher.match(descriptors, key.descriptors, reverse);
            std::vector<cv::Point3f> world; std::vector<cv::Point2f> rays;
            std::vector<cv::Point2f> pixels;
            for (size_t i=0; i<matches.size(); ++i) {
                const auto& m=matches[i];
                if (m.size()<2 || m[0].distance >= .8f*m[1].distance || m[0].distance > .7f) continue;
                const int j=m[0].trainIdx;
                if (reverse[j].trainIdx != int(i)) continue;
                world.push_back(key.points[i]); rays.push_back(features[j].ray); pixels.push_back(features[j].pixel);
            }
            if (world.size()<20) continue;
            cv::Mat rv, tv, accepted;
            if (!cv::solvePnPRansac(world, rays, cv::Mat::eye(3,3,CV_64F), cv::Mat(), rv, tv,
                    false, 150, float(2.5/focal), .999, accepted, cv::SOLVEPNP_EPNP)) continue;
            if (accepted.rows<20 || accepted.rows < int(world.size()*.55)) continue;
            std::vector<cv::Point3f> wi; std::vector<cv::Point2f> ri;
            std::set<std::pair<int,int>> cells;
            for (int i=0;i<accepted.rows;++i) { int j=accepted.at<int>(i); wi.push_back(world[j]); ri.push_back(rays[j]);
                cells.insert({int(std::floor(rays[j].x*6)), int(std::floor(rays[j].y*6))}); }
            if (cells.size()<6) continue;
            cv::solvePnPRefineLM(wi,ri,cv::Mat::eye(3,3,CV_64F),cv::Mat(),rv,tv);
            cv::Mat rc; cv::Rodrigues(rv,rc); Eigen::Matrix3d rw;
            Eigen::Vector3d tc; for(int i=0;i<3;++i) {tc[i]=tv.at<double>(i);for(int j=0;j<3;++j)rw(i,j)=rc.at<double>(i,j);}
            Pose candidate{rw.transpose(),-rw.transpose()*tc};
            if (!candidate.r.allFinite() || !candidate.p.allFinite()) continue;
            double error=0; bool front=true;
            for(size_t i=0;i<wi.size();++i) { const auto& w=wi[i]; Eigen::Vector3d c=rw*Eigen::Vector3d(w.x,w.y,w.z)+tc;
                if(c.z()<=.15) {front=false;break;} error+=(cv::norm(cv::Point2f(float(c.x()/c.z()),float(c.y()/c.z()))-ri[i])); }
            error=error/wi.size()*focal;
            if (!front || error>2.0) continue;
            if (accepted.rows>bestCount || (accepted.rows==bestCount && error<bestError)) {best=candidate;bestCount=accepted.rows;bestError=error;bestIndex=(searchCursor+a)%frames.size();}
        }
        searchCursor=(searchCursor+attempts)%frames.size();
        if (!bestCount) {confirmations=0;return false;}
        // Keep the matching reference in the next bounded search window so a
        // large bank does not prevent consecutive confirmations.
        searchCursor=bestIndex;
        Eigen::Matrix3d proposed=best.r*local.r.transpose();
        // Both VIO roots have gravity aligned. Reject a bad tilt/scale hypothesis.
        if ((proposed*Eigen::Vector3d::UnitY()-Eigen::Vector3d::UnitY()).norm()>.0872) {confirmations=0;return false;}
        const double yaw=std::atan2(proposed(0,2),proposed(0,0));
        Eigen::Matrix3d yawRotation=Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitY()).toRotationMatrix();
        if(Eigen::AngleAxisd(yawRotation.transpose()*proposed).angle()>.035) {confirmations=0;return false;}
        const Eigen::Vector3d offset=best.p-yawRotation*local.p;
        bool consistent=lastMatch>=0 && t>lastMatch && t-lastMatch<1.5 &&
            (offset-pendingTranslation).norm()<.12 && Eigen::AngleAxisd(pendingRotation.transpose()*yawRotation).angle()<.035;
        if(consistent) {
            const double dl=(local.p-firstLocal.p).norm(), dm=(best.p-firstMap.p).norm();
            if(dl>.15 && (dm/dl<.8 || dm/dl>1.2)) consistent=false;
        }
        if (!consistent) {confirmations=0;firstTime=t;firstLocal=local;firstMap=best;}
        ++confirmations;lastMatch=t;pendingRotation=yawRotation;pendingTranslation=offset;
        inliers=bestCount;residual=bestError;
        if(confirmations>=3 && t-firstTime>=.4 && (local.p-firstLocal.p).norm()>.15) {rotation=yawRotation;translation=offset;aligned=true;}
        return aligned;
    }

    bool remember(double t, const Pose& local, const std::vector<Feature>& features, const cv::Mat& descriptors,
                  const std::vector<Eigen::Vector3d>& landmarks, double focal) {
        if(!aligned || descriptors.rows<20) return false;
        const Pose pose=mapped(local);
        if(!frames.empty()) {
            if(t-frames.back().t<1.0) return false;
            if((pose.p-frames.back().pose.p).norm()<.25 && Eigen::AngleAxisd(pose.r.transpose()*frames.back().pose.r).angle()<.2) return false;
        }
        Keyframe key{t,pose,{},{}}; std::set<int> used; std::set<std::pair<int,int>> cells;
        for(const auto& point:landmarks) {
            Eigen::Vector3d pc=local.r.transpose()*(point-local.p);
            if(!pc.allFinite() || pc.z()<.3 || pc.z()>8) continue;
            cv::Point2f ray(float(pc.x()/pc.z()),float(pc.y()/pc.z())); int best=-1; double dist=3.0/focal;
            for(size_t j=0;j<features.size();++j) {double d=cv::norm(ray-features[j].ray);if(d<dist && !used.count(j)) {best=int(j);dist=d;}}
            if(best<0) continue;
            used.insert(best); cells.insert({int(std::floor(ray.x*6)),int(std::floor(ray.y*6))});
            const Eigen::Vector3d pw=mapped(point); key.points.emplace_back(pw.x(),pw.y(),pw.z());key.descriptors.push_back(descriptors.row(best));
            if(key.points.size()>=128) break;
        }
        if(key.points.size()<20 || cells.size()<6) return false;
        frames.push_back(std::move(key)); if(frames.size()>32) frames.pop_front(); return true;
    }
private:
    size_t searchCursor=0;
    double lastMatch=-1,firstTime=0;
    Pose firstLocal,firstMap;
    Eigen::Matrix3d pendingRotation=Eigen::Matrix3d::Identity();
    Eigen::Vector3d pendingTranslation=Eigen::Vector3d::Zero();
};
}

#pragma once
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

struct KltEvidence { int input=0, forward=0, photometric=0, roundTrip=0, geometric=0; bool homography=false; };

// Filter invalid optical-flow pairs BEFORE RANSAC. A round-trip check prevents
// ambiguous patches from being kept solely because they fit a geometric model.
// Homography support handles stationary/pure-rotation/planar imagery where a
// fundamental matrix is degenerate. This only selects tracks, never a pose.
template<class Normalize0, class Normalize1>
KltEvidence robustKlt(const std::vector<cv::Mat> &pyramid0, const std::vector<cv::Mat> &pyramid1,
                     std::vector<cv::Point2f> &p0, std::vector<cv::Point2f> &p1,
                     cv::Size window, int levels, double focal,
                     Normalize0 normalize0, Normalize1 normalize1, std::vector<uchar> &mask) {
    KltEvidence evidence; evidence.input=p0.size(); mask.assign(p0.size(),0);
    if (p0.size()!=p1.size() || p0.empty() || pyramid0.empty() || pyramid1.empty() || !std::isfinite(focal) || focal<=0) return evidence;
    const auto inside=[](const cv::Point2f &p, const cv::Mat &image) {
        return std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.y>=0 && p.x<image.cols && p.y<image.rows;
    };
    const cv::TermCriteria term(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,.01);
    std::vector<uchar> forward; std::vector<float> error;
    cv::calcOpticalFlowPyrLK(pyramid0,pyramid1,p0,p1,forward,error,window,levels,term,cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<size_t> indices; std::vector<cv::Point2f> from,to;
    for(size_t i=0;i<p0.size();++i) if(i<forward.size() && forward[i] && inside(p0[i],pyramid0[0]) && inside(p1[i],pyramid1[0])) {
        evidence.forward++;
        // LK's mean absolute patch error is in grayscale intensity units with
        // these flags. Round-trip consistency alone can retain occluded patches.
        if(i>=error.size() || !std::isfinite(error[i]) || error[i]>30.0f) continue;
        indices.push_back(i); from.push_back(p1[i]); to.push_back(p0[i]);
    }
    evidence.photometric=indices.size(); if(indices.size()<10) return evidence;
    std::vector<uchar> backward;
    cv::calcOpticalFlowPyrLK(pyramid1,pyramid0,from,to,backward,error,window,levels,term,cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<size_t> valid; std::vector<cv::Point2f> normalized0,normalized1;
    for(size_t j=0;j<indices.size();++j) {
        const size_t i=indices[j];
        if(j>=backward.size() || !backward[j] || !inside(to[j],pyramid0[0]) || cv::norm(to[j]-p0[i])>1.0) continue;
        const auto a=normalize0(p0[i]), b=normalize1(p1[i]);
        if(!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(b.x)||!std::isfinite(b.y)) continue;
        valid.push_back(i); normalized0.push_back(a); normalized1.push_back(b);
    }
    evidence.roundTrip=valid.size(); if(valid.size()<10) return evidence;
    std::vector<uchar> fundamental,homography;
    const auto F=cv::findFundamentalMat(normalized0,normalized1,cv::FM_RANSAC,2.0/focal,.999,fundamental);
    const auto H=cv::findHomography(normalized0,normalized1,cv::RANSAC,1.0/focal,homography,2000,.999);
    const auto support=[&](const std::vector<uchar> &m,const cv::Mat &model) {
        return model.empty() || !cv::checkRange(model) || m.size()!=valid.size() ? 0 : int(std::count_if(m.begin(),m.end(),[](uchar v){return v!=0;}));
    };
    const int nf=support(fundamental,F), nh=support(homography,H);
    // Prefer the more constrained image model when it explains a substantial
    // majority. On a plane, F can accept arbitrarily wrong motion along an
    // epipolar line (even a self-consistent occlusion match). General multi-depth
    // scenes without dominant H support still use the fundamental matrix.
    evidence.homography=nh>=10 && nh>=int(std::ceil(.8*valid.size()));
    const auto &chosen=evidence.homography ? homography:fundamental;
    evidence.geometric=evidence.homography ? nh:nf;
    if(evidence.geometric<10) { evidence.geometric=0; return evidence; }
    for(size_t j=0;j<valid.size();++j) if(chosen[j]) mask[valid[j]]=1;
    return evidence;
}

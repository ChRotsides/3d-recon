#include "FeatureDiagnostics.h"
#include "XFeatBridge.h"
#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"
#include "feat/FeatureInitializer.h"
#include "core/VioManager.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstring>
#include "tsdf.h"

int main(int argc,char **argv) {
    int tests=0;
    { live_xfeat::Store store; live_xfeat::Frame found;
      assert(store.lookup(1.0,960,540,found)==live_xfeat::Lookup::Disabled);
      const float candidates[]={120,80,.2f, 400,200,.9f, -1,40,.8f};
      assert(store.submit(1.0,960,540,candidates,3));
      assert(store.lookup(1.0+1.0/30.0,960,540,found)==live_xfeat::Lookup::Missing);
      assert(store.submit(2.0,960,540,candidates,3));
      assert(store.lookup(2.0,959,540,found)==live_xfeat::Lookup::Missing);
      assert(store.lookup(2.0,960,540,found)==live_xfeat::Lookup::Found);
      assert(found.candidates.size()==2 && found.candidates[0].x==400 && found.candidates[1].x==120);
      assert(store.lookup(2.0,960,540,found)==live_xfeat::Lookup::Missing);
      store.setEnabled(false); assert(store.lookup(2.0,960,540,found)==live_xfeat::Lookup::Disabled); ++tests; }
    assert(!retireOldestFeatureTracks(6,11)); assert(!retireOldestFeatureTracks(11,11));
    assert(retireOldestFeatureTracks(12,11)); assert(retireOldestFeatureTracks(6,5)); ++tests;
    cv::setNumThreads(1); cv::setRNGSeed(7);
    cv::Mat texture(320,480,CV_8UC1); cv::randu(texture,0,255); cv::GaussianBlur(texture,texture,{3,3},.7);
    cv::Mat roi=cv::Mat::zeros(texture.size(),CV_8UC1); cv::rectangle(roi,{30,30,420,260},cv::Scalar(255),cv::FILLED);
    std::vector<cv::Point2f> corners; cv::goodFeaturesToTrack(texture,corners,120,.03,15,roi);
    assert(corners.size()>=100);
    const auto normalized=[](cv::Point2f p) { return cv::Point2f((p.x-240)/320,(p.y-160)/320); };
    const auto match=[&](const cv::Mat &target,std::vector<cv::Point2f> &tracked,std::vector<uchar> &mask) {
        std::vector<cv::Mat> a,b; cv::buildOpticalFlowPyramid(texture,a,{21,21},3); cv::buildOpticalFlowPyramid(target,b,{21,21},3);
        auto initial=corners; tracked=corners;
        const auto e=robustKlt(a,b,initial,tracked,{21,21},3,320,normalized,normalized,mask);
        std::cerr<<"KLT input/forward/photometric/roundtrip/geometric="<<e.input<<"/"<<e.forward<<"/"<<e.photometric<<"/"<<e.roundTrip<<"/"<<e.geometric<<" H="<<e.homography<<"\n";
        return e;
    };
    std::vector<cv::Point2f> tracked; std::vector<uchar> mask;
    { const auto e=match(texture,tracked,mask); assert(e.geometric>=100); ++tests; }
    cv::Mat translated; const cv::Mat shift=(cv::Mat_<double>(2,3)<<1,0,4,0,1,1);
    cv::warpAffine(texture,translated,shift,texture.size());
    { const auto e=match(translated,tracked,mask); assert(e.geometric>=90);
      double error=0; int n=0; for(size_t i=0;i<mask.size();++i) if(mask[i]) { error+=cv::norm(tracked[i]-corners[i]-cv::Point2f(4,1)); ++n; }
      assert(error/n<.2); std::cout<<"translation inliers="<<n<<" mean error="<<error/n<<" px\n"; ++tests; }
    { cv::Mat rotated; const auto rotation=cv::getRotationMatrix2D({240,160},3,1);
      cv::warpAffine(texture,rotated,rotation,texture.size());
      const auto e=match(rotated,tracked,mask); assert(e.geometric>=60);
      double error=0; int n=0;
      for(size_t i=0;i<mask.size();++i) if(mask[i]) {
          const auto p=corners[i];
          const cv::Point2f expected(rotation.at<double>(0,0)*p.x+rotation.at<double>(0,1)*p.y+rotation.at<double>(0,2),rotation.at<double>(1,0)*p.x+rotation.at<double>(1,1)*p.y+rotation.at<double>(1,2));
          error+=cv::norm(tracked[i]-expected); ++n;
      }
      assert(error/n<.3); ++tests; }
    { cv::Mat brighter; translated.convertTo(brighter,CV_8U,1.08,3);
      const auto e=match(brighter,tracked,mask); assert(e.geometric>=80); ++tests; }
    { cv::Mat layer, mixed=translated.clone(); const cv::Mat nearShift=(cv::Mat_<double>(2,3)<<1,0,10,0,1,1);
      cv::warpAffine(texture,layer,nearShift,texture.size()); layer(cv::Rect(240,0,240,320)).copyTo(mixed(cv::Rect(240,0,240,320)));
      const auto e=match(mixed,tracked,mask); assert(!e.homography);
      int left=0,right=0; for(size_t i=0;i<mask.size();++i) if(mask[i]) {
          if(corners[i].x<210) ++left; if(corners[i].x>270) ++right;
      }
      assert(left>=15 && right>=15); ++tests;
    } // real parallax: keep tracks from both depths, don't always choose a planar model
    { cv::Mat near, mixed=translated.clone(); const cv::Mat nearShift=(cv::Mat_<double>(2,3)<<1,0,12,0,1,1);
      cv::warpAffine(texture,near,nearShift,texture.size());
      near(cv::Rect(380,0,100,320)).copyTo(mixed(cv::Rect(380,0,100,320)));
      const auto e=match(mixed,tracked,mask);
      int nearTracks=0;
      for(size_t i=0;i<mask.size();++i) if(mask[i] && corners[i].x>410)
          nearTracks+=cv::norm(tracked[i]-corners[i]-cv::Point2f(12,1))<.5;
      std::cerr<<"dominant wall with near object: retained near tracks="<<nearTracks<<" H="<<e.homography<<"\n";
      assert(nearTracks>=3); ++tests;
    }
    { cv::Mat occluded=translated.clone(); cv::rectangle(occluded,{100,60,150,180},cv::Scalar(0),cv::FILLED);
      // Coarse pyramid patches overlap the occlusion beyond its pixel boundary;
      // require enough accurate survivors, not acceptance of those ambiguous patches.
      const auto e=match(occluded,tracked,mask); assert(e.geometric>=20 && e.geometric<int(corners.size()));
      double worst=0; for(size_t i=0;i<mask.size();++i) if(mask[i]) worst=std::max(worst,cv::norm(tracked[i]-corners[i]-cv::Point2f(4,1)));
      std::cerr<<"occlusion worst retained error="<<worst<<" px\n"; assert(worst<1);
      ++tests; }
    { const auto e=match(cv::Mat::zeros(texture.size(),CV_8UC1),tracked,mask); assert(e.geometric==0); ++tests; }
    { std::vector<cv::Mat> a,b; std::vector<cv::Point2f> p,q; std::vector<uchar> m;
      assert(robustKlt(a,b,p,q,{21,21},3,320,normalized,normalized,m).geometric==0); ++tests; }

    auto camera=std::make_shared<ov_core::CamRadtan>(480,320);
    Eigen::VectorXd intrinsics(8); intrinsics<<320,320,240,160,0,0,0,0; camera->set_value(intrinsics);
    ov_core::TrackKLT tracker({{0,camera}},120,0,false,ov_core::TrackBase::HistogramMethod::NONE,15,6,4,12);
    live_xfeat::Store trackerXFeat; std::vector<float> learned;
    for(size_t i=0;i<corners.size();++i) { learned.push_back(corners[i].x); learned.push_back(corners[i].y); learned.push_back(1.0f-i/(float)corners.size()); }
    assert(trackerXFeat.submit(0.0,480,320,learned.data(),learned.size()/3));
    for(int frame=0;frame<20;++frame) {
        live_diagnostics::current={}; live_diagnostics::current.previousCameraTime=frame ? (frame-1)/30.0:NAN;
        cv::Mat image; cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,frame*.8,0,1,frame*.2);
        cv::warpAffine(texture,image,transform,texture.size());
        ov_core::CameraData data; data.timestamp=frame/30.0; data.sensor_ids={0}; data.images={image}; data.masks={cv::Mat::zeros(image.size(),CV_8UC1)};
        live_xfeat::FeedScope xfeatScope(&trackerXFeat);
        tracker.feed_new_camera(data);
        const auto ids=tracker.get_last_ids().at(0); assert(ids.size()>60);
        int longTracks=0;
        for(const auto id:ids) {
            const auto f=tracker.get_feature_database()->get_feature(id); assert(f);
            const auto &t=f->timestamps.at(0); assert(t.back()==data.timestamp);
            for(size_t i=1;i<t.size();++i) assert(t[i]>t[i-1]);
            longTracks+=t.size()>=15;
        }
        if(frame==0) { assert(live_diagnostics::current.stationaryDetails[54]>60); ++tests; }
        if(frame==19) { std::cout<<"real TrackKLT: "<<longTracks<<" tracks with >=15 observations\n"; assert(longTracks>50); ++tests; }
    }
    { const auto stats=trackerXFeat.stats(); assert(stats.consumed==1 && stats.selected>60); ++tests; }
    // XFeat stays enabled after its first result, but runs at a fraction of the
    // camera rate. Complete occlusion must recover on frames with no learned result.
    for(int frame=20;frame<26;++frame) {
        live_diagnostics::current={}; live_diagnostics::current.previousCameraTime=(frame-1)/30.0;
        ov_core::CameraData data; data.timestamp=frame/30.0; data.sensor_ids={0};
        data.images={frame<22 ? cv::Mat::zeros(texture.size(),CV_8UC1) : texture};
        data.masks={cv::Mat::zeros(texture.size(),CV_8UC1)};
        live_xfeat::FeedScope xfeatScope(&trackerXFeat);
        tracker.feed_new_camera(data);
        if(frame==21) assert(tracker.get_last_ids().at(0).empty());
        if(frame==25) { assert(tracker.get_last_ids().at(0).size()>60); ++tests; }
    }
    ov_core::FeatureInitializerOptions options; ov_core::FeatureInitializer initializer(options);
    const auto triangulate=[&](double baseline) {
        auto feature=std::make_shared<ov_core::Feature>();
        std::unordered_map<size_t,std::unordered_map<double,ov_core::FeatureInitializer::ClonePose>> clones;
        const Eigen::Matrix3d identity=Eigen::Matrix3d::Identity();
        for(int i=0;i<11;++i) {
            const Eigen::Vector3d position(i*baseline,0,0); const Eigen::Vector3d point=Eigen::Vector3d(.2,-.1,3)-position;
            clones[0].emplace(i,ov_core::FeatureInitializer::ClonePose(identity,position));
            feature->timestamps[0].push_back(i); feature->uvs_norm[0].push_back(Eigen::Vector2f(point.x()/point.z(),point.y()/point.z()));
        }
        live_diagnostics::MsckfScope scope;
        const bool ok=initializer.single_triangulation(feature,clones);
        if(ok) assert((feature->p_FinG-Eigen::Vector3d(.2,-.1,3)).norm()<1e-4);
        return ok;
    };
    assert(triangulate(.02)); ++tests;
    assert(!triangulate(0)); ++tests;
    assert(live_diagnostics::current.stationaryDetails[51]+live_diagnostics::current.stationaryDetails[47]>0); ++tests;
    if(argc>1) {
        // Full estimator smoke test: a textured plane at 3 m, analytic smooth
        // translation, and physically consistent IMU acceleration. No real phone
        // motion is required. This checks that visual EKF updates can execute.
        ov_core::Printer::setPrintLevel("ERROR");
        auto parser=std::make_shared<ov_core::YamlParser>(argv[1]);
        ov_msckf::VioManagerOptions params; params.print_and_load(parser); assert(parser->successful());
        params.use_multi_threading_subs=false; params.zupt_only_at_beginning=false;
        params.init_options.init_max_disparity=2.0;
        params.init_options.init_window_time=3.0; params.init_options.init_dyn_use=true;
        params.init_options.init_max_features=120; params.init_options.init_dyn_mle_opt_calib=false;
        params.init_options.init_dyn_mle_max_iter=50; params.init_options.init_dyn_mle_max_time=.5;
        params.init_options.init_dyn_mle_max_threads=2;
        if(argc>2) params.state_options.max_clone_size=std::stoi(argv[2]);
        auto manager=std::make_shared<ov_msckf::VioManager>(params);
        StationaryDetector detector; live_diagnostics::detector=&detector;
        cv::Mat scene(540,960,CV_8UC1); cv::randu(scene,0,255); cv::GaussianBlur(scene,scene,{3,3},.7);
        const Eigen::Matrix3d cameraToImu=manager->get_state()->_calib_IMUtoCAM.at(0)->Rot().transpose();
        int accepted=0,attempted=0,startupDiagnostics=0; double imuTime=1.0;
        std::vector<double> frameMs;
        Eigen::Vector3d holdStart=Eigen::Vector3d::Zero();
        for(int frame=0;frame<300;++frame) {
            const double t=1.0+frame/30.0;
            while(imuTime<=t+.01) {
                // Four seconds of rest for the three-second bias guard, then move.
                const double s=imuTime-5.0;
                const double acceleration=s>0 && s<4 ? .2*std::pow(3.141592653589793/4,2)*std::cos(3.141592653589793*s/4):0;
                ov_core::ImuData m; m.timestamp=imuTime; m.wm.setZero();
                m.am=Eigen::Vector3d(0,9.81,0)+cameraToImu*Eigen::Vector3d(acceleration,0,0);
                manager->feed_measurement_imu(m); detector.imu(imuTime,{0,0,0},{m.am.x(),m.am.y(),m.am.z()}); imuTime+=.005;
            }
            const double s=std::clamp(t-5.0,0.0,4.0), x=.2*(1-std::cos(3.141592653589793*s/4));
            const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-639.2*x/3,0,1,0);
            cv::Mat frameImage; cv::warpAffine(scene,frameImage,transform,scene.size());
            ov_core::CameraData data; data.timestamp=t; data.sensor_ids={0}; data.images={frameImage}; data.masks={cv::Mat::zeros(scene.size(),CV_8UC1)};
            live_diagnostics::current={}; live_diagnostics::current.previousCameraTime=frame ? 1.0+(frame-1)/30.0:NAN;
            const auto start=std::chrono::steady_clock::now();
            manager->feed_measurement_camera(data);
            frameMs.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            if(frame==270) holdStart=manager->get_state()->_imu->pos();
            const auto &d=live_diagnostics::current;
            accepted+=d.msckfAccepted+d.slamAccepted; attempted+=d.msckfAttempted+d.slamAttempted;
            startupDiagnostics+=d.stationaryDetails[29]==1;
            if(frame==119) {
                // Pure rest after initialization must expose the actual current
                // stationary-updated state, even before a first visual update.
                assert(manager->initialized() && d.stationary);
                assert(std::abs(manager->get_state()->_timestamp-t)<1e-6);
                ++tests;
            }
            if(frame%60==59) std::cerr<<"VIO frame="<<frame<<" initialized="<<manager->initialized()<<" tracks="<<d.tracks<<" accepted="<<accepted<<" attempted="<<attempted<<" startup="<<startupDiagnostics<<" p="<<manager->get_state()->_imu->pos().transpose()<<" tri(condition/behind/near/far/nan)="<<d.stationaryDetails[47]<<"/"<<d.stationaryDetails[48]<<"/"<<d.stationaryDetails[49]<<"/"<<d.stationaryDetails[50]<<"/"<<d.stationaryDetails[51]<<"\n";
        }
        std::cerr<<"synthetic full VIO initialized="<<manager->initialized()<<" accepted="<<accepted<<"/"<<attempted<<" startup handoffs="<<startupDiagnostics<<"\n";
        std::sort(frameMs.begin(),frameMs.end());
        std::cerr<<"camera processing median/p95="<<frameMs[frameMs.size()/2]<<"/"<<frameMs[size_t(.95*(frameMs.size()-1))]<<" ms; final 0.97s hold net displacement="<<(manager->get_state()->_imu->pos()-holdStart).norm()<<" m; final speed="<<manager->get_state()->_imu->vel().norm()<<" m/s\n";
        assert(manager->initialized() && accepted>=20 && startupDiagnostics==1); ++tests;
        assert((manager->get_state()->_imu->pos()-holdStart).norm()<.02);
        assert(manager->get_state()->_imu->vel().norm()<.02);
        assert(std::abs(manager->get_state()->_imu->pos().norm()-.4)<.10); ++tests;
        live_diagnostics::detector=nullptr;
    }
    if(argc>3) {
        VoxelHashTSDF map;
        std::vector<uint16_t> depth(64*48,1500);
        std::vector<uint8_t> rgb(64*48*4,128);
        const float k[4]={60,60,32,24},r[9]={1,0,0,0,1,0,0,0,1},c[3]={0,0,0};
        for(int i=0;i<6;++i) map.integrate(depth.data(),64,48,k,r,c,rgb.data(),64,48,1);
        std::vector<float> cloud; const auto count=map.surface(cloud,10000000);
        assert(count>0);
        assert(map.saveSurface(argv[3])==int64_t(count));
        std::ifstream saved(argv[3],std::ios::binary); std::string header,line;
        while(std::getline(saved,line)) {header+=line+"\n"; if(line=="end_header") break;}
        assert(header.find("independent_local_frame")!=std::string::npos);
        assert(header.find("element vertex "+std::to_string(count)+"\n")!=std::string::npos);
        for(size_t i=0;i<count;++i) {
            for(int axis=0;axis<3;++axis) {float v; saved.read(reinterpret_cast<char*>(&v),4);assert(v==cloud[6*i+axis]);}
            for(int color=3;color<6;++color) assert(saved.get()==std::lround(cloud[6*i+color]*255));
        }
        assert(saved.peek()==std::char_traits<char>::eof());
        assert(map.saveSurface((std::string(argv[3])+"/invalid").c_str())==-1);
        std::vector<float> after;map.surface(after,10000000);assert(after==cloud);
        std::cerr<<"archived complete surface "<<count<<" points; failed save preserved map\n"; ++tests;
    }
    std::cout<<tests<<" feature tracking/geometry tests passed\n";
}

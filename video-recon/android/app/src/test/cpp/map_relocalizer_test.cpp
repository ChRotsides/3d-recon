#include "MapRelocalizer.h"
#include <cassert>
#include <iostream>

int main() {
    live_map::Relocalizer startup;
    startup.lost(); startup.restart(false); assert(startup.aligned);
    startup.restart(true); assert(!startup.aligned);
    live_map::Relocalizer map;
    std::vector<Eigen::Vector3d> world;
    std::vector<live_map::Feature> features;
    cv::Mat descriptors(64,64,CV_32F,cv::Scalar(0));
    for(int i=0;i<64;++i) {
        const double z=3.+.15*(i%5),x=(i%8-3.5)*.35,y=(i/8-3.5)*.25;
        world.emplace_back(x,y,z);
        cv::Point2f ray(x/z,y/z);features.push_back({ray*500.f,ray});descriptors.at<float>(i,i)=1;
    }
    live_map::Pose origin;
    assert(map.remember(1,origin,features,descriptors,world,500));
    map.restart(false); assert(!map.aligned); // References still require alignment.
    map.aligned=true;
    assert(!map.remember(1.1,origin,features,descriptors,world,500));
    map.lost(); assert(!map.aligned);
    Eigen::Matrix3d rotation=Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Vector3d translation(1,0,-.5);
    for(int frame=0;frame<3;++frame) {
        live_map::Pose actual; actual.p=Eigen::Vector3d(.1*frame,0,0);
        live_map::Pose local{rotation.transpose(),rotation.transpose()*(actual.p-translation)};
        for(int i=0;i<64;++i) {const Eigen::Vector3d p=world[i]-actual.p;features[i].ray=cv::Point2f(p.x()/p.z(),p.y()/p.z());}
        const bool matched=map.locate(2.+.4*frame,local,features,descriptors,500);
        assert(matched==(frame==2));
        if(matched) assert((map.mapped(local).p-actual.p).norm()<1e-3);
    }
    assert((map.rotation-rotation).norm()<1e-3);assert((map.translation-translation).norm()<1e-3);
    map.lost(); cv::Mat wrong(64,64,CV_32F,cv::Scalar(1.f/8));
    assert(!map.locate(4,origin,features,wrong,500));assert(map.confirmations==0);
    // A gravity-inconsistent solution must never be allowed to rotate the room.
    live_map::Pose tilted; tilted.r=Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitX()).toRotationMatrix();
    for(int i=0;i<64;++i) features[i].ray=cv::Point2f(world[i].x()/world[i].z(),world[i].y()/world[i].z());
    for(int i=0;i<4;++i) assert(!map.locate(5+.4*i,tilted,features,descriptors,500));
    std::cout<<"map relocalization tests passed\n";
}

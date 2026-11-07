#include "tsdf.h"
#include <algorithm>
#include <cassert>
#include <iostream>

int main() {
    const int w=80,h=60;
    float k[4]={80,80,40,30},r[9]={1,0,0,0,1,0,0,0,1},c[3]={0,0,0};
    std::vector<uint16_t> depth(w*h,2000);std::vector<uint8_t> weights(w*h,64);
    VoxelHashTSDF map;std::vector<float> points;
    for(int i=0;i<12;++i) map.integrate(depth.data(),w,h,k,r,c,nullptr,0,0,2,weights.data());
    assert(map.surface(points,300000)==0);
    std::fill(weights.begin(),weights.end(),255);
    for(int i=0;i<4;++i)map.integrate(depth.data(),w,h,k,r,c,nullptr,0,0,2,weights.data());
    assert(map.surface(points,300000)>100);
    for(size_t i=2;i<points.size();i+=6)assert(std::abs(points[i]-2)<.04);
    for(float x:{-5.f,5.f}) { c[0]=x;for(int i=0;i<4;++i)map.integrate(depth.data(),w,h,k,r,c,nullptr,0,0,2,weights.data()); }
    const auto dense=map.surface(points,300000);assert(dense>300);
    assert(map.surface(points,100)<=100);
    bool left=false,centre=false,right=false;
    for(size_t i=0;i<points.size();i+=6) {left|=points[i]<-3;centre|=std::abs(points[i])<2;right|=points[i]>3;}
    assert(left&&centre&&right);
    assert(map.surface(points,1)==1);assert(map.surface(points,0)==0);
    // Rejected geometry cannot hang stride loops or allocate a corrupt volume.
    const auto before=map.blocks();
    assert(map.integrate(depth.data(),w,h,k,r,c,nullptr,0,0,0).status==2);
    float badK[4]={NAN,80,40,30};
    assert(map.integrate(depth.data(),w,h,badK,r,c,nullptr,0,0,1).status==2);
    assert(map.blocks()==before);
    VoxelHashTSDF bounded(.03f,4,16,2,6,1);
    assert(bounded.integrate(depth.data(),w,h,k,r,c,nullptr,0,0,1).status==1);
    assert(bounded.blocks()==0); // capacity rejection is atomic
    std::cout<<"TSDF quality and coverage tests passed\n";
}

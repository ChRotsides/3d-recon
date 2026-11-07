// Read-only replay of the production IMU startup guard, not a full VIO replay.
// Usage: startup_replay imu.csv [reference_boot_time]
#include "StartupMotionCheck.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char **argv) {
    if (argc<2) { std::cerr << "usage: startup_replay imu.csv [reference_boot_time]\n"; return 2; }
    struct Raw { double t; std::array<double,3> v; };
    std::vector<Raw> accel,gyro; std::ifstream input(argv[1]); std::string line;
    if (!input) return 2;
    std::getline(input,line);
    while (std::getline(input,line)) {
        std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
        Raw s; int type;
        if (!(row>>s.t>>type>>s.v[0]>>s.v[1]>>s.v[2])) return 2;
        if (type==1) accel.push_back(s); else if(type==4) gyro.push_back(s);
    }
    if (accel.empty() || gyro.size()<2) return 2;
    const auto byTime=[](const Raw &a,const Raw &b) { return a.t<b.t; };
    std::sort(accel.begin(),accel.end(),byTime); std::sort(gyro.begin(),gyro.end(),byTime);
    std::vector<StartupMotionCheck::Sample> samples; size_t j=0;
    for (const auto &a:accel) {
        while(j+1<gyro.size() && gyro[j+1].t<a.t) ++j;
        if(j+1==gyro.size() || gyro[j].t>a.t || gyro[j+1].t-gyro[j].t>.03) continue;
        const double f=(a.t-gyro[j].t)/(gyro[j+1].t-gyro[j].t);
        StartupMotionCheck::Sample s{a.t,{},a.v};
        for(int k=0;k<3;++k) s.gyro[k]=gyro[j].v[k]+f*(gyro[j+1].v[k]-gyro[j].v[k]);
        samples.push_back(s);
    }
    const double reference=argc>2 ? std::stod(argv[2]) : samples.front().t;
    std::cout << "end_t,relative_t,imu_guard_accepted,older_gyro_mean,newer_gyro_mean,gyro_rms,accel_std,mean_change,bgx,bgy,bgz\n";
    double last=-INFINITY;
    for (size_t k=0;k<samples.size();++k) {
        const double t=samples[k].t;
        if(t-samples.front().t<3.0 || t-last<.1) continue;
        last=t; std::vector<StartupMotionCheck::Sample> older,newer;
        for(size_t n=k+1;n>0;) {
            const auto &s=samples[--n]; if(s.t<=t-3.0) break;
            (s.t<=t-1.5 ? older:newer).push_back(s);
        }
        std::reverse(older.begin(),older.end()); std::reverse(newer.begin(),newer.end());
        const StartupMotionCheck c(older,newer,1.5);
        std::cout << std::setprecision(12) << t << ',' << t-reference << ',' << c.accepted << ','
            << c.older.gyroMean << ',' << c.newer.gyroMean << ','
            << std::max(c.older.gyroRms,c.newer.gyroRms) << ','
            << std::max(c.older.accelStd,c.newer.accelStd) << ',' << c.meanChange << ','
            << c.bias[0] << ',' << c.bias[1] << ',' << c.bias[2] << '\n';
    }
}

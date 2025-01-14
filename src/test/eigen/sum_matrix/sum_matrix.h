
#pragma once 
#include <Eigen/Eigen>
class SumMartix{
public: 
    SumMartix() = default;
    ~SumMartix() = default;
    Eigen::Matrix3d Sum(const  Eigen::Matrix3d  &a,const  Eigen::Matrix3d  & b);
};
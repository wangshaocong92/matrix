
#include "sum_matrix.h"


class SumMartix{
public: 
    SumMartix() = default;
    ~SumMartix() = default;
    Eigen::Matrix3d Sum(Eigen::Matrix3d &a, Eigen::Matrix3d &b);
};
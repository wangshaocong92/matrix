
#include "sum_matrix.h"
#include <iostream>
#include <vector>

Eigen::Matrix3d SumMartix::Sum(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
    std::vector<int> vec{0, 1, 2, 3, 4};
    for (auto i = 0; i < vec.size(); i++) {
        std::cout << vec[ i ] << std::endl;
    }
    return a + b;
}
#pragma once
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace format {
template<typename... Args>
std::string dyna_print(std::string_view rt_fmt_str, Args&&... args)
{
    std::vector<bool> bvec;
    return std::vformat(rt_fmt_str, std::make_format_args(args...));
}

void print();
}

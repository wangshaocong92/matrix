#include "test_format.h"

namespace format {
void print()
{
    std::cout << std::format("Hello {}!\n", "world");
 
    std::string fmt;
    for (int i{}; i != 3; ++i)
    {
        fmt += "{} "; // constructs the formatting string
        std::cout << fmt << " : ";
        std::cout << dyna_print(fmt, "alpha", 'Z', 3.14, "unused");
        std::cout << '\n';
    }
}
}

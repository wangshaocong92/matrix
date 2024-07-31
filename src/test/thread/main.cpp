#include "thread/mul_thread_error.h"
#include <iostream>

int main()
{
    thread::multi_thread_error m;
    int                        count = 10;
    auto                       now1  = std::chrono::system_clock::now();
    for (auto i = 0; i < count; i++) {
        m.run(100000000, thread::multi_thread_error::run_type<false, true>());
    }
    auto now2 = std::chrono::system_clock::now();
    std::cout << "no one core mut spend time :" << (now2 - now1).count();

    for (auto i = 0; i < count; i++) {
        m.run(100000000, thread::multi_thread_error::run_type<false, false>());
    }
    auto now3 = std::chrono::system_clock::now();
    std::cout << "no one core no mut spend time :" << (now3 - now2).count();

    for (auto i = 0; i < count; i++) {
        m.run(100000000, thread::multi_thread_error::run_type<true, false>());
    }
    auto now4 = std::chrono::system_clock::now();
    std::cout << "one core no mut spend time :" << (now4 - now3).count();
    return 0;
}
#include "thread/memory_barrier.h"
#include "thread/mul_thread_error.h"
#include <atomic>
#include <iostream>
#include <iterator>

int main()
{
#define COUNT 1000000
#if TEST_THREAD

    thread::multi_thread_error m;
    int                        count = 10;
    auto                       now1  = std::chrono::system_clock::now();
    for (auto i = 0; i < count; i++) {
        m.run(COUNT, thread::multi_thread_error::run_type<false, true>());
    }
    auto now2 = std::chrono::system_clock::now();
    std::cout << "spend time :" << (now2 - now1).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.run(COUNT, thread::multi_thread_error::run_type<false, false>());
    }
    auto now3 = std::chrono::system_clock::now();
    std::cout << "spend time :" << (now3 - now2).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.run(COUNT, thread::multi_thread_error::run_type<true, false>());
    }
    auto now4 = std::chrono::system_clock::now();
    std::cout << "spend time :" << (now4 - now3).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.run(COUNT, thread::multi_thread_error::run_type<true, true>());
    }
    auto now5 = std::chrono::system_clock::now();
    std::cout << "spend time :" << (now5 - now2).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.run(COUNT);
    }
    auto now6 = std::chrono::system_clock::now();
    std::cout << "spend time :" << (now6 - now5).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.atomic_run(COUNT);
    }
    auto now7 = std::chrono::system_clock::now();
    std::cout << "one cpu spend time :" << (now7 - now5).count() << std::endl;

    for (auto i = 0; i < count; i++) {
        m.atomic_run(COUNT, false);
    }
    auto now8 = std::chrono::system_clock::now();
    std::cout << "not one spend time :" << (now8 - now7).count() << std::endl;
#endif
    thread::memory_barrier mb;
    std::cout << "memory_order_acquire:" << std::endl;
    mb.run(COUNT, std::memory_order_acquire, std::memory_order_release);
    return 0;
}
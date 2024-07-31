#pragma once



#include <iostream>
#include <mutex>
#include <thread>

#include <sched.h>
#include <unistd.h>
namespace thread {

class multi_thread_error
{
    public:
     multi_thread_error() =default;

     ~multi_thread_error() = default;
     template <bool use_one_core, bool use_mutex> struct run_type {};
     template <bool use_one_core, bool use_mutex>
     void run(int count, run_type<use_one_core, use_mutex> type) {}

 private:
     int value_ {0};
     std::mutex  mut_;
     std::thread t1_;
     std::thread t2_;
};

template <> void multi_thread_error::run<false, false>(int count, run_type<false, false> type) {
    t1_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第0个核心
        int core = 0;
        CPU_SET(core, &cpuset);

        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            value_++;
        }
    });
    t2_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第10个核心
        int core = 10;
        CPU_SET(core, &cpuset);
        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "one core no mut value: " << value_ << std::endl;
}

template <> void multi_thread_error::run<false, true>(int count, run_type<false, true> type) {
    t1_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第0个核心
        int core = 0;
        CPU_SET(core, &cpuset);

        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    t2_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第10个核心
        int core = 10;
        CPU_SET(core, &cpuset);
        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "one core mut value: " << value_ << std::endl;
}

template <> void multi_thread_error::run<true, false>(int count, run_type<true, false> type) {
    t1_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第0个核心
        int core = 0;
        CPU_SET(core, &cpuset);

        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            // std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    t2_ = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第10个核心
        int core = 0;
        CPU_SET(core, &cpuset);
        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        // 执行线程的工作
        std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            // std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "one core value: " << value_ << std::endl;
}
}
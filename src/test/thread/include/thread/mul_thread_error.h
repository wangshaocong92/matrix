#pragma once

#include <atomic>
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
     void run(int count);
     void atomic_run(int count, bool one_cpu = true);

 private:
     int value_ {0};
     std::atomic_int atomic_value_;
     std::atomic_int atomic_flag_{0};
     std::mutex  mut_;
     std::thread t1_;
     std::thread t2_;
};
}
#include "thread/mul_thread_error.h"
#include <atomic>
#include <iostream>

namespace thread {
/*
  * 加锁与不加锁之间时间花费在此case中，存在两个数量级的差距
  * 单核双线程单变量也会存在异常
    * 执行过程中，此进程抛给了其他cpu?
    * 单核也不线程安全?
  * 单核加锁明显快于多核，应该是不同cpu的catch同步造成的
  * for 或者 while，锁应当在它们之前锁住，避免多次 lock和unlock
  * compare_exchange_strong 若是不一致，则将当前值写入到比较变量中，需要注意比较变量的值

mutil core mut value: 200000000,200000000,200000000 ...
spend time :87,254,111,229
mutil core no mut value: 104042389,105027143,102964352,104139902，103922542 ...
spend time :1,807,233,561
one core no mut value: 136648582,141987286，175159128，179159830，119694664，200000000 ...
spend time :775,311,274
one core mut value: 200000000,200000000,200000000 ...
spend time :26,256,321,522
one core mut fore while value: 200000000,200000000,200000000 ...
spend time :734,918,242
one core atomic value: 200000000,200000000,200000000 ...
spend time :9,128,612,757
mutil core atomic value: 200000000,200000000,200000000 ...
spend time :14,742,440,743
自旋锁 one cpu spend time :31,836,065,598   // 单cpu慢于互斥锁 1.
单cpu，自旋锁当前进程时间片未用完的情况下，是不会主动放弃cpu的。
                                                                持有锁的进程在当前进程时间片为消耗完的情况下无法继续执行,
                                                                即会出现时间片浪费的问题
自旋锁 not one spend time :67,087,654,133   // 多cpu看起来快于互斥锁 1.自旋减少了进程调度消费 2.
自旋锁不需要内核态和用户态切换
*/
template <> void multi_thread_error::run<false, false>(int count, run_type<false, false> type) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            value_++;
        }
    });
    t2_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "mutil core no mut value: " << value_ << std::endl;
}

void multi_thread_error::atomic_run(int count, bool one_cpu) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;

        while (c--) {
            bool ret;
            do {
                int expected = 0;
                ret = atomic_flag_.compare_exchange_strong(expected, 1, std::memory_order_acquire);
            } while (!ret);
            value_++;
            atomic_flag_.store(0, std::memory_order_release);
        }
    });
    t2_    = std::thread([ & ]() {
        pid_t tid = syscall(SYS_gettid);

        // 创建cpu集合
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 假设我们要将线程绑定到第0个核心
        int core = one_cpu ? 0 : 10;
        CPU_SET(core, &cpuset);
        // 设置亲和性
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        // 执行线程的工作
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            bool ret;
            do {
                int expected = 0; ////  一个简单的自旋锁而已，感觉不仅干爆了当前cpu。收益也不高
                ret = atomic_flag_.compare_exchange_strong(expected, 1, std::memory_order_acquire);
            } while (!ret);
            value_++;
            atomic_flag_.store(0, std::memory_order_release);
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "atomic value: " << value_ << std::endl;
}

void multi_thread_error::run(int count) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto                         c = count;
        std::unique_lock<std::mutex> locker(mut_);
        while (c--) {
            value_++;
        }
    });
    t2_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;

        std::unique_lock<std::mutex> locker(mut_);
        while (c--) {
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "one core mut fore while value: " << value_ << std::endl;
}

template <> void multi_thread_error::run<true, true>(int count, run_type<true, true> type) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    t2_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
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

template <> void multi_thread_error::run<false, true>(int count, run_type<false, true> type) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    t2_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "mutil core mut value: " << value_ << std::endl;
}

template <> void multi_thread_error::run<true, false>(int count, run_type<true, false> type) {
    value_ = 0;
    t1_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            // std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    t2_    = std::thread([ & ]() {
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
        // std::cout << "Thread is running on CPU " << sched_getcpu() << std::endl;
        auto c = count;
        while (c--) {
            // std::unique_lock<std::mutex> locker(mut_);
            value_++;
        }
    });
    if (t1_.joinable()) t1_.join();
    if (t2_.joinable()) t2_.join();
    std::cout << "one core no mut value: " << value_ << std::endl;
}
} // namespace thread
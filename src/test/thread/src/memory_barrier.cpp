#include "thread/memory_barrier.h"
#include <iostream>
#include <thread>
#include <unistd.h>
namespace thread {
std::atomic<int> counter(0); // 原子计数器
// 一个简单的线程函数，增加计数器
void increment2(int id) {
    for (int i = 0; i < 100; ++i) {
        // 使用memory_order_acq_rel保证这个操作对其他线程是可见的
        // 并且看到的是一个连续的、未被重排的操作序列
        int old_count = counter.fetch_add(1, std::memory_order_seq_cst);
        std::cout << "Thread " << id << " incremented counter to " << old_count + 1 << std::endl;
    }
}
void increment(int id) {
    pid_t tid = syscall(SYS_gettid);

    // 创建cpu集合
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // 假设我们要将线程绑定到第0个核心
    int core = id + 1;
    CPU_SET(core, &cpuset);

    // 设置亲和性
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    for (int i = 0; i < 100000; ++i) {
        int old_count =
            counter.fetch_add(1, std::memory_order_relaxed); // 使用memory_order_relaxed增加计数器
        std::cout << "线程ID " << id << ": " << counter.load(std::memory_order_relaxed)
                  << std::endl;
    }
}
void memory_barrier::run(int value, std::memory_order load, std::memory_order store) {
    const int                num_threads = 10;
    std::vector<std::thread> threads;

    // 创建并启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment, i);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        t.join();
    }

    // 输出最终的计数器值
    std::cout << "最终的计数结果：-----" << counter.load(std::memory_order_relaxed) << std::endl;
}
} // namespace thread
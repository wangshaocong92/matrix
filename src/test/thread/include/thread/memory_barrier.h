#include <atomic>
#include <thread>
#include <vector>
#pragma

namespace thread {

class memory_barrier {
public:
    memory_barrier()  = default;
    ~memory_barrier() = default;

    void run(int value, std::memory_order load, std::memory_order store);

private:
    std::vector<int> vec_value_;
    int              value_{0};
    std::atomic_int atomic_{0};
    std::atomic_bool atomic2_{false};
    std::thread      t1_;
    std::thread      t2_;
};

} // namespace thread
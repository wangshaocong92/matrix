#include <omp.h>
#include <stdio.h>

/*
threadprivate 用于声明一个变量为线程私有，每个线程都有一个独立的变量副本
1. 变量必须为全局变量
2. 必须关闭动态线程分配
*/

int         a = 5;
#pragma omp threadprivate(a)

int main() {
    printf("a is %d\n", a);
    omp_set_num_threads(3);
    omp_set_dynamic(0); // 关闭动态线程
#pragma omp parallel    // 第一个并行区域
    {
        printf("a is %d\n", a); /// threadprivate 会继承外部的值
        a += omp_get_thread_num() + 1;
    }
#pragma omp parallel // 第二个并行区域
    { printf("thread %d : a is %d\n", omp_get_thread_num(), a); }

    /// 看起来会用上并行区间内最小的值来重新初始化a
    printf("a is %d\n", a);
    /*
        而 copyin 子句专门用于 threadprivate 变量，可以为所有线程的 threadprivate 变量分配相同的值。

        用 copyin 子句设置的 threadprivate
       变量，在进入并行区域时，会用主线程变量值为每个线程的该变量副本初始化。
    */

#pragma omp parallel copyin(a) // 第三个并行区域
    { printf("thread %d : a is %d\n", omp_get_thread_num(), a); }
    return 0;
}
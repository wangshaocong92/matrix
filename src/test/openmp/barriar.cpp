#include <omp.h>
#include <stdio.h>

/*
barrier 用于并行域内代码的线程同步，线程执行到 barrier 时要停下等待，直到所有线程都执行到 barrier
时才继续往下执行

如下代码，会在所有线程都执行完 barrier 前的语句后才执行 barrier 语句
*/
int main() {
    omp_set_num_threads(3);
#pragma omp parallel
    {
        printf("hhh, here is thread %d\n", omp_get_thread_num());
#pragma omp barrier
        printf("thread %d cross barrier\n", omp_get_thread_num());
    }

#pragma omp parallel
    {
        printf("hhh, here is thread %d\n", omp_get_thread_num());
        printf("thread %d cross barrier\n", omp_get_thread_num());
    }

#pragma omp parallel // 第一个并行区域
    {
#pragma omp for nowait
        for (int i = 1; i <= 3; i++) {
            printf("nowait thread %d : hhh\n", omp_get_thread_num());
        }
        printf("nowait thread %d : www\n", omp_get_thread_num());
    }

#pragma omp parallel // 第一个并行区域
    {
#pragma omp for
        for (int i = 1; i <= 3; i++) {
            printf("wait thread %d : hhh\n", omp_get_thread_num());
        }
        printf("wait thread %d : www\n", omp_get_thread_num());
    }
}

// parallel 和 for 创建区域结束时都有隐式同步 barrier

/*
nowait 子句
上面介绍 barrier 时提到： parallel 和 for 创建区域结束时事实上都有隐式同步 barrier

nowait 子句即用于取消 parallel 和 for 中的默认隐含 barrier
，使一个线程完成指定工作后无需等待其它线程，直接进行后续的工作，如下：
*/
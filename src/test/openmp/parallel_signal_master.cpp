#include <omp.h>
#include <stdio.h>

int main() {
    omp_set_num_threads(3);
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < 3; ++i) {
            printf("hhh, here is thread %d\n", omp_get_thread_num());
        }
#pragma omp master // 只在主线程执行
        for (int i = 0; i < 3; ++i) {
            printf("master %d run", omp_get_thread_num());
        }
#pragma omp single // 只在单一线程执行，具体是哪个线程不确定
        for (int i = 0; i < 3; ++i) {
            printf("the only thread to run single is %d\n", omp_get_thread_num());
        }
    }
}
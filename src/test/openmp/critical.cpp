#include <omp.h>
#include <stdio.h>

/*
critical 用在一段代码临界区之前，保证每次只有一个 OpenMP
线程进入，即保证程序的特定区域一次（时刻）只有一个线程执行

critical 的作用和 atomic 非常相似，区别是 atomic 只作用于单个数据操作（原子操作），而 critical
作用域是一段代码块
*/
int main() {
    omp_set_num_threads(12);
    int a = 0;
#pragma omp parallel shared(a)
    {
#pragma omp critical
        {
            a++;
            printf("thread %d : ++a is %d\n", omp_get_thread_num(), a);
        }
    }
}

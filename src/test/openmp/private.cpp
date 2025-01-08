#include <omp.h>
#include <stdio.h>

/*
private 用于声明变量为线程私有，每个线程都有一个独立的变量副本

private 和 threadprivate 的区别：
threadprivate 需要在全局区域声明，而 private 可以在任何并行快之外的地方声明
threadprivate 可以在多个并行区域内使用，而 private 只能在一个并行区域内使用，出了并行区域就失效
threadprivate 不可以在动态线程分配下使用，而 private 可以
threadprivate 不需要在并行区域内初始化，而 private 需要

private 和 firstprivate 的区别：
firstprivate 可以继承外部变量的值，而 private 不可以

private 和 lastprivate 的区别：
lastprivate 可以将退出并行部分时将计算结果赋值回原变量，而 private 不可以
*/

int main() {
    omp_set_num_threads(3);
    int a = 5;
#pragma omp parallel private(a)
    {
        a = 1;
        a++;
#pragma omp for
        for (int i = 1; i <= 3; i++) {
            printf("one, here is thread %d and a is %d\n", omp_get_thread_num(), a);
        }
    }

#pragma omp parallel private(a)
    {
#pragma omp for
        for (int i = 1; i <= 3; i++) {
            printf("two, here is thread %d and a is %d\n", omp_get_thread_num(), a);
        }
    }

    printf("a is %d\n", a);
}

/*
output:
one, here is thread 1 and a is 2
one, here is thread 2 and a is 2
one, here is thread 0 and a is 2
two, here is thread 0 and a is 0
two, here is thread 1 and a is 0
two, here is thread 2 and a is 0
a is 5
*/
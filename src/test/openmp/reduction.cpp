#include <omp.h>
#include <stdio.h>

/*
在每个线程为全局的原始变量创建属性为 private 的线程局部变量副本
将各个线程的局部变量副本进行指定的操作，并将操作后的结果返回全局的原始变量

并不继承外部的值，reduction 线程局部变量副本，需要在内部初始化，否则其值是不确定的
*/
int main() {
    omp_set_num_threads(3);
    int a = 2;
#pragma omp parallel reduction(+ : a)
    {
        printf("a is %d\n", a);
        a = 4;
#pragma omp for
        for (int i = 1; i <= 3; i++) {
            a += i;
            // printf("hhh, here is thread %d, a is %d and i is %d\n", omp_get_thread_num(), a, i);
        }
    }
    printf("at last, a is %d\n", a); // 2 + 1 + 2 + 3 = 8
    a = 2;
#pragma omp parallel reduction(* : a)
    {
        printf("a is %d\n", a);
        // a = 4;
#pragma omp for
        for (int i = 1; i <= 3; i++) {
            a += i;
            // printf("hhh, here is thread %d, a is %d and i is %d\n", omp_get_thread_num(), a, i);
        }
    }
    printf("at last, a is %d\n", a);

    return 0;
}

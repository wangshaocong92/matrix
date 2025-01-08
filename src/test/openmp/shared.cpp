#include <omp.h>
#include <stdio.h>

/*
    该子句用于声明变量列表中的所有变量都是进程公共的。即需要处理线程间的竞争问题
    shared 子句是带有继承初值和将最终值传回给原变量的作用的。
*/

int main() {
    int a = 3;
#pragma omp parallel for shared(a)
    for (auto i = 1; i <= 28; i++) {
#pragma omp atomic
        a += i;
        // printf("hhh, here is thread %d, a is %d and i is %d\n", omp_get_thread_num(), a, i);
    }
    printf("at last, a is %d\n", a);

    a = 3;
    for (auto i = 1; i <= 28; i++) a += i;
    printf("at last, a is %d\n", a);
}

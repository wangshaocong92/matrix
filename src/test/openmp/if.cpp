#include <omp.h>
#include <stdio.h>

/*
if 语句中的参数为真，则并行执行，否则串行执行
*/
void        test(int val) {
#pragma omp parallel if (val) // val是否需要编译期可知
    {
        if (omp_in_parallel()) {
#pragma omp single
            { printf("val = %d, parallelized with %d threads\n", val, omp_get_num_threads()); }
        } else {
            printf("val = %d, serialized\n", val);
        }
    }
}

int main() {
    omp_set_num_threads(2);
    test(0);
    test(2);
}
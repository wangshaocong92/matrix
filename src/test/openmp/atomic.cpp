#include <omp.h>
#include <stdio.h>

/// a
int main() {
    omp_set_num_threads(12);

    for (auto i = 0; i < 1000; i++) {
        int a = 0;
#pragma omp parallel shared(a)
        {
#pragma omp atomic
            a += a + 1;
        }
        if (a != 268435455) printf("a is %d\n", a);
        int b = 0;
#pragma omp parallel shared(b)
        { b += b + 1; }
        if (b != 268435455) printf("b is %d\n", b);
    }

    return 0;
}

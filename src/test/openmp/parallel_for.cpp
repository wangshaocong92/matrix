#include <omp.h>
#include <stdio.h>

int main() {
    omp_set_num_threads(3);
#pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        printf("hhh, here is thread %d\n", omp_get_thread_num());
    }
}

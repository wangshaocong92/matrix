
#include <omp.h>
#include <stdio.h>

int main() {
    omp_set_num_threads(3);
#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int i = 0; i < 3; ++i) {
                printf("hhh, here is thread %d and section 1\n", omp_get_thread_num());
            }
        }
#pragma omp section
        {
            for (int i = 0; i < 3; ++i) {
                printf("hhh, here is thread %d and section 2\n", omp_get_thread_num());
            }
        }
#pragma omp section
        {
            for (int i = 0; i < 3; ++i) {
                printf("hhh, here is thread %d and section 3\n", omp_get_thread_num());
            }
        }
    }
}

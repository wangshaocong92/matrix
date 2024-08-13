#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
int main(void) {
    int i;
    int buf[ 100 ] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (i = 0; i < 2; i++) {
        fork();
        printf("+");
        // write("/home/pi/code/test_fork/test_fork.txt",buf,8);
        write(STDOUT_FILENO, "-", 1);
    }
    return 0;
}
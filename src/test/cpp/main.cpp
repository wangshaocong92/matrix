#include "test_concept.h"
#include "test_format.h"
#include "test_generator.h"
#include "test_coroutine.h"
 
int main()
{
    format::print();
    generator::print();
    test_concept::print();
    std::cout<< "Start main()\n";

    test_coroutine::run();
    return 0;
}
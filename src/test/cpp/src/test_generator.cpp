#include "test_generator.h"


namespace generator {

void print()
{
    Tree<char> tree[] {
        {'D',tree + 1, tree + 2},    {'B',tree + 3, tree + 4},   {'F',tree + 5, tree + 6}, 
          {'A'},   {'C'},{'E'},{'G'}
    };
    for(auto x : tree->traverse_inorder())
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

} // namespace generator
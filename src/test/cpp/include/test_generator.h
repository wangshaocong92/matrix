#pragma once
#include <bits/elements_of.h>
#include <generator>
#include <iostream>
#include <ranges>

namespace generator {

template <class T> struct Tree {
  T value;
  Tree *left{nullptr};
  Tree *right{nullptr};

  std::generator<const T &> traverse_inorder() const {
    if (left)
      co_yield std::ranges::elements_of(left->traverse_inorder());
    co_yield value;
    if (right) {
      for (const auto &x : right->traverse_inorder()) {
        co_yield x;
      }
    }
  }
};

void print();

} // namespace generator
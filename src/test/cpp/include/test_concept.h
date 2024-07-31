#pragma once

#include <iostream>
namespace test_concept {
    template <typename T> concept r = requires(T x,T y) { 
        x.add(y); 
    }; 
    template <typename T> requires r<T>
    class SelfAddition
    {
        public:
        SelfAddition(T &x)
        {
        }
    };

    struct test_s{
      void  add(const test_s & s)
      {
        a += s.a;
      }
      int a = 0;

    };


    void print()
    {
        // int a = 0;
        // SelfAddition<int> si(a);
        test_s s;
        SelfAddition<test_s> ts(s);
    }

}
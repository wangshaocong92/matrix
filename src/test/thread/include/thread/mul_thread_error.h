#pragma once



#include <iostream>
#include <thread>
namespace thread {

class multi_thread_error
{
    public:
     multi_thread_error() =default;

     ~multi_thread_error()
     {
        if(t1_.joinable())
            t1_.join();
        if(t2_.joinable())
            t2_.join();
        std::cout << "value: " << value_ <<std::endl;
     }


    void run(int count)
    {
        t1_ = std::thread([&](){
            auto c = count;
            while (c--) {
                value_++;
            }
        });
        t2_ = std::thread([&](){
            auto c = count;
            while (c--) {
                value_++;
            }
        });
    }
     private:
     int value_ {0};
     std::thread t1_;
     std::thread t2_;
};
}
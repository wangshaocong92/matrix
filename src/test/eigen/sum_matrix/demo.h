#pragma once

namespace demo{
class democlass{
public:
    democlass(int a):a_(a) {};
    ~democlass() = default;
    democlass & operator+(const democlass &a)
    {
        a_ += a.a_;
        return *this;
    }
    int a() const
    {
        return a_;
    }
private:
    int a_ = 0;
};
};
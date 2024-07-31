#pragma once

namespace matrix {
template<typename Type, int W,int H> class Matrix{
    public:
    static constexpr bool IsSquare = (W == H);  
    private:
    Type * mptr_;
};
}

// Inline exp and finiteness for the search's hot loops. MinGW's msvcrt exp runs an x87 long-double path
// (__expl_internal) and std::isfinite calls __fpclassify; together they took about a third of a Play producer's CPU.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fastmath {

// True unless x is infinite or NaN.
inline bool finite(double x){uint64_t b;std::memcpy(&b,&x,sizeof b);return (b&0x7ff0000000000000ull)!=0x7ff0000000000000ull;}

// e^x within about 2 ulp: x = k ln2 + r with |r| <= ln2/2 (Cody-Waite split of ln2), e^r by its degree-13 Taylor
// polynomial (truncation below 1e-17 relative), scaled by 2^k through the exponent bits. NaN stays NaN, x < -746
// (and -inf) gives 0, x > 709.78 gives +inf; scales outside the normal exponent range go through std::ldexp.
inline double exp(double x){
 if(!(x>-746.))return x!=x?x:0.;
 if(x>709.782712893384)return std::numeric_limits<double>::infinity();
 const double k=std::floor(x*1.4426950408889634+.5);
 const double r=(x-k*.6931471803691238)-k*1.9082149292705877e-10;
 double p=1./6227020800.;
 p=p*r+1./479001600.;p=p*r+1./39916800.;p=p*r+1./3628800.;p=p*r+1./362880.;p=p*r+1./40320.;p=p*r+1./5040.;
 p=p*r+1./720.;p=p*r+1./120.;p=p*r+1./24.;p=p*r+1./6.;p=p*r+.5;p=p*r+1.;p=p*r+1.;
 const int n=int(k);
 if(n<-1022 || n>1023)return std::ldexp(p,n);
 const uint64_t bits=uint64_t(n+1023)<<52;double scale;std::memcpy(&scale,&bits,sizeof scale);
 return p*scale;
}

}  // namespace fastmath

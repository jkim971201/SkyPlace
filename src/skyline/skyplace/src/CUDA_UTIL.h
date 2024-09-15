#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cstdio>
#include <cuda.h>
#include <cufft.h> 

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

#define FFT_PI 3.141592653589793238462643383279502884197169

#ifndef CUDA_CHECK
#define CUDA_CHECK(status) __CUDA_CHECK(status, __FILE__, __LINE__)
#endif

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                   \
    do {                                                                      \
        cusolverStatus_t err_ = (err);                                        \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                       \
        }                                                                     \
    } while (0)

namespace skyplace
{

template<typename T>
struct negate
{
  __host__ __device__
  T operator() (T& x) const
  {
    return -x;
  }
};

struct minusAndDiv
{
  float e;

  minusAndDiv(float e) : e(e) {}

  __host__ __device__ 
  float operator() (float x, float y)
  {
    return (x - y) / e;
  }
};

struct saxpy
{
  float a;

  saxpy(float a) : a(a) {}

  __host__ __device__ 
  float operator() (float x, float y)
  {
    return x + a * y;
  }
};

struct axby
{
  float a;
  float b;

  axby(float a, float b) : a(a), b(b) {}

  __host__ __device__ 
  float operator() (float x, float y)
  {
    return a * x + b * y; 
  }
};

struct axby2
{
  float a;
  float b;

  axby2(float a, float b) : a(a), b(b) {}

  __host__ __device__ 
  float operator() (float x, float y)
  {
    return a * x + b * y * y; 
  }
};

struct quadratic 
{
  quadratic() {}

  __host__ __device__ 
  float operator() (float x, float A)
  {
    return x * A * x;
  }
};

struct rectify 
{
  rectify() {}

  __host__ __device__ 
  float operator() (float x)
  {
    return (x < 1e-2) ? 1e-2 : x;
  }
};

template<typename T>
struct myAbs
{
  __host__ __device__
  T operator() (const T& x) const
  {
    if(x >= 0)
      return x;
    else
      return x * -1;
  }
};

struct mulA
{
  float a;

  mulA(float a) : a(a) {}

  __host__ __device__ 
  float operator() (float x, float y)
  {
    return a * x + a * y;
  }
};

struct scalarAdd
{
  float a;

  scalarAdd(float a) : a(a) {}

  __host__ __device__ 
  float operator() (float x)
  {
    return x + a;
  }
};

struct scalarMul
{
  float a;

  scalarMul(float a) : a(a) {}

  __host__ __device__ 
  float operator() (float x)
  {
    return a * x;
  }
};

template <typename T>
struct square
{
  __host__ __device__
  T operator()(const T& x) const 
  { 
    return x * x;
  }
};

struct checkAndMultiply
{
	float coeff_;

	checkAndMultiply(float coeff) : coeff_(coeff) {}

  __host__ __device__
  float operator()(float density_scale, bool flag) 
  { 
    return (flag == true) ? density_scale * coeff_ : density_scale;
  }
};

inline void vectorAdd(float x, thrust::device_vector<float>& a)
{
  thrust::transform(a.begin(), a.end(), a.begin(), scalarAdd(x));
}

inline void vectorAdd(float x, const thrust::device_vector<float>& a,
                                     thrust::device_vector<float>& b)
{
  thrust::transform(a.begin(), a.end(), b.begin(), scalarAdd(x));
}

inline void vectorAdd(const thrust::device_vector<float>& a, 
                      const thrust::device_vector<float>& b,
                            thrust::device_vector<float>& c)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::plus<float>());
}

inline void vectorSub(const thrust::device_vector<float>& a, 
                      const thrust::device_vector<float>& b,
                            thrust::device_vector<float>& c)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::minus<float>());
}

inline void vectorMul(const thrust::device_vector<float>& a, 
                      const thrust::device_vector<float>& b,
                            thrust::device_vector<float>& c)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::multiplies<float>());
}

inline void vectorDiv(const thrust::device_vector<float>& a, 
                      const thrust::device_vector<float>& b,
                            thrust::device_vector<float>& c)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::divides<float>());
}

inline void vectorAddAxBy(float a, float b,
                          const thrust::device_vector<float>& x, 
                          const thrust::device_vector<float>& y,
                                thrust::device_vector<float>& z)
{
  thrust::transform(x.begin(), x.end(),
                    y.begin(), 
                    z.begin(),
                    axby(a, b) );
}

inline void vectorAddAxBy2(float a, float b,
                           const thrust::device_vector<float>& x, 
                           const thrust::device_vector<float>& y,
                                 thrust::device_vector<float>& z)
{
  thrust::transform(x.begin(), x.end(),
                    y.begin(), 
                    z.begin(),
                    axby2(a, b) );
}

// Compute x^T * A * x
inline float vectorQuadratic(const thrust::device_vector<float>& x,
                             const thrust::device_vector<float>& A,
                                   thrust::device_vector<float>& workSpace)
{
  float sum = 0.0;

  thrust::transform(x.begin(), x.end(), 
                    A.begin(),
                    workSpace.begin(),
                    quadratic() );

  sum = thrust::reduce(workSpace.begin(), workSpace.end());

  return sum;
}

inline void vectorRectify(thrust::device_vector<float>& a)
{
  thrust::transform(a.begin(), a.end(), a.begin(), rectify() );
}

template<typename T>
inline T vectorMin(const thrust::device_vector<T>& a)
{
  return *thrust::min_element(thrust::device, a.begin(), a.end());
}

template<typename T>
inline T vectorMax(const thrust::device_vector<T>& a)
{
  return *thrust::max_element(thrust::device, a.begin(), a.end());
}

// - > + or + -> -
inline void invertSign(thrust::device_vector<float>& a)
{
  thrust::transform(a.begin(), a.end(), a.begin(), thrust::negate<float>());
}

inline void invertSign(const thrust::device_vector<float>& a, 
                             thrust::device_vector<float>& b)
{
  thrust::transform(a.begin(), a.end(), b.begin(), thrust::negate<float>());
}

// Scalar Multiplication
// a <- ka
inline void vectorScalarMul(float k, thrust::device_vector<float>& a)
{
  thrust::transform(a.begin(), a.end(), a.begin(), scalarMul(k));
}

// b <- ka
inline void vectorScalarMul(float k, const thrust::device_vector<float>& a,
                                           thrust::device_vector<float>& b)
{
  thrust::transform(a.begin(), a.end(), b.begin(), scalarMul(k));
}

inline void vectorInit(thrust::device_vector<float>& a, float x)
{
  thrust::fill(a.begin(), a.end(), x);
}

// Compute ||a||_1
inline float compute1Norm(const thrust::device_vector<float>& a)
{
  float sumAbs = thrust::transform_reduce(a.begin(), a.end(), 
                 myAbs<float>(), 0.0, thrust::plus<float>());
  return sumAbs;
}

// Compute ( ||a-b||_2 )^2
// c is workspace vector to store c[i] = a[i] - b[i]
inline float compute2NormSquare(const thrust::device_vector<float>& a,
                                const thrust::device_vector<float>& b,
                                      thrust::device_vector<float>& c)
{
  float sumDist = 0.0;

  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::minus<float>());

  sumDist = thrust::transform_reduce(c.begin(), c.end(), 
                                     square<float>(), 
                                     0.0,
                                     thrust::plus<float>());
  return sumDist;
}

// Compute ( ||a||_2 )^2
// if b is zero, then workspace vector is not necessary
inline float compute2NormSquare(const thrust::device_vector<float>& a)
{
  float sum = 0.0;

  sum = thrust::transform_reduce(a.begin(), a.end(), 
                                 square<float>(), 
                                 0.0,
                                 thrust::plus<float>());
  return sum;
}

// Compute ( a^T * b) 
inline float innerProduct(const thrust::device_vector<float>& a,
                          const thrust::device_vector<float>& b)
{
  float sum = 0.0;

  sum = thrust::inner_product(a.begin(), a.end(),
                              b.begin(), 
                              0.0);
  return sum;
}

// CUDA Kernel Functions should be defined outside of the C++ Class
__device__ inline float getXCoordiInsideLayoutDevice(const float cellCx, 
                                                     const float cellDx, 
                                                     const float dieLx,  
                                                     const float dieUx)
{
  float newCx = cellCx;

  if(cellCx - cellDx / 2 < dieLx)
    newCx = dieLx + cellDx / 2;
  if(cellCx + cellDx / 2 > dieUx)
    newCx = dieUx - cellDx / 2;

  return newCx;
}

__device__ inline float getYCoordiInsideLayoutDevice(const float cellCy, 
                                                     const float cellDy, 
                                                     const float dieLy,  
                                                     const float dieUy)
{
  float newCy = cellCy;

  if(cellCy - cellDy / 2 < dieLy)
    newCy = dieLy + cellDy / 2;
  if(cellCy + cellDy / 2 > dieUy)
    newCy = dieUy - cellDy / 2;

  return newCy;
}

__device__ inline float getPartialDerivative(const float cellMin, 
                                             const float cellMax,
                                             const float  binMin,
                                             const float  binMax)
{
  if(cellMax >= binMin && cellMax <= binMax)
    return 1.0;
  else if(cellMax > binMax && cellMin <= binMin)
    return 0.0;
  else if(cellMin > binMin && cellMin <= binMax)
    return -1.0;
  else
    return 0.0;
}                                             

}; // namespace skyplace

inline __device__ __host__ bool isPowerOf2(int val)
{
  return val && (val & (val - 1)) == 0;
}

template<typename T>
inline T* setThrustVector(const size_t size, 
                          thrust::device_vector<T>& d_vector) 
{
  d_vector.resize(size);
  return thrust::raw_pointer_cast(&d_vector[0]);
}

template<typename T>
inline const T* getRawPointer(const thrust::device_vector<T>& thrustVector)
{
  return thrust::raw_pointer_cast(&thrustVector[0]);
}

template<typename T>
inline T* getRawPointer(thrust::device_vector<T>& thrustVector)
{
  return thrust::raw_pointer_cast(&thrustVector[0]);
}

inline void __CUDA_CHECK(cudaError_t status, const char *file, const int line)
{
  if(status != cudaSuccess)
  {
    fprintf(stderr, "[CUDA-ERROR] Error %s at line %d in file %s\n", cudaGetErrorString(status), line, file);
    exit(status);
  }
}

inline __device__ int INDEX(const int hid, const int wid, const int N)
{
  return (hid * N + wid);
}

inline __device__ cufftDoubleComplex complexMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  cufftDoubleComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

inline __device__ cufftComplex complexMul(const cufftComplex &x, const cufftComplex &y)
{
  cufftComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

inline __device__ cufftDoubleReal RealPartOfMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  return x.x * y.x - x.y * y.y;
}

inline __device__ cufftReal RealPartOfMul(const cufftComplex &x, const cufftComplex &y)
{
  return x.x * y.x - x.y * y.y;
}

inline __device__ cufftDoubleReal ImaginaryPartOfMul(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  return x.x * y.y + x.y * y.x;
}

inline __device__ cufftReal ImaginaryPartOfMul(const cufftComplex &x, const cufftComplex &y)
{
  return x.x * y.y + x.y * y.x;
}

inline __device__ cufftDoubleComplex complexAdd(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  cufftDoubleComplex res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

inline __device__ cufftComplex complexAdd(const cufftComplex &x, const cufftComplex &y)
{
  cufftComplex res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

inline __device__ cufftDoubleComplex complexSubtract(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  cufftDoubleComplex res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

inline __device__ cufftComplex complexSubtract(const cufftComplex &x, const cufftComplex &y)
{
  cufftComplex res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

inline __device__ cufftDoubleComplex complexConj(const cufftDoubleComplex &x)
{
  cufftDoubleComplex res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

inline __device__ cufftComplex complexConj(const cufftComplex &x)
{
  cufftComplex res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

inline __device__ cufftDoubleComplex complexMulConj(const cufftDoubleComplex &x, const cufftDoubleComplex &y)
{
  cufftDoubleComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

inline __device__ cufftComplex complexMulConj(const cufftComplex &x, const cufftComplex &y)
{
  cufftComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

#endif

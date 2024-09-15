#include <cstdio>
#include <cmath>
#include <chrono>
#include <climits> // for MAX FLOAT
#include <cfloat>

// CUDA
#include <cuda_runtime.h>

#include "CUDA_UTIL.h"
#include "WireLengthGradient.h"

namespace skyplace
{

// CUDA Kernel Functions should be defined outside of the C++ Class
__global__ void computeMinMax(const int    numNet,
                              const int*   netStart,
                              const float* pinX, 
                              const float* pinY, 
                                    float* maxPinXArr, 
                                    float* minPinXArr,
                                    float* maxPinYArr, 
                                    float* minPinYArr)
{
  // i := netID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numNet)
  {
    float maxPinX =       0;
    float maxPinY =       0;
    float minPinX = FLT_MAX;
    float minPinY = FLT_MAX;

    for(int j = netStart[i]; j < netStart[i+1]; j++)
    {
      maxPinX = max(pinX[j], maxPinX);
      minPinX = min(pinX[j], minPinX);
      maxPinY = max(pinY[j], maxPinY);
      minPinY = min(pinY[j], minPinY);
    }

    maxPinXArr[i] = maxPinX;
    minPinXArr[i] = minPinX;
    maxPinYArr[i] = maxPinY;
    minPinYArr[i] = minPinY;
  }
}

__global__ void computeExp(const int    numPin,
                           const float  gammaInv, 
                           const int*   pin2Net,
                           const float* pinX, 
                           const float* pinY, 
                           const float* maxPinX,
                           const float* maxPinY,
                           const float* minPinX,
                           const float* minPinY,
                                 float* apX, 
                                 float* apY, 
                                 float* amX, 
                                 float* amY)
{
  // i := pinID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numPin)
  {
    int netID = pin2Net[i];
    apX[i] = exp(+(pinX[i] - maxPinX[netID]) * gammaInv);  
    apY[i] = exp(+(pinY[i] - maxPinY[netID]) * gammaInv);  
    amX[i] = exp(-(pinX[i] - minPinX[netID]) * gammaInv);  
    amY[i] = exp(-(pinY[i] - minPinY[netID]) * gammaInv);  
  }
}

__global__ void computeExpSum(const int    numNet,
                              const int*   netStart,
                              const float* apX, 
                              const float* apY,
                              const float* amX, 
                              const float* amY,
                                    float* bpX, 
                                    float* bpY, 
                                    float* bmX, 
                                    float* bmY)
{
  // i := netID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numNet)
  {
    bpX[i] = 0;
    bpY[i] = 0;
    bmX[i] = 0;
    bmY[i] = 0;
    
    for(int j = netStart[i]; j < netStart[i+1]; j++)
    {
      bpX[i] += apX[j];
      bpY[i] += apY[j];
      bmX[i] += amX[j];
      bmY[i] += amY[j];
    }
  }
}

__global__ void computeXYExpSum(const int    numNet,
                                const int*   netStart,
                                const float* pinX,
                                const float* pinY,
                                const float* apX, 
                                const float* apY,
                                const float* amX, 
                                const float* amY,
                                      float* cpX, 
                                      float* cpY,
                                      float* cmX, 
                                      float* cmY)
{
  // i := netID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(i < numNet)
  {
    cpX[i] = 0;
    cpY[i] = 0;
    cmX[i] = 0;
    cmY[i] = 0;

    for(int j = netStart[i]; j < netStart[i + 1]; j++)
    {
      cpX[i] += pinX[j] * apX[j];
      cpY[i] += pinY[j] * apY[j];
      cmX[i] += pinX[j] * amX[j];
      cmY[i] += pinY[j] * amY[j];
    }  
  }
}

__global__ void computePinGrad(const int    numPin,
                               const float  gInv,
                               const int*   pin2Net,
                               const float* pinX,
                               const float* pinY,
                               const float* apX, 
                               const float* apY,
                               const float* amX, 
                               const float* amY,
                               const float* bpX, 
                               const float* bpY,
                               const float* bmX, 
                               const float* bmY,
                               const float* cpX, 
                               const float* cpY,
                               const float* cmX, 
                               const float* cmY,
                               const float* netWeight,
                                     float* pinGradX,  
                                     float* pinGradY)
{
  // i := pinID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numPin)
  {
    int nID = pin2Net[i]; // netID
    pinGradX[i] 
      = ((1 - pinX[i] * gInv) * bmX[nID] + gInv * cmX[nID]) * amX[i] / bmX[nID] / bmX[nID]
      - ((1 + pinX[i] * gInv) * bpX[nID] - gInv * cpX[nID]) * apX[i] / bpX[nID] / bpX[nID];

    pinGradY[i] 
      = ((1 - pinY[i] * gInv) * bmY[nID] + gInv * cmY[nID]) * amY[i] / bmY[nID] / bmY[nID]
      - ((1 + pinY[i] * gInv) * bpY[nID] - gInv * cpY[nID]) * apY[i] / bpY[nID] / bpY[nID];

    // Applying Net Weight
    float weightForThisNet = netWeight[nID];
    pinGradX[i] *= weightForThisNet; 
    pinGradY[i] *= weightForThisNet;
  }
}

__global__ void addPinGrad(const int numCell,
                           const int* cellStart,
                           const int* cellNumPin,
                           const int* pinListCell,
                           const float* pinGradX,
                           const float* pinGradY,
                                 float* cellGradX,
                                 float* cellGradY)
{
  // i := CellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    cellGradX[i] = 0;
    cellGradY[i] = 0;
  
    int firstPinID = cellStart[i];
  
    for(int j = 0; j < cellNumPin[i]; j++)
    {
      int pinID = pinListCell[firstPinID + j];
      cellGradX[i] += pinGradX[pinID];
      cellGradY[i] += pinGradY[pinID];
    }
  }
}

__global__ void updatePinCoordinateKernel(const int    numPin, 
                                          const int    numMovable,
                                          const int*   cellStart,
                                          const int*   cellNumPin,
                                          const int*   pinListCell,
                                          const float* newCellX,
                                          const float* newCellY, 
                                          const float* pinOffsetX,
                                          const float* pinOffsetY,
                                                float* pinX,
                                                float* pinY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numMovable)
  {
    int firstPinID = cellStart[i];
  
    for(int j = 0; j < cellNumPin[i]; j++)
    {
      int pinID = pinListCell[firstPinID + j];
      pinX[pinID] = newCellX[i] + pinOffsetX[pinID];
      pinY[pinID] = newCellY[i] + pinOffsetY[pinID];
    }
  }
}

__global__ void computeNetBBox(const int    numNet,
                               const int*   netStart,
                               const float* pinX,
                               const float* pinY,
                                     float* netBBoxWidth,
                                     float* netBBoxHeight)
{
  // i := netID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numNet)
  {
    float maxPinX =       0;
    float maxPinY =       0;
    float minPinX = FLT_MAX;
    float minPinY = FLT_MAX;

    // j = pinID
    for(int j = netStart[i]; j < netStart[i+1]; j++)
    {
      maxPinX = max(pinX[j], maxPinX);
      minPinX = min(pinX[j], minPinX);
      maxPinY = max(pinY[j], maxPinY);
      minPinY = min(pinY[j], minPinY);
    }

    netBBoxWidth[i]  = maxPinX - minPinX;
    netBBoxHeight[i] = maxPinY - minPinY;
  }
}

__global__ void computeWAForEachNet(const int numNet,
                                    const float* bpX,
                                    const float* bpY,
                                    const float* bmX,
                                    const float* bmY,
                                    const float* cpX,
                                    const float* cpY,
                                    const float* cmX,
                                    const float* cmY,
                                          float* waForEachNetX,
                                          float* waForEachNetY)
{
  // i := netID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numNet)
  {
    waForEachNetX[i] = 
      cpX[i] / bpX[i] - cmX[i] / bmX[i];

    waForEachNetY[i] = 
      cpY[i] / bpY[i] - cmY[i] / bmY[i];
  }
}

WireLengthGradient::WireLengthGradient()
  :  
    db_                  (nullptr),
    numCell_             (0),
    numNet_              (0),
    numPin_              (0),
    gammaInv_            (0.0),

    d_ptr_pin2Net_       (nullptr),
    d_ptr_netStart_      (nullptr),
    d_ptr_pinListCell_   (nullptr),
    d_ptr_cellStart_     (nullptr),
    d_ptr_cellNumPin_    (nullptr),

    d_ptr_maxPinX_       (nullptr),
    d_ptr_minPinX_       (nullptr),
    d_ptr_maxPinY_       (nullptr),
    d_ptr_minPinY_       (nullptr),

    d_ptr_apX_           (nullptr),
    d_ptr_bpX_           (nullptr),
    d_ptr_cpX_           (nullptr),
    d_ptr_apY_           (nullptr),
    d_ptr_bpY_           (nullptr),
    d_ptr_cpY_           (nullptr),
    d_ptr_amX_           (nullptr),
    d_ptr_bmX_           (nullptr),
    d_ptr_cmX_           (nullptr),
    d_ptr_amY_           (nullptr),
    d_ptr_bmY_           (nullptr),
    d_ptr_cmY_           (nullptr),

    d_ptr_pinGradX_      (nullptr),
    d_ptr_pinGradY_      (nullptr),

    d_ptr_waForEachNetX_ (nullptr),
    d_ptr_waForEachNetY_ (nullptr),

    wlGradTime_          (0.0)
{}

WireLengthGradient::WireLengthGradient(std::shared_ptr<SkyPlaceDB> db)
  : WireLengthGradient()
{
  printf("[WireLengthGradient] Start Initialization. \n");

  db_ = db;

  numCell_ = db_->numMovable();
  numNet_  = db_->numNet();
  numPin_  = db_->numPin();

  initForCUDAKernel();

  printf("[WireLengthGradient] Initialization Finished.\n");
}

void
WireLengthGradient::computeGrad(float* wlGradX, 
                                float* wlGradY)
{
  int numThread    = 64;
  int numBlockPin  = (numPin_      - 1 + numThread) / numThread; 
  int numBlockNet  = (numNet_  * 2 - 1 + numThread) / numThread; 
  int numBlockCell = (numCell_ * 2 - 1 + numThread) / numThread; 

  auto t1 = std::chrono::high_resolution_clock::now();

  // Step #1 : Compute Min Max
  computeMinMax<<<numBlockNet, numThread>>>(numNet_, 
                                            d_ptr_netStart_, 
                                            d_ptr_pinX_,
                                            d_ptr_pinY_,
                                            d_ptr_maxPinX_, 
                                            d_ptr_minPinX_, 
                                            d_ptr_maxPinY_, 
                                            d_ptr_minPinY_);

  // Step #2 : Compute a+ & a-
  computeExp<<<numBlockPin, numThread>>>(numPin_, 
                                         gammaInv_, 
                                         d_ptr_pin2Net_, 
                                         d_ptr_pinX_, 
                                         d_ptr_pinY_, 
                                         d_ptr_maxPinX_, 
                                         d_ptr_maxPinY_, 
                                         d_ptr_minPinX_, 
                                         d_ptr_minPinY_,
                                         d_ptr_apX_, 
                                         d_ptr_apY_, 
                                         d_ptr_amX_, 
                                         d_ptr_amY_);

  // Step #3 : Compute b+ & b-
  computeExpSum<<<numBlockNet, numThread>>>(numNet_, 
                                            d_ptr_netStart_, 
                                            d_ptr_apX_, 
                                            d_ptr_apY_,
                                            d_ptr_amX_, 
                                            d_ptr_amY_,
                                            d_ptr_bpX_, 
                                            d_ptr_bpY_, 
                                            d_ptr_bmX_, 
                                            d_ptr_bmY_);

  // Step #4 : Compute c+ & c-
  computeXYExpSum<<<numBlockNet, numThread>>>(numNet_, 
                                              d_ptr_netStart_, 
                                              d_ptr_pinX_, 
                                              d_ptr_pinY_,
                                              d_ptr_apX_, 
                                              d_ptr_apY_, 
                                              d_ptr_amX_, 
                                              d_ptr_amY_,
                                              d_ptr_cpX_, 
                                              d_ptr_cpY_, 
                                              d_ptr_cmX_, 
                                              d_ptr_cmY_);

  // Step #5 : Compute Pin Gradient
  computePinGrad<<<numBlockPin, numThread>>>(numPin_,
                                             gammaInv_, 
                                             d_ptr_pin2Net_,
                                             d_ptr_pinX_, 
                                             d_ptr_pinY_,
                                             d_ptr_apX_, 
                                             d_ptr_apY_, 
                                             d_ptr_amX_, 
                                             d_ptr_amY_,
                                             d_ptr_bpX_, 
                                             d_ptr_bpY_, 
                                             d_ptr_bmX_, 
                                             d_ptr_bmY_,
                                             d_ptr_cpX_, 
                                             d_ptr_cpY_, 
                                             d_ptr_cmX_, 
                                             d_ptr_cmY_,
                                             d_ptr_netWeight_,
                                             d_ptr_pinGradX_, 
                                             d_ptr_pinGradY_);

  // Step #6 : Add each pin gradients of cells
  addPinGrad<<<numBlockCell, numThread>>>(numCell_, 
                                          d_ptr_cellStart_, 
                                          d_ptr_cellNumPin_, 
                                          d_ptr_pinListCell_,
                                          d_ptr_pinGradX_, 
                                          d_ptr_pinGradY_, 
                                          wlGradX, 
                                          wlGradY);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  wlGradTime_ += runtime.count();
}

void 
WireLengthGradient::updatePinCoordinates(const float* d_cellCx, 
                                         const float* d_cellCy)
{
  int numThread = 64;
  int numBlockCell = (numCell_ - 1 + numThread) / numThread;

  updatePinCoordinateKernel<<<numBlockCell, numThread>>>(numPin_, 
                                                         numCell_,
                                                         d_ptr_cellStart_,
                                                         d_ptr_cellNumPin_,
                                                         d_ptr_pinListCell_,
                                                         d_cellCx,
                                                         d_cellCy,
                                                         d_ptr_pinOffsetX_,
                                                         d_ptr_pinOffsetY_,
                                                         d_ptr_pinX_,
                                                         d_ptr_pinY_);
}

float
WireLengthGradient::computeHPWL()
{
  int numThread = 64;
  int numBlockNet = (numNet_ - 1 + numThread) / numThread;

  computeNetBBox<<<numBlockNet, numThread>>>(numNet_, 
                                             d_ptr_netStart_,
                                             d_ptr_pinX_,
                                             d_ptr_pinY_,
                                             d_ptr_netBBoxWidth_, 
                                             d_ptr_netBBoxHeight_);

  float hpwlX = thrust::reduce(d_netBBoxWidth_.begin(),
                               d_netBBoxWidth_.end());

  float hpwlY = thrust::reduce(d_netBBoxHeight_.begin(),
                               d_netBBoxHeight_.end());

  return (hpwlX + hpwlY);
}

void
WireLengthGradient::computeGradAfterValue(float* wlGradX, float* wlGradY)
{
  int numThread    = 64;
  int numBlockPin  = (numPin_      - 1 + numThread) / numThread; 
  int numBlockCell = (numCell_ * 2 - 1 + numThread) / numThread; 

  auto t1 = std::chrono::high_resolution_clock::now();

  // Step #5 : Compute Pin Gradient
  computePinGrad<<<numBlockPin, numThread>>>(numPin_,
                                             gammaInv_, 
                                             d_ptr_pin2Net_,
                                             d_ptr_pinX_, 
                                             d_ptr_pinY_,
                                             d_ptr_apX_, 
                                             d_ptr_apY_, 
                                             d_ptr_amX_, 
                                             d_ptr_amY_,
                                             d_ptr_bpX_, 
                                             d_ptr_bpY_, 
                                             d_ptr_bmX_, 
                                             d_ptr_bmY_,
                                             d_ptr_cpX_, 
                                             d_ptr_cpY_, 
                                             d_ptr_cmX_, 
                                             d_ptr_cmY_,
                                             d_ptr_netWeight_,
                                             d_ptr_pinGradX_, 
                                             d_ptr_pinGradY_);

  // Step #6 : Add each pin gradients of cells
  addPinGrad<<<numBlockCell, numThread>>>(numCell_, 
                                          d_ptr_cellStart_, 
                                          d_ptr_cellNumPin_, 
                                          d_ptr_pinListCell_,
                                          d_ptr_pinGradX_, 
                                          d_ptr_pinGradY_, 
                                          wlGradX, 
                                          wlGradY);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  wlGradTime_ += runtime.count();
}

float
WireLengthGradient::getWA()
{
  int numThread = 64;

  int numBlockPin  = (numPin_ - 1 + numThread) / numThread; 
  int numBlockNet  = (numNet_ - 1 + numThread) / numThread;

  // Step #1 : Compute Min Max
  computeMinMax<<<numBlockNet, numThread>>>(numNet_, 
                                            d_ptr_netStart_, 
                                            d_ptr_pinX_,
                                            d_ptr_pinY_,
                                            d_ptr_maxPinX_, 
                                            d_ptr_minPinX_, 
                                            d_ptr_maxPinY_, 
                                            d_ptr_minPinY_);

  // Step #2 : Compute a+ & a-
  computeExp<<<numBlockPin, numThread>>>(numPin_, 
                                         gammaInv_, 
                                         d_ptr_pin2Net_, 
                                         d_ptr_pinX_, 
                                         d_ptr_pinY_, 
                                         d_ptr_maxPinX_, 
                                         d_ptr_maxPinY_, 
                                         d_ptr_minPinX_, 
                                         d_ptr_minPinY_,
                                         d_ptr_apX_, 
                                         d_ptr_apY_, 
                                         d_ptr_amX_, 
                                         d_ptr_amY_);

  // Step #3 : Compute b+ & b-
  computeExpSum<<<numBlockNet, numThread>>>(numNet_, 
                                            d_ptr_netStart_, 
                                            d_ptr_apX_, 
                                            d_ptr_apY_,
                                            d_ptr_amX_, 
                                            d_ptr_amY_,
                                            d_ptr_bpX_, 
                                            d_ptr_bpY_, 
                                            d_ptr_bmX_, 
                                            d_ptr_bmY_);

  // Step #4 : Compute c+ & c-
  computeXYExpSum<<<numBlockNet, numThread>>>(numNet_, 
                                              d_ptr_netStart_, 
                                              d_ptr_pinX_, 
                                              d_ptr_pinY_,
                                              d_ptr_apX_, 
                                              d_ptr_apY_, 
                                              d_ptr_amX_, 
                                              d_ptr_amY_,
                                              d_ptr_cpX_, 
                                              d_ptr_cpY_, 
                                              d_ptr_cmX_, 
                                              d_ptr_cmY_);

  computeWAForEachNet<<<numBlockNet, numThread>>>(numNet_, 
                                                  d_ptr_bpX_,
                                                  d_ptr_bpY_,
                                                  d_ptr_bmX_,
                                                  d_ptr_bmY_,
                                                  d_ptr_cpX_,
                                                  d_ptr_cpY_,
                                                  d_ptr_cmX_,
                                                  d_ptr_cmY_,
                                                  d_ptr_waForEachNetX_,
                                                  d_ptr_waForEachNetY_);

  float waX = thrust::reduce(d_waForEachNetX_.begin(),
                             d_waForEachNetX_.end());

  float waY = thrust::reduce(d_waForEachNetY_.begin(),
                             d_waForEachNetY_.end());

  return (waX + waY);
}

float
WireLengthGradient::getWAafterGrad()
{
  int numThread = 64;
  int numBlockNet = (numNet_ - 1 + numThread) / numThread;

  computeWAForEachNet<<<numBlockNet, numThread>>>(numNet_, 
                                                  d_ptr_bpX_,
                                                  d_ptr_bpY_,
                                                  d_ptr_bmX_,
                                                  d_ptr_bmY_,
                                                  d_ptr_cpX_,
                                                  d_ptr_cpY_,
                                                  d_ptr_cmX_,
                                                  d_ptr_cmY_,
                                                  d_ptr_waForEachNetX_,
                                                  d_ptr_waForEachNetY_);

  float waX = thrust::reduce(d_waForEachNetX_.begin(),
                             d_waForEachNetX_.end());

  float waY = thrust::reduce(d_waForEachNetY_.begin(),
                             d_waForEachNetY_.end());

  return (waX + waY);
}

void
WireLengthGradient::initForCUDAKernel()
{
  d_ptr_maxPinX_   = setThrustVector<float>(numNet_, d_maxPinX_);
  d_ptr_minPinX_   = setThrustVector<float>(numNet_, d_minPinX_);

  d_ptr_maxPinY_   = setThrustVector<float>(numNet_, d_maxPinY_);
  d_ptr_minPinY_   = setThrustVector<float>(numNet_, d_minPinY_);

  d_ptr_apX_       = setThrustVector<float>(numPin_, d_apX_);
  d_ptr_bpX_       = setThrustVector<float>(numNet_, d_bpX_);
  d_ptr_cpX_       = setThrustVector<float>(numNet_, d_cpX_);

  d_ptr_apY_       = setThrustVector<float>(numPin_, d_apY_);
  d_ptr_bpY_       = setThrustVector<float>(numNet_, d_bpY_);
  d_ptr_cpY_       = setThrustVector<float>(numNet_, d_cpY_);

  d_ptr_amX_       = setThrustVector<float>(numPin_, d_amX_);
  d_ptr_bmX_       = setThrustVector<float>(numNet_, d_bmX_);
  d_ptr_cmX_       = setThrustVector<float>(numNet_, d_cmX_);

  d_ptr_amY_       = setThrustVector<float>(numPin_, d_amY_);
  d_ptr_bmY_       = setThrustVector<float>(numNet_, d_bmY_);
  d_ptr_cmY_       = setThrustVector<float>(numNet_, d_cmY_);

  d_ptr_pinGradX_  = setThrustVector<float>(numPin_, d_pinGradX_);
  d_ptr_pinGradY_  = setThrustVector<float>(numPin_, d_pinGradY_);

  d_ptr_netBBoxWidth_  = setThrustVector<float>(numNet_, d_netBBoxWidth_);
  d_ptr_netBBoxHeight_ = setThrustVector<float>(numNet_, d_netBBoxHeight_);

  d_ptr_waForEachNetX_ = setThrustVector<float>(numNet_, d_waForEachNetX_);
  d_ptr_waForEachNetY_ = setThrustVector<float>(numNet_, d_waForEachNetY_);

  // We don't want to store these data permanently
  thrust::host_vector<int> h_pin2Net(numPin_);
  thrust::host_vector<int> h_netStart(numNet_ + 1);
  thrust::host_vector<int> h_pinListCell(numPin_);
  thrust::host_vector<int> h_cellStart(numCell_);
  thrust::host_vector<int> h_cellNumPin(numCell_);

  thrust::host_vector<float> h_pinX(numPin_);
  thrust::host_vector<float> h_pinY(numPin_);

  thrust::host_vector<float> h_pinOffsetX(numPin_);
  thrust::host_vector<float> h_pinOffsetY(numPin_);

  thrust::host_vector<float> h_netWeight(numNet_);

  d_ptr_pin2Net_     = setThrustVector<int>(numPin_    , d_pin2Net_);
  d_ptr_netStart_    = setThrustVector<int>(numNet_ + 1, d_netStart_);
  d_ptr_pinListCell_ = setThrustVector<int>(numPin_    , d_pinListCell_);
  d_ptr_cellStart_   = setThrustVector<int>(numCell_   , d_cellStart_);
  d_ptr_cellNumPin_  = setThrustVector<int>(numCell_   , d_cellNumPin_);

  d_ptr_pinX_        = setThrustVector<float>(numPin_  , d_pinX_);
  d_ptr_pinY_        = setThrustVector<float>(numPin_  , d_pinY_);

  d_ptr_pinOffsetX_  = setThrustVector<float>(numPin_  , d_pinOffsetX_);
  d_ptr_pinOffsetY_  = setThrustVector<float>(numPin_  , d_pinOffsetY_);

  d_ptr_netWeight_   = setThrustVector<float>(numNet_  , d_netWeight_);

  // pinListCell is a list of pin,
  // whose order is determined by cells
  // [Example]
  //   pinListCell[0] : pinID of 1st pin of cell 0
  //   pinListCell[1] : pinID of 2nd pin of cell 0
  //   pinListCell[2] : pinID of 3rd pin of cell 0

  //   pinListCell[3] : pinID of 1st pin of cell 1
  //   pinListCell[4] : pinID of 2nd pin of cell 1
  //   pinListCell[5] : pinID of 3rd pin of cell 1

  //   ...

  // cellNumPin contains the number of pins 
  // in each cells as a list
  // Since the order of PinID is aligned with NetID 
  // We need extra data structure so that we can access 
  // to the correct PinID with Cell ID

  // Build Netlist Information Vectors
  for(auto& net : db_->nets())
  {
    int netID = net->id();
    h_netStart[netID]  = net->pins()[0]->id();
    h_netWeight[netID] = net->weight();
    for(auto& pin : net->pins())
      h_pin2Net[pin->id()] = netID;
  }

  h_netStart[numNet_] = numPin_;
  // we should not leave netStart_[numNet_ + 1]
  // as a garbage value

  int pinListCellIdx = 0;
  for(auto& cell : db_->movableCells())
  {
    int cellID = cell->id();

    h_cellNumPin[cellID] = cell->pins().size();
    h_cellStart[cellID]  = pinListCellIdx;

    for(auto& pin : cell->pins())
    {
      int pinID = pin->id();
      h_pinListCell[pinListCellIdx] = pinID;
      pinListCellIdx++;
    }
  }

  for(auto& pin : db_->pins())
  {
    int pinID = pin->id();
    h_pinX[pinID] = pin->cx();
    h_pinY[pinID] = pin->cy();
    h_pinOffsetX[pinID] = pin->offsetX();
    h_pinOffsetY[pinID] = pin->offsetY();
  }

  thrust::copy(h_pinX.begin(),
               h_pinX.end(),
               d_pinX_.begin());

  thrust::copy(h_pinY.begin(),
               h_pinY.end(),
               d_pinY_.begin());

  thrust::copy(h_pin2Net.begin(),
               h_pin2Net.end(),
               d_pin2Net_.begin());

  thrust::copy(h_netStart.begin(),
               h_netStart.end(),
               d_netStart_.begin());

  thrust::copy(h_pinListCell.begin(),
               h_pinListCell.end(),
               d_pinListCell_.begin());

  thrust::copy(h_cellStart.begin(),
               h_cellStart.end(),
               d_cellStart_.begin());

  thrust::copy(h_cellNumPin.begin(),
               h_cellNumPin.end(),
               d_cellNumPin_.begin());

  thrust::copy(h_pinOffsetX.begin(),
               h_pinOffsetX.end(),
               d_pinOffsetX_.begin());

  thrust::copy(h_pinOffsetY.begin(),
               h_pinOffsetY.end(),
               d_pinOffsetY_.begin());

  thrust::copy(h_netWeight.begin(),
               h_netWeight.end(),
               d_netWeight_.begin());
}

}; // namespace skyplace

#ifndef WIRELENGTH_GRAD_H
#define WIRELENGTH_GRAD_H

#include <memory>
#include <ctime>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "SkyPlaceDB.h"

namespace skyplace
{

class WireLengthGradient
{
  public:
    
    WireLengthGradient();
    WireLengthGradient(std::shared_ptr<SkyPlaceDB> db);

    void computeGrad(float* wlGradX, float* wlGradY);

    void computeGradAfterValue(float* wlGradX, float* wlGradY);

    void updatePinCoordinates(const float* d_cellCx, 
                              const float* d_cellCy);

    float computeHPWL();

		// NOTE: getWA() has to be called after computeGrad. 
		// Otherwise, it will give you a garbage value
    float getWA(); 
    float getWAafterGrad();

		// Getters
    double wlGradTime() const { return wlGradTime_; }

		// Setters
    void setGammaInv (const float gammaInv) { gammaInv_  = gammaInv; }

  private:

    std::shared_ptr<SkyPlaceDB> db_;

    int numNet_;
    int numPin_;
    int numCell_;

    float gammaInv_;

    thrust::device_vector<float> d_pinX_;
    thrust::device_vector<float> d_pinY_;

    float* d_ptr_pinX_;
    float* d_ptr_pinY_;
  
    thrust::device_vector<int> d_pin2Net_;
    int* d_ptr_pin2Net_;
    
    thrust::device_vector<int> d_netStart_;
    int* d_ptr_netStart_;

    thrust::device_vector<int> d_pinListCell_;
    int* d_ptr_pinListCell_;

    thrust::device_vector<int> d_cellStart_;
    int* d_ptr_cellStart_;

    thrust::device_vector<int> d_cellNumPin_;
    int* d_ptr_cellNumPin_;

    thrust::device_vector<float> d_pinOffsetX_;
    float* d_ptr_pinOffsetX_;

    thrust::device_vector<float> d_pinOffsetY_;
    float* d_ptr_pinOffsetY_;

    thrust::device_vector<float> d_netBBoxWidth_;
    float* d_ptr_netBBoxWidth_;

    thrust::device_vector<float> d_netBBoxHeight_;
    float* d_ptr_netBBoxHeight_;

    thrust::device_vector<float> d_waForEachNetX_;
    float* d_ptr_waForEachNetX_;

    thrust::device_vector<float> d_waForEachNetY_;
    float* d_ptr_waForEachNetY_;

    thrust::device_vector<float> d_maxPinX_;
    thrust::device_vector<float> d_minPinX_;

    thrust::device_vector<float> d_maxPinY_;
    thrust::device_vector<float> d_minPinY_;

    float* d_ptr_maxPinX_;
    float* d_ptr_minPinX_;

    float* d_ptr_maxPinY_;
    float* d_ptr_minPinY_;

    thrust::device_vector<float> d_apX_;
    thrust::device_vector<float> d_bpX_;
    thrust::device_vector<float> d_cpX_;

    thrust::device_vector<float> d_apY_;
    thrust::device_vector<float> d_bpY_;
    thrust::device_vector<float> d_cpY_;

    thrust::device_vector<float> d_amX_;
    thrust::device_vector<float> d_bmX_;
    thrust::device_vector<float> d_cmX_;

    thrust::device_vector<float> d_amY_;
    thrust::device_vector<float> d_bmY_;
    thrust::device_vector<float> d_cmY_;

    float* d_ptr_apX_;
    float* d_ptr_bpX_;
    float* d_ptr_cpX_;

    float* d_ptr_apY_;
    float* d_ptr_bpY_;
    float* d_ptr_cpY_;

    float* d_ptr_amX_;
    float* d_ptr_bmX_;
    float* d_ptr_cmX_;

    float* d_ptr_amY_;
    float* d_ptr_bmY_;
    float* d_ptr_cmY_;

    thrust::device_vector<float> d_pinGradX_;
    thrust::device_vector<float> d_pinGradY_;

    float* d_ptr_pinGradX_;
    float* d_ptr_pinGradY_;

    thrust::device_vector<float> d_netWeight_;
    float* d_ptr_netWeight_;

    double wlGradTime_;

    // Methods
    void initForCUDAKernel();
};

}; // namespace skyplace 

#endif

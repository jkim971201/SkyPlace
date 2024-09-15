#ifndef NESTEROV_OPTIMIZER_H
#define NESTEROV_OPTIMIZER_H

#include <memory>

#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "HyperParam.h"
#include "SkyPlaceDB.h"

#include "Painter.h"
#include "WireLengthGradient.h"
#include "DensityGradient.h"
#include "TargetFunction.h"
#include "skyplace/SkyPlace.h"

namespace skyplace
{

struct Stat;

class NesterovOptimizer
{
  public:

    NesterovOptimizer();
    NesterovOptimizer(std::shared_ptr<HyperParam>     param,
                      std::shared_ptr<SkyPlaceDB>     db,
                      std::shared_ptr<TargetFunction> func,
                      std::shared_ptr<Painter>        painter);

    ~NesterovOptimizer();

    // Main Loop
    Stat startOptimize(bool plotMode);

  private:

    int numCell_;
    int numNet_;
    int numPin_;

    float dieLx_;
    float dieLy_;
    float dieUx_;
    float dieUy_;

    std::shared_ptr<SkyPlaceDB>       db_;
    std::shared_ptr<Painter>        painter_;
    std::shared_ptr<TargetFunction> targetFunction_;

    // Main Nesterov Loop
    void initOptimizer();

    // Placement Metric
    float wa_;
    float prevHpwl_;
    float hpwl_;
    float overflow_;
    float macroOverflow_;
    float sumPenalty_;

    // Self-Adaptive Parameters
    float curA_;
    float stepLength_;

    void updateOneIteration(int iter, int backTrackIter);

    float backTracking(int iter, float coeff, int& backTrackIter);

    // Hyper Parameters
		std::shared_ptr<HyperParam> param_;

    // For WireLength Gradient & HPWL Computation
    // Host Array
    thrust::host_vector<float> h_cellCx_;
    thrust::host_vector<float> h_cellCy_;

    // Device Array
    float* d_ptr_cellWidth_;
    float* d_ptr_cellHeight_;
		thrust::device_vector<float> d_cellWidth_;
		thrust::device_vector<float> d_cellHeight_;

    // Placer Coordinates
    // Previous Predicted Coordinates
    thrust::device_vector<float> d_prevPreX_;
    thrust::device_vector<float> d_prevPreTotalGradX_;

    thrust::device_vector<float> d_prevPreY_;
    thrust::device_vector<float> d_prevPreTotalGradY_;

    // Current Predicted Coordinates
    thrust::device_vector<float> d_curPreX_;
    thrust::device_vector<float> d_curPreTotalGradX_;

    thrust::device_vector<float> d_curPreY_;
    thrust::device_vector<float> d_curPreTotalGradY_;

    // Next Predicted Coordinates
    thrust::device_vector<float> d_nextPreX_;
    thrust::device_vector<float> d_nextPreTotalGradX_;

    thrust::device_vector<float> d_nextPreY_;
    thrust::device_vector<float> d_nextPreTotalGradY_;

    // Main Coordinates
    thrust::device_vector<float> d_curX_;
    thrust::device_vector<float> d_nextX_;

    thrust::device_vector<float> d_curY_;
    thrust::device_vector<float> d_nextY_;

    thrust::device_vector<float> d_workSpaceForStepLength_;

    // Raw Pointers
    float* d_ptr_prevPreX_;
    float* d_ptr_prevPreTotalGradX_;

    float* d_ptr_prevPreY_;
    float* d_ptr_prevPreTotalGradY_;

    float* d_ptr_curPreX_;
    float* d_ptr_curPreTotalGradX_;

    float* d_ptr_curPreY_;
    float* d_ptr_curPreTotalGradY_;

    float* d_ptr_nextPreX_;
    float* d_ptr_nextPreTotalGradX_;

    float* d_ptr_nextPreY_;
    float* d_ptr_nextPreTotalGradY_;

    float* d_ptr_curX_;
    float* d_ptr_nextX_;

    float* d_ptr_curY_;
    float* d_ptr_nextY_;

    // ETC...
    bool onlyGradMode_;
    bool isDiverge_;
    void diverge();
    void opt2db();

    void copyPotential2db();
    void copyBinDensity2db();
    void copyDensityGrad2db();

    // GPU-related Functions
    void moveForward(const float stepLength, 
                     const float stepLengthForPrediction,
                     const float* d_ptr_curCoordiX,
                     const float* d_ptr_curCoordiY,
                     const float* d_ptr_curPreCoordiX,
                     const float* d_ptr_curPreCoordiY,
                     const float* d_ptr_directionX,
                     const float* d_ptr_directionY,
                           float* d_ptr_nextCoordiX,
                           float* d_ptr_nextCoordiY,
                           float* d_ptr_nextPreCoordiX,
                           float* d_ptr_nextPreCoordiY);

    void moveBackward(const float  stepLength, 
                      const float* d_ptr_curCoordiX,
                      const float* d_ptr_curCoordiY,
                      const float* d_ptr_directionX,
                      const float* d_ptr_directionY,
                            float* d_ptr_nextCoordiX,
                            float* d_ptr_nextCoordiY);

    float computeLipschitz(const thrust::device_vector<float>& d_curPreX, 
                           const thrust::device_vector<float>& d_curPreY,
                           const thrust::device_vector<float>& d_curPreTotalGradX,
                           const thrust::device_vector<float>& d_curPreTotalGradY,
                           const thrust::device_vector<float>& d_nextPreX, 
                           const thrust::device_vector<float>& d_nextPreY, 
                           const thrust::device_vector<float>& d_nextPreTotalGradX,
                           const thrust::device_vector<float>& d_nextPreTotalGradY);

    // Runtime
    double nesterovTime_;
    double initTime_;

    void initForCUDAKernel();
    void freeDeviceMemory();
};

}; // namespace skyplace 

#endif

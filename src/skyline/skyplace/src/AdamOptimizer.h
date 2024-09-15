#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include <memory>

#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

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

class AdamOptimizer 
{
  public:

    AdamOptimizer();
    AdamOptimizer(std::shared_ptr<HyperParam>      param,
                  std::shared_ptr<SkyPlaceDB>      db,
                  std::shared_ptr<TargetFunction>  func,
                  std::shared_ptr<Painter>         painter);

    ~AdamOptimizer();

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

    std::shared_ptr<SkyPlaceDB>      db_;
    std::shared_ptr<Painter>         painter_;
    std::shared_ptr<TargetFunction>  targetFunction_;

    void initOptimizer();

    // Placement Metric
    float wa_;
    float hpwl_;
    float prevHpwl_;
    float overflow_;
    float macroOverflow_;
    float sumPenalty_;
  
    // Self-Adaptive Parameters
    float gammaInv_;
    float lambda_;
    float initialGammaInv_;

    void updateGamma(float overflow);
    void updateLambda();
    void updateOneIteration(int iter);
    void updatePlacementMetric();

    void moveForward(const float stepLength,
                     const thrust::device_vector<float>& d_curX,
                     const thrust::device_vector<float>& d_curY,
                     const thrust::device_vector<float>& d_curDirectionX,
                     const thrust::device_vector<float>& d_curDirectionY,
                           thrust::device_vector<float>& d_nextX,
                           thrust::device_vector<float>& d_nextY);

    // GradSum
    float wlGradSum_;
    float densityGradSum_;

    // Hyper Parameters
		std::shared_ptr<HyperParam> param_;

    // For WireLength Gradient & HPWL Computation
    // Host Array
    thrust::host_vector<float> h_cellCx_;
    thrust::host_vector<float> h_cellCy_;

    // Device Array
    thrust::device_vector<float> d_cellWidth_;
    thrust::device_vector<float> d_cellHeight_;

    // Placer Coordinates
    thrust::device_vector<float> d_curX_;
    thrust::device_vector<float> d_curY_;

    thrust::device_vector<float> d_nextX_;
    thrust::device_vector<float> d_nextY_;

    thrust::device_vector<float> d_curDirectionX_;
    thrust::device_vector<float> d_curDirectionY_;

    // Gradient
    thrust::device_vector<float> d_curGradX_;
    thrust::device_vector<float> d_curGradY_;

    // Adam Optimizer
    float alpha_;
    float beta1_;
    float beta2_;

    float epsilon_;

    float beta1k_; // beta1_ ^ (k+1)
    float beta2k_; // beta2_ ^ (k+1)

    // First Moment
    thrust::device_vector<float> d_curMX_;
    thrust::device_vector<float> d_curMY_;

    thrust::device_vector<float> d_nextMX_;
    thrust::device_vector<float> d_nextMY_;

    // Second Moment 
    thrust::device_vector<float> d_curNX_;
    thrust::device_vector<float> d_curNY_;

    thrust::device_vector<float> d_nextNX_;
    thrust::device_vector<float> d_nextNY_;

    // Bias Corrected
    thrust::device_vector<float> d_bcMX_;
    thrust::device_vector<float> d_bcMY_;

    thrust::device_vector<float> d_bcNX_;
    thrust::device_vector<float> d_bcNY_;

    void updateMoment(const thrust::device_vector<float>& d_curMX,
                      const thrust::device_vector<float>& d_curMY,
                      const thrust::device_vector<float>& d_curNX,
                      const thrust::device_vector<float>& d_curNY,
                      const thrust::device_vector<float>& d_curGradX,
                      const thrust::device_vector<float>& d_curGradY,
                            thrust::device_vector<float>& d_nextMX,
                            thrust::device_vector<float>& d_nextMY,
                            thrust::device_vector<float>& d_nextNX,
                            thrust::device_vector<float>& d_nextNY);

    void updateDirection(const thrust::device_vector<float>& d_nextMX,
                         const thrust::device_vector<float>& d_nextMY,
                         const thrust::device_vector<float>& d_nextNX,
                         const thrust::device_vector<float>& d_nextNY,
                               thrust::device_vector<float>& d_bcMX,
                               thrust::device_vector<float>& d_bcMY,
                               thrust::device_vector<float>& d_bcNX,
                               thrust::device_vector<float>& d_bcNY,
                               thrust::device_vector<float>& d_curDirectionX,
                               thrust::device_vector<float>& d_curDirectionY);

    // ETC...
    bool onlyGradMode_;
    bool isDiverge_;
    void diverge();
    void opt2db();

    void copyPotential2db();
    void copyBinDensity2db();
    void copyDensityGrad2db();

    // Runtime
    double adTime_;
    double initTime_;

    void initForCUDAKernel();
    void freeDeviceMemory();
};

}; // namespace skyline

#endif

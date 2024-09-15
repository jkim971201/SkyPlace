#ifndef TARGET_FUNCTION_H
#define TARGET_FUNCTION_H

#include <memory>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "HyperParam.h"
#include "SkyPlaceDB.h"
#include "WireLengthGradient.h"
#include "DensityGradient.h"

namespace skyplace 
{

class TargetFunction
{
  public:

    // Constructor
    TargetFunction() {}
    TargetFunction(std::shared_ptr<SkyPlaceDB>         db,
                   std::shared_ptr<WireLengthGradient> wireLength,
                   std::shared_ptr<DensityGradient>    density,
                   std::shared_ptr<HyperParam>         param);

    // APIs
    void updatePointAndGetGrad(const thrust::device_vector<float>& d_curX,
                               const thrust::device_vector<float>& d_curY,
                                     thrust::device_vector<float>& d_gradX,
                                     thrust::device_vector<float>& d_gradY,
                               bool  precondition = true);

    float updatePointAndGetValue(const thrust::device_vector<float>& d_curX,
                                 const thrust::device_vector<float>& d_curY);

    float getValueAfterGrad();

    void computeGradAfterValue(const thrust::device_vector<float>& d_curX,
                               const thrust::device_vector<float>& d_curY,
                               thrust::device_vector<float>& d_gradX,
                               thrust::device_vector<float>& d_gradY);

    void getInitialGrad(const float* d_ptr_initX,
                        const float* d_ptr_initY,
                              float* d_ptr_initGradX,
                              float* d_ptr_initGradY);

    void multiplyPreconditioner(thrust::device_vector<float>& inputVectorX,
                                thrust::device_vector<float>& inputVectorY);

    void updateParameters(bool onlyGrad = true);

    // Setters
    void setMacroDensityWeight (bool mode);

    // Getters
    float getWA             () const { return wa_;             }
    float getHPWL           () const { return hpwl_;           }
    float getPenalty        () const { return sumPenalty_;     }
    float getOverflow       () const { return overflow_;       }
    float getMacroOverflow  () const { return macroOverflow_;  }
    float getLambda         () const { return lambda_;         }

    std::shared_ptr<DensityGradient>    density   () const { return density_;    }
    std::shared_ptr<WireLengthGradient> wirelength() const { return wireLength_; }

    // To Visualize Density Gradient
    // These will be passed to Painter
    thrust::device_vector<float>& densityGradX() { return d_densityGradX_; }
    thrust::device_vector<float>& densityGradY() { return d_densityGradY_; }

    double getDensityTime  ()  const { return density_->densityTime();   }
    double getPoissonTime  ()  const { return density_->poissonTime();   }
    double getBinDenUpTime ()  const { return density_->binDenUpTime();  }
    double getWLGradTime   ()  const { return wireLength_->wlGradTime(); }

  private:

    int   N_;
    float gammaInv_;
    float initGammaInv_;
    float lambda_;
    float minPrecond_;
    float refHPWL_;
    float minPhi_;
    float maxPhi_;

    float wa_;
    float hpwl_;
    float prevHpwl_;
    float overflow_;
    float macroOverflow_;
    float sumPenalty_;

    bool macroDensityWeight_;
    float coeffSoFar_;

    void updateGammaInv  (float overflow);
    void updateLambda    (float prevHpwl, float curHPWL);

    thrust::device_vector<float> d_wlGradX_;
    thrust::device_vector<float> d_wlGradY_;
    float* d_ptr_wlGradX_;
    float* d_ptr_wlGradY_;

    thrust::device_vector<float> d_densityGradX_;
    thrust::device_vector<float> d_densityGradY_;
    float* d_ptr_densityGradX_;
    float* d_ptr_densityGradY_;

    thrust::device_vector<float> d_numPin_;
    float* d_ptr_numPin_;

    thrust::device_vector<float> d_cellArea_;
    float* d_ptr_cellArea_;

    // This is also in the DensityGradient
    // Waste of Memory => TODO: remove this
    thrust::device_vector<bool> d_isMacro_;
    bool* d_ptr_isMacro_;

    thrust::device_vector<float> d_macroDecay_;
    float* d_ptr_macroDecay_;

    float wlGradSum_;
    float densityGradSum_;

    std::shared_ptr<SkyPlaceDB>         db_;
    std::shared_ptr<WireLengthGradient> wireLength_;
    std::shared_ptr<DensityGradient>    density_;

    void diverge();
};

}; // namespace skyplace

#endif

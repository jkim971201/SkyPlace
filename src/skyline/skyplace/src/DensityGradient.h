#ifndef DENSITY_GRAD_H
#define DENSITY_GRAD_H

#include <memory>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "SkyPlaceDB.h"
#include "PoissonSolver.h"

namespace skyplace
{

class DensityGradient
{
  public:
    
    DensityGradient();
    DensityGradient(std::shared_ptr<SkyPlaceDB> db);

    ~DensityGradient();

    // API
    void computeGrad(float* densityGradX,
                     float* densityGradY,
                     const float* cellCx,
                     const float* cellCy);

    void computeGradAfterValue(float* densityGradX,
                               float* densityGradY,
                               const float* cellCx,
                               const float* cellCy);

    float computePenaltyAfterGrad();
    float computePenalty(const float* d_ptr_cellCx, 
                         const float* d_ptr_cellCy);

    void updateStdDensityWeight    (float coeffStd    );
    void updateMacroDensityWeight  (float coeffMacro  );
    void updateFillerDensityWeight (float coeffFiller );

		void resetMacroDensityWeight();

    // Setters
    void setLocalLambdaMode(bool mode) { localLambdaMode_ = mode; }
		void setSigma          (float val) { sigma_           = val;  }

    // Getters
    int    numBinX        () const { return numBinX_;       }
    int    numBinY        () const { return numBinY_;       }
    float  overflow       () const { return overflow_;      }
    float  macroOverflow  () const { return macroOverflow_; }
    double densityTime    () const { return densityTime_;   }
    double poissonTime    () const { return poissonTime_;   }
    double binDenUpTime   () const { return binDenUpTime_;  }
		float  getSigma       () const { return sigma_;         }

    // To Visualize Density Information
    // These will be passed to Painter
    const float* getDevicePotential     () const { return d_ptr_binPotential_;          }
    const float* getDeviceBinDensity    () const { return d_ptr_binDensity_;            }
    const float* getDevicePreconditioner() const { return d_ptr_densityPreconditioner_; }

  private:

    int numBinX_;
    int numBinY_;
    int numMovable_;
    float sumMovableArea_;

    float overflow_;
    float macroOverflow_;
    float sumPenalty_;

    float dieLx_;
    float dieLy_;
    float dieUx_;
    float dieUy_;

    float targetDensity_;

    float binWidth_;
    float binHeight_;
		float sigma_;

    bool localLambdaMode_;

    std::shared_ptr<SkyPlaceDB>      db_;
    std::unique_ptr<PoissonSolver> poissonSolver_;

    // Device Data
    thrust::device_vector<float> d_cellDensityWidth_; 
    thrust::device_vector<float> d_cellDensityHeight_; 
    thrust::device_vector<float> d_cellDensityScale_;

    float* d_ptr_cellDensityWidth_;
    float* d_ptr_cellDensityHeight_;
    float* d_ptr_cellDensityScale_;

    thrust::device_vector<bool> d_isStd_;
    bool* d_ptr_isStd_;

    thrust::device_vector<bool> d_isFiller_;
    bool* d_ptr_isFiller_;

    thrust::device_vector<bool> d_isMacro_;
    bool* d_ptr_isMacro_;

    thrust::device_vector<float> d_fixedArea_;
    float* d_ptr_fixedArea_;

    thrust::device_vector<float> d_macroArea_;
    float* d_ptr_macroArea_;

    thrust::device_vector<float> d_scaledBinArea_;
    float* d_ptr_scaledBinArea_;

    thrust::device_vector<float> d_binLx_;
    float* d_ptr_binLx_;

    thrust::device_vector<float> d_binLy_;
    float* d_ptr_binLy_;

    thrust::device_vector<float> d_binUx_;
    float* d_ptr_binUx_;

    thrust::device_vector<float> d_binUy_;
    float* d_ptr_binUy_;

    thrust::device_vector<float> d_movableArea_;
    float* d_ptr_movableArea_;

    thrust::device_vector<float> d_fillerArea_;
    float* d_ptr_fillerArea_;

    thrust::device_vector<float> d_overflowArea_;
    float* d_ptr_overflowArea_;

    thrust::device_vector<float> d_macroOverflowArea_;
    float* d_ptr_macroOverflowArea_;

    thrust::device_vector<float> d_binDensity_;
    float* d_ptr_binDensity_;

    thrust::device_vector<float> d_binPenalty_;
    float* d_ptr_binPenalty_;

    thrust::device_vector<float> d_binLambda_;
    float* d_ptr_binLambda_;

    thrust::device_vector<float> d_densityPreconditioner_;
    float* d_ptr_densityPreconditioner_;

    thrust::device_vector<float> d_binPotential_;
    float* d_ptr_binPotential_;

    thrust::device_vector<float> d_electroForceX_;
    float* d_ptr_electroForceX_;

    thrust::device_vector<float> d_electroForceY_;
    float* d_ptr_electroForceY_;

    // Runtime
    double densityTime_;
    double poissonTime_;
    double binDenUpTime_;

    void initForCUDAKernel();

    void freeDeviceMemory();
};

} // namespace skyplace

#endif

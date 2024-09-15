#include "CUDA_UTIL.h"
#include "TargetFunction.h"

namespace skyplace
{

__global__ void noPrecondition(const int    numCell,
                               const float  lambda,
                               const float* wlGradX,
                               const float* wlGradY,
                               const float* densityGradX,
                               const float* densityGradY,
                               float* totalGradX,
                               float* totalGradY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    totalGradX[i] = (wlGradX[i] + lambda * densityGradX[i]);
    totalGradY[i] = (wlGradY[i] + lambda * densityGradY[i]);
  }
}

__global__ void jacobianPrecondition(const int    numCell,
                                     const float  lambda,
                                     const float  minPrecond,
                                     const float* numPinForEachCell,
                                     const float* densityPreconditioner,
																		 const bool*  isMacro,
																		 const float* macroDecay,
                                     const float* wlGradX,
                                     const float* wlGradY,
                                     const float* densityGradX,
                                     const float* densityGradY,
                                           float* totalGradX,
                                           float* totalGradY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    int   wirelengthPrecond = numPinForEachCell[i];
    float densityPrecond    = densityPreconditioner[i];

    float precond  = wirelengthPrecond + lambda * densityPrecond;

    if(precond < minPrecond)
      precond = minPrecond;

		if(isMacro[i])
		{
			float decay = macroDecay[i];
			// printf("macroDecay : %f\n", macroDecay[i]);
			totalGradX[i] = (wlGradX[i] + lambda * densityGradX[i]) / precond * decay;
			totalGradY[i] = (wlGradY[i] + lambda * densityGradY[i]) / precond * decay;
		}
		else
		{
			totalGradX[i] = (wlGradX[i] + lambda * densityGradX[i]) / precond;
			totalGradY[i] = (wlGradY[i] + lambda * densityGradY[i]) / precond;
		}
  }
}

__global__ void multiplyJacobianPrecondition(const int    numCell,
                                             const float  lambda,
                                             const float  minPrecond,
                                             const float* numPinForEachCell,
                                             const float* densityPreconditioner,
																						 const bool*  isMacro,
																						 const float* macroDecay,
                                                   float* inputVectorX,
                                                   float* inputVectorY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    int   wirelengthPrecond = numPinForEachCell[i];
    float densityPrecond    = densityPreconditioner[i];

    float precond  = wirelengthPrecond + lambda * densityPrecond;

    if(precond < minPrecond)
      precond = minPrecond;

		if(isMacro[i])
		{
			inputVectorX[i] /= (precond / macroDecay[i]);
			inputVectorY[i] /= (precond / macroDecay[i]);
		}
		else
		{
			inputVectorX[i] /= precond;
			inputVectorY[i] /= precond;
		}
  }
}

void
TargetFunction::multiplyPreconditioner(thrust::device_vector<float>& inputVectorX,
                                       thrust::device_vector<float>& inputVectorY)
{
  int numThread = 64;
  int numBlockCell = (N_ - 1 + numThread) / numThread;

  multiplyJacobianPrecondition<<<numBlockCell, numThread>>>(N_, 
                                                            lambda_,
                                                            minPrecond_,
                                                            d_ptr_numPin_,
                                                            density_->getDevicePreconditioner(),
																										        d_ptr_isMacro_,
																										        d_ptr_macroDecay_,
                                                            getRawPointer(inputVectorX),
                                                            getRawPointer(inputVectorY));
}

TargetFunction::TargetFunction(std::shared_ptr<SkyPlaceDB> db, 
                               std::shared_ptr<WireLengthGradient> wireLength,
                               std::shared_ptr<DensityGradient> density,
                               std::shared_ptr<HyperParam>      param)
{
  db_ = db;
  
  wireLength_ = wireLength;
  density_ = density;

  N_ = db_->numMovable();

  gammaInv_ = 2.0 * param->initGammaInvCoef 
            / (db_->binX() + db_->binY());

  wireLength_->setGammaInv(gammaInv_);

  lambda_       = param->initLambda;
  initGammaInv_ = param->initGammaInvCoef;
  minPrecond_   = param->minPrecond;
  refHPWL_      = param->referenceHpwl;
  minPhi_       = param->minPhiCoef;
  maxPhi_       = param->maxPhiCoef;

	macroDensityWeight_ = false;
	coeffSoFar_ = 1.0;

  wlGradSum_      = 0.0;
  densityGradSum_ = 0.0;

  d_ptr_wlGradX_      = setThrustVector(N_, d_wlGradX_);
  d_ptr_wlGradY_      = setThrustVector(N_, d_wlGradY_);
  d_ptr_densityGradX_ = setThrustVector(N_, d_densityGradX_);
  d_ptr_densityGradY_ = setThrustVector(N_, d_densityGradY_);

  d_ptr_numPin_     = setThrustVector(N_, d_numPin_);
  d_ptr_cellArea_   = setThrustVector(N_, d_cellArea_);

	d_ptr_isMacro_    = setThrustVector(N_, d_isMacro_);
	d_ptr_macroDecay_ = setThrustVector(N_, d_macroDecay_);

  std::vector<float> h_numPin;
	h_numPin.reserve(db_->numPin());

  std::vector<float> h_cellArea;
	h_cellArea.reserve(N_);

	std::vector<bool> h_isMacro;
	h_isMacro.reserve(N_);

	std::vector<float> h_macroDecay;
	h_macroDecay.reserve(N_);

  for(auto& cell : db_->movableCells())
  {
    h_numPin.push_back(static_cast<float>(cell->pins().size()));
    h_cellArea.push_back(cell->dx() * cell->dy());
		
		if(cell->isMacro())
		{
			h_isMacro.push_back(true);
			h_macroDecay.push_back(1.0);
		}
		else
		{
			h_isMacro.push_back(false);
			h_macroDecay.push_back(1.0);
		}
  }

  thrust::copy(h_numPin.begin(), h_numPin.end(),
               d_numPin_.begin());

  thrust::copy(h_cellArea.begin(), h_cellArea.end(),
               d_cellArea_.begin());

	thrust::copy(h_isMacro.begin(), h_isMacro.end(),
			         d_isMacro_.begin() );

	thrust::copy(h_macroDecay.begin(), h_macroDecay.end(),
			         d_macroDecay_.begin() );
}

float
TargetFunction::getValueAfterGrad()
{
  // For WireLength Grad
  wa_ = wireLength_->getWAafterGrad();

  // For Density Grad
  sumPenalty_ = density_->computePenaltyAfterGrad();

  float val = wa_ + lambda_ * sumPenalty_;
  return val;
}

void
TargetFunction::diverge()
{
  printf("Divergence Detected!\n");
  printf("Terminate Placer...\n");
  exit(0);
}

void
TargetFunction::computeGradAfterValue(const thrust::device_vector<float>& d_curX,
                                      const thrust::device_vector<float>& d_curY,
                                      thrust::device_vector<float>& d_gradX,
                                      thrust::device_vector<float>& d_gradY)
{
  // For WireLength Grad
  wireLength_->computeGradAfterValue(d_ptr_wlGradX_, d_ptr_wlGradY_);

  // For Density Grad
  density_->computeGrad(d_ptr_densityGradX_,
                        d_ptr_densityGradY_,
                        getRawPointer(d_curX),
                        getRawPointer(d_curY) );

  wlGradSum_  = compute1Norm(d_wlGradX_);
  wlGradSum_ += compute1Norm(d_wlGradY_);

  densityGradSum_  = compute1Norm(d_densityGradX_);
  densityGradSum_ += compute1Norm(d_densityGradY_);

  if(std::isnan(wlGradSum_) || std::isnan(densityGradSum_) ||
     std::isinf(wlGradSum_) || std::isinf(densityGradSum_))
  {
    printf("DensityGradSum   : %E\n", densityGradSum_);
    printf("WireLengthGradSum: %E\n", wlGradSum_);
    diverge();
  }

  int numThread = 64;
  int numBlockCell = (N_ - 1 + numThread) / numThread;

  jacobianPrecondition<<<numBlockCell, numThread>>>(N_, 
                                                    lambda_,
                                                    minPrecond_,
                                                    d_ptr_numPin_,
                                                    density_->getDevicePreconditioner(),
																										d_ptr_isMacro_,
																										d_ptr_macroDecay_,
                                                    d_ptr_wlGradX_,
                                                    d_ptr_wlGradY_,
                                                    d_ptr_densityGradX_,
                                                    d_ptr_densityGradY_,
                                                    getRawPointer(d_gradX),
                                                    getRawPointer(d_gradY));
}

float
TargetFunction::updatePointAndGetValue(const thrust::device_vector<float>& d_curX,
                                       const thrust::device_vector<float>& d_curY)
{
  // For WireLength Grad
  wireLength_->updatePinCoordinates(getRawPointer(d_curX), 
                                    getRawPointer(d_curY));

  wa_ = wireLength_->getWA();

  // For Density Grad
  sumPenalty_ = density_->computePenalty(getRawPointer(d_curX),
                                         getRawPointer(d_curY));

  float val = wa_ + lambda_ * sumPenalty_;
  //printf("functionValue: %E\n", val); 
  return val;
}

void
TargetFunction::updatePointAndGetGrad(const thrust::device_vector<float>& d_curX,
                                      const thrust::device_vector<float>& d_curY,
                                      thrust::device_vector<float>& d_gradX,
                                      thrust::device_vector<float>& d_gradY,
                                      bool precondition)
{
  // For WireLength Grad
  wireLength_->updatePinCoordinates(getRawPointer(d_curX), 
                                    getRawPointer(d_curY));

  wireLength_->computeGrad(d_ptr_wlGradX_, d_ptr_wlGradY_);

  // For Density Grad
  density_->computeGrad(d_ptr_densityGradX_, 
                        d_ptr_densityGradY_, 
                        getRawPointer(d_curX),
                        getRawPointer(d_curY));

  wlGradSum_  = compute1Norm(d_wlGradX_);
  wlGradSum_ += compute1Norm(d_wlGradY_);

  densityGradSum_  = compute1Norm(d_densityGradX_);
  densityGradSum_ += compute1Norm(d_densityGradY_);

  if(std::isnan(wlGradSum_) || std::isnan(densityGradSum_) ||
     std::isinf(wlGradSum_) || std::isinf(densityGradSum_))
  {
    printf("DensityGradSum   : %E\n", densityGradSum_);
    printf("WireLengthGradSum: %E\n", wlGradSum_);
    diverge();
  }

  int numThread = 64;
  int numBlockCell = (N_ - 1 + numThread) / numThread;

  if(precondition)
  {
    jacobianPrecondition<<<numBlockCell, numThread>>>(N_, 
                                                      lambda_,
                                                      minPrecond_,
                                                      d_ptr_numPin_,
                                                      density_->getDevicePreconditioner(),
																											d_ptr_isMacro_,
																											d_ptr_macroDecay_,
                                                      d_ptr_wlGradX_,
                                                      d_ptr_wlGradY_,
                                                      d_ptr_densityGradX_,
                                                      d_ptr_densityGradY_,
                                                      getRawPointer(d_gradX),
                                                      getRawPointer(d_gradY));
  }
  else
  {
    noPrecondition<<<numBlockCell, numThread>>>(N_, 
                                                lambda_,
                                                d_ptr_wlGradX_,
                                                d_ptr_wlGradY_,
                                                d_ptr_densityGradX_,
                                                d_ptr_densityGradY_,
                                                getRawPointer(d_gradX),
                                                getRawPointer(d_gradY));
  }
}

void
TargetFunction::getInitialGrad(const float* d_ptr_initX,
                               const float* d_ptr_initY,
                                     float* d_ptr_initGradX,
                                     float* d_ptr_initGradY)
{
  // We have to first compute Density Gradient
  // because we do not yet know overflow that is required to update gamma
  density_->computeGrad(d_ptr_densityGradX_, 
                        d_ptr_densityGradY_,
                        d_ptr_initX,
                        d_ptr_initY);

  overflow_ = density_->overflow();
  updateGammaInv(overflow_);

  wireLength_->updatePinCoordinates(d_ptr_initX, d_ptr_initY);

  wireLength_->computeGrad(d_ptr_wlGradX_,
                           d_ptr_wlGradY_);

  int numThread    = 64;
  int numBlockCell = (N_ - 1 + numThread) / numThread;

  jacobianPrecondition<<<numBlockCell, numThread>>>(N_, 
                                                    0.0,
                                                    minPrecond_,
                                                    d_ptr_numPin_,
                                                    density_->getDevicePreconditioner(),
																										d_ptr_isMacro_,
																										d_ptr_macroDecay_,
                                                    d_ptr_wlGradX_,
                                                    d_ptr_wlGradY_,
                                                    d_ptr_densityGradX_,
                                                    d_ptr_densityGradY_,
                                                    d_ptr_initGradX,
                                                    d_ptr_initGradY);

  wlGradSum_  = compute1Norm(d_wlGradX_);
  wlGradSum_ += compute1Norm(d_wlGradY_);

  densityGradSum_  = compute1Norm(d_densityGradX_);
  densityGradSum_ += compute1Norm(d_densityGradY_);

  if(std::isnan(wlGradSum_) || std::isnan(densityGradSum_) ||
     std::isinf(wlGradSum_) || std::isinf(densityGradSum_))
  {
    printf("DensityGradSum   : %f\n", densityGradSum_);
    printf("WireLengthGradSum: %f\n", wlGradSum_);
    exit(0);
  }

  lambda_ *= wlGradSum_ / densityGradSum_;
  // Set the initial value of lambda
  // same as RePlAce Paper TCAD 2019
}

void
TargetFunction::updateGammaInv(float overflow)
{
  if(overflow > 1.0)
    gammaInv_ = 0.1;
  else if(overflow < 0.1)
    gammaInv_ = 10.0;
  else
    gammaInv_ = 1.0 / std::pow(10.0, 20 * (overflow - 0.1) / 9.0 - 1.0);

  gammaInv_ *= initGammaInv_;
  // 10, 20, 9 are also hyper-parameters...

  wireLength_->setGammaInv(gammaInv_);
}

void
TargetFunction::updateLambda(float prevHpwl, float curHpwl)
{
  float p = (curHpwl - prevHpwl) / refHPWL_;
  float coef = 1;

  if(p < 0) 
    coef = maxPhi_;
  else
    coef = std::max(minPhi_, maxPhi_* std::pow(maxPhi_, -p));
  lambda_ *= coef;
}

void
TargetFunction::updateParameters(bool onlyGrad)
{
  prevHpwl_      = hpwl_;
  hpwl_          = wireLength_->computeHPWL();
  overflow_      = density_->overflow();
  macroOverflow_ = density_->macroOverflow();

  if(!onlyGrad)
  {
    wa_          = wireLength_->getWAafterGrad();
    sumPenalty_  = density_->computePenaltyAfterGrad();
  }

//	if(overflow_ < 0.20)
//		density_->updateFillerDensityWeight(0.999);
//	if(overflow_ < 0.30)
//		density_->updateStdDensityWeight(1.001);

	if(macroDensityWeight_)
	{
		//if(overflow_ < 0.05)
		//	vectorScalarMul(0.9, d_macroDecay_);
		// float sigma = density_->getSigma();
		// printf("Sigma : %f\n", sigma);
		//density_->setSigma(sigma * 1.005);
		if(overflow_ < 0.10)
		{
			coeffSoFar_ *= 0.9999;
			printf("coeffSoFar_ = %f\n", coeffSoFar_);
			density_->updateMacroDensityWeight(0.9999);
		}
	}

  updateGammaInv(overflow_);
  updateLambda(prevHpwl_, hpwl_);
}

void 
TargetFunction::setMacroDensityWeight(bool mode) 
{ 
	macroDensityWeight_ = true; 
	density_->resetMacroDensityWeight();
}

}; // namespace skyplace

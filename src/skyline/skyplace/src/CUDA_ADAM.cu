#include <cstdio>
#include <memory>
#include <chrono>
#include <cmath>

#include "CUDA_UTIL.h"
#include "AdamOptimizer.h"

namespace skyplace
{

__global__ void moveForwardKernelAdam(const int    numCell,
                                      const float  dieLx,
                                      const float  dieLy,
                                      const float  dieUx,
                                      const float  dieUy,
                                      const float  stepLength,
                                      const float* cellDx,
                                      const float* cellDy,
                                      const float* curX,
                                      const float* curY,
                                      const float* curDirectionX,
                                      const float* curDirectionY,
                                            float* nextX,
                                            float* nextY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    float nextXtemp = 
      curX[i] - stepLength * curDirectionX[i];
  
    float nextYtemp = 
      curY[i] - stepLength * curDirectionY[i];

    nextX[i] 
      = getXCoordiInsideLayoutDevice(nextXtemp, cellDx[i], dieLx, dieUx);

    nextY[i] 
      = getYCoordiInsideLayoutDevice(nextYtemp, cellDy[i], dieLy, dieUy);
  }
}

__global__ void updateDirectionKernelAdam(const int    numCell,
                                          const float  epsilon,
                                          const float* d_ptr_bcMX,
                                          const float* d_ptr_bcMY,
                                          const float* d_ptr_bcNX,
                                          const float* d_ptr_bcNY,
                                                float* d_ptr_curDirectionX,
                                                float* d_ptr_curDirectionY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    float bcnx = d_ptr_bcNX[i];
    float bcny = d_ptr_bcNY[i];

    if(bcnx < 0.0)
      bcnx = 0.0;

    if(bcny < 0.0)
      bcny = 0.0;

    d_ptr_curDirectionX[i] = d_ptr_bcMX[i] 
                           / ( sqrtf(bcnx) + epsilon );

    d_ptr_curDirectionY[i] = d_ptr_bcMY[i] 
                           / ( sqrtf(bcny) + epsilon );
  }
}

AdamOptimizer::AdamOptimizer()
  : numCell_                    (0),
    numNet_                     (0),
    numPin_                     (0),
    
    dieLx_                      (0.0),
    dieLy_                      (0.0),
    dieUx_                      (0.0),
    dieUy_                      (0.0),

    wa_                         (0.0),
    hpwl_                       (0.0),
    prevHpwl_                   (0.0),
    overflow_                   (0.0),
    sumPenalty_                 (0.0),

    param_                      (nullptr),
    alpha_                      (100.0),
    beta1_                      (0.9  ),
    beta2_                      (0.999),

    onlyGradMode_               (true   ),
    isDiverge_                  (false  ),
    db_                         (nullptr),
    painter_                    (nullptr),
    targetFunction_             (nullptr),

    initTime_                   (0.0),
    adTime_                     (0.0)
{}

AdamOptimizer::AdamOptimizer(std::shared_ptr<HyperParam>     param,
                             std::shared_ptr<SkyPlaceDB>     db,
                             std::shared_ptr<TargetFunction> func,
                             std::shared_ptr<Painter>        painter) 
  : AdamOptimizer()
{
  param_          = param;
  db_             = db;
  painter_        = painter;
  targetFunction_ = func;

  dieLx_ = db_->die()->lx();
  dieLy_ = db_->die()->ly();
  dieUx_ = db_->die()->ux();
  dieUy_ = db_->die()->uy();

  alpha_  = param->adam_alpha;
  beta1_  = param->adam_beta1;
  beta2_  = param->adam_beta2;
  beta1k_ = beta1_;
  beta2k_ = beta2_;
}

AdamOptimizer::~AdamOptimizer()
{
  freeDeviceMemory();
}

void
AdamOptimizer::initOptimizer()
{
  printf("[Adam] Optimizer Initialization\n");

  auto t1 = std::chrono::high_resolution_clock::now();

  param_->printHyperParameters();

  initForCUDAKernel();

  targetFunction_->getInitialGrad(getRawPointer(d_curX_),
                                  getRawPointer(d_curY_),
                                  getRawPointer(d_curGradX_),
                                  getRawPointer(d_curGradY_) );

  targetFunction_->updateParameters(onlyGradMode_);
  prevHpwl_ = hpwl_;
  hpwl_     = targetFunction_->getHPWL();
  overflow_ = targetFunction_->getOverflow();

  if(!onlyGradMode_)
    sumPenalty_ = targetFunction_->getPenalty();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  initTime_ += runtime.count();

  printf("[Adam] Optimizer Initialization Finished.\n");
}

void
AdamOptimizer::moveForward(const float stepLength,
                           const thrust::device_vector<float>&  d_curX,
                           const thrust::device_vector<float>&  d_curY,
                           const thrust::device_vector<float>&  d_curDirectionX,
                           const thrust::device_vector<float>&  d_curDirectionY,
                                 thrust::device_vector<float>&  d_nextX,
                                 thrust::device_vector<float>&  d_nextY)
{
  int numThread = 64;
  int numBlockCell = (numCell_ - 1 + numThread) / numThread;

  moveForwardKernelAdam<<<numBlockCell, numThread>>>(numCell_, 
                                                     dieLx_, 
                                                     dieLy_, 
                                                     dieUx_, 
                                                     dieUy_,
                                                     stepLength,
                                                     getRawPointer(d_cellWidth_), 
                                                     getRawPointer(d_cellHeight_),
                                                     getRawPointer(d_curX), 
                                                     getRawPointer(d_curY), 
                                                     getRawPointer(d_curDirectionX),
                                                     getRawPointer(d_curDirectionY),
                                                     getRawPointer(d_nextX), 
                                                     getRawPointer(d_nextY));
}

void
AdamOptimizer::updateMoment(const thrust::device_vector<float>& d_curMX,
                            const thrust::device_vector<float>& d_curMY,
                            const thrust::device_vector<float>& d_curNX,
                            const thrust::device_vector<float>& d_curNY,
                            const thrust::device_vector<float>& d_curGradX,
                            const thrust::device_vector<float>& d_curGradY,
                                  thrust::device_vector<float>& d_nextMX,
                                  thrust::device_vector<float>& d_nextMY,
                                  thrust::device_vector<float>& d_nextNX,
                                  thrust::device_vector<float>& d_nextNY)
{
  vectorAddAxBy(beta1_, -(1.0 - beta1_), 
                d_curMX, d_curGradX, d_nextMX);

  vectorAddAxBy(beta1_, -(1.0 - beta1_), 
                d_curMY, d_curGradY, d_nextMY);

  vectorAddAxBy2(beta2_, (1.0 - beta2_), 
                 d_curNX, d_curGradX, d_nextNX);

  vectorAddAxBy2(beta2_, (1.0 - beta2_), 
                 d_curNY, d_curGradY, d_nextNY);
}

void
AdamOptimizer::updateDirection(const thrust::device_vector<float>& d_nextMX,
                               const thrust::device_vector<float>& d_nextMY,
                               const thrust::device_vector<float>& d_nextNX,
                               const thrust::device_vector<float>& d_nextNY,
                                     thrust::device_vector<float>& d_bcMX,
                                     thrust::device_vector<float>& d_bcMY,
                                     thrust::device_vector<float>& d_bcNX,
                                     thrust::device_vector<float>& d_bcNY,
                                     thrust::device_vector<float>& d_curDirectionX,
                                     thrust::device_vector<float>& d_curDirectionY)
{
  float coeff1 = 1.0 / (1.0 - beta1k_);
  float coeff2 = 1.0 / (1.0 - beta2k_);

  vectorScalarMul(coeff1, d_nextMX, d_bcMX);
  vectorScalarMul(coeff1, d_nextMY, d_bcMY);

  vectorScalarMul(coeff2, d_nextNX, d_bcNX);
  vectorScalarMul(coeff2, d_nextNY, d_bcNY);

  int numThread = 64;
  int numBlockCell = (numCell_ - 1 + numThread) / numThread;

  updateDirectionKernelAdam<<<numBlockCell, numThread>>>(numCell_, 
                                                         epsilon_,
                                                         getRawPointer(d_bcMX),
                                                         getRawPointer(d_bcMY),
                                                         getRawPointer(d_bcNX),
                                                         getRawPointer(d_bcNY),
                                                         getRawPointer(d_curDirectionX),
                                                         getRawPointer(d_curDirectionY));
}

Stat
AdamOptimizer::startOptimize(bool plotMode)
{
  auto t1 = std::chrono::high_resolution_clock::now();

  initOptimizer();

  printf("[Adam] [StartGP] HPWL : %-6.3f  Overflow : %-3.3f\n",
                                  hpwl_ / 1e6, overflow_);

  int iter = 0;

  if(plotMode)
    painter_->prepareForPlot();

  for(; iter < param_->maxOptIter; iter++)
  {
    // Step #1: Compute Next Gradient
    targetFunction_->updatePointAndGetGrad(d_curX_,
                                           d_curY_,
                                           d_curGradX_,
                                           d_curGradY_,
                                           false);

    // Step #2: Update Moment
    updateMoment(d_curMX_,
                 d_curMY_,
                 d_curNX_,
                 d_curNY_,
                 d_curGradX_,
                 d_curGradY_,
                 d_nextMX_,
                 d_nextMY_,
                 d_nextNX_,
                 d_nextNY_);

    // Step #3: Update Direction
    updateDirection(d_nextMX_,
                    d_nextMY_,
                    d_nextNX_,
                    d_nextNY_,
                    d_bcMX_,
                    d_bcMY_,
                    d_bcNX_,
                    d_bcNY_,
                    d_curDirectionX_,
                    d_curDirectionY_);
    
    // Step #4: Move forward with the direction vector
    moveForward(alpha_,
                d_curX_, 
                d_curY_,
                d_curDirectionX_,
                d_curDirectionY_,
                d_nextX_,
                d_nextY_);

    updateOneIteration(iter);

    if(plotMode && (iter % 2 == 0))
    {
      opt2db();
      copyPotential2db();
      painter_->saveImage(iter, hpwl_, overflow_, false);
    }

    if(overflow_ <= param_->targetOverflow)
    {
      if(hpwl_ < prevHpwl_)
        continue;
      isDiverge_ = false;
      printf("[Adam] Convergence!\n");
      break;
    }
  }

  opt2db();

  // For Visualization
  copyPotential2db();
  copyDensityGrad2db();
  copyBinDensity2db();

  if(plotMode)
    painter_->saveImage(iter, hpwl_, overflow_, false); // no Filler

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  adTime_ = runtime.count();

  Stat finalStat 
    = {!isDiverge_, 
        hpwl_, 
        overflow_, 
        iter, 
        adTime_, 
        targetFunction_->getWLGradTime(),
        targetFunction_->getDensityTime(),
        targetFunction_->getPoissonTime(),
        targetFunction_->getBinDenUpTime(),
        initTime_};

  return finalStat;  
}

void
AdamOptimizer::initForCUDAKernel()
{
  printf("[CUDA-Adam] Start Initialization.\n");

  numCell_ = db_->movableCells().size();
  numNet_  = db_->nets().size();
  numPin_  = db_->pins().size();

  h_cellCx_.resize(numCell_);
  h_cellCy_.resize(numCell_);

  std::vector<float> h_cellWidth_(numCell_);
  std::vector<float> h_cellHeight_(numCell_);
  std::vector<float> h_cellArea_(numCell_);

  std::vector<float> h_pinX_(numPin_);
  std::vector<float> h_pinY_(numPin_);

  // Not sure cellID is in the same order with
  // cellVector
  for(auto& cell : db_->movableCells())
  {
    int cellID = cell->id();

    h_cellCx_[cellID] = cell->cx();
    h_cellCy_[cellID] = cell->cy();

    h_cellWidth_[cellID]  = cell->dx();
    h_cellHeight_[cellID] = cell->dy();
    h_cellArea_[cellID]   = cell->area();
  }

  // Step #1. Resize Thrust Vector
  d_cellWidth_.resize(numCell_);
  d_cellHeight_.resize(numCell_);

  d_curX_.resize(numCell_);
  d_curY_.resize(numCell_);

  d_curGradX_.resize(numCell_);
  d_curGradY_.resize(numCell_);

  d_curDirectionX_.resize(numCell_);
  d_curDirectionY_.resize(numCell_);

  d_nextX_.resize(numCell_);
  d_nextY_.resize(numCell_);

  // Step #2. Synchronize Host<->Device Data 
  thrust::copy(h_cellWidth_.begin() , h_cellWidth_.end() , d_cellWidth_.begin()  );
  thrust::copy(h_cellHeight_.begin(), h_cellHeight_.end(), d_cellHeight_.begin() );

  thrust::copy(h_cellCx_.begin(), h_cellCx_.end(), d_curX_.begin());
  thrust::copy(h_cellCy_.begin(), h_cellCy_.end(), d_curY_.begin());

  // Step #3. Vectors for Adam
  d_curMX_.resize(numCell_);
  d_curMY_.resize(numCell_);

  d_nextMX_.resize(numCell_);
  d_nextMY_.resize(numCell_);

  d_curNX_.resize(numCell_);
  d_curNY_.resize(numCell_);

  d_nextNX_.resize(numCell_);
  d_nextNY_.resize(numCell_);

  d_bcMX_.resize(numCell_);
  d_bcMY_.resize(numCell_);

  d_bcNX_.resize(numCell_);
  d_bcNY_.resize(numCell_);

  thrust::fill(d_curMX_.begin(), d_curMX_.end(), 0.0);
  thrust::fill(d_curMY_.begin(), d_curMY_.end(), 0.0);

  thrust::fill(d_curNX_.begin(), d_curNX_.end(), 0.0);
  thrust::fill(d_curNY_.begin(), d_curNY_.end(), 0.0);

  beta1k_ = beta1_;
  beta2k_ = beta2_;

  epsilon_ = 1e-8;

  printf("[CUDA-Adam] Alpha: %E \n", alpha_);
  printf("[CUDA-Adam] Beta1: %E \n", beta1_);
  printf("[CUDA-Adam] Beta2: %E \n", beta2_);
  printf("[CUDA-Adam] Finish Initialization.\n");
}

void
AdamOptimizer::updateOneIteration(int iter)
{
  // Current <= Next
  d_curX_.swap(d_nextX_);
  d_curY_.swap(d_nextY_);

  d_curMX_.swap(d_nextMX_);
  d_curMY_.swap(d_nextMY_);

  d_curNX_.swap(d_nextNX_);
  d_curNY_.swap(d_nextNY_);

  alpha_ *= 0.997;

  beta1k_ *= beta1_;
  beta2k_ *= beta2_;

  targetFunction_->updateParameters(onlyGradMode_);
  prevHpwl_      = hpwl_;
  hpwl_          = targetFunction_->getHPWL();
  overflow_      = targetFunction_->getOverflow();
  macroOverflow_ = targetFunction_->getMacroOverflow();

  printf("[Adam] [Iter%04d] HPWL : %-6.3f  Overflow : %-3.3f\n", 
                      iter,      hpwl_ / 1e6 / db_->getDbu(),         overflow_); 

  //printf("[Adam] [Iter%04d] HPWL : %-6.3f  Overflow : %-3.3f MacroOverflow : %-4.4f\n", 
  //                    iter,      hpwl_ / 1e6,         overflow_,          macroOverflow_); 

  if(!onlyGradMode_)
  {
    wa_          = targetFunction_->getWA();
    sumPenalty_  = targetFunction_->getPenalty();
    float lambda = targetFunction_->getLambda();
    float functionValue = wa_ + lambda * sumPenalty_;
    printf("WA + lambda * Penalty = %E + %E * %E = %E\n", 
            wa_, lambda, sumPenalty_, functionValue);
  }
}

void
AdamOptimizer::opt2db()
{
  thrust::copy(d_curX_.begin(), d_curX_.end(), h_cellCx_.begin());
  thrust::copy(d_curY_.begin(), d_curY_.end(), h_cellCy_.begin());

  int cIdx = 0;
  for(auto cell : db_->movableCells())
  {
    cell->setCenterLocation(h_cellCx_[cIdx], h_cellCy_[cIdx]);
    cIdx++;
  }
}

void
AdamOptimizer::copyPotential2db()
{
  int numBin = static_cast<int>(db_->bins().size());

  float* temp = new float[numBin];

  CUDA_CHECK(
    cudaMemcpy(temp, 
    targetFunction_->density()->getDevicePotential(),
    numBin * sizeof(float),
    cudaMemcpyDeviceToHost));

  int binID = 0;
  for(auto& bin : db_->bins())
    bin->setElectroPotential(temp[binID++]);

  delete [] temp;
}

void
AdamOptimizer::copyBinDensity2db()
{
  int numBin = static_cast<int>(db_->bins().size());

  float* temp = new float[numBin];

  CUDA_CHECK(
    cudaMemcpy(temp, 
    targetFunction_->density()->getDeviceBinDensity(),
    numBin * sizeof(float),
    cudaMemcpyDeviceToHost));

  int binID = 0;
  for(auto& bin : db_->bins())
    bin->setDensity(temp[binID++]);

  delete [] temp;
}

void
AdamOptimizer::copyDensityGrad2db()
{
  db_->densityGradX().resize(numCell_);
  db_->densityGradY().resize(numCell_);

  thrust::copy(targetFunction_->densityGradX().begin(), 
               targetFunction_->densityGradX().end(), 
               db_->densityGradX().begin());

  thrust::copy(targetFunction_->densityGradY().begin(), 
               targetFunction_->densityGradY().end(), 
               db_->densityGradY().begin());
}

void
AdamOptimizer::diverge()
{
  isDiverge_ = true;
  printf("[Adam] Divergence Detected!\n");
  printf("[Adam] Terminate Placer...\n");
  exit(0);
}

void
AdamOptimizer::freeDeviceMemory()
{
}

}; // namespace skyplace

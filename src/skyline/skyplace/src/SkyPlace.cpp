#include <cstdio>
#include <chrono>

#include "skyplace/SkyPlace.h"
#include "InitialPlacer.h"
#include "TargetFunction.h"
#include "DensityGradient.h"
#include "WireLengthGradient.h"
#include "NesterovOptimizer.h"
#include "AdamOptimizer.h"
#include "HyperParam.h"
#include "Painter.h"

namespace skyplace 
{

SkyPlace::SkyPlace() {}
SkyPlace::SkyPlace(std::shared_ptr<dbDatabase> db)
  : dbDatabase_        (db),
    param_             (nullptr),
    db_                (nullptr),
    painter_           (nullptr),
    initialPlacer_     (nullptr),
    wireLength_        (nullptr),
    ntOptimizer_       (nullptr),
    adOptimizer_       (nullptr),
    density_           (nullptr),
    localLambdaMode_   (false  ),
    plotMode_          (false  ),
    dbTime_            (0.0    )
{
  opt_   = OptimizerType::Nesterov;
  db_    = std::make_shared<SkyPlaceDB>();
  param_ = std::make_shared<HyperParam>();
}

SkyPlace::~SkyPlace() {}

void 
SkyPlace::setTargetOverflow(float val) 
{ 
  param_->targetOverflow = val;      
}

void 
SkyPlace::setInitLambda(float val) 
{ 
  param_->initLambda = val;
}

void 
SkyPlace::setMaxPhiCoef(float val) 
{ 
  param_->maxPhiCoef = val;
}

void 
SkyPlace::setRefHpwl(float val) 
{ 
  param_->referenceHpwl = val;
}

void 
SkyPlace::setInitGammaInv(float val) 
{ 
  param_->initGammaInvCoef = val;
}

void 
SkyPlace::setAdamAlpha(float val) 
{ 
  param_->adam_alpha = val;           
}

void 
SkyPlace::setAdamBeta1(float val) 
{ 
  param_->adam_beta1 = val;           
}

void 
SkyPlace::setAdamBeta2(float val) 
{ 
  param_->adam_beta2 = val;           
}

// Though targetDensity is also a hyper-parameter,
// this has to be initialized before SkyPlaceDB is initialized.
void 
SkyPlace::setTargetDensity(float density)  
{ 
  if(density > 1.00 || density <= 0.0)
  {
    printf("Warning - Invalid target density.\n");
    printf("Given value will be ignored.\n");
  }
  else
    db_->setTargetDensity(density);
}

void
SkyPlace::preamble()
{
  auto t1 = std::chrono::high_resolution_clock::now();

  // 1. SkyPlaceDB Initialization
  db_->init(dbDatabase_);

  // 2. Make SubTools
    // Make InitialPlacer
    initialPlacer_ = std::make_unique<InitialPlacer>(db_);

    // Make Density Function
    density_ = std::make_shared<DensityGradient>(db_);
    density_->setLocalLambdaMode(localLambdaMode_);

    // Make WireLength Function
    wireLength_ = std::make_shared<WireLengthGradient>(db_);

    // Make Target Function
    targetFunc_ 
      = std::make_shared<TargetFunction>(db_, wireLength_, density_, param_);

    // Make Painter
    painter_ = std::make_shared<Painter>();
    painter_->setDB(db_);

  auto t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> runtime = t2 - t1;
  dbTime_ = runtime.count();
}

void
SkyPlace::setInitMethod(const std::string& init_method)
{
  if(initialPlacer_ != nullptr)
    initialPlacer_->setInitMethod(init_method);
}

void
SkyPlace::setOptimizer(const std::string& opt_type)
{
  if(opt_type == "Adam" || opt_type == "adam")
    opt_ = OptimizerType::Adam;
  else if(opt_type == "Nesterov" || opt_type == "nesterov")
    opt_ = OptimizerType::Nesterov;
  else
  {
    printf("Undefined Optimizer %s\n", opt_type.c_str());
    printf("Argument is ignored and Nesterov will be used...\n");
  }
}

void
SkyPlace::setMacroWeight(float macroWeight)
{
  db_->updateMacroDensityWeight(macroWeight);
  targetFunc_->setMacroDensityWeight(true);
}

void
SkyPlace::run()
{
  // Prepare for Global Placement
  // (Import dbDatabase to SkyPlaceDB + Make sub-tools...)
  preamble();

  // Make Optimizer
  if(opt_ == OptimizerType::Adam)
    adOptimizer_ = std::make_unique<AdamOptimizer>(param_, db_, targetFunc_, painter_);
  else if(opt_ == OptimizerType::Nesterov)
    ntOptimizer_ =std::make_unique<NesterovOptimizer>(param_, db_, targetFunc_, painter_);
  else
    assert(0);

  printf("[SkyPlace] Local Lambda Mode : %s\n", (localLambdaMode_  ? "ON" : "OFF"));
  printf("[SkyPlace] GIF Plot Mode     : %s\n", (plotMode_         ? "ON" : "OFF"));
  printf("[SkyPlace] Start Placement!\n");

  // Step#1: Initial Placement
  doInitialPlace();

  auto t1 = std::chrono::high_resolution_clock::now();

  // Step#2: Main Placement Iteration
  Stat finalStat; 
  if(opt_ == OptimizerType::Nesterov)
    finalStat = ntOptimizer_->startOptimize(plotMode_);
  else if(opt_ == OptimizerType::Adam)
    finalStat = adOptimizer_->startOptimize(plotMode_);
  else
    assert(0);

  db_->exportDB(dbDatabase_);  // Deliver new coorindates to dbDatabase

  auto t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float> runtime = t2 - t1;

  totalTime_ = runtime.count();

  printf("[SkyPlace] Placement Finished!\n");

  // TODO : Fix mismatch between hpwl_ of Optimizer class and SkyPlaceDB
  db_->updateHpwl();

  // Finish : Print Statistic
  printStat(finalStat);
}

void
SkyPlace::doInitialPlace()
{
  printf("[SkyPlace] Start Initial Place.\n");
  if(initialPlacer_ != nullptr)
    initialPlacer_->doInitialPlace();
  else
  {
    printf("InitialPlacer is not made...\n");
    exit(1);
  }
  printf("[SkyPlace] Finish Initial Place.\n");
}

void
SkyPlace::printStat(Stat finalStat)
{
  double dbTime         = dbTime_;
  double ipTime         = initialPlacer_->getRuntime();
  double gpTime         = finalStat.totalPlaceTime;
  double wlGradTime     = finalStat.wlGradTime;
  double denGradTime    = finalStat.denGradTime;
  double poissonTime    = finalStat.poissonTime;
  double binDensityTime = finalStat.binDensityTime;
  double initTime       = finalStat.initTime;

  double totalTime = totalTime_ + dbTime + ipTime;

  int hpwl_micron = static_cast<int>(finalStat.hpwl / db_->getDbu());

  printf(" ==================================================\n");
  printf(" Final Statistic\n");
  printf(" ==================================================\n");
  printf("  | Benchmark      | %-10s \n", db_->designName().c_str());
  printf("  | HPWL (um)      | %-10d \n", hpwl_micron);
  printf("  | Overflow       | %-10.3f  \n", finalStat.overflow);
  printf("  | # Iteration    | %-10d    \n", finalStat.iter);
  printf("  | Time DB        | %-5.1f   \n", dbTime);
  printf("  | Time IP        | %-5.1f   \n", ipTime);
  printf("  | Time NonLinOpt | %-5.1f   \n", gpTime);
  printf("  | Time WL Grad   | %-5.1f (%3.1f%%) \n", wlGradTime, wlGradTime / gpTime * 100);
  printf("  | Time Den Grad  | %-5.1f (%3.1f%%) \n", denGradTime, denGradTime / gpTime * 100);
  printf("    - Time Poisson   | %-5.1f (%3.1f%%) \n", poissonTime, poissonTime/ gpTime * 100);
  printf("    - Time BinDenUp  | %-5.1f (%3.1f%%) \n", binDensityTime, binDensityTime/ gpTime * 100);
  printf("  | Time OptInit   | %-5.1f (%3.1f%%) \n", initTime, initTime / gpTime * 100);
  printf("  | Time Total     | %-5.1f   \n", totalTime);
  printf("  | Converge?      | %-5d     \n", finalStat.converge);
  printf(" ==================================================\n");
}

} // namespace SkyPlace

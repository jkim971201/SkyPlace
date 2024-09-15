#ifndef SKYPLACE_H
#define SKYPLACE_H

#include <string>
#include <memory>

namespace db {
  class dbDatabase;
}

namespace skyplace 
{

enum class OptimizerType 
{
  Nesterov,
  Adam
};

typedef struct Stat
{
  bool    converge;
  float   hpwl;
  float   overflow;
  int     iter;

  double  totalPlaceTime;
  double  wlGradTime;
  double  denGradTime;
  double  poissonTime;
  double  binDensityTime;
  double  initTime;
} Stat;

class SkyPlaceDB;
class InitialPlacer;
class TargetFunction;
class DensityGradient;
class WireLengthGradient;
class NesterovOptimizer;
class AdamOptimizer;
class HyperParam;
class Painter;

class SkyPlace 
{
  public:

    SkyPlace();
    SkyPlace(std::shared_ptr<db::dbDatabase> db);
 
    ~SkyPlace();

    // APIs
    void run(); // global_place

    void setPlotMode()        { plotMode_        = true; }
    void setLocalLambdaMode() { localLambdaMode_ = true; }
    void setInitMethod(const std::string& init_method);
    void setOptimizer (const std::string& opt_type);

    // Hyper-Parameter Setting command
    void setTargetOverflow  (float val);
    void setInitLambda      (float val);
    void setMaxPhiCoef      (float val);
    void setRefHpwl         (float val);
    void setInitGammaInv    (float val);
    void setAdamAlpha       (float val);
    void setAdamBeta1       (float val);
    void setAdamBeta2       (float val);
    void setMacroWeight     (float val);
    void setTargetDensity   (float density);

  private:

    // dbDatabase from SkyLine Core
    std::shared_ptr<db::dbDatabase> dbDatabase_;

    // Hyper-Parameters
    std::shared_ptr<HyperParam> param_;

    // 1. Import dbDatabase to SkyPlaceDB
    // 2. Make Sub Tools 
    // (InitialPlacer / Density / WireLength / TargetFunction / Painter)
    // 3. Construct TargetFunction f (WireLength & Density) 
    void preamble();

    // Sub-Tools
    std::shared_ptr<SkyPlaceDB> db_;
    std::shared_ptr<Painter>    painter_;

    // For main Placement Stage
    std::unique_ptr<InitialPlacer>      initialPlacer_;
    std::shared_ptr<DensityGradient>    density_;
    std::shared_ptr<WireLengthGradient> wireLength_;
    std::shared_ptr<TargetFunction>     targetFunc_;
    std::unique_ptr<NesterovOptimizer>  ntOptimizer_;
    std::unique_ptr<AdamOptimizer>      adOptimizer_;

    void doInitialPlace();

    // Configuration
    OptimizerType opt_;

    // ETC
    void printStat(Stat finalStat);
    bool plotMode_;
    bool localLambdaMode_;

    // RunTime
    double dbTime_;
    double totalTime_;
};

} // namespace skyplace 

#endif

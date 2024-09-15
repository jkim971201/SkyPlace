#ifndef INITIAL_PLACE_H
#define INITIAL_PLACE_H

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include "fusion.h"

#include <memory>
#include <set>
#include <algorithm>
#include <string>
#include "SkyPlaceDB.h"
#include "FMPartitioner.h"

namespace skyplace 
{

using Eigen::BiCGSTAB;
using Eigen::IdentityPreconditioner;

using namespace mosek::fusion;
using namespace monty;

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SMatrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::Triplet<float> Triplet;

class InitialPlacer
{
  public:
    
    InitialPlacer();                               // Default Constructor
    InitialPlacer(std::shared_ptr<SkyPlaceDB> db); // Constructor really used

    ~InitialPlacer();

    // APIs
    void doInitialPlace();

    // Setters
    void setInitMethod(const std::string& init_method);

    // Getters
    double getRuntime() const { return runtime_; }

  private:

    std::shared_ptr<SkyPlaceDB> db_;

    bool randomInit_;
    bool qpInit_;
    int maxFanout_;
    int maxBiCGIter_; // MaxIter for one BiCG
    int maxIter_;     // MaxIter for entire flow
    int minIter_;     // MinIter for entire flow
    float netWeightScale_;
    float minLength_;
    float minError_;

    Vector xM_; // Movable X
    Vector yM_; // Movalbe Y

    Vector LMFxF_; // L_MF * xF
    Vector LMFyF_; // L_MF * yF

    SMatrix LxMM_; // Laplacian between movable cells
    SMatrix LyMM_; // Laplacian between movable cells

    void doRandomInit();
    void doClusterQuadratic();

    void placeCenter(bool include_filler = false); 
    // if filler is true, then place all cells in the center of die including fillers
    void updateLaplacian();
    void updateCoordinates();

    double runtime_;

    // Clustering-based Initialization
    int numCluster_;
    Vector areaVector_;

    int garbageClusterID_; // To ignore sparse cluster

    float avgClusterArea_;

      // Step #1
    void doClustering();
    void refineCluster();
    void ignoreSparse(int minSize);
    void computeAreaMap();
    void simpleBiPartitioning(std::vector<Cell*>& smallGraph, int& numCluster);
      // Step #2
    void createClusterLaplacian(SMatrix& Lmm, Vector& Lmf_xf, Vector& Lmf_yf);
      // Step #3
    Vector solveQCQP(const SMatrix& A, const Vector& b, const Vector& v, double K) const;

    float LL2CC_X(float llx) const;
    float LL2CC_Y(float lly) const;

    float CC2LL_X(float ccx) const;
    float CC2LL_Y(float ccy) const;

    float getMirrorX(float locX, float dieCx, float dieLx, float dieUx) const;
    float getMirrorY(float locY, float dieCy, float dieLy, float dieUy) const;
};

}; // namespace skyplace 

#endif

#include <cstdio>
#include <vector>
#include <map>
#include <random>
#include <memory>
#include <chrono>

#include "gpulouvain.h"
#include "InitialPlacer.h"

namespace skyplace 
{

std::shared_ptr<ndarray<int,1>>     nint(const std::vector<int>    &X) { return new_array_ptr<int>(X);    }
std::shared_ptr<ndarray<double,1>>  ndou(const std::vector<double> &X) { return new_array_ptr<double>(X); }

Matrix::t eigen2fusion(SMatrix& mat)
{
  int N   = mat.rows();
  int NNZ = mat.nonZeros();

  std::vector<int>    rows(NNZ);
  std::vector<int>    cols(NNZ);
  std::vector<double> vals(NNZ);

  int idx = 0;
  for(int i = 0; i < mat.outerSize(); i++)
  {
    for(SMatrix::InnerIterator it(mat, i); it; ++it)
    {
      rows[idx] = it.row();
      cols[idx] = it.col();
      vals[idx] = static_cast<double>(it.value());
      idx++;

      //std::cout << "row: " << it.row()   << std::endl;
      //std::cout << "col: " << it.col()   << std::endl;
      //std::cout << "val: " << it.value() << std::endl;
    }
  }

  return Matrix::sparse(N, N, nint(rows), nint(cols), ndou(vals) );
}

inline bool isLargeMacro(Cell* cell, float dieDx, float dieDy)
{
  return (cell->dx() > dieDx * 0.2) || (cell->dy() > dieDy * 0.2);
}


InitialPlacer::InitialPlacer()
  : db_          (nullptr),
    numCluster_  (0),
    runtime_     (0),
    randomInit_  (true)
{}

InitialPlacer::InitialPlacer(std::shared_ptr<SkyPlaceDB> db)
  : InitialPlacer()
{
  db_ = db;

  // Hyper-parameters
  maxFanout_      =         200;
  maxBiCGIter_    =         150; // 150
  maxIter_        =          30; // 30 or 15
  minIter_        =           5; 
  netWeightScale_ =           1; // 800?
  minError_       =      1.0e-5;
  minLength_      =        1500;
}

InitialPlacer::~InitialPlacer()
{
  // Default Destructor
}

void
InitialPlacer::doInitialPlace()
{
  auto t1 = std::chrono::high_resolution_clock::now();

  if(!randomInit_)
    doRandomInit();
  else
    doClusterQuadratic();

  auto t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> runtime = t2 - t1;
  runtime_ = runtime.count();
}

void
InitialPlacer::setInitMethod(const std::string& init_method)
{
  using namespace std;
  if(init_method == "Random" || init_method == "random")
  {
    randomInit_ = true; 
    qpInit_ = false;
  }
  else if(init_method == "qp" || init_method == "quadratic")
  {
    randomInit_ = false; 
    qpInit_ = true;
  }
  else
  {
    cout << "[WARN] Undefined Initialization method " << init_method << endl;
    cout << "[WARN] Argument is ignored and Random init will be used...\n";
  }
}

void
InitialPlacer::doRandomInit()
{
  printf("[InitialPlacer] Place all movable cells in the center with Gaussian Noise.\n");

  // See DREAMPlace TCAD`21 for details
  // The authors argue that random center initiailization
  // does not degrade the placement quality
  float dieCx = db_->die()->cx();
  float dieCy = db_->die()->cy();

  float dieDx = db_->die()->dx();
  float dieDy = db_->die()->dy();

  float meanX = dieCx;  
  float meanY = dieCy;  

  float devX = dieDx * 0.05;
  float devY = dieDy * 0.05;

  std::default_random_engine gen;
  std::normal_distribution<float> gaussianNoiseX(meanX, devX);
  std::normal_distribution<float> gaussianNoiseY(meanY, devY);

  for(auto& cell : db_->movableCells())
  {
    float locX = gaussianNoiseX(gen);
    float locY = gaussianNoiseY(gen);

    //if(!cell->isFiller())
    cell->setCenterLocation(locX, locY);
  }

  for(auto& cell : db_->movableCells())
    db_->moveCellInsideLayout(cell);

  db_->updateHpwl();

  printf("[InitialPlacer] Initial HPWL: %6.1f\n",
                          db_->getHPWL() / 1e6);

  printf("[InitialPlacer] Finish Random Center Initialization.\n");
}

void                      // Default = false
InitialPlacer::placeCenter(bool include_filler)
{
  float dieCx = db_->die()->cx();
  float dieCy = db_->die()->cy();

  for(auto& cell : db_->movableCells())
  {
    if(!cell->isFiller() || include_filler)
      cell->setCenterLocation(dieCx, dieCy);
  }
}

void
InitialPlacer::updateLaplacian()
{
  db_->updatePinBound();

  int numMovable = db_->numMovable() - db_->numFiller();

  xM_.resize(numMovable);
  yM_.resize(numMovable);

  LMFxF_ = Eigen::MatrixXf::Constant(numMovable, 1, 0.0);
  LMFyF_ = Eigen::MatrixXf::Constant(numMovable, 1, 0.0);

  LxMM_.resize(numMovable, numMovable);
  LyMM_.resize(numMovable, numMovable);

  // setFromTripets() automatically
  // sum up duplicated triplets
  std::vector<Triplet> tripletVectorX;
  std::vector<Triplet> tripletVectorY;

  for(auto& net : db_->nets())
  {  
    if(net->deg() < 2 || net->deg() > maxFanout_) 
      continue;

    float wB2B = netWeightScale_ / (net->deg() - 1);

    // Select two pins of each net
    for(int p1 = 0; p1 < net->deg() - 1; p1++)
    {
      Pin* pin1 = net->pins()[p1];
      for(int p2 = p1 + 1; p2 < net->deg(); p2++)
      {
        Pin* pin2 = net->pins()[p2];

        int c1 = pin1->cell()->id();
        int c2 = pin2->cell()->id();

        if(pin1->cell() == pin2->cell())
          continue;
  
        // B2B Net Model X Coordinate
        if(pin1->isMinPinX() || pin1->isMaxPinX() ||
           pin2->isMinPinX() || pin2->isMaxPinX() ) 
        {
          float lengthX = std::abs(pin1->cx() - pin2->cx());
          float weightX = 0.0;

          if(lengthX > minLength_)
            weightX = wB2B / lengthX; 
          else 
            weightX = wB2B / minLength_;

          // Case 1 : p1: Movable <-> p2: Movable
          if(!pin1->cell()->isFixed() && !pin2->cell()->isFixed())
          {
            tripletVectorX.push_back(Triplet(c1, c1, weightX));
            tripletVectorX.push_back(Triplet(c2, c2, weightX));
            tripletVectorX.push_back(Triplet(c1, c2, -weightX));
            tripletVectorX.push_back(Triplet(c2, c1, -weightX));
          }

          // Case 2 : p1: Movable <-> p2: Fixed
          else if(!pin1->cell()->isFixed() && pin2->cell()->isFixed())
          {
            tripletVectorX.push_back(Triplet(c1, c1, weightX));
            LMFxF_(c1) += weightX * pin2->cell()->cx();
          }

          // Case 3 : p1: Fixed <-> p2: Movable
          else if(pin1->cell()->isFixed() && !pin2->cell()->isFixed())
          {
            tripletVectorX.push_back(Triplet(c2, c2, weightX));
            LMFxF_(c2) += weightX * pin1->cell()->cx();
          }
        }

        // B2B Net Model Y Coordinate
        if(pin1->isMinPinY() || pin1->isMaxPinY() ||
           pin2->isMinPinY() || pin2->isMaxPinY() ) 
        {
          float lengthY = std::abs(pin1->cy() - pin2->cy());
          float weightY = 0.0; 

          if(lengthY > minLength_)
            weightY = wB2B / lengthY; 
          else 
            weightY = wB2B / minLength_;

          // Case 1 : Movable <-> Movable
          if(!pin1->cell()->isFixed() && !pin2->cell()->isFixed())
          {
            tripletVectorY.push_back(Triplet(c1, c1, weightY));
            tripletVectorY.push_back(Triplet(c2, c2, weightY));
            tripletVectorY.push_back(Triplet(c1, c2, -weightY));
            tripletVectorY.push_back(Triplet(c2, c1, -weightY));
          }

          // Case 2 : Movable <-> Fixed
          else if(!pin1->cell()->isFixed() && pin2->cell()->isFixed())
          {
            tripletVectorY.push_back(Triplet(c1, c1, weightY));
            LMFyF_(c1) += weightY * pin2->cell()->cy();
          }

          // Case 3 : Fixed <-> Movable
          else if(pin1->cell()->isFixed() && !pin2->cell()->isFixed())
          {
            tripletVectorY.push_back(Triplet(c2, c2, weightY));
            LMFyF_(c2) += weightY * pin1->cell()->cy();
          }
        }
      }
    }
  }
  LxMM_.setFromTriplets(tripletVectorX.begin(), tripletVectorX.end());
  LyMM_.setFromTriplets(tripletVectorY.begin(), tripletVectorY.end());
}

void
InitialPlacer::updateCoordinates()
{
  for(auto& cell : db_->movableCells())
  {
    // only update movable cells
    if(!cell->isFiller())
    {
      cell->setCenterLocation(static_cast<int>(xM_(cell->id())), 
                              static_cast<int>(yM_(cell->id())));

      //std::cout << xM_(cell->id()) << std::endl;
      //std::cout << yM_(cell->id()) << std::endl;
    }
  }
}

void 
InitialPlacer::simpleBiPartitioning(std::vector<Cell*>& smallGraph, int& numCluster)
{
  FMPartitioner fm(smallGraph, 0.3, numCluster);
}

void 
InitialPlacer::refineCluster()
{
     // clusterID          CellPtr
  std::map<int, std::vector<Cell*>> clusterMap;

  float dieDx = db_->die()->dx();
  float dieDy = db_->die()->dy();

  for(auto& cell : db_->movableCells() )
  {
    if(!cell->isFiller() && cell->pins().size() != 0)
      clusterMap[cell->clusterID()].push_back(cell);
  }

  for(auto& kv : clusterMap)
  {
    int cID = kv.first;
    int numMacro = 0;
    int numCell  = kv.second.size();

    // std::cout << "Cluster " << cID << " : " << numCell << std::endl;
    for(auto& cell : kv.second)
    {
      if( cell->isMacro() )
        numMacro++;
      if( isLargeMacro(cell, dieDx, dieDy) && numCell != 1)
        cell->setClusterID( numCluster_++ );
    }

    if(numMacro > 15 && numMacro < 100)
    {
      std::cout << "Too many macros are detected in a cluster..." << std::endl;
      std::vector<Cell*> smallGraph; // only has Macros
      
      for(auto& cell : kv.second)
      {
        if(cell->isMacro())
          smallGraph.push_back(cell);
      }

      simpleBiPartitioning(smallGraph, numCluster_);
    }
  }
}

// Wrapper for GPU Louvain
void
InitialPlacer::doClustering()
{
  int V = db_->numMovable() - db_->numFiller();

  using namespace Louvain;

  host_structures hostStructures;

  hostStructures.originalV = V;
  hostStructures.V = V;

  cudaHostAlloc((void**)&hostStructures.vertexCommunity, V * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostStructures.communityWeight, V * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostStructures.originalToCommunity, V * sizeof(int), cudaHostAllocDefault);

  std::vector<std::map<int, float>> neighbours(V);
  // Here is assumption that graph is undirected.

  for(auto& net : db_->nets() )
  {
    if( net->deg() < 2 || net->deg() > 24 ) // 25?
      continue;

    float cliqueWeight = 1.0 / (float(net->deg()) - 1.0);

    for(int p1 = 0; p1 < net->deg() - 1; p1++)
    {
      Pin*   pin1 = net->pins()[p1];
      Cell* cell1 = pin1->cell(); 

      if( cell1->isIO() || cell1->isFixed() )
        continue;

      int v1 = cell1->id();

      for(int p2 = p1 + 1; p2 < net->deg(); p2++)
      {
        Pin*   pin2 = net->pins()[p2];
        Cell* cell2 = pin2->cell(); 
        
        int v2 = cell2->id();

        if( v1 == v2 || cell2->isIO() || cell2-> isFixed() )
          continue;

        hostStructures.communityWeight[v1] += cliqueWeight;
        hostStructures.communityWeight[v2] += cliqueWeight;

        hostStructures.M += cliqueWeight;

        neighbours[v1][v2] += cliqueWeight;
        neighbours[v2][v1] += cliqueWeight;
      }
    }
  }

  int E = 0;
  for(auto& m : neighbours)
    E += m.size();

  std::cout << "V: " << V << " E:  " << E << std::endl;

  hostStructures.E = E;
  cudaHostAlloc((void**)&hostStructures.edges,   E * sizeof(  int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hostStructures.weights, E * sizeof(float), cudaHostAllocDefault);

  std::cout << "Memory Allocation has successed." << std::endl;

  int index = 0;
  for(int v = 0; v < V; v++) 
  {
    hostStructures.edgesIndex[v] = index;
    for(auto & it : neighbours[v]) 
    {
      // std::cout << index << " / " << E << std::endl;
      hostStructures.edges[index] = it.first;
      hostStructures.weights[index] = it.second;
      index++;
    }
  }

  hostStructures.edgesIndex[V] = E;

  // GPU Louvain Clustering
  gpu_louvain(hostStructures, 0.01, 3);

  std::cout << "Num Cluster : " << hostStructures.V << std::endl;
  numCluster_ = hostStructures.V;

  for(auto& cell : db_->movableCells() )
  {
    if(cell->isFiller())
      continue;

    assert(cell->pins().size() != 0);

    int clusterID = hostStructures.originalToCommunity[cell->id()];
    cell->setClusterID(clusterID);
  }

  cudaFreeHost(hostStructures.vertexCommunity);
  cudaFreeHost(hostStructures.communityWeight);
  cudaFreeHost(hostStructures.edges);
  cudaFreeHost(hostStructures.weights);
  cudaFreeHost(hostStructures.edgesIndex);
  cudaFreeHost(hostStructures.originalToCommunity);
}

//std::tuple<SMatrix, Vector>
void
InitialPlacer::createClusterLaplacian(SMatrix& Lmm, Vector& Lmf_xf, Vector& Lmf_yf)
{
  int numFixed = db_->numFixed();

  // vID  vID   edgeWeight
  std::vector<Triplet> tripletVector;

  // M2M : Movable to Movable
  // M2F : Movable to Fixed
  float weightForM2M = 1.0;
  float weightForM2F = 4.0; // 4.0?

  for(auto& net : db_->nets() )
  {
    // Since we are ignoring too large nets,
    // there can be some clusters that are not connected to any other vertex
    if( net->deg() < 2 || net->deg() > 200 ) // 25?
      continue;

    for(int p1 = 0; p1 < net->deg() - 1; p1++)
    {
      Pin*   pin1 = net->pins()[p1];
      Cell* cell1 = pin1->cell(); 

      int v1 = 0;
      if(cell1->isFixed())
        v1 = cell1->id() + numCluster_;
      else
        v1 = cell1->clusterID();

      for(int p2 = p1 + 1; p2 < net->deg(); p2++)
      {
        Pin*   pin2 = net->pins()[p2];
        Cell* cell2 = pin2->cell(); 
        
        int v2 = 0;
        if( cell2->isFixed() )
          v2 = cell2->id() + numCluster_;
        else
          v2 = cell2->clusterID();

        if( v1 == v2 )
          continue;

        float weight = 1.0;
        if(cell1->isFixed() || cell2->isFixed())
          weight = weightForM2F;
        else
          weight = weightForM2M;

        // L = D - A
        // For Diagonal Matrix
        tripletVector.push_back( Triplet(v1, v1, +weight) );
        tripletVector.push_back( Triplet(v2, v2, +weight) );

        // For Adjacency Matrix
        tripletVector.push_back( Triplet(v1, v2, -weight) );
        tripletVector.push_back( Triplet(v2, v1, -weight) );
      }
    }
  }

  int V = numCluster_ + numFixed;

  //for(auto& cell : db_->fixedCells() )
  //  printf("CellID: %d V: %d numCluster: %d numFixed: %d \n", cell->id(), V, numCluster_, db_->numFixed());

  SMatrix laplacian;
  laplacian.resize(V, V);
  laplacian.setFromTriplets( tripletVector.begin(), tripletVector.end() );

  SMatrix Lmf = laplacian.block(0,            // Start Index of Row (Y)
                                numCluster_,  // Start Index of Col (X)
                                numCluster_,  // Size of Y (numRow)
                                numFixed);    // Size of X (numCol)

  // printf("Debug: Lmf Rows: %d / %d\n", Lmf.rows(), numCluster_);
  // printf("Debug: Lmf Cols: %d / %d\n", Lmf.cols(), numFixed);

  Vector fixedX(numFixed);
  Vector fixedY(numFixed);

  int fixedIdx = 0;
  for(auto& cell : db_->fixedCells() )
  {
    fixedX(fixedIdx) = LL2CC_X(cell->cx());
    fixedY(fixedIdx) = LL2CC_Y(cell->cy());

    fixedIdx++;
  }

  Lmm.resize(numCluster_, numCluster_);
  Lmf_xf.resize(numCluster_);
  Lmf_yf.resize(numCluster_);

  Lmm = laplacian.block(0, 0, numCluster_, numCluster_);

  assert(Lmf.cols() == fixedX.size() );
  assert(Lmf.cols() == fixedY.size() );
  assert(Lmm.cols() == numCluster_   );
  assert(Lmm.rows() == numCluster_   );

  Lmf_xf = Lmf * fixedX;
  Lmf_yf = Lmf * fixedY;
}

Vector 
InitialPlacer::solveQCQP(const SMatrix& A, const Vector& b, const Vector& v, double K) const
{
  printf("Start to Solve QCQP\n");

  int N   = A.rows();
  int NNZ = A.nonZeros();

  SMatrix M_1, M_2, M_3;
  M_1.resize(N + 1, N + 1);
  M_2.resize(N + 1, N + 1);
  M_3.resize(N + 1, N + 1);

  // M_1.block(1, 1, N, N) = A; -> SubBlock is read-only for SparseMatrix
  for(int i = 0; i < A.outerSize(); i++)
  {
    for(SMatrix::InnerIterator it(A, i); it; ++it)
      M_1.coeffRef( it.row() + 1, it.col() + 1 ) = it.value();
  }

  // M_i(0, 0) == 0.0
  for(int i = 0; i < N; i++)
  {
    if( i < N )
    {
      M_1.coeffRef(i + 1, 0) = b(i);
      M_1.coeffRef(0, i + 1) = b(i);
    }

    M_2.coeffRef(i + 1, i + 1) = v(i);
    M_3.coeffRef(0,     i + 1) = v(i); // v(i)
    M_3.coeffRef(i + 1,     0) = v(i); // v(i)

    //M_2.coeffRef(i + 1, i + 1) = 1.0;
    //  M_3.coeffRef(0,     i + 1) = 1.0; // v(i)
    //  M_3.coeffRef(i + 1,     0) = 1.0; // v(i)
  }

  auto M_1_fusion = eigen2fusion(M_1);
  auto M_2_fusion = eigen2fusion(M_2);
  auto M_3_fusion = eigen2fusion(M_3);

  printf("Matrix create finished.\n");

  //std::cout << "Fusion M_1 = " << M_1_fusion->toString() << std::endl;
  //std::cout << "Fusion M_2 = " << M_1_fusion->toString() << std::endl;
  //std::cout << "Fusion M_3 = " << M_1_fusion->toString() << std::endl;

  Model::t M = new Model("QCQP_SDP Relaxation"); 
  auto _M = finally([&]() { M->dispose(); }); 
  // This is like delete or free of C/C++
  // but I don't know why this has to be called first

  // Setting up the variables
  Variable::t X = M->variable("X", Domain::inPSDCone(N + 1));

  // Objective
  M->objective(ObjectiveSense::Minimize, Expr::dot(M_1_fusion, X));

  // Constraints
  M->constraint("Constraint1", Expr::dot(M_2_fusion, X), Domain::equalsTo(K)   );
  M->constraint("Constraint2", Expr::dot(M_3_fusion, X), Domain::equalsTo(0.0) );
  M->constraint("Constraint3", X->index(nint({0,0}))   , Domain::equalsTo(1.0) );

  M->setSolverParam("numThreads", "16");

  printf("Solving QCQP...\n");

  M->solve();

  printf("Solve finished.\n");

  auto sol = *(X->slice( nint( {0, 1} ), nint( {1, N + 1} ))->level());

  Vector eigenX(N);

  int idx = 0;
  for(auto& val : sol)
    eigenX(idx++) = val;

  return eigenX;
}

void 
InitialPlacer::computeAreaMap()
{
  avgClusterArea_ = 0.0;
  areaVector_.resize(numCluster_);
  for(int i = 0; i < numCluster_; i++)
    areaVector_(i) = 0.0;

  for(auto& cell : db_->movableCells() )
  {
    if(!cell->isFiller())
    {
      float area = cell->area();
      areaVector_(cell->clusterID()) += area;
      avgClusterArea_ += area;
    }
  }

  avgClusterArea_ /= static_cast<float>(numCluster_);
}

void
InitialPlacer::ignoreSparse(int minSize)
{
     // clusterID          CellPtr
  std::map<int, std::vector<Cell*>> clusterMap;

  std::map<int, int> macroNumMap;

  // Initialization
  for(int i = 0; i < numCluster_; i++)
    macroNumMap[i] = 0;

  for(auto& cell : db_->movableCells() )
  {
    if(!cell->isFiller() && cell->pins().size() != 0)
    {
      clusterMap[cell->clusterID()].push_back(cell);

      if(cell->isMacro())
        macroNumMap[cell->clusterID()] += 1;
    }
  }

  // ClutserID before removal -> ClusterID after removal
  std::map<int, int> before2after;
  std::set<int> sparseClusterList;

  int effClusterID = 0;
  for(auto& kv : clusterMap)
  {
    int cID = kv.first;
    int numCell  = kv.second.size();
    int numMacro = macroNumMap[cID];

    // Detect Sparse Cluster
    if(numMacro == 0 && numCell < minSize)
      sparseClusterList.insert(cID);
    else
      before2after[cID] = effClusterID++;
  }

  garbageClusterID_ = effClusterID;
  numCluster_       = effClusterID + 1;

  for(auto& cell : db_->movableCells() )
  {
    if(!cell->isFiller() && cell->pins().size() != 0)
    {
      int cID = cell->clusterID();
      if(sparseClusterList.count(cID))
        cell->setClusterID(garbageClusterID_);
      else
        cell->setClusterID(before2after[cID]);
    }
  }

  printf("Removed %d sparse clusters...\n", sparseClusterList.size());
  printf("NumCluster after Removal: %d\n", numCluster_);

  // ForDebugging
  //for(auto& kv : before2after)
  //  printf("oldID : %d -> newID: %d\n", kv.first, kv.second);

//  // For Debugging
//  std::vector<int> forDebug(numCluster_);
//
//  for(auto& cell : db_->movableCells() )
//  {
//    if(!cell->isFiller() && cell->clusterID() != garbageClusterID_)
//      forDebug[cell->clusterID()] += 1;
//  }
//
//  for(int i = 0; i < numCluster_; i++)
//    printf("Cluster %03d - numCell : %d\n", i, forDebug[i]);
}

void
InitialPlacer::doClusterQuadratic()
{
  // Step #1
  doClustering();
  refineCluster();
  ignoreSparse(4);
  computeAreaMap();
  db_->setNumCluster(numCluster_);

  // Step #2
  SMatrix Lmm;
  Vector  Lmf_xf;
  Vector  Lmf_yf;

  createClusterLaplacian(Lmm, Lmf_xf, Lmf_yf);

  // Step #3
  Vector solX(numCluster_);
  Vector solY(numCluster_);

  float dieLx = LL2CC_X(db_->die()->lx());
  float dieLy = LL2CC_Y(db_->die()->ly());
  float dieUx = LL2CC_X(db_->die()->ux());
  float dieUy = LL2CC_Y(db_->die()->uy());

  float dieCx = LL2CC_X(db_->die()->cx());
  float dieCy = LL2CC_Y(db_->die()->cy());

  std::cout << "dieLx: " << dieLx << std::endl;
  std::cout << "dieLy: " << dieLy << std::endl;
  std::cout << "dieUx: " << dieUx << std::endl;
  std::cout << "dieUy: " << dieUy << std::endl;

  std::cout << "dieCx: " << dieCx << std::endl;
  std::cout << "dieCy: " << dieCy << std::endl;

  //solX = solveLinearSystemCPU(Lmm, Lmf_xf);
  //solY = solveLinearSystemCPU(Lmm, Lmf_yf);

  //solX = solveQP(Lmm, Lmf_xf, dieLx, dieUx, dieCx);
  //solY = solveQP(Lmm, Lmf_yf, dieLy, dieUy, dieCy);

  //float Kx = 0.25 * avgClusterArea_ * numCluster_ * (dieUx - dieLx) * (dieUx - dieLx);
  //float Ky = 0.25 * avgClusterArea_ * numCluster_ * (dieUy - dieLy) * (dieUy - dieLy);

  double Kx = 0.25 * static_cast<double>(dieUx - dieLx) 
                   * static_cast<double>(dieUx - dieLx);

  double Ky = 0.25 * static_cast<double>(dieUy - dieLy) 
                   * static_cast<double>(dieUy - dieLy);

  //double Kx = 0.4 * static_cast<double>(dieUx - dieLx) 
  //                * static_cast<double>(dieUx - dieLx)
  //                * static_cast<double>(avgClusterArea_)
  //                * static_cast<double>(numCluster_);

  //double Ky = 0.4 * static_cast<double>(dieUy - dieLy) 
  //                * static_cast<double>(dieUy - dieLy)
  //                * static_cast<double>(avgClusterArea_)
  //                * static_cast<double>(numCluster_);

  std::cout << "Kx : " << Kx << std::endl;
  std::cout << "Ky : " << Ky << std::endl;

  solX = solveQCQP(Lmm, Lmf_xf, areaVector_, Kx);
  solY = solveQCQP(Lmm, Lmf_yf, areaVector_, Ky);

  //for(int i = 0; i < numCluster_; i++)
  //  std::cout << "Debug newX newY : " << solX(i) << " " << solY(i) << std::endl;

  // Step #4
  float dieLxLL = db_->die()->lx();
  float dieLyLL = db_->die()->ly();

  float dieUxLL = db_->die()->ux();
  float dieUyLL = db_->die()->uy();

  float dieCxLL = db_->die()->cx();
  float dieCyLL = db_->die()->cy();

  float dieDxLL = db_->die()->dx();
  float dieDyLL = db_->die()->dy();

  float meanX = dieCxLL;  
  float meanY = dieCyLL;  

  // 0.3?
  float devX = dieDxLL * 0.3;
  float devY = dieDyLL * 0.3;

  std::default_random_engine gen;
  std::normal_distribution<float> gaussianNoiseX(meanX, devX);
  std::normal_distribution<float> gaussianNoiseY(meanY, devY);

  float newX = 0.0;
  float newY = 0.0;

  int numMacro = db_->numMacro();

  std::normal_distribution<float> noiseGen1(0.0, 2 * 500.0);
  std::normal_distribution<float> noiseGen2(0.0, 2 * 200.0);

  for(auto& cell : db_->movableCells() )
  {
    if(cell->isFiller())
    {
      float locX = gaussianNoiseX(gen);
      float locY = gaussianNoiseY(gen);

      locX = getMirrorX(locX, dieCxLL, dieLxLL, dieUxLL);
      locY = getMirrorY(locY, dieCyLL, dieLyLL, dieUyLL);

      cell->setCenterLocation( locX, locY );
      db_->moveCellInsideLayout(cell);
      continue;
    }

    int cID = cell->clusterID();

    newX = CC2LL_X(solX(cID));
    newY = CC2LL_Y(solY(cID));

    // We have to put some noise to prevent 
    // macros in the same cluster stick together
    // ( for macro-heavy designs (e.g. bigblue2) )
    if(cell->isMacro() && numMacro > 500)
    {
      float noiseX = noiseGen1(gen);
      float noiseY = noiseGen2(gen);

      cell->setCenterLocation( newX + noiseX, newY + noiseY );
    }
    else
      cell->setCenterLocation( newX, newY );

    db_->moveCellInsideLayout(cell);
  }
}

float
InitialPlacer::LL2CC_X(float llx) const
{
  return llx - db_->die()->cx();
}

float
InitialPlacer::LL2CC_Y(float lly) const
{
  return lly - db_->die()->cy();
}

float
InitialPlacer::CC2LL_X(float ccx) const
{
  return ccx + db_->die()->cx();
}

float
InitialPlacer::CC2LL_Y(float ccy) const
{
  return ccy + db_->die()->cy();
}

float
InitialPlacer::getMirrorX(float locX, float dieCx, float dieLx, float dieUx) const
{
  if(locX <= dieCx) 
    return dieLx + (dieCx - locX);
  else
    return dieUx - (locX - dieCx);
}

float
InitialPlacer::getMirrorY(float locY, float dieCy, float dieLy, float dieUy) const
{
  if(locY <= dieCy) 
    return dieLy + (dieCy - locY);
  else
    return dieUy - (locY - dieCy);
}

}; // namespace skyplace 

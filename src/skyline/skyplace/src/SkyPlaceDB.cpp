#include <cmath>
#include <cassert>
#include <map>
#include <random>    // For mt19937
#include <algorithm> // For sort
#include <stdio.h>
#include <iostream>
#include <climits>   // For INT_MAX, INT_MIN
#include <cfloat>    // For FLT_MAX
#include <fstream>

#include "db/dbTech.h"
#include "db/dbDesign.h"
#include "db/dbTypes.h"
#include "db/dbDie.h"
#include "db/dbInst.h"
#include "db/dbNet.h"
#include "db/dbITerm.h"
#include "db/dbBTerm.h"
#include "db/dbRow.h"

#include "SkyPlaceDB.h"
#include "db/dbDatabase.h"

namespace skyplace
{

//////////////////////////////////////////////////////////
static float getOverlapArea(const Bin* bin, const Cell* cell)
{
  float rectLx = std::max(bin->lx(), cell->lx());
  float rectLy = std::max(bin->ly(), cell->ly());
  float rectUx = std::min(bin->ux(), cell->ux());
  float rectUy = std::min(bin->uy(), cell->uy());

  if(rectLx >= rectUx || rectLy >= rectUy)
    return 0;
  else
    return (rectUx - rectLx) * (rectUy - rectLy);
}

// getOverlapArea should use int64_t ideally,
// but runtime will be doubled (referred to OpenROAD comment)
static float getOverlapAreaWithDensitySize(const Bin* bin, const Cell* cell)
{
  float rectLx = std::max(bin->lx(), cell->dLx());
  float rectLy = std::max(bin->ly(), cell->dLy());
  float rectUx = std::min(bin->ux(), cell->dUx());
  float rectUy = std::min(bin->uy(), cell->dUy());

  if(rectLx >= rectUx || rectLy >= rectUy)
    return 0;
  else
    return ((rectUx - rectLx) * (rectUy - rectLy));
}
//////////////////////////////////////////////////////////

// Cell // 
Cell::Cell()
  : id_            (      0),
    cluster_id_    (      0),
    cx_            (      0), 
    cy_            (      0),
    dx_            (      0), 
    dy_            (      0),
    dDx_           (      0), 
    dDy_           (      0),
    densityScale_  (      1),
    isIO_          (  false),
    isMacro_       (  false), 
    isFixed_       (  false),
    isFiller_      (  false),
    dbInst_        (nullptr),
    dbBTerm_       (nullptr)
{}

Cell::Cell(dbInst* inst) : Cell()
{
  dbInst_  = inst;
  isFixed_ = inst->isFixed();
  isMacro_ = inst->isMacro();

  if(!isMacro_)
  {
    dx_ = static_cast<float>(inst->dx());
    dy_ = static_cast<float>(inst->dy());
    cx_ = static_cast<float>(inst->lx() + dx_ / 2);
    cy_ = static_cast<float>(inst->ly() + dy_ / 2);
  }
  else
  {
    dx_ = static_cast<float>(inst->dx() + inst->haloR() + inst->haloL());
    dy_ = static_cast<float>(inst->dy() + inst->haloT() + inst->haloB());
    cx_ = static_cast<float>(inst->lx() - inst->haloL() + dx_ / 2);
    cy_ = static_cast<float>(inst->ly() - inst->haloB() + dy_ / 2);
  }
}

Cell::Cell(dbBTerm* bterm) : Cell()
{
  dbBTerm_ = bterm;
  isIO_    = true;
  isFixed_ = true;

  dx_ = static_cast<float>(bterm->dx());
  dy_ = static_cast<float>(bterm->dy());

  cx_ = static_cast<float>(bterm->cx());
  cy_ = static_cast<float>(bterm->cy());
}

// Constructor for Filler Cell
Cell::Cell(float cx, float cy, float dx, float dy) : Cell()
{
  cx_ = cx;
  cy_ = cy;
  dx_ = dx;
  dy_ = dy;

  isFixed_  = false;
  isMacro_  = false;
  isFiller_ = true;
}

void 
Cell::setCenterLocation(float newCx, float newCy)
{
  cx_ = newCx;
  cy_ = newCy;

  for(auto &pin : pins_)
    pin->updatePinLocation(this);
}

void 
Cell::setDensitySize(float dWidth, float dHeight, float dScale)
{
  dDx_ = dWidth;
  dDy_ = dHeight;
  densityScale_ = dScale;
}

// Pin // 
Pin::Pin()
  : id_           (      0), 
    cx_           (      0), 
    cy_           (      0),
    isIO_         (  false),
    offsetX_      (      0), 
    offsetY_      (      0),
    isMinPinX_    (  false), 
    isMinPinY_    (  false),
    isMaxPinX_    (  false), 
    isMaxPinY_    (  false),
    net_          (nullptr), 
    cell_         (nullptr)
{}

Pin::Pin(dbITerm* iterm, int id) : Pin()
{
  id_ = id;
  offsetX_ = static_cast<float>(iterm->getMTerm()->cx());
  offsetY_ = static_cast<float>(iterm->getMTerm()->cy());
}

Pin::Pin(dbBTerm* bterm, int id) : Pin()
{
  id_ = id;
  int btermCx = bterm->cx();
  int btermCy = bterm->cy();

  cx_ = static_cast<float>(btermCx);
  cy_ = static_cast<float>(btermCy);
  offsetX_ = static_cast<float>(btermCx - bterm->lx());
  offsetY_ = static_cast<float>(btermCy - bterm->ly());
  isIO_ = true;
}

void
Pin::updatePinLocation(Cell* cell) 
{
  // if we use cell_ instead of given Cell*,
  // we have to check whether cell_ is not nullptr 
  // everytime we call this function
  cx_ = cell->cx() + offsetX_;
  cy_ = cell->cy() + offsetY_;
}

// Net //
Net::Net()
  : id_      (0      ),
    lx_      (INT_MAX),
    ly_      (INT_MAX), 
    ux_      (INT_MIN), 
    uy_      (INT_MIN),
    weight_  (1.0    ),
    dbNet_   (nullptr)
{}

Net::Net(dbNet* net, int id) : Net()
{
  id_     =  id;
  weight_ = 1.0;
  dbNet_  = net;
}

void
Net::updateBBox()
{
  if(pins_.size() == 0)
  {
    std::cout << "Warning - " << dbNetPtr()->name() << " has no pins.\n";
    lx_ = ly_ = ux_ = uy_ = 0;
    return;
  }

  // To detect an error,
  // We initilize them as large integer.
  // so that an un-initilized net will
  // make total HPWL invalid.
  lx_ = ly_ = std::numeric_limits<float>::max();
  ux_ = uy_ = 0;

  for(auto& p : pins_)
  {
    lx_ = std::min(p->cx(), lx_);
    ly_ = std::min(p->cy(), ly_);
    ux_ = std::max(p->cx(), ux_);
    uy_ = std::max(p->cy(), uy_);
  }
}

// Bin //
Bin::Bin()
  : row_    (0), 
    col_    (0),
    lx_     (0), 
    ly_     (0),
    ux_     (0), 
    uy_     (0),
    lambda_           (1.0),
    fixedArea_        (0),
    movableArea_      (0),
    density_          (0),
    targetDensity_    (0),
    electroForceX_    (0),
    electroForceY_    (0),
    electroPotential_ (0)
{}

Bin::Bin(int row , int col , 
         float lx, float ly, 
         float ux, float uy,
         float targetDensity) : Bin()
{
  row_ = row;
  col_ = col;
  lx_ = lx;
  ly_ = ly;
  ux_ = ux;
  uy_ = uy;
  targetDensity_ = targetDensity;
}

// Row //
Row::Row(dbRow* row)
{
  numSiteX_ = row->numSiteX();
  numSiteY_ = row->numSiteY();

  lx_ = row->origX();
  ly_ = row->origY();

  ux_ = lx_ + row->dx();
  uy_ = ly_ + row->dy();

  stepX_ = row->stepX();
  stepY_ = row->stepY();

  siteWidth_ = row->siteWidth();
}

// SkyPlaceDB // 
SkyPlaceDB::SkyPlaceDB()
  : designName_           (     ""),
    targetDensity_        (    1.0),
    dbu_                  (      0),
    numStdCells_          (      0), 
    numMacro_             (      0),
    numFixed_             (      0), 
    numMovable_           (      0), 
    numMovableMacro_      (      0),
    numFiller_            (      0),
    numBinX_              (      0), 
    numBinY_              (      0), 
    numIO_                (      0),
    numCluster_           (      0),
    sumMovableArea_       (      0), 
    sumMovableStdArea_    (      0), 
    sumMovableMacroArea_  (      0), 
    sumScaledMovableArea_ (      0), 
    sumFixedArea_         (      0),
    hpwl_                 (      0), 
    binX_                 (      0), 
    binY_                 (      0),
    diePtr_               (nullptr),
    fillerWidth_          (      0), 
    fillerHeight_         (      0),
    numInitStep_          (      5)
{
  reset();
}

void
SkyPlaceDB::reset()
{
  dbu_ = 0;
  
  designName_ = std::string();

  diePtr_ = nullptr;

  cellPtrs_.clear();
  cellInsts_.clear();

  movableMacroPtrs_.clear();
  fixedPtrs_.clear();
  movablePtrs_.clear();

  netPtrs_.clear();
  netInsts_.clear();

  pinPtrs_.clear();
  pinInsts_.clear();

  binPtrs_.clear();
  binInsts_.clear();

  rowPtrs_.clear();
  rowInsts_.clear();

  dbInst2Cell_.clear();
  dbBTerm2Cell_.clear();

  dbNet2Net_.clear();
  dbITerm2Pin_.clear();
  dbBTerm2Pin_.clear();
}

void
SkyPlaceDB::exportDB(std::shared_ptr<dbDatabase> _dbDatabase)
{
  for(auto cell : movablePtrs_)
  {
    if(!cell->isFiller())
    {
      dbInst* inst_ptr = cell->dbInstPtr();
      int newLx = static_cast<int>( cell->lx() );
      int newLy = static_cast<int>( cell->ly() );
      inst_ptr->setLocation(newLx, newLy);
    }
  }
}

void
SkyPlaceDB::importDB(std::shared_ptr<dbDatabase> _dbDatabase)
{
  const auto design = _dbDatabase->getDesign();

  const std::vector<dbNet*>&   db_nets   = design->getNets();
  const std::vector<dbRow*>&   db_rows   = design->getRows();
  const std::vector<dbInst*>&  db_insts  = design->getInsts();
  const std::vector<dbITerm*>& db_iterms = design->getITerms();
  const std::vector<dbBTerm*>& db_bterms = design->getBTerms();

  // Step #1: Initialize Die
  die_.setLx(static_cast<float>(design->coreLx()));
  die_.setLy(static_cast<float>(design->coreLy()));
  die_.setUx(static_cast<float>(design->coreUx()));
  die_.setUy(static_cast<float>(design->coreUy()));
  diePtr_ = &die_;

  // Step #2: Initialize Cell
  int numInst  = db_insts.size();
  int numBTerm = db_bterms.size();

  int cIdx = 0;
  int fixedIdx   = 0;
  int movableIdx = 0;

  // We should make "Cell" for both insts and bterms
  cellInsts_.resize(numInst + numBTerm);
  cellPtrs_.reserve(numInst + numBTerm);

  for(int iterI = 0; iterI < numInst; iterI++)
  {
    Cell& cellInst = cellInsts_[iterI];
    dbInst* inst_ptr = db_insts[cIdx];
    cellInst = Cell(inst_ptr);

    dbInst2Cell_[inst_ptr] = &cellInst;
    cellPtrs_.push_back(&cellInst);

    cIdx++;

    if(cellInst.isFixed()) 
    {
      cellInst.setID(fixedIdx++);
      fixedPtrs_.push_back(&cellInst);
      if(!isOutsideDie(&cellInst))
        sumFixedArea_ += cellInst.area();
      if(cellInst.isMacro())
        numMacro_++;
    }
    else                    
    {
      cellInst.setID(movableIdx++);
      movablePtrs_.push_back(&cellInst);
      auto cellArea = cellInst.area();
      sumMovableArea_ += cellArea;

      if(cellInst.isMacro())
      {
        sumMovableMacroArea_  += cellArea;
        sumScaledMovableArea_ += cellArea * targetDensity_;
        numMacro_++;
        numMovableMacro_++;
      }
      else 
      {
        sumMovableStdArea_    += cellArea;
        sumScaledMovableArea_ += cellArea;
        numStdCells_++;
      }
    }
  }

  // Step #2: Make Cells for IOs
  for(int iterB = 0; iterB < numBTerm; iterB++)
  {
    Cell& cellInst = cellInsts_[numInst + iterB];
    dbBTerm* bterm = db_bterms[iterB];
    
    cellInst = Cell(bterm);
    cellInst.setID(fixedPtrs_.size());
    cellPtrs_.push_back(&cellInst);
    fixedPtrs_.push_back(&cellInst);
    dbBTerm2Cell_[bterm] = &cellInst;
  }

  // Step #3: Initialize Net
  int numNets = db_nets.size();
  netInsts_.resize(numNets);
  netPtrs_.reserve(numNets);

  int nIdx = 0;
  for(auto& netInst : netInsts_)
  {
    dbNet* net_ptr = db_nets[nIdx];
    netInst = Net(net_ptr, nIdx);
    netPtrs_.push_back(&netInst);
    dbNet2Net_[net_ptr] = &netInst;
    nIdx++;
  }

  // Step #4: Initialize Pin
  int numPins = db_iterms.size() + db_bterms.size();
  pinInsts_.resize(numPins);
  pinPtrs_.reserve(numPins);

  int dbIOID   = 0;
  int dbNetID  = 0;
  int dbCellID = 0;

  int  netIDForThisPin = 0;
  int cellIDForThisPin = 0;

  int pIdx = 0;

  for(auto& dbNetPtr : db_nets)
  {
    auto findNet = dbNet2Net_.find(dbNetPtr);
    Net* netPtr = nullptr;

    if(findNet != dbNet2Net_.end())
      netPtr = findNet->second;
    else
    {
      std::cout << "Net " << dbNetPtr->name() << " does not exist in SkyPlaceDB.\n";
      exit(1);
    }

    // For dbBTerm
    Cell* cellFromBTerm = nullptr;
    for(auto& bterm : dbNetPtr->getBTerms())
    {
      Pin& pinInstanceB = pinInsts_[pIdx];
      pinInstanceB = Pin(bterm, pIdx);
      pinInstanceB.setNet(netPtr);
      netPtr->addNewPin(&pinInstanceB);
      pinPtrs_.push_back(&pinInstanceB);
      dbBTerm2Pin_[bterm] = &pinInstanceB;
      pIdx++;

      auto findBTermCell = dbBTerm2Cell_.find(bterm);
      if(findBTermCell != dbBTerm2Cell_.end())
      {
        cellFromBTerm = findBTermCell->second;
        pinInstanceB.setCell(cellFromBTerm);
        cellFromBTerm->addNewPin(&pinInstanceB);
      }
    }

    dbInst* dbInstPtr = nullptr;
    Cell* cellPtr = nullptr;

    // For dbITerm
    for(auto& iterm : dbNetPtr->getITerms())
    {
      Pin& pinInstanceI = pinInsts_[pIdx];
      pinInstanceI = Pin(iterm, pIdx);
      pinInstanceI.setNet(netPtr);
      netPtr->addNewPin(&pinInstanceI);
      pinPtrs_.push_back(&pinInstanceI);
      dbITerm2Pin_[iterm] = &pinInstanceI;
      dbInstPtr = iterm->getInst();
      pIdx++;

      auto findCell = dbInst2Cell_.find(dbInstPtr);
      if(findCell != dbInst2Cell_.end())
      {
        cellPtr = findCell->second;
        pinInstanceI.setCell(cellPtr);
        cellPtr->addNewPin(&pinInstanceI);
      }
      else
      {
        std::cout << "Cell " << dbInstPtr->name() << " does not exist in SkyPlaceDB.\n";
        exit(1);
      }
    }
  }

  // Step #5: Initialize Pin Location and NetBBox
  updateHpwl();

  // Step #6: Make Rows
  int numRow = db_rows.size();
  rowInsts_.resize(numRow);
  rowPtrs_.reserve(numRow);

  int rowIdx = 0;

  for(auto& row : rowInsts_)
  {
    row = Row(db_rows[rowIdx++]);
    rowPtrs_.push_back(&row);
  }

  // Finish -> assign number variables
  numMovable_ = movablePtrs_.size();
  numFixed_   = fixedPtrs_.size();
}

void
SkyPlaceDB::init(std::shared_ptr<dbDatabase> db)
{
  printf("Start SkyPlaceDB Initialization.\n");

  reset();

  dbu_ = db->getTech()->getDbu();
  designName_ = db->getDesign()->name();

  // Step#1: Import dbDatabase to SkyPlaceDB
  importDB(db);
  printf("Import Database         ---> Finished. (Step 1 / %d)\n", numInitStep_);

  // Step#2: Bin Initialization
  createBins();
  printf("Bin Grid Initialization ---> Finished. (Step 2 / %d)\n", numInitStep_); 
  
  // Step#3: Filler Insertion
  createFillers();
  printf("Filler Cell Insertion   ---> Finished. (Step 3 / %d)\n", numInitStep_); 

  // Step#4: Update Density Size
  updateDensitySize();
  printf("Density Size Update     ---> Finished. (Step 4 / %d)\n", numInitStep_); 

  // Step#5: Update Fixed Overlap Area
  updateFixedOverlapArea();
  printf("FixedOverlapArea Update ---> Finished. (Step 5 / %d)\n", numInitStep_); 

  printf("SkyPlaceDB is initilazed successfully!\n");

  printInfo();

	// Check Target Density
  if(targetDensity_ < util_)
	{
    printf("Error - target density (%f) must be larger than utilization (%f).\n",
				    targetDensity_, util_);
		exit(0);
	}
}

void
SkyPlaceDB::updateHpwl()
{
  // Initialize Pin Location
  for(auto& c : cellPtrs_)
    for(auto& p : c->pins())
      p->updatePinLocation(c);

  hpwl_ = 0;

  // Initilize Net BBox
  for(auto& n : netPtrs_)
  {
    n->updateBBox();
    hpwl_ += n->hpwl();
  }
}

bool
SkyPlaceDB::isOutsideDie(const Cell* cell)
{
  bool isOutside = false;

  if(cell->ux() <= die_.lx()) isOutside = true;
  if(cell->lx() >= die_.ux()) isOutside = true;
  if(cell->uy() <= die_.ly()) isOutside = true;
  if(cell->ly() >= die_.uy()) isOutside = true;

  return isOutside;
}

void
SkyPlaceDB::updatePinBound()
{
  for(auto& net : netPtrs_)
  {
    Pin* minPinX = nullptr;
    Pin* minPinY = nullptr;
    Pin* maxPinX = nullptr;
    Pin* maxPinY = nullptr;

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = 0;
    float maxY = 0;

    for(auto& pin : net->pins())
    {
      float cx = pin->cx();
      float cy = pin->cy();

      if(cx <= minX)
      {
        if(minPinX != nullptr)
          minPinX->unsetMinPinX();
        minX = cx;
        pin->setMinPinX();
        minPinX = pin;
      }

      if(cy <= minY)
      {
        if(minPinY != nullptr)
          minPinY->unsetMinPinY();
        minY = cy;
        pin->setMinPinY();
        minPinY = pin;
      }

      if(cx > maxX)
      {
        if(maxPinX != nullptr)
          maxPinX->unsetMaxPinX();
        maxX = cx;
        pin->setMaxPinX();
        maxPinX = pin;
      }

      if(cy > maxY)
      {
        if(maxPinY != nullptr)
          maxPinY->unsetMaxPinY();
        maxY = cy;
        pin->setMaxPinY();
        maxPinY = pin;
      }
    }
  }
}

void
SkyPlaceDB::createBins()
{
  sumTotalInstArea_ = sumFixedArea_ + sumMovableArea_;
  dieArea_ = die_.area();
  density_ = sumTotalInstArea_ / dieArea_;
  util_    = sumMovableArea_   / (dieArea_ - sumFixedArea_);

  avgCellArea_    =  (sumMovableArea_ + sumFixedArea_) / (numMovable_  + numFixed_);
  avgStdCellArea_ =  (sumMovableArea_) / static_cast<float>(numMovable_);

  // Do not use density_, rather use TargetDensity 
  idealBinAreaForAvgCellArea_    = avgCellArea_    / targetDensity_;
  idealBinAreaForAvgStdCellArea_ = avgStdCellArea_ / targetDensity_;

  idealBinCountForAvgCellArea_    = dieArea_ / idealBinAreaForAvgCellArea_;
  idealBinCountForAvgStdCellArea_ = dieArea_ / idealBinAreaForAvgStdCellArea_;

  idealNumBinForAvgCellArea_      = std::sqrt(idealBinCountForAvgCellArea_);
  idealNumBinForAvgStdCellArea_   = std::sqrt(idealBinCountForAvgStdCellArea_);

  //idealNumBin_ = std::ceil(idealNumBinForAvgCellArea_);
  idealNumBin_ = std::ceil(idealNumBinForAvgStdCellArea_);

  float dieX = die_.dx();
  float dieY = die_.dy();

  float aspect_ratio = std::max(dieX / dieY, dieY / dieX);
  int ratio = 1;

  if(aspect_ratio >= 4.0)
    ratio = 4;
  else if(aspect_ratio >= 2.0)
    ratio = 2;
  else 
    ratio = 1;

  int numBin = 4; 
  while(4 * numBin * (numBin * ratio) < idealBinCountForAvgStdCellArea_)
  {
    numBin *= 2;
    if(numBin >= 8192)
      break;
  }
  // numBin /= 2;

  if(dieX > dieY)
  {
    numBinX_ = numBin * ratio;
    numBinY_ = numBin;
  }
  else
  {
    numBinX_ = numBin;
    numBinY_ = numBin * ratio;
  }

  binInsts_.resize(numBinX_ * numBinY_);
  binPtrs_.reserve(numBinX_ * numBinY_);

  float binSizeX = die_.dx() / static_cast<float>(numBinX_);
  float binSizeY = die_.dy() / static_cast<float>(numBinY_);

  float dieUx = die_.ux();
  float dieLx = die_.lx();
  float dieUy = die_.uy();
  float dieLy = die_.ly();

  float lx = dieLx, ly = dieLy;
  int numCreated = 0;

  for(auto& bin : binInsts_)
  {  
    float binWidth  = binSizeX;
    float binHeight = binSizeY;

    int row = numCreated / numBinX_;
    int col = numCreated % numBinX_;

    if(col == (numBinX_ - 1) || (lx + binSizeX > dieUx))
      binWidth  = dieUx - lx;

    if(row == (numBinY_ - 1) || (ly + binSizeY > dieUy))
      binHeight = dieUy - ly;

    bin = Bin(row, col, lx, ly, 
              lx + binWidth, ly + binHeight, 
              targetDensity_);

    lx += binWidth;
    
    if(lx >= dieUx)
    {
      ly += binHeight;
      lx = dieLx;
    }

    binPtrs_.push_back(&bin);
    numCreated++;
  }

  binX_  = binSizeX;
  binY_  = binSizeY;
}

void
SkyPlaceDB::createFillers()
{
  float dxSum = 0;
  float dySum = 0;

  std::vector<float> dxList;
  std::vector<float> dyList;

  dxList.reserve(numMovable_);
  dyList.reserve(numMovable_);

  for(auto& cell : movablePtrs_)
  {
    dxList.push_back(cell->dx());
    dyList.push_back(cell->dy());
  }

  std::sort(dxList.begin(), dxList.end());

  int minIdx = static_cast<int>(static_cast<float>(dxList.size()) * 0.10);
  int maxIdx = static_cast<int>(static_cast<float>(dxList.size()) * 0.90);

  // if numMovable is too small
  if(minIdx == maxIdx) 
  {
    minIdx = 0;
    maxIdx = numMovable_;
  }

  for(int i = minIdx; i < maxIdx; i++)
  {
    dxSum += dxList[i];
    dySum += dyList[i];
  }

  fillerWidth_   = dxSum / static_cast<float>(maxIdx - minIdx);
  fillerHeight_  = dySum / static_cast<float>(maxIdx - minIdx);

  avgFillerArea_  = fillerWidth_ * fillerHeight_;
  whiteSpaceArea_ = (die_.area() - sumFixedArea_);
  fillerArea_     = whiteSpaceArea_ * targetDensity_ 
                  - sumScaledMovableArea_;

  if(fillerArea_ < 0)
  {
    printf("Error - FillerArea is smaller than 0!\n");
    printf("Use higher placement density...\n");
    exit(0);
  }

  numFiller_ = static_cast<int>(fillerArea_ / avgFillerArea_);

  std::mt19937 randVal(0);

  int dieLx = static_cast<int>(die_.lx());
  int dieLy = static_cast<int>(die_.ly());
  int dieWidth  = static_cast<int>(die_.dx());
  int dieHeight = static_cast<int>(die_.dy());

  // Rest of this function is incredibly stupid...
  // TODO: Fix these stupid parts
  for(int i = 0; i < numFiller_; i++)
  {
    auto randX = randVal();
    auto randY = randVal();
    
    // Random distribution over the entire layout
    Cell filler(randX % dieWidth  + dieLx, 
                randY % dieHeight + dieLy,
                fillerWidth_, fillerHeight_);

    cellInsts_.push_back(filler);
  }

  cellPtrs_.clear();
  cellPtrs_.reserve(cellInsts_.size());

  numMovable_ += numFiller_;
  
  movableMacroPtrs_.clear();
  movableMacroPtrs_.reserve(numMovableMacro_);

  movablePtrs_.clear();
  movablePtrs_.reserve(numMovable_);

  fixedPtrs_.clear();
  fixedPtrs_.reserve(numFixed_);

  int fixedID   = 0;
  int movableID = 0;

  for(auto& cell : cellInsts_)
  {
    cellPtrs_.push_back(&cell);
    if(!cell.isFixed())
    {
      cell.setID(movableID++);
      movablePtrs_.push_back(&cell);

      if( cell.isMacro() )
        movableMacroPtrs_.push_back(&cell);
    }
    else
    {
      cell.setID(fixedID++);
      fixedPtrs_.push_back(&cell);
    }
    for(auto & pin : cell.pins())
      pin->setCell(&cell);
  }
}

OverlapBins
SkyPlaceDB::findBin(const Cell* cell)
{
  float lx = cell->lx();
  float ux = cell->ux();
  
  int minX = std::floor((lx - die_.lx()) / binX_);
  int maxX = std::ceil((ux - die_.lx()) / binX_);

  minX = std::max(minX, 0);
  maxX = std::min(numBinX_, maxX);

  std::pair<int, int> minMaxX = std::make_pair(minX, maxX);

  float ly = cell->ly();
  float uy = cell->uy();
  
  int minY = std::floor((ly - die_.ly()) / binY_);
  int maxY = std::ceil((uy - die_.ly()) / binY_);

  minY = std::max(minY, 0);
  maxY = std::min(numBinY_, maxY);

  std::pair<int, int> minMaxY = std::make_pair(minY, maxY);

  return std::make_pair(minMaxX, minMaxY);
}

OverlapBins
SkyPlaceDB::findBinWithDensitySize(const Cell* cell)
{
  float lx = cell->dLx();
  float ux = cell->dUx();
  
  int minX = std::floor((lx - die_.lx()) / binX_);
  int maxX = std::ceil((ux - die_.lx()) / binX_);

  minX = std::max(minX, 0);
  maxX = std::min(numBinX_, maxX);

  std::pair<int, int> minMaxX = std::make_pair(minX, maxX);

  float ly = cell->dLy();
  float uy = cell->dUy();

  int minY = std::floor((ly - die_.ly()) / binY_);
  int maxY = std::ceil((uy - die_.ly()) / binY_);

  minY = std::max(minY, 0);
  maxY = std::min(numBinY_, maxY);

  std::pair<int, int> minMaxY = std::make_pair(minY, maxY);

  return std::make_pair(minMaxX, minMaxY);
}

bool
SkyPlaceDB::isOverlap(const Cell* cell, const Bin* bin)
{
  bool checkX = false;
  bool checkY = false;

  if((cell->lx() >= bin->lx()) && (cell->lx() <= bin->ux()))
    checkX = true;
  if((cell->ux() >= bin->lx()) && (cell->ux() <= bin->ux()))
    checkX = true;
  if((cell->ly() >= bin->ly()) && (cell->ly() <= bin->uy()))
    checkY = true;
  if((cell->uy() >= bin->ly()) && (cell->uy() <= bin->uy()))
    checkY = true;

  return checkX && checkY;
}

void
SkyPlaceDB::moveCellInsideLayout(Cell* cell)
{
  cell->setCenterLocation(getXCoordiInsideLayout(cell), 
                          getYCoordiInsideLayout(cell));
}

float
SkyPlaceDB::getXCoordiInsideLayout(const Cell* cell)
{
  float newCx = cell->cx();

  if(cell->lx() < die_.lx())
    newCx = die_.lx() + cell->dx()/2;
  if(cell->ux() > die_.ux())
    newCx = die_.ux() - cell->dx()/2;

  return newCx;
}

float
SkyPlaceDB::getYCoordiInsideLayout(const Cell* cell)
{
  float newCy = cell->cy();

  if(cell->ly() < die_.ly())
    newCy = die_.ly() + cell->dy()/2;
  if(cell->uy() > die_.uy())
    newCy = die_.uy() - cell->dy()/2;

  return newCy;
}

float
SkyPlaceDB::getXCoordiInsideLayout(const Cell* cell, float cx)
{
  float newCx = cx;

  if(cx - cell->dx()/2 < die_.lx())
    newCx = die_.lx() + cell->dx()/2;
  if(cx + cell->dx()/2 > die_.ux())
    newCx = die_.ux() - cell->dx()/2;
  return newCx;
}

float
SkyPlaceDB::getYCoordiInsideLayout(const Cell* cell, float cy)
{
  float newCy = cy;

  if(cy - cell->dy()/2 < die_.ly())
    newCy = die_.ly() + cell->dy()/2;
  if(cy + cell->dy()/2 > die_.uy())
    newCy = die_.uy() - cell->dy()/2;
  return newCy;
}

float
SkyPlaceDB::getXDensityCoordiInsideLayout(const Cell* cell)
{
  float newCx = cell->cx();

  if(cell->dLx() < die_.lx())
    newCx = die_.lx() + cell->dDx()/2;
  if(cell->dUx() > die_.ux())
    newCx = die_.ux() - cell->dDx()/2;

  return newCx;
}

float
SkyPlaceDB::getYDensityCoordiInsideLayout(const Cell* cell)
{
  float newCy = cell->cy();

  if(cell->dLy() < die_.ly())
    newCy = die_.ly() + cell->dDy()/2;
  if(cell->dUy() > die_.uy())
    newCy = die_.uy() - cell->dDy()/2;

  return newCy;
}

void
SkyPlaceDB::updateDensitySize()
{
  float scaleX = 0, scaleY = 0;
  float densityW = 0, densityH = 0;

  for(auto& cell: cellPtrs_)
  {
    if(cell->dx() > DENSITY_SCALE * binX_)
    {
      scaleX   = 1.0;
      densityW = cell->dx();
    }
    else
    {
      scaleX   = cell->dx() / (DENSITY_SCALE * binX_);
      densityW = DENSITY_SCALE * binX_;
    }

    if(cell->dy() > DENSITY_SCALE * binY_)
    {
      scaleY   = 1.0;
      densityH = cell->dy();
    }
    else
    {
      scaleY   = cell->dy() / (DENSITY_SCALE * binY_);
      densityH = DENSITY_SCALE * binY_;
    }
    cell->setDensitySize(densityW, densityH, scaleX * scaleY);
  }
}

void
SkyPlaceDB::updateMacroDensityWeight(float macroWeight)
{
  if(numMovableMacro_ == 0)
    return;

  float avgMacroArea = sumMovableMacroArea_ 
                     / static_cast<float>(numMovableMacro_);

  float maxMacroArea = 0.0;
  float minMacroArea = FLT_MAX;

  float sumMacroAreaVarianceSquare = 0.0;

  for(auto& macro : movableMacroPtrs_)
  {
    float area = macro->area();
    sumMacroAreaVarianceSquare += (area - avgMacroArea) 
                                * (area - avgMacroArea);
  
    if(area > maxMacroArea)
      maxMacroArea = area;
    if(area < minMacroArea)
      minMacroArea = area;
  }

  float std_dev = std::sqrt( sumMacroAreaVarianceSquare / 
                             static_cast<float>( numMovableMacro_ ) );

  float maxRatio = maxMacroArea / avgMacroArea;
  float minRatio = minMacroArea / avgMacroArea;

  float ratio_std_macro = sumMovableMacroArea_ / sumMovableStdArea_;

  std::cout << "Avg MacroArea   : " << avgMacroArea << std::endl;
  std::cout << "Std_Dev         : " << std_dev << std::endl;
  std::cout << "Max Ratio       : " << maxRatio << std::endl;
  std::cout << "Min Ratio       : " << minRatio << std::endl;
  std::cout << "A_Macro / A_Std : " << ratio_std_macro << std::endl;
  std::cout << "Density Weight  : " << macroWeight << std::endl;
  std::cout << "Max MacroArea   : " << maxMacroArea << std::endl;
  std::cout << "Min MacroArea   : " << minMacroArea << std::endl;

  // if(targetDensity_ < 1.0 && (maxRatio > 1.2 || minRatio < 0.2) )
  for(auto& macro : movableMacroPtrs_)
  {
    //if(macro->dx() > die_.dx() * 0.2 || macro->dy() > die_.dy() * 0.2)
    {
      //printf("large Macro!\n");
      float weight = macro->densityScale();
      macro->setDensityScale(weight * macroWeight); // 1.09??
    }
  }
}

void
SkyPlaceDB::updateFixedOverlapArea()
{
  for(auto& bin : binPtrs_)
    bin->setFixedArea(0);

  for(auto& cell : fixedPtrs_) 
  {
    if(isOutsideDie(cell))
      continue;

    OverlapBins ovBins = findBin(cell);
    int minX  = ovBins.first.first;
    int maxX  = ovBins.first.second;
    int minY  = ovBins.second.first;
    int maxY  = ovBins.second.second;

    for(int i = minX; i < maxX; i++)
    {
      for(int j = minY; j < maxY; j++)
      {
        Bin* bin = binPtrs_[j * numBinX_ + i];
        const float overlapArea = getOverlapArea(bin, cell);
        bin->addFixedArea(overlapArea * bin->targetDensity());
        // FixedArea should be scaled-down with target density
        // (according to the OpenROAD RePlAce comment)
      }
    }
  }
}

void
SkyPlaceDB::printInfo() const 
{
  using namespace std;

  float initHpwl = hpwl_ / static_cast<float>(dbu_);
  // Zero Check is done in init()

  cout << endl;
  cout << "*** Summary of SkyPlaceDB ***" << endl;
  cout << "---------------------------------------------" << endl;
  cout << " DESIGN NAME        : " << designName_         << endl;
  cout << " NUM CELL (TOTAL)   : " << numMovable_ + numFixed_ << endl;
  cout << " NUM CELL (MOVABLE) : " << numMovable_         << endl;
  cout << " NUM CELL (FIXED)   : " << numFixed_           << endl;
  cout << " NUM CELL (FILLER)  : " << numFiller_          << endl;
  cout << " NUM MOVABLE MACRO  : " << numMovableMacro_    << endl;
  cout << " NUM NET            : " << numNet()            << endl;
  cout << " NUM PIN            : " << numPin()            << endl;
  cout << " NUM IO             : " << numIO()             << endl;
  cout << " NUM BIN   (IDEAL)  : " << idealNumBin_        << endl;
  cout << " NUM BIN   (TOTAL)  : " << numBinX_ * numBinY_ << endl;
  cout << " NUM BIN_X (USED)   : " << numBinX_            << endl;
  cout << " NUM BIN_Y (USED)   : " << numBinY_            << endl;
  cout << " BIN WIDTH          : " << binX_               << endl;
  cout << " BIN HEIGHT         : " << binY_               << endl;
  cout << " FILLER WIDTH       : " << fillerWidth_        << endl;
  cout << " FILLER HEIGHT      : " << fillerHeight_       << endl;
  cout << " FILLER AREA        : " << avgFillerArea_      << endl;
  cout << " AREA (TOTAL)       : " << sumTotalInstArea_   << endl;
  cout << " AREA (MOVABLE)     : " << sumMovableArea_     << endl;
  cout << " AREA (STD)         : " << sumMovableStdArea_  << endl;
  cout << " AREA (MACRO)       : " << sumMovableMacroArea_<< endl;
  cout << " AREA (FIXED)       : " << sumFixedArea_       << endl;
  cout << " AREA (FILLER)      : " << fillerArea_         << endl;
  cout << " AREA (WHITESPACE)  : " << whiteSpaceArea_     << endl;
  cout << " AREA (CORE)        : " << dieArea_            << endl;
  cout << " TARGET DENSITY     : " << targetDensity_ * 100.0 << "%\n";
  cout << " DENSITY            : " << density_ * 100.0    << "%\n";
  cout << " UTIL               : " << util_    * 100.0    << "%\n";
  cout << " INITIAL HPWL       : " << initHpwl << endl;
  cout << "---------------------------------------------" << endl;
  cout << endl;
}

void
SkyPlaceDB::printFixedOverlapArea()
{
  std::ofstream output;
  output.open("fixedOverlapArea.txt");

  for(auto& bin : binPtrs_)
    output << bin->fixedArea() << std::endl;

  output.close();
}

void
SkyPlaceDB::printBinDensity()
{
  std::ofstream output;
  output.open("BinDensity.txt");

  for(auto& bin : binPtrs_)
  {
    output << bin->col() << "," << bin->row() << ",";
    output << bin->density() << std::endl;
  }

  output.close();
}

void
SkyPlaceDB::printBinPotential()
{
  std::ofstream output;
  output.open("Potential.txt");

  for(auto& bin : binPtrs_)
  {
    output << bin->col() << "," << bin->row() << ",";
    output << bin->potential() << std::endl;
  }

  output.close();
}

void
SkyPlaceDB::printBinElectroFieldXY()
{
  std::ofstream output;
  output.open("ElectroField.txt");

  for(auto& bin : binPtrs_)
  {
    output << bin->col() << "," << bin->row() << ",";
    output << bin->electroForceX() << ",";
    output << bin->electroForceY() <<  std::endl;
  }

  output.close();
}

void
SkyPlaceDB::debugCell() const
{
  printf("=== Debug Cell ===\n");

  for(auto &cell : cellPtrs_)
    printCellInfo(cell);
}

void
SkyPlaceDB::debugNet() const
{
  printf("=== Debug Net ===\n");

  for(auto &net : netPtrs_)
    printNetInfo(net);
}

void
SkyPlaceDB::printCellInfo(Cell* cell) const
{
  std::cout << "Cell ID: " << cell->id() << std::endl;
  printf("(%f, %f) - (%f, %f)\n", 
      cell->lx(), cell->ly(), cell->ux(), cell->uy() );

  printf("Pin Coordinate\n");
  for(auto &pin: cell->pins())
    printf("ID: %d (%f, %f) Net: %d\n", pin->id(), pin->cx(), pin->cy(), pin->net()->id());
}

void
SkyPlaceDB::printNetInfo(Net* net) const
{
  std::cout << "Net ID: " << net->id() << std::endl;
  printf("(%f, %f) - (%f, %f)\n", 
      net->lx(), net->ly(), net->ux(), net->uy() );

  printf("Pin Coordinate\n");
  for(auto &pin: net->pins())
    printf("ID: %d (%f, %f) Cell: %d\n", pin->id(), pin->cx(), pin->cy(), pin->cell()->id());
}

} // namespace skyplace

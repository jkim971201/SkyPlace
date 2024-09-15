#ifndef SKYPLACE_DB_H
#define SKYPLACE_DB_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#define SQRT2 1.414213562373095048801L
#define DENSITY_SCALE SQRT2

namespace db 
{
  class dbDatabase;
  class dbTech;
  class dbDesign;
  class dbTypes;
  class dbDie;
  class dbInst;
  class dbNet;
  class dbITerm;
  class dbBTerm;
  class dbRow;
}

namespace skyplace 
{

using namespace db;
using OverlapBins = std::pair<std::pair<int, int>, std::pair<int, int>>;

// C- Style Struct Version For CUDA Kernel Functions
typedef struct OVBIN
{
  int lxID = 0;
  int lyID = 0;
  int uxID = 0;
  int uyID = 0;
} OVBIN;

class Pin;
class Net;

class Cell
{
  public:

    Cell();
    Cell(dbInst* inst);
    Cell(dbBTerm* bterm);
    Cell(float cx, float cy, float dx, float dy);
    // Contructor for Filler Cell

    // Getters
    int   id() const { return id_;           }
    float lx() const { return cx_ - dx_ / 2; }
    float ly() const { return cy_ - dy_ / 2; }
    float ux() const { return cx_ + dx_ / 2; }
    float uy() const { return cy_ + dy_ / 2; }

    float cx() const { return cx_; }
    float cy() const { return cy_; }
    float dx() const { return dx_; }
    float dy() const { return dy_; }

    float dLx() const { return cx_ - dDx_ / 2; }
    float dLy() const { return cy_ - dDy_ / 2; }
    float dUx() const { return cx_ + dDx_ / 2; }
    float dUy() const { return cy_ + dDy_ / 2; }
    float dDx() const { return dDx_; }
    float dDy() const { return dDy_; }

    float area() const { return dx_ * dy_; }
    float densityScale() const { return densityScale_; }

    bool isIO     () const { return isIO_;       }
    bool isFixed  () const { return isFixed_ ;   }
    bool isMacro  () const { return isMacro_ ;   }
    bool isFiller () const { return isFiller_;   }

    int  clusterID() const { return cluster_id_; }

    const std::vector<Pin*>& pins() const { return pins_; }

    // These will return nullptr,
    // if there is no corresponding dbInst (e.g. fillerCell)
    dbInst* dbInstPtr() const { return dbInst_;  }

    // If this cell is from dbInst,
    // this function will return nullptr
    dbBTerm* dbBTermPtr() const { return dbBTerm_;  }

    // Setters
    void setID       (int id) { id_  = id; }
    void setClusterID(int id) { cluster_id_ = id; }

    void setCenterLocation (float newCx,  float newCy);
    void setDensitySize    (float dWidth, float dHeight, float dScale);
    void setDensityScale   (float dScale) { densityScale_ = dScale; }

    void addNewPin(Pin* pin) { pins_.push_back(pin); }

  private:

    dbInst*  dbInst_;
    dbBTerm* dbBTerm_;

    // ID is required to compute Laplacian
    // Fixed and Movables seperately
    int id_;
    int cluster_id_;

    float cx_;
    float cy_;

    float dx_;
    float dy_;

    float dDx_;
    float dDy_;

    float densityScale_;

    bool isIO_;
    bool isMacro_;
    bool isFixed_;
    bool isFiller_;

    std::vector<Pin*> pins_;
};

class Net
{
  public:
    Net();

    Net(dbNet* net, int id); 

    // Getters
    float     lx() const { return lx_;                     }
    float     ly() const { return ly_;                     }
    float     ux() const { return ux_;                     }
    float     uy() const { return uy_;                     }
    float     cx() const { return (ux_ + lx_) / 2.0;       }
    float     cy() const { return (uy_ + ly_) / 2.0;       }
    float     dx() const { return ux_ - lx_;               }
    float     dy() const { return uy_ - ly_;               }
    int       id() const { return id_;                     }
    int      deg() const { return pins_.size();            }
    float   hpwl() const { return (ux_ - lx_ + uy_ - ly_); } 
    float weight() const { return weight_;                 }

    const std::vector<Pin*>& pins() const { return pins_; }

    dbNet* dbNetPtr() const { return dbNet_; }

    // Setters
    void setWeight  (float weight) { weight_ = weight; }
    void addNewPin  (Pin* pin) { pins_.push_back(pin); }

    void updateBBox();

  private:

    dbNet* dbNet_;

    int id_;
    float lx_; // Lx of BBox
    float ly_; // Ly of BBox
    float ux_; // Ux of BBox
    float uy_; // Uy of BBox

    float weight_;

    std::vector<Pin*> pins_;
};

class Pin
{
  public:

    Pin();
    Pin(dbBTerm* bterm, int id); // Constructor for dbBTerm
    Pin(dbITerm* iterm, int id); // Constructor for dbITerm

    // Getters
    int     id() const { return id_;   }
    float   cx() const { return cx_;   }
    float   cy() const { return cy_;   }
    bool  isIO() const { return isIO_; }

    Net*   net() const { return net_;  }
    Cell* cell() const { return cell_; } 
    // returns nullptr if Pin is made from dbBTerm (== IO pin)

    float offsetX() const { return offsetX_; }
    float offsetY() const { return offsetY_; }

    bool isMinPinX() const { return isMinPinX_; }
    bool isMinPinY() const { return isMinPinY_; }
    bool isMaxPinX() const { return isMaxPinX_; }
    bool isMaxPinY() const { return isMaxPinY_; }

    // Setters
    void setNet (Net*   net) { net_  =  net; }
    void setCell(Cell* cell) { cell_ = cell; }

    void updatePinLocation(Cell* cell);

    void   setMinPinX() { isMinPinX_ =  true; } 
    void   setMinPinY() { isMinPinY_ =  true; }
    void   setMaxPinX() { isMaxPinX_ =  true; }
    void   setMaxPinY() { isMaxPinY_ =  true; }
    void unsetMinPinX() { isMinPinX_ = false; }
    void unsetMinPinY() { isMinPinY_ = false; }
    void unsetMaxPinX() { isMaxPinX_ = false; }
    void unsetMaxPinY() { isMaxPinY_ = false; }

  private:

    int   id_;
    float cx_;
    float cy_;
    bool  isIO_;

    float offsetX_;
    float offsetY_;

    bool isMinPinX_;
    bool isMinPinY_;
    bool isMaxPinX_;
    bool isMaxPinY_;

    Net* net_;
    Cell* cell_;
};

// In the current implementation,
// lx ly ux uy are coodinates of Core,
// where we just assume that die ly ly is (0, 0).
// So, if (lx, ly) of DEF DIEAREA is not (0, 0),
// then this will make bug.
class Die
{
  public:
    Die() { lx_ = ly_ = ux_ = uy_ = 0; }

    // These are core Lx Ly Ux Uy
    void setLx(float val) { lx_ = val; }
    void setLy(float val) { ly_ = val; }
    void setUx(float val) { ux_ = val; }
    void setUy(float val) { uy_ = val; }

    float lx() const { return lx_; }
    float ly() const { return ly_; }
    float ux() const { return ux_; }
    float uy() const { return uy_; }

    float cx() const { return (lx_ + ux_) / 2 ; }
    float cy() const { return (ly_ + uy_) / 2 ; }

    float dx() const { return ux_ - lx_; }
    float dy() const { return uy_ - ly_; }

    float area() const { return (ux_ - lx_) * (uy_ - ly_); }

  private:
    float lx_;
    float ly_;
    float ux_;
    float uy_;
};

class Bin
{
  public:
    Bin();
    Bin(int row, int col, 
        float lx, float ly, 
        float ux, float uy, 
        float targetDensity);

    int row() const { return row_; }
    int col() const { return col_; }

    float lx() const { return lx_; }
    float ly() const { return ly_; }
    float ux() const { return ux_; }
    float uy() const { return uy_; }
    float dx() const { return ux_ - lx_; }
    float dy() const { return uy_ - ly_; }
    float cx() const { return (ux_ + lx_)/2; }
    float cy() const { return (uy_ + ly_)/2; }

    float density       () const { return density_;          }
    float targetDensity () const { return targetDensity_;    }

    // Local Lagrange Multipler
    float lambda()        const { return lambda_; }

    float potential()     const { return electroPotential_; }

    float electroForceX() const { return electroForceX_;    }
    float electroForceY() const { return electroForceY_;    }

    void setLambda(float lambda) { lambda_ = lambda; }

    void setElectroPotential(float potential) { electroPotential_ = potential; }

    float area() const { return (ux_ - lx_) * (uy_ - ly_); }

    float fixedArea   () const { return fixedArea_;   }
    float movableArea () const { return movableArea_; }
    float fillerArea  () const { return fillerArea_;  }

    void setDensity     (float density    ) { density_     = density;      }

    void setFixedArea   (float fixedArea  ) { fixedArea_   = fixedArea;    }
    void setMovableArea (float movableArea) { movableArea_ = movableArea;  }
    void setFillerArea  (float fillerArea ) { fillerArea_  = fillerArea;   }

    void addFixedArea   (float fixedArea  ) { fixedArea_   += fixedArea;   }
    void addMovableArea (float movableArea) { movableArea_ += movableArea; }
    void addFillerArea  (float fillerArea ) { fillerArea_  += fillerArea;  }

  private:
    int row_;
    int col_;

    float lx_;
    float ly_;
    float ux_;
    float uy_;

    float lambda_;

    float fixedArea_;
    float movableArea_;
    float fillerArea_;

    float density_;
    float targetDensity_;

    float electroPotential_;

    float electroForceX_;
    float electroForceY_;
};

class Row
{
  public:

    Row() {};
    Row(dbRow* row);

    // Getters
    int numSiteX() const { return numSiteX_; }
    int numSiteY() const { return numSiteY_; }
    // numSiteY is usually 1

    int siteWidth() const { return siteWidth_; }

    int lx() const { return lx_; }
    int ly() const { return ly_; }
    int ux() const { return ux_; }
    int uy() const { return uy_; }
    int dx() const { return ux_ - lx_; } 
    int dy() const { return uy_ - ly_; }

    int stepX() const { return stepX_; }
    int stepY() const { return stepY_; }

  private:
  
    int numSiteX_;
    int numSiteY_; // most cases, this is just 1

    int siteWidth_;

    int lx_;
    int ly_;
    int ux_;
    int uy_;

    int stepX_;
    int stepY_;
};

class SkyPlaceDB 
{
  public: 

    SkyPlaceDB(); // Constructor (Just for Initialization)

    // Important APIs
    void reset();
    void exportDB(std::shared_ptr<dbDatabase> _dbDatabase); 
    void init(std::shared_ptr<dbDatabase> db); // Initialization to run placement 
    void setTargetDensity (float density) { targetDensity_ = density; }
    void setNumBinX       (int   numBinX) { numBinX_       = numBinX; }     
    void setNumBinY       (int   numBinY) { numBinY_       = numBinY; }     

    void updateMacroDensityWeight(float macroWeight);

    // Used by Initial Placer
    void moveCellInsideLayout(Cell* cell); 
    void setNumCluster(int numCluster) { numCluster_  = numCluster; }

    // Getters
    const std::string&   designName() const { return designName_;  }
    const std::string&   designDir () const { return  designDir_;  }
    const std::vector<Cell*>& cells() const { return   cellPtrs_;  }
    const std::vector<Net*>&   nets() const { return    netPtrs_;  }
    const std::vector<Pin*>&   pins() const { return    pinPtrs_;  }
    const std::vector<Bin*>&   bins() const { return    binPtrs_;  }
    const std::vector<Row*>&   rows() const { return    rowPtrs_;  }

    const std::vector<Cell*>&   fixedCells() const { return fixedPtrs_;   }
    const std::vector<Cell*>& movableCells() const { return movablePtrs_; }

    Die* die() const { return diePtr_; }

    int numFixed   () const { return numFixed_;        }
    int numMovable () const { return numMovable_;      }
    int numFiller  () const { return numFiller_;       }
    int numNet     () const { return netPtrs_.size();  }
    int numPin     () const { return pinPtrs_.size();  }
    int numIO      () const { return numIO_;           }
    int numCluster () const { return numCluster_;      }
    int numMacro   () const { return numMovableMacro_; }
    int getDbu     () const { return dbu_;             }

    float util                 () const { return util_;                 }
    float density              () const { return density_;              }
    float targetDensity        () const { return targetDensity_;        }
    float sumFixedArea         () const { return sumFixedArea_;         }
    float sumMovableArea       () const { return sumMovableArea_;       }
    float sumScaledMovableArea () const { return sumScaledMovableArea_; }
    // sumMovableArea does not include fillerArea
    // sumScaledMovableArea is used for createFillers
    // sumScaledMovableArea = sumStdCells + sumMacroArea * targetDensity

    float getHPWL () const { return hpwl_;    }
    int   numBinX () const { return numBinX_; } // ex) 512x256 Grid -> numBinX = 512
    int   numBinY () const { return numBinY_; } // ex) 512x256 Grid -> numBinY = 256
    float binX    () const { return binX_;    }
    float binY    () const { return binY_;    }

    void  updateHpwl();
    void  updatePinBound(); // For B2B Model (CG-based Initialization)
  
    // To plot density gradient arrows, these will be delivered to Painter
    std::vector<float>& densityGradX() { return densityGradX_; }
    std::vector<float>& densityGradY() { return densityGradY_; }

    // For Debugging
    void debugCell() const;
    void debugNet () const;

  private:

    std::string designName_;
    std::string designDir_;

    float targetDensity_;

    int dbu_; // This is from dbDatabase->dbTech->getDbu()

    // Number of Objects
    int numStdCells_;
    int numMacro_;
    int numFixed_;
    int numMovable_;
    int numMovableMacro_;
    int numFiller_;
    int numBinX_;
    int numBinY_;
    int numIO_;
    int numCluster_;

    // For BookShelf format, this will be used to define macro cell
    // Technology-Related
    float maxRowHeight_;

    // Sub-Routines of init                         
    void importDB(std::shared_ptr<dbDatabase> _dbDatabase); // Step#1
    void createBins            ();  // Step#2
    void createFillers         ();  // Step#3
    void updateDensitySize     ();  // Step#4
    void updateFixedOverlapArea();  // Step#5
    int  numInitStep_;              // For messaging

    // Bin-related Methods
    OverlapBins findBin                  (const Cell* cell);
    OverlapBins findBinWithDensitySize   (const Cell* cell);

    bool isOutsideDie                    (const Cell* cell);
    bool isOverlap                       (const Cell* cell, const Bin* bin);

    float getXCoordiInsideLayout         (const Cell* cell);
    float getYCoordiInsideLayout         (const Cell* cell);

    float getXCoordiInsideLayout         (const Cell* cell, float x);
    float getYCoordiInsideLayout         (const Cell* cell, float y);

    float getXDensityCoordiInsideLayout  (const Cell* cell);
    float getYDensityCoordiInsideLayout  (const Cell* cell);

    // Bin-related
    float sumFixedArea_;
    float sumMovableArea_;
    float sumMovableStdArea_;
    float sumMovableMacroArea_;
    float sumScaledMovableArea_;

    float sumTotalInstArea_;
    float dieArea_;
    float density_;
    float util_;

    float avgCellArea_;
    float avgStdCellArea_;
    float idealBinAreaForAvgCellArea_;
    float idealBinAreaForAvgStdCellArea_;
    float idealBinCountForAvgCellArea_;
    float idealBinCountForAvgStdCellArea_;
    float idealNumBinForAvgCellArea_;
    float idealNumBinForAvgStdCellArea_;

    int   idealNumBin_;
    float binX_;
    float binY_;

    float avgFillerArea_;
    float whiteSpaceArea_;
    float fillerArea_;

    float fillerWidth_;
    float fillerHeight_;
    ///////////////////////////

    float hpwl_;

    // Database
    std::vector<Cell*> cellPtrs_; 
    std::vector<Cell>  cellInsts_; 

    std::vector<Cell*> movableMacroPtrs_; 
    std::vector<Cell*> fixedPtrs_; 
    std::vector<Cell*> movablePtrs_; 

    std::vector<Net*>  netPtrs_; 
    std::vector<Net>   netInsts_; 

    std::vector<Pin*>  pinPtrs_; 
    std::vector<Pin>   pinInsts_; 

    std::vector<Bin*>  binPtrs_;
    std::vector<Bin>   binInsts_;

    std::vector<Row*>  rowPtrs_;
    std::vector<Row>   rowInsts_;

    std::unordered_map<dbInst*,  Cell*> dbInst2Cell_;
		std::unordered_map<dbBTerm*, Cell*> dbBTerm2Cell_;
    std::unordered_map<dbNet*,    Net*> dbNet2Net_;
		std::unordered_map<dbITerm*,  Pin*> dbITerm2Pin_;
		std::unordered_map<dbBTerm*,  Pin*> dbBTerm2Pin_;

    Die  die_;
    Die* diePtr_;

    // To plot density gradient arrows...
    std::vector<float> densityGradX_;
    std::vector<float> densityGradY_;

    // For Debug
    void printInfo() const;
    void printBinDensity();
    void printBinPotential();
    void printBinElectroFieldXY();
    void printFixedOverlapArea();

    void printCellInfo(Cell* cell) const;
    void printNetInfo(Net* net) const;
};

} // namespace skyplace 

#endif

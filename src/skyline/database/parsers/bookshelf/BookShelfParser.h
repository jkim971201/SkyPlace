#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <climits>   // For std::numeric_limits
#include <filesystem>

#define MAX_FILE_NAME 256

// By JKIm
// 02. 28. 2023
// BookShelfDB does not have ability to do anything itself
// it's just a container of raw values from original
// input file ( .nodes .pl .scl .nets ) 
// You don't have to worry about some APIs are missing
// because these are enough for building a PlacerDB
// ** stupid implementations are remaining ...

namespace bookshelf
{

class BsPin;
class BsNet;
class BsCell;

class BookShelfDB;

class BsDie // should it be named as BsCore?
{
  public:
    BsDie() {}

    // Getters
    int ux() const { return ux_; }
    int uy() const { return uy_; }
    int lx() const { return lx_; }
    int ly() const { return ly_; }

    int dx() const { return ux_ - lx_; }
    int dy() const { return uy_ - ly_; }

    int area() const { return dx() * dy(); }

    // Setters
    void setUxUy(int ux, int uy) 
    {
      ux_ = ux;
      uy_ = uy;
    };

    void setLxLy(int lx, int ly) 
    {
      lx_ = lx;
      ly_ = ly;
    };

  private:
    int ux_;
    int uy_;
    int lx_;
    int ly_;
};

class BsRow
{
  public:
    BsRow(int idx, 
          int ly, 
          int rowHeight, 
          int siteWidth, 
          int siteSpacing, 
          int offsetX, 
          int numSites);

    // Getters
    int id() const { return idx_;                   }
    int dx() const { return rowWidth_;              }
    int dy() const { return rowHeight_;             }
    int lx() const { return offsetX_;               }
    int ly() const { return ly_;                    }
    int ux() const { return offsetX_ + rowWidth_;   }
    int uy() const { return ly_ + rowHeight_;       }

    int siteWidth   () const { return siteWidth_;   }
    int siteSpacing () const { return siteSpacing_; }
    int numSites    () const { return numSites_;    }

  private:
    int  idx_;

    // These are written in .scl file
    int  ly_;           // 1. Coordinate of ly
    int  rowHeight_;    // 2. Height
    int  siteWidth_;    // 3. Site Width
    int  siteSpacing_;  // 4. Site Spacing
    bool siteOrient_;   // 5. Siteorient
    bool siteSymmetry_; // 6. Sitesymmetry
    int  offsetX_;      // 7. SubrowOrigin
    int  numSites_;     // 8. NumSites

    // RowWidth = numSites * siteWidth (siteWidth == siteSpacing?)
    int  rowWidth_; 
};

class BsCell
{
  public:
    BsCell();
    BsCell(std::string& name, 
           int id,
           int width, 
           int height, 
           bool Terminal, 
           bool TerminalNI);

    // Getters
    const std::string& name() const { return name_; }

    int id() const { return id_; }

    int lx() const { return lx_; }
    int ly() const { return ly_; }
    int ux() const { return ux_; }
    int uy() const { return uy_; }

    int dx() const { return dx_; }
    int dy() const { return dy_; }

    int area() const { return dx_ * dy_; }

    char orient()       const { return orient_;       }
    bool isTerminal()   const { return isTerminal_;   }
    bool isTerminalNI() const { return isTerminalNI_; }

    bool isFixed()      const { return isFixed_;      }
    bool isFixedNI()    const { return isFixedNI_;    }

    // Setters
    void setXY(int x, int y)
    {
      lx_ = x;
      ly_ = y;
      ux_ = lx_ + dx_;
      uy_ = ly_ + dy_;
    }

    void setFixed()           
    { 
      isFixed_   = true;  
      isFixedNI_ = false; 
    }

    void setFixedNI() 
    { 
      isFixed_   = true;
      isFixedNI_ = true; 
    }

    void setOrient(char orient)
    {
      orient_ = orient;
    }

    void addNewPin(BsPin* pin) { pins_.push_back(pin); }

    const std::vector<BsPin*>& pins() const { return pins_; }

  private:
    std::string name_;

    int id_;

    int lx_;
    int ly_;
    int ux_;
    int uy_;

    int dx_;
    int dy_;

    char orient_;

    bool isTerminal_;
    bool isTerminalNI_;

    bool isFixed_;
    bool isFixedNI_;

    std::vector<BsPin*> pins_;
};

class BsPin
{
  public:
    BsPin(BsCell* cell, int netID, double offsetX_, double offsetY_, char IO);

    int netID() const { return netID_; }

    BsCell* cell() const { return cell_; }
    BsNet*  net()  const { return net_;  }

    double offsetX() const { return offsetX_; }
    double offsetY() const { return offsetY_; }

    char IO() const { return io_; }

    void setNet(BsNet* net) { net_ = net; }

  private:
    int netID_;

    BsCell* cell_;
    BsNet* net_;
 
    double offsetX_;
    double offsetY_;

    // if Sink    == I
    // if Driver  == O
    // if Both(?) == B
    char io_; 
};

class BsNet
{
  public:
    BsNet(int id) : id_(id), name_ ("")
    {
      pins_.clear();
    }

    void addNewPin(BsPin* pin) { pins_.push_back(pin); }
    void setName(const std::string& name) { name_ = name; }

    int id() const { return id_; }
    int getDegree() const { return pins_.size(); }

    const std::string& name() const { return name_; }
    const std::vector<BsPin*>& pins() const { return pins_; }

  private:
    int id_;
    std::string name_;
    std::vector<BsPin*> pins_;
};

class BookShelfDB
{
  public:
    BookShelfDB();
    BookShelfDB(int numNodes);

    void makeBsCell(std::string& name, int lx, int ly, bool Terminal, bool TerminalNI);
    void makeBsNet(int netID, const std::string& name);
    void makeBsPin(BsCell* cell, int netID, double offX, double offY, char IO);

    void makeBsRow(int idx, int ly, int rowHeight, 
                   int siteWidth, int siteSpacing, int offsetX, int numSites);

    const std::vector<BsCell*>& cellVector() const { return cellPtrs_; }
    const std::vector<BsRow*>&   rowVector() const { return  rowPtrs_; }
    const std::vector<BsNet*>&   netVector() const { return  netPtrs_; }
    const std::vector<BsPin*>&   pinVector() const { return  pinPtrs_; }

    BsNet*  getBsNetByID(int id)              { return netMap_[id];    }
    BsCell* getBsCellByName(std::string name) { return cellMap_[name]; }
    BsRow*  getBsRowbyID(int id)              { return rowMap_[id];    }

    BsDie* getDie() const { return bsDiePtr_; };

    void buildBsCellMap();
    void buildBsRowMap();
    void finishPinsAndNets();

    void verifyMap();
    void verifyVec();
    void verifyPtrVec();
 
    int numRows()    const { return numRows_;    }
    int numCells()   const { return numBsCells_; }
    int numFixed()   const { return numFixed_;   }
    int numFixedNI() const { return numFixedNI_; }
    int numMovable() const { return numMovable_; }

    int getDieWidth()  const { return bsDiePtr_->dx(); }
    int getDieHeight() const { return bsDiePtr_->dy(); }

    int rowHeight() const { return rowHeight_; }
    void setHeight(int rowHeight) { rowHeight_ = rowHeight; }

  private:

    int numRows_;
    int numBsCells_;
    int numFixed_;
    int numFixedNI_;
    int numMovable_;
    int numInst_;

    int rowHeight_;

    std::vector<BsRow*> rowPtrs_; 
    std::vector<BsRow>  rowInsts_; 

    std::unordered_map<int, BsRow*> rowMap_;

    std::vector<BsPin*> pinPtrs_; 
    std::vector<BsPin>  pinInsts_; 

    std::vector<BsNet*> netPtrs_; 
    std::vector<BsNet>  netInsts_; 

    std::unordered_map<int, BsNet*> netMap_;

    std::vector<BsCell*> cellPtrs_; 
    std::vector<BsCell>  cellInsts_; 

    std::unordered_map<std::string, BsCell*> cellMap_;

    BsDie bsDie_;
    BsDie* bsDiePtr_;
};

class BookShelfParser
{
  public:
    BookShelfParser();

    // APIs
    void parse     (const std::filesystem::path& aux_name);
    void printInfo ();

    // Getters
    int rowHeight  () const { return bookShelfDB_->rowHeight();   }
    int numFixed   () const { return bookShelfDB_->numFixed();    }
    int numFixedNI () const { return bookShelfDB_->numFixedNI();  }
    int numMovable () const { return bookShelfDB_->numMovable();  }
    int numCells   () const { return bookShelfDB_->numCells();    }
    int numRows    () const { return bookShelfDB_->numRows();     }
    const BsDie* getDie() const { return bookShelfDB_->getDie();  }

    const std::vector<BsCell*>& cellVector() const { return bookShelfDB_->cellVector(); }
    const std::vector<BsRow*>&   rowVector() const { return bookShelfDB_->rowVector();  }
    const std::vector<BsNet*>&   netVector() const { return bookShelfDB_->netVector();  }
    const std::vector<BsPin*>&   pinVector() const { return bookShelfDB_->pinVector();  }

    const char* getBenchName() const { return benchName_; }
    const char* getDir()       const { return dir_;       }

    std::shared_ptr<BookShelfDB> const getDB() { return bookShelfDB_; }

    bool isOutsideDie(BsCell* cell);

  private:

    std::shared_ptr<BookShelfDB> bookShelfDB_;

    void init(const char* aux_name);

    void read_aux();
    void read_nodes();
    void read_pl();
    void read_scl();
    void read_nets();

    int numFixed_;
    int numFixedNI_;
    int numMovable_;

    int maxRowHeight_;

    char benchName_[MAX_FILE_NAME];

    // From .aux
    char dir_  [MAX_FILE_NAME];
    char aux_  [MAX_FILE_NAME];
    char nodes_[MAX_FILE_NAME];
    char nets_ [MAX_FILE_NAME];
    char pl_   [MAX_FILE_NAME];
    char scl_  [MAX_FILE_NAME];
};

} // namespace BookShelf

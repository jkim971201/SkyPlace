#ifndef DB_MTERM_H
#define DB_MTERM_H

#include <string>
#include <vector>

#include "dbTypes.h"
#include "dbLayer.h"
#include "dbMacro.h"

namespace db
{

class dbMacro;
class dbLayer;
class dbMTerm;

// TODO : Make this class as a child-class of dbPolygon
class dbMTermPort
{
  public:

    dbMTermPort()
      : layer_ (nullptr),
        mterm_ (nullptr)
    {}

    // Setters
    void setLayer(dbLayer* layer) { layer_ = layer; }
    void setMTerm(dbMTerm* mterm) { mterm_ = mterm; }
    void addPoint(int newX, int newY) 
    { 
      auto new_point = std::pair<int, int>(newX, newY);
      shape_.push_back(new_point);
    }

    // Getters
    const dbLayer* layer()    const { return layer_; }
    const dbMTerm* getMTerm() const { return mterm_; }
    const std::vector<std::pair<int, int>>& getShape() const { return shape_; }

  private:

    dbLayer* layer_;
    dbMTerm* mterm_;
    std::vector<std::pair<int, int>> shape_;
};

class dbMTerm
{
  public: 

    dbMTerm();

    void print() const;
    void setBoundary(); // Set Boundary Box

    // Setters
    void setName         (std::string  pinName ) { name_     = pinName;    }
    void setMacro        (dbMacro*     lefMacro) { macro_    = lefMacro;   }
    void setPinUsage     (PinUsage     pinUsage) { pinUsage_ = pinUsage;   }
    void setPinDirection (PinDirection pinDir  ) { pinDir_   = pinDir;     }
    void setPinShape     (PinShape     pinShape) { pinShape  = pinShape_;  }
    void addPort         (dbMTermPort*  newPort) { newPort->setMTerm(this); ports_.push_back(newPort); }

    // Getters
    // lx ly ux uy are coordinates of BBox of LEF PIN PORT (dbMTermPort)
    int lx() const { return lx_; }
    int ly() const { return ly_; }
    int ux() const { return ux_; }
    int uy() const { return uy_; }
    int dx() const { return ux_ - lx_; }
    int dy() const { return uy_ - ly_; }
    int cx() const { return (ux_ + lx_) / 2; } // this works as offsetX
    int cy() const { return (uy_ + ly_) / 2; } // this works as offsetY

    dbMacro*          macro() const { return macro_;    }
    const std::string& name() const { return name_;     }
    PinUsage          usage() const { return pinUsage_; }
    PinDirection  direction() const { return pinDir_;   }
    PinShape          shape() const { return pinShape_; }

    const std::vector<dbMTermPort*>& ports() const { return ports_; }

  private:

    dbMacro* macro_;

    std::string  name_;
    PinUsage     pinUsage_;
    PinDirection pinDir_;
    PinShape     pinShape_;

    std::vector<dbMTermPort*> ports_; 

    int lx_;
    int ly_;
    int ux_;
    int uy_;
};

}

#endif

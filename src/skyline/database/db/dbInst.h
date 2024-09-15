#ifndef DB_INST_H
#define DB_INST_H

#include <vector>
#include <cstdlib> // for int64_t
#include <unordered_map>

#include "dbTypes.h"

namespace db
{

class dbMacro;
class dbITerm;

class dbInst
{
  public:

    dbInst();

    void print() const;

    // Setters
    void setName   (const std::string& name) { name_   = name;   }
    void setMacro  (dbMacro* macro)          { macro_  = macro;  }
    void setOrient (const Orient  orient)    { orient_ = orient; }
    void setSource (const Source  source)    { source_ = source; }
    void setStatus (const Status  status)    { status_ = status; }

    void addITerm   (dbITerm* iterm);
    void setLocation(int placementX, int placementY); 
    // placementX, placementY is the point from .def

    // Halo is set to 0 by default
    void setHalo(int hT, int hB, int hL, int hR) 
    {
      haloT_ = hT;
      haloB_ = hB; 
      haloL_ = hL; 
      haloR_ = hR;
    }

    // Getters
          dbMacro* macro()       { return macro_; }
    const dbMacro* macro() const { return macro_; }
    const std::string& name() const { return name_;  }
    const std::vector<dbITerm*>& getITerms() const { return iterms_; }
    int lx() const { return lx_; }
    int ly() const { return ly_; }
    int ux() const { return lx_ + dx_; }
    int uy() const { return ly_ + dy_; }
    int cx() const { return lx_ + dx_ / 2; }
    int cy() const { return ly_ + dy_ / 2; }
    int dx() const { return dx_; }
    int dy() const { return dy_; }
    int haloT() const { return haloT_; }
    int haloB() const { return haloB_; }
    int haloL() const { return haloL_; }
    int haloR() const { return haloR_; }
    int64_t area() const { return static_cast<int64_t>(dx_) 
                                * static_cast<int64_t>(dy_); }

    bool isFixed()   const;
    bool isStdCell() const;
    bool isMacro()   const;
    bool isDummy()   const;
    Orient orient()  const { return orient_; }
    Source source()  const { return source_; }
    Status status()  const { return status_; }

    dbITerm* getITermByMTermName(const std::string& name);

  private:

    dbMacro* macro_;

    std::string name_;

    Orient orient_;
    Source source_;
    Status status_;

    int lx_;
    int ly_;

    int dx_;
    int dy_;

    int haloT_;
    int haloB_;
    int haloL_;
    int haloR_;

    std::vector<dbITerm*> iterms_;
    std::unordered_map<std::string, dbITerm*> itermMap_;
};

}

#endif

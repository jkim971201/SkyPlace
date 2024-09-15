#ifndef DB_SITE_H
#define DB_SITE_H

#include "dbTypes.h"

namespace db
{

class dbSite
{
  public:

    dbSite();

    void print() const;

    // Setters
    void setName(const char* name)  { name_ = std::string(name); }
    void setSiteClass(SiteClass cl) { siteClass_ = cl;  }
    void setSymmetryX   (bool sym)  { symX_   = sym;    }
    void setSymmetryY   (bool sym)  { symY_   = sym;    }
    void setSymmetryR90 (bool sym)  { symR90_ = sym;    }
    void setSizeX(int val)          { sizeX_  = val;    }
    void setSizeY(int val)          { sizeY_  = val;    }

    // Getters
    const std::string& name() const { return name_; }
    SiteClass siteClass() const { return siteClass_; }
    bool isSymmetryX()    const { return symX_;      }
    bool isSymmetryY()    const { return symY_;      }
    bool isSymmetryR90()  const { return symR90_;    }

    int sizeX() const { return sizeX_; }
    int sizeY() const { return sizeY_; }

  private:

    // LEF Syntax
    std::string name_;

    SiteClass siteClass_;

    bool symX_;
    bool symY_;
    bool symR90_;

    int sizeX_;
    int sizeY_;
};

}

#endif

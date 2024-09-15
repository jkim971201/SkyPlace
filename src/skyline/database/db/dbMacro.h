#ifndef DB_MACRO
#define DB_MACRO

#include <string>
#include <vector>

#include "dbMTerm.h"
#include "dbObs.h"
#include "dbTypes.h"

namespace db
{

class dbMTerm;
class dbSite;
class dbObs;

class dbMacro
{

  public:

    dbMacro();
    ~dbMacro();

    void print() const;

    // Setters
    void setName(const char* name)    { name_ = std::string(name); }
    void setMacroClass(MacroClass cl) { macroClass_ = cl;    }
    void setSite (dbSite* site)       { site_   = site;      }
    void setSizeX(int sizeX)          { sizeX_  = sizeX;     }
    void setSizeY(int sizeY)          { sizeY_  = sizeY;     }
    void setOrigX(int origX)          { origX_  = origX;     }
    void setOrigY(int origY)          { origY_  = origY;     }
    void setSymmetryX   (bool sym)    { symX_   = sym;       }
    void setSymmetryY   (bool sym)    { symY_   = sym;       }
    void setSymmetryR90 (bool sym)    { symR90_ = sym;       }
    void addObs(dbObs* obs)           { obs_.push_back(obs); }
    void addMTerm(dbMTerm* mterm);

    // Getters
    MacroClass   macroClass() const { return macroClass_; }
    dbSite*            site() const { return site_;       }
    const std::string& name() const { return name_;       }

    int sizeX() const { return sizeX_; }
    int sizeY() const { return sizeY_; }
    int origX() const { return origX_; }
    int origY() const { return origY_; }

    bool isSymmetryX()   const { return symX_;   }
    bool isSymmetryY()   const { return symY_;   }
    bool isSymmetryR90() const { return symR90_; }

    const std::vector<dbMTerm*>& getMTerms() const { return mterms_; }
    const std::vector<dbObs*>&   getObs()    const { return obs_;    }

    dbMTerm* getMTermByName(const std::string& pinName);

  private:

    // LEF Syntax
    std::string           name_;
    MacroClass            macroClass_;
    dbSite*               site_;
    std::vector<dbMTerm*> mterms_;
    std::vector<dbObs*>   obs_;

    std::unordered_map<std::string, dbMTerm*> mtermMap_;

    int sizeX_;
    int sizeY_;
    
    int origX_;
    int origY_;

    bool symX_;
    bool symY_;
    bool symR90_;
};

}

#endif

#ifndef DB_ITERM_H
#define DB_ITERM_H

#include <string>

#include "dbRect.h"

namespace db
{

class dbNet;
class dbInst;
class dbMTerm;

class dbITerm
{
  public:

    dbITerm();
    dbITerm(const std::string& name, dbInst* inst, dbMTerm* mterm);

    bool isSignal() const;
    void print() const;

		// getBBox
    const dbRect getRect() const;

    // Setters
    void setNet   (dbNet*     net) { net_    = net;    }
    void setInst  (dbInst*   inst) { inst_   = inst;   }
    void setMTerm (dbMTerm* mterm) { mterm_  = mterm;  }
    void setName  (const std::string& name) { name_ = name; }

    // Getters
    const std::string& name() const { return name_; }

          dbNet*   getNet()         { return net_;   }
    const dbNet*   getNet()   const { return net_;   }

          dbInst*  getInst()        { return inst_;  }
    const dbInst*  getInst()  const { return inst_;  }

          dbMTerm* getMTerm()       { return mterm_; }
    const dbMTerm* getMTerm() const { return mterm_; }

  private:

    // Name of dbITerm : 
    // "InstanceName" + "DividerCharacter" + "MTermName" 
    // (this rule is not valid for bookshelf)
    std::string name_;

    dbNet*   net_;
    dbInst*  inst_;
    dbMTerm* mterm_;
};

}

#endif

#ifndef DB_NET_H
#define DB_NET_H

#include "dbTypes.h"
#include "dbWire.h"

#include <vector>
#include <string>

namespace db
{

class dbWire;
class dbITerm;
class dbBTerm;
class dbNonDefaultRule;

class dbNet
{
  public:

    dbNet()
      : name_      (""),
        ndr_       (nullptr),
        wire_      (new dbWire),
        use_       (NetUse::SIGNAL_NET),
        source_    (Source::NETLIST),
        isSpecial_ (false)
    {}

    ~dbNet()
    {
      delete wire_;
    }

    // Setters
    void setName   (const std::string& name) { name_ = name; }
    void setUse    (const NetUse use) { use_    = use; }
    void setSource (const Source src) { source_ = src; }
    void setSpecial() { isSpecial_ = true; }

    void addITerm  (dbITerm* iterm) { iterms_.push_back(iterm); }
    void addBTerm  (dbBTerm* bterm) { bterms_.push_back(bterm); }
    void setNonDefaultRule(dbNonDefaultRule* rule) { ndr_ = rule; }

    // Getters
    const std::string& name() const { return name_; }
    const std::vector<dbITerm*>& getITerms() const { return iterms_; }
    const std::vector<dbBTerm*>& getBTerms() const { return bterms_; }
    
    int numTerms() const { return iterms_.size() + bterms_.size(); }

    NetUse use()    const { return use_;    }
    Source source() const { return source_; }

    const dbWire* getWire() const { return wire_; }
          dbWire* getWire()       { return wire_; }

    bool  isSpecial() const { return isSpecial_; }
    bool  hasNonDefaultRule() const { return ndr_ != nullptr; }
    const dbNonDefaultRule* getNonDefaultRule() const { return ndr_; }

  private:

    std::string name_;
    std::vector<dbITerm*> iterms_;
    std::vector<dbBTerm*> bterms_;

    NetUse use_;
    Source source_;

    bool isSpecial_;
    dbWire* wire_;
    dbNonDefaultRule* ndr_;
};

}

#endif

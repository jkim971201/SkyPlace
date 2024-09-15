#ifndef DB_VIAINST_H
#define DB_VIAINST_H

namespace db
{

class dbViaMaster;

// Instance of a via
class dbViaInst : public dbRect // Rect = BBox of this via
{
  public:

    dbViaInst() 
      : master_   (nullptr), 
        isTechVia_(false),
        botMask_  (0),
        cutMask_  (0),
        topMask_  (0)
    {}

    // Setters
    void setBotMask(int val) { botMask_ = val; }
    void setCutMask(int val) { cutMask_ = val; }
    void setTopMask(int val) { topMask_ = val; }
    void setMaster(dbViaMaster* master) { master_ = master; }

    // Getters
    int botMask() const { return botMask_; }
    int cutMask() const { return cutMask_; }
    int topMask() const { return topMask_; }

    bool isTechVia() const { return isTechVia_; }
    const dbViaMaster* getMaster() const { return master_; }

  private:
  
    int botMask_;
    int cutMask_;
    int topMask_;
    bool isTechVia_;
    dbViaMaster* master_;
};

}

#endif

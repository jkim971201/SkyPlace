#ifndef DB_ROW_H
#define DB_ROW_H

#include "dbSite.h"

namespace db
{

class dbRow
{
  public:

    dbRow()
      : name_      (""),
        site_      (nullptr),
        origX_     (0),
        origY_     (0),
        numSiteX_  (0),
        numSiteY_  (0),
        stepX_     (0),
        stepY_     (0)
    {
      orient_ = Orient::N;
    }

    void print() const;

    int lx() const { return origX_; }
    int ly() const { return origY_; }
    int ux() const { return origX_ + stepX_ * numSiteX_ + siteWidth_;  }
    int uy() const { return origY_ + stepY_ * numSiteY_ + siteHeight_; }
    int dx() const { return numSiteX_ * stepX_ + siteWidth_;  }
    int dy() const { return numSiteY_ * stepY_ + siteHeight_; }

    int siteWidth()  const { return siteWidth_;  }
    int siteHeight() const { return siteHeight_; }

    // Setters
    void setName(const std::string& name) { name_ = name; }
    void setSite(dbSite* site) 
    { 
      site_ = site; 
      siteWidth_  = site_->sizeX();
      siteHeight_ = site_->sizeY();
    }

    // This is only for converting bookshelf Row to dbRow
    void setSiteSize(int w, int h)
    {
      siteWidth_  = w;
      siteHeight_ = h;
    }

    void setOrigX(int val) { origX_ = val; }
    void setOrigY(int val) { origY_ = val; }

    void setNumSiteX(int val) { numSiteX_ = val; }
    void setNumSiteY(int val) { numSiteY_ = val; }

    void setStepX(int val) { stepX_ = val; }
    void setStepY(int val) { stepY_ = val; }

    void setOrient(Orient orient) { orient_ = orient; }

    // Getters
    const std::string& name() const { return name_; }
    const dbSite* site() const { return site_; }

    int origX() const { return origX_; }
    int origY() const { return origY_; }

    int numSiteX() const { return numSiteX_; }
    int numSiteY() const { return numSiteY_; }

    int stepX() const { return stepX_; }
    int stepY() const { return stepY_; }

    Orient orient() const { return orient_; }

  private:

    std::string name_;
    dbSite*     site_;

    int origX_;
    int origY_;

    int numSiteX_;
    int numSiteY_;

    int stepX_;
    int stepY_;

    int siteWidth_;
    int siteHeight_;

    Orient orient_;
};

}

#endif

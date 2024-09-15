#ifndef DB_BLOCKAGE_H
#define DB_BLOCKAGE_H

#include "dbPolygon.h"

namespace db
{

class dbBlockage : public dbPolygon
{
  public:

    dbBlockage() : spacing_(0), isPlacement_(false) {}

    // Getters
    int spacing() const { return spacing_; }
    bool isPlacementBlockage() const { return isPlacement_; }

    // Setters
    void setSpacing(int val) { spacing_ = val; }
    void setPlacementBlockage() { isPlacement_ = true; }

  private:
 
    int spacing_;

    // PLACEMENT Blockage
    bool isPlacement_;
};

}

#endif

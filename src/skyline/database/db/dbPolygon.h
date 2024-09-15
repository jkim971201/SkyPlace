#ifndef DB_POLYGON_H
#define DB_POLYGON_H

#include <limits>
#include <vector>

#include "dbRect.h"

namespace db
{

class dbLayer;

class dbPolygon : public dbRect // dbRect represents bbox of this polygon
{
  public:

    dbPolygon() {}

    void updateBBox()
    {
      int lx = std::numeric_limits<int>::max();
      int ly = std::numeric_limits<int>::max();
      int ux = std::numeric_limits<int>::min();
      int uy = std::numeric_limits<int>::min();

      for(const auto& [x, y] : getShape())
      {
        if(x < lx) lx = x;
        if(y < ly) ly = y;
        if(x > ux) ux = x;
        if(y > uy) uy = y;
      }

      setLx(lx);
      setLy(ly);
      setUx(ux);
      setUy(uy);
    }

    // Setters
    void addPoint(int newX, int newY) 
    { 
      auto new_point = std::pair<int, int>(newX, newY);
      shape_.push_back(new_point);
    }

    // Getters
    const std::vector<std::pair<int, int>>& getShape() const { return shape_; }

  private:

    std::vector<std::pair<int, int>> shape_;
};

}

#endif

#ifndef DB_OBS
#define DB_OBS

#include <vector>

#include "dbLayer.h"
#include "dbMacro.h"

namespace db
{

class dbLayer;
class dbMacro;

class dbObs
{
  public:

    dbObs()
      : layer_ (nullptr),
        macro_ (nullptr)
    {}

    // Setters
    void setLayer(dbLayer* layer) { layer_ = layer; }
    void setMacro(dbMacro* macro) { macro_ = macro; }
    void addPoint(int newX, int newY) 
    { 
      auto new_point = std::pair<int, int>(newX, newY);
      shape_.push_back(new_point);
    }

    // Getters
    const dbLayer* layer()    const { return layer_; }
    const dbMacro* getMacro() const { return macro_; }
    const std::vector<std::pair<int, int>>& getShape() const { return shape_; }

  private:

    dbLayer* layer_;
    dbMacro* macro_;
    std::vector<std::pair<int, int>> shape_;
};

}

#endif
